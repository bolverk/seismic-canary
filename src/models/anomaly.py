"""Rule-based anomaly model for explosion-vs-earthquake assessment.

Implements a transparent, interpretable scoring system that assesses
how consistent an event's characteristics are with an explosion versus
a tectonic earthquake. Each rule's contribution is individually accessible
for transparency.

IMPORTANT: Output is labeled as "experimental anomaly score" — not
a calibrated probability. An explosion-like seismic signature does NOT
establish that an event was nuclear.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class RuleContribution:
    """Contribution of a single rule to the assessment."""
    feature_name: str
    raw_value: Optional[float]
    contribution_earthquake: float
    contribution_explosion: float
    confidence: float
    explanation: str


@dataclass
class Assessment:
    """Complete anomaly assessment for an event.

    This is an EXPERIMENTAL anomaly score, not a calibrated probability.
    """
    earthquake_consistency: float  # 0-1 scale
    explosion_consistency: float  # 0-1 scale
    confidence: float  # How much evidence is available (0-1)
    alert_level: int  # 0=ordinary, 1=unusual, 2=probable explosion
    contributing_features: List[RuleContribution] = field(default_factory=list)
    model_version: str = Config.MODEL_VERSION
    is_experimental: bool = True
    insufficient_data: bool = False
    summary: str = ""


class AnomalyModel(ABC):
    """Abstract base class for anomaly detection models."""

    @abstractmethod
    def assess(self, event: Dict[str, Any]) -> Assessment:
        """Assess an event for explosion-like characteristics.

        Args:
            event: Dictionary with event fields (from DataFrame row).

        Returns:
            Assessment with scores and contributing features.
        """
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Return model version string."""
        ...


class RuleBasedModel(AnomalyModel):
    """Transparent rule-based anomaly model.

    Rules:
    - Depth: shallow events score higher for explosion
    - P/S ratio: high P/S scores higher for explosion
    - mb-Ms: large positive mb-Ms scores higher for explosion
    - Location: distance from known faults (future implementation)

    Each rule contributes independently. The final score is a weighted
    combination of rule contributions.
    """

    VERSION = "baseline-001"

    # Rule weights (sum should equal 1.0 for normalization)
    WEIGHTS = {
        "depth": 0.25,
        "p_s_ratio": 0.25,
        "mb_ms": 0.20,
        "source_mechanism": 0.20,
        "location": 0.10,
    }

    def get_version(self) -> str:
        return self.VERSION

    def assess(self, event: Dict[str, Any]) -> Assessment:
        """Assess an event using rule-based criteria."""
        contributions = []
        total_earthquake = 0.0
        total_explosion = 0.0
        total_confidence = 0.0
        total_weight = 0.0

        # Depth rule
        depth_contrib = self._assess_depth(event.get("depth_km"))
        contributions.append(depth_contrib)
        if depth_contrib.confidence > 0:
            w = self.WEIGHTS["depth"]
            total_earthquake += depth_contrib.contribution_earthquake * w
            total_explosion += depth_contrib.contribution_explosion * w
            total_confidence += depth_contrib.confidence * w
            total_weight += w

        # P/S ratio rule
        ps_contrib = self._assess_p_s_ratio(event.get("p_s_ratio"))
        contributions.append(ps_contrib)
        if ps_contrib.confidence > 0:
            w = self.WEIGHTS["p_s_ratio"]
            total_earthquake += ps_contrib.contribution_earthquake * w
            total_explosion += ps_contrib.contribution_explosion * w
            total_confidence += ps_contrib.confidence * w
            total_weight += w

        # mb-Ms rule
        mbms_contrib = self._assess_mb_ms(event.get("mb_ms"))
        contributions.append(mbms_contrib)
        if mbms_contrib.confidence > 0:
            w = self.WEIGHTS["mb_ms"]
            total_earthquake += mbms_contrib.contribution_earthquake * w
            total_explosion += mbms_contrib.contribution_explosion * w
            total_confidence += mbms_contrib.confidence * w
            total_weight += w

        # Source mechanism rule
        mech_contrib = self._assess_source_mechanism(
            event.get("magnitude_type"), event.get("event_type")
        )
        contributions.append(mech_contrib)
        if mech_contrib.confidence > 0:
            w = self.WEIGHTS["source_mechanism"]
            total_earthquake += mech_contrib.contribution_earthquake * w
            total_explosion += mech_contrib.contribution_explosion * w
            total_confidence += mech_contrib.confidence * w
            total_weight += w

        # Location rule (placeholder - always low confidence for now)
        loc_contrib = self._assess_location(
            event.get("latitude"), event.get("longitude")
        )
        contributions.append(loc_contrib)
        if loc_contrib.confidence > 0:
            w = self.WEIGHTS["location"]
            total_earthquake += loc_contrib.contribution_earthquake * w
            total_explosion += loc_contrib.contribution_explosion * w
            total_confidence += loc_contrib.confidence * w
            total_weight += w

        # Normalize
        if total_weight > 0:
            earthquake_score = total_earthquake / total_weight
            explosion_score = total_explosion / total_weight
            confidence = total_confidence / total_weight
        else:
            earthquake_score = 0.0
            explosion_score = 0.0
            confidence = 0.0

        # Clamp to [0, 1]
        earthquake_score = max(0.0, min(1.0, earthquake_score))
        explosion_score = max(0.0, min(1.0, explosion_score))
        confidence = max(0.0, min(1.0, confidence))

        # Determine alert level
        insufficient_data = confidence < 0.25
        if insufficient_data:
            alert_level = -1  # insufficient data
        elif explosion_score >= Config.ALERT_LEVEL_THRESHOLDS[2]:
            alert_level = 2
        elif explosion_score >= Config.ALERT_LEVEL_THRESHOLDS[1]:
            alert_level = 1
        else:
            alert_level = 0

        # Generate summary
        summary = self._generate_summary(alert_level, explosion_score, contributions)

        return Assessment(
            earthquake_consistency=earthquake_score,
            explosion_consistency=explosion_score,
            confidence=confidence,
            alert_level=alert_level,
            contributing_features=contributions,
            model_version=self.VERSION,
            insufficient_data=insufficient_data,
            summary=summary,
        )

    def _assess_depth(self, depth_km: Optional[float]) -> RuleContribution:
        """Assess depth criterion.

        Explosions are typically shallow (< 5 km), often < 2 km.
        Deep events (> 10 km) are almost certainly tectonic.
        """
        if depth_km is None:
            return RuleContribution(
                feature_name="depth",
                raw_value=None,
                contribution_earthquake=0.0,
                contribution_explosion=0.0,
                confidence=0.0,
                explanation="Depth not available",
            )

        if depth_km < 2.0:
            eq_score = 0.1
            ex_score = 0.9
            explanation = f"Very shallow ({depth_km:.1f} km) — consistent with explosion"
        elif depth_km < 5.0:
            eq_score = 0.3
            ex_score = 0.7
            explanation = f"Shallow ({depth_km:.1f} km) — unusual, possibly explosion-like"
        elif depth_km < 10.0:
            eq_score = 0.6
            ex_score = 0.4
            explanation = f"Moderate depth ({depth_km:.1f} km) — ambiguous"
        else:
            eq_score = 0.9
            ex_score = 0.1
            explanation = f"Deep ({depth_km:.1f} km) — consistent with tectonic earthquake"

        return RuleContribution(
            feature_name="depth",
            raw_value=depth_km,
            contribution_earthquake=eq_score,
            contribution_explosion=ex_score,
            confidence=0.8,
            explanation=explanation,
        )

    def _assess_p_s_ratio(self, p_s_ratio: Optional[float]) -> RuleContribution:
        """Assess P/S amplitude ratio criterion.

        Explosions produce strong P-waves relative to S-waves.
        log10(P/S) > 0.5 is suspicious; < 0 is earthquake-like.
        """
        if p_s_ratio is None:
            return RuleContribution(
                feature_name="p_s_ratio",
                raw_value=None,
                contribution_earthquake=0.0,
                contribution_explosion=0.0,
                confidence=0.0,
                explanation="P/S ratio not available (no waveform data)",
            )

        if p_s_ratio > 0.7:
            eq_score = 0.05
            ex_score = 0.95
            explanation = f"Very high P/S ratio ({p_s_ratio:.2f}) — strongly explosion-like"
        elif p_s_ratio > 0.5:
            eq_score = 0.2
            ex_score = 0.8
            explanation = f"High P/S ratio ({p_s_ratio:.2f}) — explosion-like"
        elif p_s_ratio > 0.2:
            eq_score = 0.4
            ex_score = 0.6
            explanation = f"Elevated P/S ratio ({p_s_ratio:.2f}) — slightly unusual"
        elif p_s_ratio > 0.0:
            eq_score = 0.6
            ex_score = 0.4
            explanation = f"Normal P/S ratio ({p_s_ratio:.2f}) — ambiguous"
        else:
            eq_score = 0.85
            ex_score = 0.15
            explanation = f"Low P/S ratio ({p_s_ratio:.2f}) — consistent with earthquake"

        return RuleContribution(
            feature_name="p_s_ratio",
            raw_value=p_s_ratio,
            contribution_earthquake=eq_score,
            contribution_explosion=ex_score,
            confidence=0.9,
            explanation=explanation,
        )

    def _assess_mb_ms(self, mb_ms: Optional[float]) -> RuleContribution:
        """Assess mb-Ms discriminant.

        Explosions: mb-Ms > 1.0 (strong body waves, weak surface waves)
        Earthquakes: mb-Ms < 0.5
        """
        if mb_ms is None:
            return RuleContribution(
                feature_name="mb_ms",
                raw_value=None,
                contribution_earthquake=0.0,
                contribution_explosion=0.0,
                confidence=0.0,
                explanation="mb-Ms not available",
            )

        if mb_ms > 1.5:
            eq_score = 0.05
            ex_score = 0.95
            explanation = f"Very high mb-Ms ({mb_ms:.2f}) — strongly explosion-like"
        elif mb_ms > 1.0:
            eq_score = 0.15
            ex_score = 0.85
            explanation = f"High mb-Ms ({mb_ms:.2f}) — explosion-like"
        elif mb_ms > 0.5:
            eq_score = 0.4
            ex_score = 0.6
            explanation = f"Elevated mb-Ms ({mb_ms:.2f}) — slightly unusual"
        else:
            eq_score = 0.8
            ex_score = 0.2
            explanation = f"Normal mb-Ms ({mb_ms:.2f}) — consistent with earthquake"

        return RuleContribution(
            feature_name="mb_ms",
            raw_value=mb_ms,
            contribution_earthquake=eq_score,
            contribution_explosion=ex_score,
            confidence=0.85,
            explanation=explanation,
        )

    def _assess_source_mechanism(
        self, magnitude_type: Optional[str], event_type: Optional[str]
    ) -> RuleContribution:
        """Assess source mechanism from magnitude type and event classification.

        A moment tensor solution (magnitude type mwr, mww, mwb, mwc) implies
        the source has a double-couple (shear fault) mechanism, which is
        characteristic of earthquakes. Explosions produce isotropic sources
        that do NOT yield clean moment tensor solutions.

        Additionally, if the event has been reviewed and classified as
        'earthquake' by the USGS, that carries weight.
        """
        # Magnitude types that imply a moment tensor was solved
        moment_tensor_types = {"mwr", "mww", "mwb", "mwc", "mw"}

        has_moment_tensor = (
            magnitude_type is not None
            and magnitude_type.lower() in moment_tensor_types
        )

        # Event type from catalog review
        is_classified_earthquake = (
            event_type is not None and event_type.lower() == "earthquake"
        )
        is_classified_explosion = (
            event_type is not None
            and event_type.lower() in ("explosion", "nuclear explosion", "quarry blast",
                                        "mining explosion", "chemical explosion")
        )

        if has_moment_tensor:
            # Double-couple source confirmed — strongly favors earthquake
            eq_score = 0.9
            ex_score = 0.1
            confidence = 0.85
            explanation = (
                f"Moment tensor solution available (magType={magnitude_type}) "
                f"— double-couple source consistent with earthquake"
            )
        elif is_classified_explosion:
            eq_score = 0.1
            ex_score = 0.9
            confidence = 0.9
            explanation = f"Catalog classification: {event_type}"
        elif is_classified_earthquake and magnitude_type in ("mb", "ml", "md"):
            # Body-wave or local magnitude without tensor — mild earthquake indicator
            eq_score = 0.6
            ex_score = 0.4
            confidence = 0.3
            explanation = (
                f"Classified as earthquake, magType={magnitude_type} "
                f"(no moment tensor — weak constraint)"
            )
        else:
            # No information
            return RuleContribution(
                feature_name="source_mechanism",
                raw_value=None,
                contribution_earthquake=0.0,
                contribution_explosion=0.0,
                confidence=0.0,
                explanation="No source mechanism information available",
            )

        return RuleContribution(
            feature_name="source_mechanism",
            raw_value=magnitude_type,
            contribution_earthquake=eq_score,
            contribution_explosion=ex_score,
            confidence=confidence,
            explanation=explanation,
        )

    def _assess_location(
        self, latitude: Optional[float], longitude: Optional[float]
    ) -> RuleContribution:
        """Assess location relative to known seismicity.

        Placeholder: future implementation will use known fault database.
        For now, returns low-confidence neutral assessment.
        """
        if latitude is None or longitude is None:
            return RuleContribution(
                feature_name="location",
                raw_value=None,
                contribution_earthquake=0.0,
                contribution_explosion=0.0,
                confidence=0.0,
                explanation="Location not available",
            )

        # Placeholder: neutral assessment with low confidence
        # TODO: Implement distance-to-fault calculation using
        # publicly available fault database for Iran region
        return RuleContribution(
            feature_name="location",
            raw_value=None,
            contribution_earthquake=0.5,
            contribution_explosion=0.5,
            confidence=0.2,
            explanation="Location assessment not yet implemented (placeholder)",
        )

    def _generate_summary(
        self,
        alert_level: int,
        explosion_score: float,
        contributions: List[RuleContribution],
    ) -> str:
        """Generate a human-readable assessment summary."""
        if alert_level == -1:
            return "Insufficient data for assessment."

        active_features = [c for c in contributions if c.confidence > 0]

        if alert_level == 0:
            return "Seismic characteristics consistent with ordinary tectonic earthquake."
        elif alert_level == 1:
            unusual = [c for c in active_features if c.contribution_explosion > 0.6]
            feature_names = [c.feature_name for c in unusual]
            return (
                f"Unusual seismic event. Anomalous features: "
                f"{', '.join(feature_names) if feature_names else 'multiple indicators'}."
            )
        elif alert_level == 2:
            return (
                "Seismic characteristics favor an explosion over a tectonic earthquake. "
                "Multiple independent indicators are anomalous."
            )
        else:
            return "Corroborated anomaly requiring multi-sensor confirmation."
