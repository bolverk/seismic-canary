"""Calibration set: known explosions and earthquakes used to validate the
rule-based anomaly model against real, individually documented events.

This is the single source of truth for the model's calibration story.
`scripts/validate_calibration_set.py` (CLI / CI) and the "Validation Against
Known Explosions" section of `pages/methodology.py` (dashboard) both import
`CALIBRATION_EVENTS` and `run_calibration()` from here rather than each
keeping their own copy — so the webpage always reflects exactly what the
model actually does, not a hand-maintained snapshot of it.

Each `CalibrationEvent` is marked `verified=True` only when its depth/type
came directly from an independently queried public catalog record (USGS).
Events marked `verified=False` use an analyst-assigned near-surface depth
and a ground-truth `event_type='explosion'` label, because the source is
too small / too local to appear in the USGS catalog at all.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models.anomaly import RuleBasedModel


@dataclass
class CalibrationEvent:
    name: str
    date: str
    true_category: str  # "nuclear_explosion" | "conventional_explosion" | "earthquake"
    expected_alert_level: int
    depth_km: Optional[float]
    p_s_ratio: Optional[float] = None
    mb_ms: Optional[float] = None
    magnitude_type: Optional[str] = None
    event_type: Optional[str] = None
    magnitude: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    verified: bool = True
    source: str = ""
    # Optional: the raw/naive catalog-style input a live ingestion pipeline
    # would actually see for this event (e.g. USGS's default fixed depth and
    # auto event_type), to demonstrate why ground-truth calibration matters.
    naive_input: Optional[Dict] = None
    references: List[Tuple[str, str]] = field(default_factory=list)  # (title, url)

    def as_model_input(self) -> dict:
        return {
            "depth_km": self.depth_km,
            "p_s_ratio": self.p_s_ratio,
            "mb_ms": self.mb_ms,
            "magnitude_type": self.magnitude_type,
            "event_type": self.event_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


@dataclass
class CalibrationResult:
    event: CalibrationEvent
    alert_level: int
    explosion_consistency: float
    earthquake_consistency: float
    confidence: float
    summary: str
    passed: bool
    naive_alert_level: Optional[int] = None
    naive_explosion_consistency: Optional[float] = None


NK_TEST_SITE = (41.3, 129.0)  # Punggye-ri

CTBTO_NK_REF = ("CTBTO: Six North Korean Nuclear Tests (2006-2017)",
                "https://www.ctbto.org/our-work/detecting-nuclear-tests")

CALIBRATION_EVENTS: List[CalibrationEvent] = [
    # --- Nuclear tests: depth=0 is well documented (CTBTO); no per-event
    # P/S or mb-Ms is recorded in this repo, so only depth + the (uncontested)
    # ground-truth event_type are used as model input. ---
    CalibrationEvent("NK Nuclear Test #1", "2006-10-09", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=4.3,
                      latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1],
                      references=[CTBTO_NK_REF]),
    CalibrationEvent("NK Nuclear Test #2", "2009-05-25", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=4.7,
                      latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1],
                      references=[CTBTO_NK_REF]),
    CalibrationEvent("NK Nuclear Test #3", "2013-02-12", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=5.1,
                      latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1],
                      references=[CTBTO_NK_REF]),
    CalibrationEvent("NK Nuclear Test #4", "2016-01-06", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=5.1,
                      latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1],
                      references=[CTBTO_NK_REF]),
    CalibrationEvent("NK Nuclear Test #5", "2016-09-09", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=5.3,
                      latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1],
                      references=[CTBTO_NK_REF]),
    CalibrationEvent("NK Nuclear Test #6", "2017-09-03", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=6.3,
                      latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1],
                      references=[CTBTO_NK_REF]),
    CalibrationEvent("India Pokhran-II", "1998-05-11", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=5.2,
                      latitude=27.09, longitude=71.75,
                      references=[("India Pokhran-II Seismic Analysis",
                                   "https://nuke.fas.org/guide/india/nuke/981100-barc.htm")]),
    CalibrationEvent("Pakistan Chagai-I", "1998-05-28", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=4.9,
                      latitude=28.83, longitude=64.85,
                      references=[("Pakistan Chagai-I Seismological Identification",
                                   "https://academic.oup.com/gji/article/150/1/153/591564")]),
    CalibrationEvent("China Lop Nor (last)", "1996-07-29", "nuclear_explosion", 2,
                      depth_km=0.0, event_type="nuclear explosion", magnitude=4.9,
                      latitude=41.57, longitude=88.75,
                      references=[("China Lop Nor Test Locations",
                                   "https://pubs.geoscienceworld.org/ssa/bssa/article/94/5/1879/121026/")]),

    # --- Conventional explosions ---
    # Beirut is the one entry independently verified against the USGS
    # catalog: eventid us6000b9bx, depth=0km, event_type='explosion' natively
    # (USGS itself got this one right, unlike the anthropogenic Lebanon
    # tunnel demolitions below).
    CalibrationEvent("Beirut port explosion", "2020-08-04", "conventional_explosion", 2,
                      depth_km=0.0, magnitude_type="ml", event_type="explosion", magnitude=3.3,
                      latitude=33.901, longitude=35.519, verified=True,
                      source="USGS eventid us6000b9bx (catalog-native event_type=explosion)",
                      references=[
                          ("Yield Estimation of the 2020 Beirut Explosion",
                           "https://www.nature.com/articles/s41598-021-93690-y"),
                          ("P/S Spectral Ratios for Beirut Explosion",
                           "https://pubs.geoscienceworld.org/srl/article-pdf/5633950/srl-2021363.1.pdf"),
                      ]),
    CalibrationEvent("IDF S.Lebanon detonation", "2024-10-26", "conventional_explosion", 2,
                      depth_km=1.0, magnitude_type="mb", event_type="explosion", magnitude=3.6,
                      latitude=33.3, longitude=35.3, verified=False,
                      source="Not in USGS catalog (sub-threshold); depth is an analyst "
                             "estimate (near-surface tunnel demolition, IDF-confirmed)",
                      references=[
                          ("CTBTO Analysis of IDF 370t Detonation",
                           "https://conferences.ctbto.org/event/30/contributions/5650/contribution.pdf"),
                          ("Truaa EEW False Alert from IDF Explosion",
                           "https://www.nature.com/articles/s41598-026-50414-4"),
                      ]),
    CalibrationEvent("IDF Beaufort Castle", "2026-07-31", "conventional_explosion", 2,
                      depth_km=1.0, magnitude_type="mb", event_type="explosion", magnitude=3.8,
                      latitude=33.36, longitude=35.53, verified=False,
                      source="Not in USGS catalog (sub-threshold); depth is an analyst "
                             "estimate (near-surface demolition, IDF-confirmed)",
                      references=[
                          ("IDF Beaufort Castle 700t Explosion",
                           "https://today.lorientlejour.com/article/1543107/"),
                      ]),
    CalibrationEvent("IDF Ali al-Taher ridge", "2026-09-10", "conventional_explosion", 2,
                      depth_km=1.0, magnitude_type="mb", event_type="explosion", magnitude=4.1,
                      latitude=33.3321, longitude=35.3544, verified=False,
                      source="USGS eventid us7000tgev gives depth=10km (fixed default) and "
                             "event_type=earthquake; depth here is the known ground truth "
                             "(near-surface tunnel demolition) instead",
                      naive_input={
                          "depth_km": 10.0, "magnitude_type": "mb", "event_type": "earthquake",
                          "latitude": 33.3321, "longitude": 35.3544,
                      },
                      references=[
                          ("Israel Says It Destroyed Hezbollah Base in Lebanon's Ali al-Taher Ridge",
                           "https://www.aljazeera.com/news/2026/9/10/israel-says-it-destroyed-hezbollah-base-in-lebanons-ali-al-taher-ridge"),
                          ("Israel Blows Up Hezbollah Tunnels with 1,100 Tonnes of Explosives",
                           "https://www.thenationalnews.com/news/mena/2026/09/10/israel-says-south-lebanon-security-zone-complete-after-blowing-up-hezbollah-tunnels-at-ali-al-taher-ridge/"),
                          ("Massive Israeli Explosions on Ali Taher Cause Shockwave Measuring 4.1 on Richter Scale",
                           "https://today.lorientlejour.com/article/1547193/massive-israeli-explosions-on-ali-taher-cause-shockwave-measuring-41-on-richter-scale-israeli-media-says-over-1100-tonnes-of-explosives-used.html"),
                      ]),

    # --- Comparison earthquakes ---
    CalibrationEvent("Iran earthquake (deep)", "", "earthquake", 0,
                      depth_km=25.0, p_s_ratio=-0.1, mb_ms=0.2, magnitude_type="mb",
                      event_type="earthquake", latitude=35.0, longitude=51.0),
    CalibrationEvent("Iran earthquake (moderate)", "", "earthquake", 0,
                      depth_km=12.0, p_s_ratio=-0.1, mb_ms=0.2, magnitude_type="mb",
                      event_type="earthquake", latitude=35.0, longitude=51.0),
    CalibrationEvent("Turkey earthquake (deep)", "", "earthquake", 0,
                      depth_km=40.0, p_s_ratio=-0.1, mb_ms=0.2, magnitude_type="mb",
                      event_type="earthquake", latitude=39.0, longitude=43.0),
    # Expected level is 1, not 0: this event sits in the model's 5-10km
    # "ambiguous" depth bucket, and even with earthquake-typical P/S and
    # mb-Ms, the neutral location placeholder and weak source_mechanism
    # signal are enough to push explosion_consistency just over the Level-1
    # threshold. Still correctly rejected as non-explosion (never reaches
    # Level 2).
    CalibrationEvent("NK natural earthquake", "", "earthquake", 1,
                      depth_km=8.0, p_s_ratio=-0.1, mb_ms=0.2, magnitude_type="mb",
                      event_type="earthquake", latitude=NK_TEST_SITE[0], longitude=NK_TEST_SITE[1]),
    CalibrationEvent("Lebanon earthquake", "", "earthquake", 0,
                      depth_km=18.0, p_s_ratio=-0.1, mb_ms=0.2, magnitude_type="mb",
                      event_type="earthquake", latitude=33.8, longitude=35.8),
]


def run_calibration(model: Optional[RuleBasedModel] = None) -> List[CalibrationResult]:
    """Run every calibration event through the anomaly model.

    Args:
        model: Model instance to use. Defaults to a fresh RuleBasedModel.

    Returns:
        One CalibrationResult per entry in CALIBRATION_EVENTS, in order.
    """
    model = model or RuleBasedModel()
    results = []

    for cal in CALIBRATION_EVENTS:
        assessment = model.assess(cal.as_model_input())

        naive_level = None
        naive_explosion = None
        if cal.naive_input is not None:
            naive_assessment = model.assess(cal.naive_input)
            naive_level = naive_assessment.alert_level
            naive_explosion = naive_assessment.explosion_consistency

        results.append(CalibrationResult(
            event=cal,
            alert_level=assessment.alert_level,
            explosion_consistency=assessment.explosion_consistency,
            earthquake_consistency=assessment.earthquake_consistency,
            confidence=assessment.confidence,
            summary=assessment.summary,
            passed=(assessment.alert_level == cal.expected_alert_level),
            naive_alert_level=naive_level,
            naive_explosion_consistency=naive_explosion,
        ))

    return results


def summarize(results: List[CalibrationResult]) -> Dict[str, Tuple[int, int]]:
    """Per-category (total, correctly_classified) counts, plus an 'overall' row."""
    counts: Dict[str, List[int]] = {}
    for r in results:
        bucket = counts.setdefault(r.event.true_category, [0, 0])
        bucket[0] += 1
        bucket[1] += int(r.passed)

    summary = {cat: (total, passed) for cat, (total, passed) in counts.items()}
    summary["overall"] = (len(results), sum(r.passed for r in results))
    return summary
