"""Tests for rule-based anomaly model."""
import pytest

from src.config import Config
from src.models.anomaly import RuleBasedModel, Assessment, AnomalyModel


@pytest.fixture
def model():
    return RuleBasedModel()


class TestInterface:
    def test_is_anomaly_model(self, model):
        assert isinstance(model, AnomalyModel)

    def test_has_version(self, model):
        assert model.get_version() == "baseline-001"


class TestTypicalEarthquake:
    def test_deep_event_level_0(self, model):
        """Deep event near fault with normal P/S → Level 0."""
        event = {
            "depth_km": 25.0,
            "p_s_ratio": -0.2,
            "mb_ms": 0.3,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)
        assert result.alert_level == 0
        assert result.earthquake_consistency > result.explosion_consistency
        assert result.confidence > 0

    def test_moderate_depth_normal_ps(self, model):
        """Moderate depth, normal P/S → Level 0."""
        event = {
            "depth_km": 12.0,
            "p_s_ratio": 0.0,
            "mb_ms": 0.2,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)
        assert result.alert_level == 0


class TestSuspiciousEvent:
    def test_shallow_high_ps(self, model):
        """Shallow event with high P/S → Level 1 or 2."""
        event = {
            "depth_km": 1.5,
            "p_s_ratio": 0.7,
            "mb_ms": 1.3,
            "latitude": 34.5,
            "longitude": 55.2,
        }
        result = model.assess(event)
        assert result.alert_level >= 1
        assert result.explosion_consistency > result.earthquake_consistency

    def test_very_shallow_only(self, model):
        """Very shallow but no waveform data → Level 0 or 1."""
        event = {
            "depth_km": 0.5,
            "p_s_ratio": None,
            "mb_ms": None,
            "latitude": 34.5,
            "longitude": 55.2,
        }
        result = model.assess(event)
        # With only depth, can still flag as unusual
        assert result.alert_level >= 0

    def test_high_mb_ms_only(self, model):
        """High mb-Ms but moderate depth."""
        event = {
            "depth_km": 8.0,
            "p_s_ratio": None,
            "mb_ms": 1.5,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)
        # Should see elevated explosion score
        assert result.explosion_consistency > 0.3


class TestInsufficientData:
    def test_all_none(self, model):
        """All features missing → insufficient data."""
        event = {
            "depth_km": None,
            "p_s_ratio": None,
            "mb_ms": None,
            "latitude": None,
            "longitude": None,
        }
        result = model.assess(event)
        assert result.insufficient_data is True
        assert result.alert_level == -1

    def test_only_location(self, model):
        """Only location available (low confidence) → insufficient data."""
        event = {
            "depth_km": None,
            "p_s_ratio": None,
            "mb_ms": None,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)
        # Location alone has very low confidence
        assert result.insufficient_data is True


class TestTransparency:
    def test_contributing_features_accessible(self, model):
        """All intermediate scores should be accessible."""
        event = {
            "depth_km": 3.0,
            "p_s_ratio": 0.4,
            "mb_ms": 0.8,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)

        assert len(result.contributing_features) == 5
        feature_names = [f.feature_name for f in result.contributing_features]
        assert "depth" in feature_names
        assert "p_s_ratio" in feature_names
        assert "mb_ms" in feature_names
        assert "source_mechanism" in feature_names
        assert "location" in feature_names

    def test_each_rule_has_explanation(self, model):
        event = {
            "depth_km": 3.0,
            "p_s_ratio": 0.4,
            "mb_ms": None,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)

        for contrib in result.contributing_features:
            assert contrib.explanation != ""

    def test_model_version_in_assessment(self, model):
        event = {"depth_km": 10.0, "latitude": 35.0, "longitude": 51.0,
                 "p_s_ratio": None, "mb_ms": None}
        result = model.assess(event)
        assert result.model_version == "baseline-001"

    def test_is_experimental_flag(self, model):
        event = {"depth_km": 10.0, "latitude": 35.0, "longitude": 51.0,
                 "p_s_ratio": None, "mb_ms": None}
        result = model.assess(event)
        assert result.is_experimental is True

    def test_summary_generated(self, model):
        event = {
            "depth_km": 1.0,
            "p_s_ratio": 0.8,
            "mb_ms": 1.5,
            "latitude": 35.0,
            "longitude": 51.0,
        }
        result = model.assess(event)
        assert len(result.summary) > 0


class TestMissingDataIsNotAMeasurement:
    """NaN must behave exactly like ``None``.

    Parquet stores numeric features as float64, so "no P/S ratio" comes back
    from a save/load as ``NaN`` rather than ``None``. A bare ``is None`` check
    lets ``NaN`` fall through every ``>`` comparison and land in the final
    branch, silently scoring missing data as a real *low* P/S ratio and
    inflating earthquake consistency for every un-enriched event.
    """

    BASE = {"depth_km": 10.0, "latitude": 35.0, "longitude": 51.0,
            "p_s_ratio": None, "mb_ms": None}

    def _rule(self, result, name):
        return next(c for c in result.contributing_features if c.feature_name == name)

    def test_nan_p_s_ratio_rule_is_skipped(self, model):
        result = model.assess({**self.BASE, "p_s_ratio": float("nan")})
        ps = self._rule(result, "p_s_ratio")
        assert ps.confidence == 0.0
        assert ps.raw_value is None
        assert "nan" not in ps.explanation

    def test_nan_p_s_scores_same_as_none(self, model):
        as_none = model.assess({**self.BASE, "p_s_ratio": None})
        as_nan = model.assess({**self.BASE, "p_s_ratio": float("nan")})
        assert as_nan.earthquake_consistency == as_none.earthquake_consistency
        assert as_nan.explosion_consistency == as_none.explosion_consistency
        assert as_nan.alert_level == as_none.alert_level

    def test_nan_p_s_does_not_shift_toward_earthquake(self, model):
        # The pre-fix behaviour: NaN scored 0.85/0.15 at confidence 0.9,
        # pushing un-enriched events towards "earthquake".
        base = model.assess({**self.BASE, "p_s_ratio": None})
        nan = model.assess({**self.BASE, "p_s_ratio": float("nan")})
        assert nan.earthquake_consistency == pytest.approx(base.earthquake_consistency)
        low_but_real = model.assess({**self.BASE, "p_s_ratio": -0.5})
        assert low_but_real.earthquake_consistency > nan.earthquake_consistency

    def test_nan_depth_is_skipped(self, model):
        with_none = model.assess({**self.BASE, "depth_km": None})
        with_nan = model.assess({**self.BASE, "depth_km": float("nan")})
        assert self._rule(with_nan, "depth").confidence == 0.0
        assert with_nan.alert_level == with_none.alert_level
        assert with_nan.confidence == with_none.confidence

    def test_nan_mb_ms_is_skipped(self, model):
        result = model.assess({**self.BASE, "mb_ms": float("nan")})
        assert self._rule(result, "mb_ms").confidence == 0.0


class TestLevel2NeedsCorroboration:
    """"Probable explosion" must not rest on catalog depth alone.

    Regional depths are often a near-surface default rather than a measured
    value, so a very shallow depth with no P/S, mb-Ms or source-mechanism
    evidence is an incomplete assessment, not a confirmed explosion.
    """

    def test_depth_only_is_held_at_level_1(self, model):
        event = {
            "depth_km": 0.5, "p_s_ratio": None, "mb_ms": None,
            "latitude": 36.85, "longitude": 49.51,
        }
        result = model.assess(event)
        assert result.explosion_consistency > Config.ALERT_LEVEL_THRESHOLDS[2]
        assert result.alert_level == 1

    def test_catalog_earthquake_does_not_corroborate(self, model):
        # Magnitude type and event type are present, but they argue *against*
        # an explosion — that is not corroboration.
        event = {
            "depth_km": 1.0, "p_s_ratio": None, "mb_ms": None,
            "magnitude_type": "ml", "event_type": "earthquake",
            "latitude": 36.85, "longitude": 49.51,
        }
        result = model.assess(event)
        assert result.explosion_consistency > Config.ALERT_LEVEL_THRESHOLDS[2]
        assert result.alert_level == 1

    def test_high_ps_reaches_level_2(self, model):
        event = {
            "depth_km": 1.5, "p_s_ratio": 0.8, "mb_ms": None,
            "latitude": 34.5, "longitude": 55.2,
        }
        assert model.assess(event).alert_level == 2

    def test_high_mb_ms_reaches_level_2(self, model):
        event = {
            "depth_km": 1.0, "p_s_ratio": None, "mb_ms": 1.5,
            "latitude": 34.5, "longitude": 55.2,
        }
        assert model.assess(event).alert_level == 2

    def test_classified_explosion_reaches_level_2(self, model):
        # Mirrors the calibration set: catalog event_type=explosion is an
        # independent, explosion-leaning signal.
        event = {
            "depth_km": 0.0, "p_s_ratio": None, "mb_ms": None,
            "event_type": "explosion", "magnitude_type": "ml",
            "latitude": 33.901, "longitude": 35.519,
        }
        assert model.assess(event).alert_level == 2

    def test_moment_tensor_blocks_level_2(self, model):
        # A solved moment tensor is a strong earthquake constraint, so it can
        # never corroborate an explosion even alongside a very shallow depth.
        event = {
            "depth_km": 0.5, "p_s_ratio": None, "mb_ms": None,
            "magnitude_type": "mww", "event_type": "earthquake",
            "latitude": 38.0, "longitude": 42.0,
        }
        assert model.assess(event).alert_level < 2

    def test_flag_can_be_disabled(self, model, monkeypatch):
        monkeypatch.setattr(Config, "ALERT_LEVEL2_REQUIRE_CORROBORATION", False)
        event = {
            "depth_km": 0.5, "p_s_ratio": None, "mb_ms": None,
            "latitude": 36.85, "longitude": 49.51,
        }
        assert model.assess(event).alert_level == 2

    def test_summary_explains_the_hold(self, model):
        event = {
            "depth_km": 1.0, "p_s_ratio": None, "mb_ms": None,
            "magnitude_type": "m", "event_type": "earthquake",
            "latitude": 36.85, "longitude": 49.51,
        }
        result = model.assess(event)
        assert result.alert_level == 1
        assert "depth alone" in result.summary


class TestKnownExplosionSites:
    """Location watchlist: proximity to a site with confirmed explosions.

    The south-Lebanon demolition campaign is the case that motivated this:
    its events are catalogued as earthquakes at an 8-10 km fallback depth,
    yet sit within ~15 km of sites where explosions were independently
    confirmed. Ordinary shallow seismicity (e.g. Iranian events) is 80-200 km
    from any such site and must not be promoted.
    """

    BEIRUT = (33.901, 35.519)
    ALI_AL_TAHER = (33.3321, 35.3544)
    IRAN = (36.85, 49.51)

    @staticmethod
    def _loc(result):
        return next(
            c for c in result.contributing_features if c.feature_name == "location"
        )

    def test_haversine_known_distance(self):
        from src.models.anomaly import _haversine_km

        assert _haversine_km(33.901, 35.519, 33.901, 35.519) == 0.0
        # Beirut -> Ali al-Taher is roughly 66 km.
        d = _haversine_km(*self.BEIRUT, *self.ALI_AL_TAHER)
        assert 60.0 < d < 72.0

    def test_location_near_site_leans_explosion(self, model):
        result = model.assess(
            {"depth_km": 6.0, "latitude": self.ALI_AL_TAHER[0],
             "longitude": self.ALI_AL_TAHER[1]}
        )
        loc = self._loc(result)
        assert loc.confidence > 0
        assert loc.contribution_explosion > loc.contribution_earthquake

    def test_location_far_from_sites_is_neutral_placeholder(self, model):
        result = model.assess(
            {"depth_km": 6.0, "latitude": self.IRAN[0], "longitude": self.IRAN[1]}
        )
        loc = self._loc(result)
        assert loc.contribution_explosion == loc.contribution_earthquake

    def test_event_just_outside_radius_is_neutral(self, model):
        # ~33 km north of Beirut, well inside Lebanon but outside the radius,
        # and not near any other watchlist site.
        result = model.assess(
            {"depth_km": 6.0, "latitude": 34.20, "longitude": 35.52}
        )
        loc = self._loc(result)
        assert loc.contribution_explosion == loc.contribution_earthquake

    def test_near_site_reaches_level_2_despite_moderate_depth(self, model):
        # The 2024-10-26 detention shape: catalogued earthquake, unconstrained
        # 8 km depth, but ~2 km from a confirmed site.
        event = {
            "depth_km": 8.0, "event_type": "earthquake", "magnitude_type": "ml",
            "latitude": 33.35, "longitude": 35.36,
        }
        assert model.assess(event).alert_level == 2

    def test_near_site_but_deep_is_not_promoted(self, model):
        event = {
            "depth_km": 15.0, "event_type": "earthquake", "magnitude_type": "ml",
            "latitude": 33.35, "longitude": 35.36,
        }
        assert model.assess(event).alert_level < 2

    def test_shallow_far_from_sites_stays_level_1(self, model):
        # A very shallow event in Iran is still only "unusual": depth alone
        # never corroborates an explosion.
        event = {"depth_km": 0.5, "latitude": self.IRAN[0], "longitude": self.IRAN[1]}
        assert model.assess(event).alert_level == 1

    def test_classified_explosion_overrides_default_depth(self, model):
        # EMSC catalogued this as a chemical explosion but handed it a fixed
        # 10 km depth; the classification must still reach Level 2.
        event = {
            "depth_km": 10.0, "event_type": "chemical explosion",
            "magnitude_type": "mb",
            "latitude": 33.33, "longitude": 35.33,
        }
        assert model.assess(event).alert_level == 2

    def test_classified_explosion_works_far_from_sites(self, model):
        event = {
            "depth_km": 10.0, "event_type": "explosion", "magnitude_type": "mb",
            "latitude": self.IRAN[0], "longitude": self.IRAN[1],
        }
        assert model.assess(event).alert_level == 2

    def test_deep_event_far_from_sites_stays_ordinary(self, model):
        event = {"depth_km": 30.0, "latitude": self.IRAN[0], "longitude": self.IRAN[1]}
        assert model.assess(event).alert_level == 0
