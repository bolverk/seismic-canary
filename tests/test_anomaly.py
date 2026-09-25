"""Tests for rule-based anomaly model."""
import pytest

from src.models.anomaly import RuleBasedModel, AnomalyModel


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


class TestSourceMechanism:
    def test_moment_tensor_magtype_lowercase(self, model):
        """Moment-tensor magnitude types strongly favor earthquake."""
        event = {
            "depth_km": 10.0, "p_s_ratio": None, "mb_ms": None,
            "latitude": 35.0, "longitude": 51.0,
            "magnitude_type": "mww", "event_type": "earthquake",
        }
        result = model.assess(event)
        mech = [c for c in result.contributing_features if c.feature_name == "source_mechanism"][0]
        assert mech.contribution_earthquake > 0.8

    def test_uppercase_body_wave_magtype_is_weak_constraint(self, model):
        """Regression: 'ML' (uppercase) classified earthquake must yield the
        weak body-wave constraint, not 'no source mechanism information'.
        Previously magnitude_type was only lower-cased in the moment-tensor
        branch, so an uppercase body-wave magType fell through to the
        'no information' branch."""
        event = {
            "depth_km": 25.0, "p_s_ratio": -0.2, "mb_ms": 0.3,
            "latitude": 35.0, "longitude": 51.0,
            "magnitude_type": "ML", "event_type": "earthquake",
        }
        result = model.assess(event)
        mech = [c for c in result.contributing_features if c.feature_name == "source_mechanism"][0]
        assert mech.confidence > 0, "Upper-case body-wave magType must be recognized"
        assert mech.contribution_earthquake > mech.contribution_explosion
        assert "not available" not in mech.explanation.lower()

    def test_classified_explosion_strong(self, model):
        """Catalog-classified explosion strongly favors explosion."""
        event = {
            "depth_km": 2.0, "p_s_ratio": None, "mb_ms": None,
            "latitude": 34.0, "longitude": 55.0,
            "magnitude_type": "mb", "event_type": "nuclear explosion",
        }
        result = model.assess(event)
        mech = [c for c in result.contributing_features if c.feature_name == "source_mechanism"][0]
        assert mech.contribution_explosion > 0.8


class TestLocationRule:
    def test_near_site_of_interest_leans_explosion(self, model):
        """An event co-located with a monitored site of interest is nudged
        toward explosion-like to surface for review."""
        # Within ~0.01 deg (~1 km) of Natanz (33.717, 51.717)
        event = {
            "depth_km": None, "p_s_ratio": None, "mb_ms": None,
            "latitude": 33.72, "longitude": 51.72,
        }
        result = model.assess(event)
        loc = [c for c in result.contributing_features if c.feature_name == "location"][0]
        assert loc.confidence > 0
        assert loc.contribution_explosion > loc.contribution_earthquake
        assert "Natanz" in loc.explanation

    def test_far_from_site_is_neutral(self, model):
        """An event far from any monitored site stays neutral."""
        # (38.0, 53.0) is ~300 km from the nearest site (Semnan)
        event = {
            "depth_km": None, "p_s_ratio": None, "mb_ms": None,
            "latitude": 38.0, "longitude": 53.0,
        }
        result = model.assess(event)
        loc = [c for c in result.contributing_features if c.feature_name == "location"][0]
        assert loc.contribution_explosion == loc.contribution_earthquake

    def test_missing_location_zero_confidence(self, model):
        """Missing coordinates produce zero-confidence location contribution."""
        event = {
            "depth_km": 10.0, "p_s_ratio": None, "mb_ms": None,
            "latitude": None, "longitude": None,
        }
        result = model.assess(event)
        loc = [c for c in result.contributing_features if c.feature_name == "location"][0]
        assert loc.confidence == 0


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
