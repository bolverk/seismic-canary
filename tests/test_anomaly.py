"""Tests for rule-based anomaly model."""
import pytest

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
