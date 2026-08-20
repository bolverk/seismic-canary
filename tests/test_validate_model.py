"""Tests for the model-validation script.

Covers the pure, offline units of ``validate_model.py``: event labeling,
metric computation, and report generation.
"""
import importlib.util
import pathlib

import pandas as pd

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_model = _load_script("validate_model")


def _sample_df():
    return pd.DataFrame([
        {"event_id": "a", "event_type": "earthquake", "confidence": 0.9,
         "predicted_explosion_score": 0.2, "true_label": 0},
        {"event_id": "b", "event_type": "quarry blast", "confidence": 0.9,
         "predicted_explosion_score": 0.8, "true_label": 1},
    ])


class TestLabelEvents:
    def test_explosion_types_labeled_one(self):
        events = pd.DataFrame([
            {"event_type": "quarry blast"},
            {"event_type": "nuclear explosion"},
            {"event_type": "earthquake"},
            {"event_type": None},
        ])
        labeled = validate_model.label_events(events)
        assert labeled["true_label"].tolist() == [1, 1, 0, 0]

    def test_returns_copy(self):
        events = pd.DataFrame([{"event_type": "earthquake"}])
        labeled = validate_model.label_events(events)
        assert "true_label" in labeled.columns
        assert "true_label" not in events.columns


class TestComputeMetrics:
    def test_perfect_classification(self):
        metrics = validate_model.compute_metrics(_sample_df(), threshold=0.5)
        assert metrics["tp"] == 1
        assert metrics["fp"] == 0
        assert metrics["tn"] == 1
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_low_confidence_events_excluded(self):
        df = pd.DataFrame([
            {"confidence": 0.1, "predicted_explosion_score": 0.9, "true_label": 1},
        ])
        metrics = validate_model.compute_metrics(df, threshold=0.5)
        # confidence > 0.2 required; this single event is dropped
        assert "error" in metrics

    def test_threshold_sensitivity(self):
        df = _sample_df()
        strict = validate_model.compute_metrics(df, threshold=0.9)
        # Event b has score 0.8, below 0.9 → predicted EQ → false negative
        assert strict["tp"] == 0
        assert strict["fn"] == 1


class TestGenerateReport:
    def test_report_contains_metrics(self):
        metrics = validate_model.compute_metrics(_sample_df(), threshold=0.5)
        report = validate_model.generate_report(metrics, "baseline-001")
        assert "baseline-001" in report
        assert "# Model Validation Report" in report
        assert "Precision" in report
        assert "Recall" in report
