"""Model validation against public catalog labels.

Validates the rule-based anomaly model by comparing its assessments
against USGS event_type labels (earthquake, quarry blast, explosion).

Usage:
    python scripts/validate_model.py [--output docs/validation_report.md]
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.models.anomaly import RuleBasedModel
from src.ingestion.seismic import USGSProvider, USGSAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Event types considered "explosion" for validation
EXPLOSION_TYPES = {"quarry blast", "explosion", "mining explosion",
                   "nuclear explosion", "chemical explosion",
                   "experimental explosion", "industrial explosion"}

EARTHQUAKE_TYPES = {"earthquake"}


def fetch_labeled_events(days: int = 365) -> pd.DataFrame:
    """Fetch events with known event_type labels from USGS.

    To get enough labeled explosions, we expand the geographic
    region and time window.

    Args:
        days: Number of days of history to fetch.

    Returns:
        DataFrame with labeled events.
    """
    provider = USGSProvider()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # Fetch earthquakes from the Iran region
    logger.info("Fetching earthquakes from monitored region...")
    try:
        earthquakes = provider.fetch_events(start, end, Config.REGION_BOUNDS)
        earthquakes = earthquakes[earthquakes["event_type"].isin(EARTHQUAKE_TYPES)]
        logger.info(f"  Found {len(earthquakes)} earthquakes")
    except USGSAPIError as e:
        logger.error(f"Failed to fetch earthquakes: {e}")
        earthquakes = pd.DataFrame()

    # Fetch known explosions (expand search)
    logger.info("Fetching known explosions (global)...")
    try:
        # USGS allows filtering by eventtype
        import requests
        params = {
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "eventtype": "explosion",
            "limit": 500,
        }
        resp = requests.get(Config.USGS_API_BASE, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("features"):
                explosions = provider._normalize_response(data)
                logger.info(f"  Found {len(explosions)} explosions")
            else:
                explosions = pd.DataFrame()
        else:
            explosions = pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to fetch explosions: {e}")
        explosions = pd.DataFrame()

    # Also fetch quarry blasts
    try:
        params["eventtype"] = "quarry blast"
        resp = requests.get(Config.USGS_API_BASE, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("features"):
                quarry = provider._normalize_response(data)
                logger.info(f"  Found {len(quarry)} quarry blasts")
                explosions = pd.concat([explosions, quarry], ignore_index=True)
            else:
                logger.info("  No quarry blasts found")
    except Exception as e:
        logger.warning(f"Failed to fetch quarry blasts: {e}")

    # Combine
    all_events = pd.concat([earthquakes, explosions], ignore_index=True)
    return all_events


def label_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add binary label: 1 for explosion-like, 0 for earthquake."""
    events = events.copy()
    events["true_label"] = events["event_type"].apply(
        lambda x: 1 if x in EXPLOSION_TYPES else 0
    )
    return events


def run_model_on_events(events: pd.DataFrame) -> pd.DataFrame:
    """Run the anomaly model on all events and add predictions."""
    model = RuleBasedModel()
    events = events.copy()

    predictions = []
    for _, row in events.iterrows():
        event_dict = row.to_dict()
        assessment = model.assess(event_dict)
        predictions.append({
            "predicted_explosion_score": assessment.explosion_consistency,
            "predicted_alert_level": assessment.alert_level,
            "confidence": assessment.confidence,
        })

    pred_df = pd.DataFrame(predictions)
    return pd.concat([events.reset_index(drop=True), pred_df], axis=1)


def compute_metrics(results: pd.DataFrame, threshold: float = 0.5) -> Dict:
    """Compute classification metrics.

    Args:
        results: DataFrame with true_label and predicted_explosion_score.
        threshold: Score threshold for explosion classification.

    Returns:
        Dictionary of metrics.
    """
    # Filter to events with sufficient confidence
    valid = results[results["confidence"] > 0.2].copy()

    if len(valid) == 0:
        return {"error": "No events with sufficient confidence"}

    y_true = valid["true_label"].values
    y_pred = (valid["predicted_explosion_score"] >= threshold).astype(int).values

    # Confusion matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(valid) if len(valid) > 0 else 0.0

    return {
        "total_events": len(valid),
        "earthquakes": int(np.sum(y_true == 0)),
        "explosions": int(np.sum(y_true == 1)),
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def generate_report(metrics: Dict, model_version: str) -> str:
    """Generate a markdown validation report."""
    report = f"""# Model Validation Report

**Model Version:** {model_version}
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Processing Version:** {Config.PROCESSING_VERSION}

## Summary

This report validates the rule-based anomaly model against events
with known `event_type` labels from the USGS earthquake catalog.

**Important limitations:**
- The model uses only catalog features (depth, location) for most events
- P/S ratio and mb-Ms require waveform processing (not available for most events)
- Quarry blasts are not nuclear explosions — this is a sanity check, not proof of capability
- The model is experimental and NOT a calibrated classifier

## Dataset

| Metric | Value |
|--------|-------|
| Total events evaluated | {metrics.get('total_events', 'N/A')} |
| Earthquakes (label=0) | {metrics.get('earthquakes', 'N/A')} |
| Explosions/blasts (label=1) | {metrics.get('explosions', 'N/A')} |

## Results (threshold={metrics.get('threshold', 0.5)})

| Metric | Value |
|--------|-------|
| Precision | {metrics.get('precision', 0):.3f} |
| Recall | {metrics.get('recall', 0):.3f} |
| F1 Score | {metrics.get('f1', 0):.3f} |
| Accuracy | {metrics.get('accuracy', 0):.3f} |

## Confusion Matrix

|  | Predicted EQ | Predicted EX |
|--|-------------|-------------|
| **Actual EQ** | {metrics.get('tn', 0)} | {metrics.get('fp', 0)} |
| **Actual EX** | {metrics.get('fn', 0)} | {metrics.get('tp', 0)} |

## Interpretation

- **Precision {metrics.get('precision', 0):.3f}**: Of events flagged as explosion-like, {metrics.get('precision', 0)*100:.0f}% actually are.
- **Recall {metrics.get('recall', 0):.3f}**: Of actual explosions, {metrics.get('recall', 0)*100:.0f}% are detected.

## Known Limitations

1. Most events lack waveform-derived features (P/S ratio, mb-Ms), so the model
   relies primarily on depth and location — which are weak discriminants alone.
2. Quarry blasts are often shallow but may not have typical explosion characteristics
   in other features because they are small and well-characterized.
3. The USGS catalog may not label all events correctly.
4. This validation does NOT prove the model can detect nuclear tests.

## Recommendations

- Priority improvement: add waveform processing to compute P/S ratio for more events
- Investigate the false positives to understand failure modes
- Consider adding frequency-based features that differentiate mine blasts from earthquakes
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate anomaly model")
    parser.add_argument("--output", default="docs/validation_report.md")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    model = RuleBasedModel()
    logger.info(f"Validating model version: {model.get_version()}")

    # Fetch labeled events
    logger.info(f"Fetching labeled events from last {args.days} days...")
    events = fetch_labeled_events(days=args.days)

    if events.empty:
        logger.error("No events fetched. Cannot validate.")
        sys.exit(1)

    # Label and predict
    labeled = label_events(events)
    logger.info(
        f"Dataset: {len(labeled)} events, "
        f"{labeled['true_label'].sum()} explosions, "
        f"{(labeled['true_label'] == 0).sum()} earthquakes"
    )

    results = run_model_on_events(labeled)

    # Compute metrics
    metrics = compute_metrics(results, threshold=0.5)
    logger.info(f"Results: {metrics}")

    # Generate report
    report = generate_report(metrics, model.get_version())

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {args.output}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Model: {model.get_version()}")
    print(f"Events: {metrics.get('total_events', 0)}")
    print(f"Precision: {metrics.get('precision', 0):.3f}")
    print(f"Recall: {metrics.get('recall', 0):.3f}")
    print(f"F1: {metrics.get('f1', 0):.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
