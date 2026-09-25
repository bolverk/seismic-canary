"""Reproduce an event assessment.

Loads an event's evidence record, re-runs feature extraction
and anomaly model, and compares results to stored values.

Usage:
    python scripts/reproduce_event.py <event_id>
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.processing.events import load_events
from src.processing.timeline import load_timeline
from src.models.anomaly import RuleBasedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reproduce_event(event_id: str) -> int:
    """Reproduce assessment for a given event.

    Args:
        event_id: The event to reproduce.

    Returns:
        Exit code: 0 if reproducible, 1 if results differ or event not found.
    """
    print(f"{'='*60}")
    print("Seismic Canary - Event Reproduction Report")
    print(f"Event ID: {event_id}")
    print(f"Model Version: {Config.MODEL_VERSION}")
    print(f"Processing Version: {Config.PROCESSING_VERSION}")
    print(f"{'='*60}\n")

    # Load event data
    events = load_events(Config.EVENTS_PARQUET)
    event_row = events[events["event_id"] == event_id]

    if event_row.empty:
        print(f"ERROR: Event '{event_id}' not found in {Config.EVENTS_PARQUET}")
        return 1

    event_data = event_row.iloc[0].to_dict()

    # Display stored event data
    print("STORED EVENT DATA:")
    print(f"  Origin Time:  {event_data.get('origin_time')}")
    print(f"  Location:     {event_data.get('latitude'):.3f}°N, {event_data.get('longitude'):.3f}°E")
    print(f"  Depth:        {event_data.get('depth_km')} km")
    print(f"  Magnitude:    {event_data.get('magnitude')} {event_data.get('magnitude_type')}")
    print(f"  Provider:     {event_data.get('provider')}")
    print(f"  Source URL:   {event_data.get('source_url')}")
    print(f"  First seen:   {event_data.get('first_seen')}")
    print(f"  Last updated: {event_data.get('last_updated')}")
    print()

    # Display stored features
    print("STORED FEATURES:")
    feature_fields = [
        "p_s_ratio", "mb_ms", "corner_frequency", "spectral_slope",
        "dominant_frequency", "snr", "station_count", "waveform_quality",
    ]
    import pandas as pd
    for field in feature_fields:
        val = event_data.get(field)
        status = f"{val}" if pd.notna(val) else "not available"
        print(f"  {field}: {status}")
    print()

    # Display stored assessment
    print("STORED ASSESSMENT:")
    print(f"  Earthquake consistency: {event_data.get('earthquake_consistency')}")
    print(f"  Explosion consistency:  {event_data.get('explosion_consistency')}")
    print(f"  Alert level:            {event_data.get('alert_level')}")
    print(f"  Model version:          {event_data.get('model_version')}")
    print(f"  Processing version:     {event_data.get('processing_version')}")
    print()

    # Re-run anomaly model
    print("RE-RUNNING ASSESSMENT:")
    model = RuleBasedModel()
    new_assessment = model.assess(event_data)

    print(f"  Earthquake consistency: {new_assessment.earthquake_consistency:.4f}")
    print(f"  Explosion consistency:  {new_assessment.explosion_consistency:.4f}")
    print(f"  Alert level:            {new_assessment.alert_level}")
    print(f"  Confidence:             {new_assessment.confidence:.4f}")
    print(f"  Model version:          {new_assessment.model_version}")
    print()

    # Compare
    print("COMPARISON:")
    stored_eq = event_data.get("earthquake_consistency")
    stored_ex = event_data.get("explosion_consistency")
    stored_level = event_data.get("alert_level")

    reproducible = True
    if pd.notna(stored_eq) and abs(stored_eq - new_assessment.earthquake_consistency) > 0.001:
        print(f"  ⚠️  Earthquake consistency differs: stored={stored_eq:.4f} vs computed={new_assessment.earthquake_consistency:.4f}")
        reproducible = False
    if pd.notna(stored_ex) and abs(stored_ex - new_assessment.explosion_consistency) > 0.001:
        print(f"  ⚠️  Explosion consistency differs: stored={stored_ex:.4f} vs computed={new_assessment.explosion_consistency:.4f}")
        reproducible = False
    if pd.notna(stored_level) and stored_level != new_assessment.alert_level:
        print(f"  ⚠️  Alert level differs: stored={stored_level} vs computed={new_assessment.alert_level}")
        reproducible = False

    if pd.isna(stored_eq) and pd.isna(stored_ex):
        print("  ℹ️  No stored assessment to compare (event not yet scored)")
        reproducible = True

    if reproducible:
        print("  ✓ Assessment is reproducible")
    print()

    # Load and display timeline
    print("TIMELINE:")
    timeline = load_timeline(event_id)
    if timeline and timeline.entries:
        for entry in timeline.entries:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if entry.timestamp else "?"
            print(f"  [{ts}] {entry.entry_type}: {entry.description}")
    else:
        print("  No timeline recorded for this event.")
    print()

    # Contributing features detail
    print("RULE CONTRIBUTIONS:")
    for contrib in new_assessment.contributing_features:
        if contrib.confidence > 0:
            print(f"  {contrib.feature_name}:")
            print(f"    Value: {contrib.raw_value}")
            print(f"    Earthquake: {contrib.contribution_earthquake:.2f}")
            print(f"    Explosion:  {contrib.contribution_explosion:.2f}")
            print(f"    Confidence: {contrib.confidence:.2f}")
            print(f"    {contrib.explanation}")
        else:
            print(f"  {contrib.feature_name}: {contrib.explanation}")
    print()

    print(f"{'='*60}")
    if reproducible:
        print("RESULT: REPRODUCIBLE ✓")
    else:
        print("RESULT: DIFFERS ⚠️")
    print(f"{'='*60}")

    return 0 if reproducible else 1


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce an event assessment."
    )
    parser.add_argument("event_id", help="Event ID to reproduce")
    args = parser.parse_args()

    sys.exit(reproduce_event(args.event_id))


if __name__ == "__main__":
    main()
