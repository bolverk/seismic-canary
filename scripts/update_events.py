"""Scheduled event update script.

This script is designed to be run by GitHub Actions on a schedule.
It fetches new seismic events, deduplicates them with existing data,
and saves the updated dataset.

Usage:
    python scripts/update_events.py [--days N] [--dry-run]
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ingestion.seismic import USGSProvider, USGSAPIError
from src.processing.events import load_events, save_events, deduplicate_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def determine_fetch_window(existing_events) -> tuple:
    """Determine the time window for fetching new events.

    If we have existing events, fetch from the most recent event time
    minus some overlap. Otherwise, fetch the last N days.

    Args:
        existing_events: DataFrame of existing events.

    Returns:
        Tuple of (start_time, end_time) as datetime objects.
    """
    end_time = datetime.now(timezone.utc)

    if existing_events.empty:
        # First run: fetch last DEFAULT_FETCH_DAYS
        start_time = end_time - timedelta(days=Config.DEFAULT_FETCH_DAYS)
        logger.info(
            f"No existing events. Fetching last {Config.DEFAULT_FETCH_DAYS} days."
        )
    else:
        # Incremental: fetch from latest event minus overlap
        latest = existing_events["origin_time"].max()
        overlap = timedelta(hours=Config.INCREMENTAL_FETCH_HOURS)
        start_time = latest - overlap
        logger.info(
            f"Incremental fetch from {start_time.isoformat()} "
            f"(latest event: {latest.isoformat()}, overlap: {overlap})"
        )

    return start_time, end_time


def run_update(days: int = None, dry_run: bool = False) -> dict:
    """Run the event update pipeline.

    Args:
        days: Override fetch window to last N days. If None, uses smart window.
        dry_run: If True, don't save results.

    Returns:
        Dictionary with update statistics.
    """
    stats = {
        "existing_count": 0,
        "fetched_count": 0,
        "new_count": 0,
        "updated_count": 0,
        "final_count": 0,
        "errors": [],
    }

    # Load existing events
    logger.info(f"Loading existing events from {Config.EVENTS_PARQUET}")
    existing = load_events(Config.EVENTS_PARQUET)
    stats["existing_count"] = len(existing)
    logger.info(f"Loaded {len(existing)} existing events.")

    # Determine fetch window
    end_time = datetime.now(timezone.utc)
    if days is not None:
        start_time = end_time - timedelta(days=days)
    else:
        start_time, end_time = determine_fetch_window(existing)

    # Fetch new events from USGS
    provider = USGSProvider()
    try:
        logger.info(
            f"Fetching from USGS: {start_time.isoformat()} to {end_time.isoformat()}"
        )
        new_events = provider.fetch_events(start_time, end_time)
        stats["fetched_count"] = len(new_events)
        logger.info(f"Fetched {len(new_events)} events from USGS.")
    except USGSAPIError as e:
        error_msg = f"USGS API error: {e}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        # Don't fail completely - we still have existing data
        new_events = existing.iloc[0:0]  # empty df with same schema

    if new_events.empty:
        logger.info("No new events fetched. Dataset unchanged.")
        stats["final_count"] = len(existing)
        return stats

    # Deduplicate and merge
    merged = deduplicate_events(existing, new_events)
    stats["final_count"] = len(merged)
    stats["new_count"] = len(merged) - len(existing)
    stats["updated_count"] = stats["fetched_count"] - stats["new_count"]

    logger.info(
        f"After deduplication: {len(merged)} total events "
        f"({stats['new_count']} new, {stats['updated_count']} updated)"
    )

    # Run anomaly model on events that haven't been scored yet
    from src.models.anomaly import RuleBasedModel
    model = RuleBasedModel()
    unscored = merged["alert_level"].isna()
    if unscored.any():
        logger.info(f"Scoring {unscored.sum()} unscored events...")
        for idx in merged[unscored].index:
            event = merged.loc[idx].to_dict()
            assessment = model.assess(event)
            merged.at[idx, "earthquake_consistency"] = assessment.earthquake_consistency
            merged.at[idx, "explosion_consistency"] = assessment.explosion_consistency
            merged.at[idx, "alert_level"] = assessment.alert_level
            merged.at[idx, "model_version"] = assessment.model_version
            merged.at[idx, "processing_version"] = Config.PROCESSING_VERSION

    # Save results
    if not dry_run:
        save_events(merged, Config.EVENTS_PARQUET)
        logger.info(f"Saved {len(merged)} events to {Config.EVENTS_PARQUET}")
    else:
        logger.info("Dry run - not saving.")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update seismic event dataset from USGS."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Fetch events from last N days (overrides smart window).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and process but don't save results.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Seismic Canary - Event Update Pipeline")
    logger.info(f"Version: {Config.VERSION}")
    logger.info(f"Region: {Config.MONITORED_REGION_DESCRIPTION}")
    logger.info("=" * 60)

    stats = run_update(days=args.days, dry_run=args.dry_run)

    # Print summary
    logger.info("")
    logger.info("Update Summary:")
    logger.info(f"  Existing events: {stats['existing_count']}")
    logger.info(f"  Fetched events:  {stats['fetched_count']}")
    logger.info(f"  New events:      {stats['new_count']}")
    logger.info(f"  Updated events:  {stats['updated_count']}")
    logger.info(f"  Final count:     {stats['final_count']}")

    if stats["errors"]:
        logger.warning(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            logger.warning(f"    - {err}")
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
