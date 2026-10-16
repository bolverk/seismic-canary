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

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ingestion.seismic import USGSProvider, SeismicAPIError
from src.ingestion.emsc import EMSCProvider
from src.processing.events import (
    load_events,
    save_events,
    deduplicate_events,
    reconcile_cross_provider_events,
)
from src.processing.enrich import (
    obspy_available,
    select_events_for_enrichment,
    enrich_event,
)

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


def run_update(
    days: int = None,
    dry_run: bool = False,
    waveform_events: int = Config.WAVEFORM_MAX_EVENTS_PER_RUN,
) -> dict:
    """Run the event update pipeline.

    Args:
        days: Override fetch window to last N days. If None, uses smart window.
        dry_run: If True, don't save results.
        waveform_events: Max events to enrich with waveform features this run.

    Returns:
        Dictionary with update statistics.
    """
    stats = {
        "existing_count": 0,
        "fetched_count": 0,
        "new_count": 0,
        "updated_count": 0,
        "enriched_count": 0,
        "final_count": 0,
        "provider_count": 0,
        "provider_failures": 0,
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

    # Fetch new events from each provider
    providers = [USGSProvider(), EMSCProvider()]
    stats["provider_count"] = len(providers)
    frames = []
    for provider in providers:
        try:
            logger.info(
                f"Fetching from {provider.provider_name.upper()}: "
                f"{start_time.isoformat()} to {end_time.isoformat()}"
            )
            fetched = provider.fetch_events(start_time, end_time)
            logger.info(
                f"Fetched {len(fetched)} events from {provider.provider_name.upper()}."
            )
            frames.append(fetched)
        except SeismicAPIError as e:
            error_msg = f"{provider.provider_name.upper()} API error: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            stats["provider_failures"] += 1

    if frames:
        new_events = pd.concat(frames, ignore_index=True)
    else:
        new_events = existing.iloc[0:0]  # empty df with same schema
    stats["fetched_count"] = len(new_events)

    if new_events.empty:
        logger.info("No new events fetched. Dataset unchanged.")
        stats["final_count"] = len(existing)
        return stats

    # Deduplicate (by event_id + provider) then collapse cross-provider
    # duplicates of the same physical event (USGS has no explosion codes,
    # EMSC does, so reconciliation can promote an EMSC record over a USGS one).
    merged = deduplicate_events(existing, new_events)
    merged = reconcile_cross_provider_events(merged)
    stats["final_count"] = len(merged)
    stats["new_count"] = max(0, len(merged) - len(existing))
    stats["updated_count"] = stats["fetched_count"] - stats["new_count"]

    logger.info(
        f"After deduplication: {len(merged)} total events "
        f"({stats['new_count']} new, {stats['updated_count']} updated)"
    )

    # Enrich a bounded batch of events with waveform-derived features (P/S,
    # SNR). Without this, the P/S and mb-Ms rules have no data, so the depth
    # rule alone dominates the score and produces shallow-event false positives.
    if waveform_events > 0 and obspy_available():
        selected = select_events_for_enrichment(merged, limit=waveform_events)
        if selected:
            from src.ingestion.stations import StationDiscovery

            # alert_level may be int64; clearing it to None needs a float column.
            if merged["alert_level"].dtype.kind in "iu":
                merged["alert_level"] = merged["alert_level"].astype("float64")

            logger.info(f"Enriching {len(selected)} events with waveform features...")
            discovery = StationDiscovery()
            for idx in selected:
                try:
                    updates = enrich_event(
                        merged.loc[idx].to_dict(), station_discovery=discovery
                    )
                except Exception as e:  # never let enrichment break the run
                    logger.warning(f"Enrichment failed for event at index {idx}: {e}")
                    continue
                if not updates:
                    continue
                for col, val in updates.items():
                    merged.at[idx, col] = val
                stats["enriched_count"] += 1
            logger.info(
                f"Enriched {stats['enriched_count']} events with waveform features."
            )
    elif waveform_events > 0:
        logger.warning("ObsPy not available — skipping waveform enrichment.")

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
        description="Update seismic event dataset from configured providers (USGS, EMSC)."
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
    parser.add_argument(
        "--waveform-events",
        type=int,
        default=Config.WAVEFORM_MAX_EVENTS_PER_RUN,
        help=(
            "Max events to enrich with waveform features this run "
            f"(default {Config.WAVEFORM_MAX_EVENTS_PER_RUN})."
        ),
    )
    parser.add_argument(
        "--no-waveforms",
        action="store_true",
        help="Skip waveform feature enrichment entirely.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Seismic Canary - Event Update Pipeline")
    logger.info(f"Version: {Config.VERSION}")
    logger.info(f"Region: {Config.MONITORED_REGION_DESCRIPTION}")
    logger.info("=" * 60)

    waveform_events = 0 if args.no_waveforms else args.waveform_events
    stats = run_update(
        days=args.days, dry_run=args.dry_run, waveform_events=waveform_events
    )

    # Print summary
    logger.info("")
    logger.info("Update Summary:")
    logger.info(f"  Existing events: {stats['existing_count']}")
    logger.info(f"  Fetched events:  {stats['fetched_count']}")
    logger.info(f"  New events:      {stats['new_count']}")
    logger.info(f"  Updated events:  {stats['updated_count']}")
    logger.info(f"  Enriched events: {stats['enriched_count']}")
    logger.info(f"  Final count:     {stats['final_count']}")

    if stats["errors"]:
        logger.warning(f"  Provider errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            logger.warning(f"    - {err}")
        # Only fail the run if every provider failed and nothing was fetched;
        # a single provider outage should not block committing the other's data.
        if (
            stats["provider_count"] > 0
            and stats["provider_failures"] >= stats["provider_count"]
            and stats["fetched_count"] == 0
        ):
            sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
