"""Waveform feature enrichment for the event update pipeline.

Wires the P/S + SNR measurement pipeline (:mod:`src.processing.waveforms`)
into the update loop. Without this step every event reaches the anomaly
model with ``p_s_ratio`` / ``snr`` / ``waveform_quality`` still NULL, so the
P/S rule (weight 0.25) and the mb-Ms rule (weight 0.20) drop out of the
weighted average and the depth rule alone dominates — which flags shallow
local-network micro-earthquakes as probable explosions.

Enrichment is deliberately bounded: only a small batch of events is
processed per run, and each event downloads from at most a handful of
stations. ``waveform_quality`` doubling as an "attempted" marker means an
enriched event is never retried, so the backlog is worked down over
successive runs.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.ingestion.stations import StationDiscovery
from src.processing.waveforms import (
    analyze_event_waveforms,
    is_eligible_for_waveforms,
)

logger = logging.getLogger(__name__)


def obspy_available() -> bool:
    """Whether the optional ObsPy dependency is importable."""
    try:
        import obspy  # noqa: F401
    except ImportError:
        return False
    return True


def select_events_for_enrichment(
    events: pd.DataFrame,
    limit: int = Config.WAVEFORM_MAX_EVENTS_PER_RUN,
) -> List[Any]:
    """Choose which events to enrich with waveform features this run.

    Only events that (a) have never been attempted (``waveform_quality`` is
    NULL), (b) are eligible for waveform retrieval, and (c) have usable
    coordinates/time are considered. Among those, higher-alert and higher
    explosion-consistency events are prioritised — they are the current
    false-positive candidates — then the most recent.

    Args:
        events: Full events DataFrame.
        limit: Maximum number of events to return.

    Returns:
        List of DataFrame index labels.
    """
    if events is None or events.empty or limit <= 0:
        return []

    if "waveform_quality" not in events.columns:
        return []

    candidates = events[events["waveform_quality"].isna()].copy()
    if candidates.empty:
        return []

    eligible = candidates.apply(
        lambda r: is_eligible_for_waveforms(r.get("magnitude"), r.get("depth_km")),
        axis=1,
    )
    candidates = candidates[eligible]
    candidates = candidates[
        candidates["latitude"].notna()
        & candidates["longitude"].notna()
        & candidates["origin_time"].notna()
    ]
    if candidates.empty:
        return []

    rank = pd.DataFrame(index=candidates.index)
    rank["alert"] = pd.to_numeric(candidates["alert_level"], errors="coerce").fillna(-1)
    rank["explosion"] = pd.to_numeric(
        candidates["explosion_consistency"], errors="coerce"
    ).fillna(0.0)
    rank["time"] = pd.to_datetime(candidates["origin_time"], utc=True)

    order = rank.sort_values(
        ["alert", "explosion", "time"], ascending=[False, False, False], kind="stable"
    ).index
    return list(order[:limit])


def _median_snr(picks) -> Optional[float]:
    """Median SNR across picks that produced a measurement."""
    values = [p.snr for p in picks if getattr(p, "snr", None) is not None]
    if not values:
        return None
    return float(np.median(values))


def enrich_event(
    event: Dict[str, Any],
    station_discovery: Optional[StationDiscovery] = None,
    max_stations: int = Config.WAVEFORM_MAX_STATIONS,
    target_traces: int = Config.WAVEFORM_TARGET_TRACES,
) -> Optional[Dict[str, Any]]:
    """Compute waveform-derived features for a single event.

    Args:
        event: Event fields as a dict.
        station_discovery: Reusable station discovery/cache instance.
        max_stations: Nearest stations to query.
        target_traces: Stop downloading once this many stations have data.

    Returns:
        Dict of column updates — always including a non-null
        ``waveform_quality`` so the event is recorded as *attempted* — or
        ``None`` if the event could not be attempted at all (ObsPy missing,
        ineligible, or missing coordinates). In the ``None`` case the event
        stays unmarked so a later run can retry it.
    """
    if not obspy_available():
        return None

    if not is_eligible_for_waveforms(event.get("magnitude"), event.get("depth_km")):
        return None

    lat = event.get("latitude")
    lon = event.get("longitude")
    origin_time = event.get("origin_time")
    if lat is None or lon is None or origin_time is None:
        return None
    if pd.isna(lat) or pd.isna(lon) or pd.isna(origin_time):
        return None

    lat, lon = float(lat), float(lon)

    discovery = station_discovery or StationDiscovery()

    stations = discovery.get_cached_stations(lat, lon)
    if stations is None:
        stations = discovery.find_stations(
            lat, lon, event_time=origin_time, max_stations=max_stations
        )
        if stations:
            discovery.cache_stations(lat, lon, stations)

    # Default outcome: attempted but no usable waveform data.
    updates: Dict[str, Any] = {
        "p_s_ratio": None,
        "snr": None,
        "station_count": 0,
        "waveform_quality": "unavailable",
        "alert_level": None,
    }

    if not stations:
        logger.info(
            "No stations found near event %s — marked waveform-unavailable.",
            event.get("event_id"),
        )
        return updates

    result = analyze_event_waveforms(
        event_id=str(event.get("event_id", "")),
        event_lat=lat,
        event_lon=lon,
        event_time=origin_time,
        stations=stations,
        target_traces=target_traces,
    )

    updates["p_s_ratio"] = result.median_p_s_ratio
    updates["snr"] = _median_snr(result.picks)
    updates["station_count"] = int(result.successful_stations or 0)
    updates["waveform_quality"] = result.quality
    return updates
