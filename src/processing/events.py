"""Event data schema and Parquet storage layer.

Defines the stable event data model and provides read/write
operations for Parquet-based storage.
"""
import math
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import Config


# Canonical column order and types for the events schema
EVENT_SCHEMA_COLUMNS = [
    "event_id",
    "provider",
    "origin_time",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "event_type",
    "place",
    "source_url",
    "first_seen",
    "last_updated",
    # Nullable feature fields (populated by processing pipeline)
    "p_s_ratio",
    "mb_ms",
    "corner_frequency",
    "spectral_slope",
    "dominant_frequency",
    "snr",
    "station_count",
    "waveform_quality",
    "source_type_score",
    "earthquake_consistency",
    "explosion_consistency",
    "alert_level",
    "model_version",
    "processing_version",
]


def get_empty_events_df() -> pd.DataFrame:
    """Create an empty DataFrame with the full event schema."""
    df = pd.DataFrame(columns=EVENT_SCHEMA_COLUMNS)
    # Set appropriate dtypes
    df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True)
    df["first_seen"] = pd.to_datetime(df["first_seen"], utc=True)
    df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True)
    for col in ["latitude", "longitude", "depth_km", "magnitude"]:
        df[col] = df[col].astype("float64")
    for col in ["p_s_ratio", "mb_ms", "corner_frequency", "spectral_slope",
                "dominant_frequency", "snr", "source_type_score",
                "earthquake_consistency", "explosion_consistency"]:
        df[col] = df[col].astype("float64")
    for col in ["station_count", "alert_level"]:
        df[col] = df[col].astype("Int64")  # nullable integer
    return df


def load_events(path: str) -> pd.DataFrame:
    """Load events from a Parquet file.
    
    If the file does not exist, returns an empty DataFrame with
    the full schema. Handles schema evolution by adding missing
    columns as null.
    
    Args:
        path: Path to the Parquet file.
        
    Returns:
        DataFrame with all schema columns.
    """
    if not os.path.exists(path):
        return get_empty_events_df()
    
    df = pd.read_parquet(path)
    
    # Schema evolution: add any missing columns as null
    for col in EVENT_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Ensure datetime columns are timezone-aware
    for col in ["origin_time", "first_seen", "last_updated"]:
        if col in df.columns and df[col].dtype != "datetime64[ns, UTC]":
            df[col] = pd.to_datetime(df[col], utc=True)
    
    return df[EVENT_SCHEMA_COLUMNS]


def save_events(df: pd.DataFrame, path: str) -> None:
    """Save events DataFrame to Parquet.
    
    Creates parent directories if they don't exist.
    Ensures consistent column ordering.
    
    Args:
        df: Events DataFrame.
        path: Output Parquet file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Ensure all schema columns exist
    for col in EVENT_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Write with consistent column order
    df[EVENT_SCHEMA_COLUMNS].to_parquet(path, index=False, engine="pyarrow")


def deduplicate_events(existing: pd.DataFrame, new_events: pd.DataFrame) -> pd.DataFrame:
    """Merge new events into existing dataset with deduplication.
    
    Rules:
    - Same event_id + same provider: update existing record (keep first_seen, update last_updated)
    - Same event_id + different provider: keep both as separate records
    - New event_id: add as new record
    - Never delete historical events
    
    Args:
        existing: Current events DataFrame.
        new_events: Newly fetched events DataFrame.
        
    Returns:
        Merged DataFrame with deduplication applied.
    """
    if existing.empty:
        return new_events.copy()
    
    if new_events.empty:
        return existing.copy()
    
    now = datetime.now(timezone.utc)
    
    # Create a composite key for matching
    existing_keys = set(
        zip(existing["event_id"].astype(str), existing["provider"].astype(str))
    )
    
    rows_to_add = []
    rows_to_update = {}  # index in existing -> new row data
    
    for _, new_row in new_events.iterrows():
        key = (str(new_row["event_id"]), str(new_row["provider"]))
        
        if key in existing_keys:
            # Update existing record: find it and update
            mask = (
                (existing["event_id"].astype(str) == key[0]) & 
                (existing["provider"].astype(str) == key[1])
            )
            idx = existing[mask].index
            if len(idx) > 0:
                # Keep first_seen from existing, update catalog fields from the
                # new row, but preserve any existing (derived/scored) values the
                # fetch does not populate (e.g. alert_level, model_version,
                # earthquake_consistency). Otherwise a re-fetch of an already
                # scored event would silently wipe its assessment.
                existing_row = existing.loc[idx[0]].to_dict()
                update_data = dict(existing_row)
                for col, val in new_row.to_dict().items():
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    update_data[col] = val
                update_data["first_seen"] = existing_row["first_seen"]
                update_data["last_updated"] = now
                rows_to_update[idx[0]] = update_data
        else:
            # New record
            row_data = new_row.to_dict()
            if pd.isna(row_data.get("first_seen")) or row_data.get("first_seen") is None:
                row_data["first_seen"] = now
            row_data["last_updated"] = now
            rows_to_add.append(row_data)
    
    # Apply updates
    result = existing.copy()
    for idx, data in rows_to_update.items():
        for col, val in data.items():
            if col in result.columns:
                result.at[idx, col] = val
    
    # Add new rows
    if rows_to_add:
        new_df = pd.DataFrame(rows_to_add)
        # Ensure schema alignment
        for col in EVENT_SCHEMA_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = None
        result = pd.concat([result, new_df[EVENT_SCHEMA_COLUMNS]], ignore_index=True)
    
    return result


# Event-type strings that indicate a (non-tectonic) explosion. This mirrors
# the anomaly model's explosion vocabulary plus the descriptive EMSC/ISF
# mappings produced in ``src.ingestion.emsc``.
EXPLOSION_EVENT_TYPES = frozenset({
    "explosion",
    "nuclear explosion",
    "chemical explosion",
    "mining explosion",
    "mine explosion",
    "quarry blast",
    "blast",
})

# Lower rank == more globally authoritative for tectonic earthquakes.
_PROVIDER_PRIORITY = {"usgs": 0, "emsc": 1}


def is_explosion_event_type(event_type: Any) -> bool:
    """Return True if a catalog event_type denotes an explosion."""
    if event_type is None:
        return False
    if isinstance(event_type, float) and math.isnan(event_type):
        return False
    text = str(event_type).strip().lower()
    if not text:
        return False
    return text in EXPLOSION_EVENT_TYPES or "explosion" in text


def _record_priority(row: pd.Series) -> tuple:
    """Rank a record for cross-provider reconciliation: higher is better.

    Explosion-labelled records win first (EMSC carries ISF explosion codes
    that USGS lacks); otherwise the more authoritative provider wins.
    """
    rank = _PROVIDER_PRIORITY.get(str(row.get("provider", "")).lower(), 99)
    return (1 if is_explosion_event_type(row.get("event_type")) else 0, -rank)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def reconcile_cross_provider_events(
    df: pd.DataFrame,
    time_tolerance_seconds: float = Config.CROSS_PROVIDER_TIME_TOLERANCE_SECONDS,
    distance_tolerance_km: float = Config.CROSS_PROVIDER_DISTANCE_TOLERANCE_KM,
) -> pd.DataFrame:
    """Collapse the same physical event reported by multiple providers.

    Different providers publish independent event IDs for the same physical
    event, so a USGS-only dataset and an EMSC dataset produce duplicate
    markers for every shared event. This groups records that are close in
    time and space and keeps a single record per group, preferring:

    1. records whose ``event_type`` indicates an explosion, then
    2. the more globally authoritative provider (USGS over EMSC), then
    3. the first-seen record.

    Records from the *same* provider are never merged (they are distinct
    catalog entries); only cross-provider duplicates are removed.

    Args:
        df: Events DataFrame (already deduplicated by id+provider).
        time_tolerance_seconds: Max origin-time difference to be considered
            the same event.
        distance_tolerance_km: Max epicentral distance to be considered the
            same event.

    Returns:
        DataFrame with cross-provider duplicates removed.
    """
    if df is None or df.empty:
        return df

    work = df.reset_index(drop=True).copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    origin_epoch = (
        pd.to_datetime(work["origin_time"], utc=True) - epoch
    ).dt.total_seconds()

    # Deterministic processing order: origin time, then provider precedence
    # (USGS anchors before EMSC so USGS wins ties), then event id. Without a
    # total order, results depend on input row order, so the same catalog can
    # merge differently from one run to the next.
    provider_rank = (
        work["provider"].astype(str).str.lower().map(_PROVIDER_PRIORITY).fillna(99)
    )
    order = (
        pd.DataFrame({
            "time": origin_epoch,
            "provider": provider_rank,
            "event_id": work["event_id"].astype(str),
        })
        .sort_values(["time", "provider", "event_id"], kind="stable")
        .index.tolist()
    )
    # Each group anchors on its first (earliest) member for matching and
    # tracks the best record chosen among members plus the providers present.
    groups: List[Dict[str, Any]] = []
    drop: set = set()

    for idx in order:
        row = work.loc[idx]
        t = int(origin_epoch.loc[idx])
        lat = row["latitude"]
        lon = row["longitude"]
        provider = str(row["provider"]).lower()

        match = None
        for group in reversed(groups):
            if t - group["anchor_time"] > time_tolerance_seconds:
                break
            if provider in group["providers"]:
                continue
            if pd.isna(lat) or pd.isna(lon) or pd.isna(group["anchor_lat"]):
                match = group
                break
            if _haversine_km(lat, lon, group["anchor_lat"], group["anchor_lon"]) \
                    <= distance_tolerance_km:
                match = group
                break

        if match is None:
            groups.append({
                "anchor_time": t,
                "anchor_lat": lat,
                "anchor_lon": lon,
                "providers": {provider},
                "best_idx": idx,
                "best_priority": _record_priority(row),
            })
            continue

        if _record_priority(row) > match["best_priority"]:
            drop.add(match["best_idx"])
            match["best_idx"] = idx
            match["best_priority"] = _record_priority(row)
        else:
            drop.add(idx)
        match["providers"].add(provider)

    return work.drop(index=list(drop)).reset_index(drop=True)
