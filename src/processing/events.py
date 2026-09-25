"""Event data schema and Parquet storage layer.

Defines the stable event data model and provides read/write
operations for Parquet-based storage.
"""
import os
from datetime import datetime, timezone

import pandas as pd


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
                # Keep first_seen from existing, update other fields
                update_data = new_row.to_dict()
                update_data["first_seen"] = existing.loc[idx[0], "first_seen"]
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
