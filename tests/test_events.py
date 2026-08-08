"""Tests for event data schema and Parquet storage layer."""
import os
import tempfile

import pandas as pd
import pytest
from datetime import datetime, timezone

from src.processing.events import (
    EVENT_SCHEMA_COLUMNS,
    get_empty_events_df,
    load_events,
    save_events,
    deduplicate_events,
)


class TestEmptySchema:
    def test_has_all_columns(self):
        df = get_empty_events_df()
        assert list(df.columns) == EVENT_SCHEMA_COLUMNS

    def test_is_empty(self):
        df = get_empty_events_df()
        assert len(df) == 0


class TestLoadSaveRoundTrip:
    def test_roundtrip(self, sample_events_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.parquet")
            save_events(sample_events_df, path)
            loaded = load_events(path)
            assert len(loaded) == len(sample_events_df)
            assert list(loaded.columns) == EVENT_SCHEMA_COLUMNS

    def test_load_nonexistent_returns_empty(self):
        df = load_events("/nonexistent/path/events.parquet")
        assert len(df) == 0
        assert list(df.columns) == EVENT_SCHEMA_COLUMNS

    def test_null_values_preserved(self, sample_events_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.parquet")
            save_events(sample_events_df, path)
            loaded = load_events(path)
            # p_s_ratio should be null for all sample events
            assert loaded["p_s_ratio"].isna().all()

    def test_schema_evolution(self, sample_events_df):
        """Loading a parquet with fewer columns should add missing ones as null."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.parquet")
            # Save with only a subset of columns
            subset_cols = ["event_id", "provider", "origin_time", "latitude",
                          "longitude", "depth_km", "magnitude", "magnitude_type"]
            subset_df = sample_events_df[subset_cols].copy()
            subset_df.to_parquet(path, index=False)
            
            # Load should still have all columns
            loaded = load_events(path)
            assert list(loaded.columns) == EVENT_SCHEMA_COLUMNS
            assert loaded["event_type"].isna().all()


class TestDeduplication:
    def test_same_event_same_provider_updates(self, sample_events_df):
        existing = sample_events_df.copy()
        # Create "new" version of same event with updated magnitude
        new_event = sample_events_df.iloc[[0]].copy()
        new_event.at[new_event.index[0], "magnitude"] = 4.5
        
        result = deduplicate_events(existing, new_event)
        # Should still have same number of rows
        assert len(result) == len(existing)
        # Magnitude should be updated
        mask = result["event_id"] == "us7000test"
        assert result.loc[mask & (result["provider"] == "usgs"), "magnitude"].iloc[0] == 4.5

    def test_same_event_different_provider_keeps_both(self, sample_events_df):
        existing = sample_events_df.iloc[[0]].copy()  # usgs provider
        new_event = existing.copy()
        new_event.at[new_event.index[0], "provider"] = "emsc"
        
        result = deduplicate_events(existing, new_event)
        assert len(result) == 2

    def test_new_event_added(self, sample_events_df):
        existing = sample_events_df.iloc[:2].copy()
        new_event = sample_events_df.iloc[[2]].copy()
        
        result = deduplicate_events(existing, new_event)
        assert len(result) == 3

    def test_empty_existing(self, sample_events_df):
        existing = get_empty_events_df()
        result = deduplicate_events(existing, sample_events_df)
        assert len(result) == len(sample_events_df)

    def test_empty_new_events(self, sample_events_df):
        new_events = get_empty_events_df()
        result = deduplicate_events(sample_events_df, new_events)
        assert len(result) == len(sample_events_df)

    def test_first_seen_preserved_on_update(self, sample_events_df):
        existing = sample_events_df.iloc[[0]].copy()
        original_first_seen = existing.iloc[0]["first_seen"]
        
        new_event = existing.copy()
        new_event.at[new_event.index[0], "magnitude"] = 4.5
        
        result = deduplicate_events(existing, new_event)
        assert result.iloc[0]["first_seen"] == original_first_seen
