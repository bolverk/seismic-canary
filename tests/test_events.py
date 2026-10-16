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
    is_explosion_event_type,
    reconcile_cross_provider_events,
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

    def test_refetch_preserves_existing_scores(self, sample_events_df):
        existing = sample_events_df.iloc[[0]].copy()
        existing.at[existing.index[0], "alert_level"] = 1
        existing.at[existing.index[0], "earthquake_consistency"] = 0.8
        existing.at[existing.index[0], "model_version"] = "1.0.0"

        # A re-fetched catalog row is unscored (all derived fields None)
        new_event = existing.copy()
        new_event.at[new_event.index[0], "alert_level"] = None
        new_event.at[new_event.index[0], "earthquake_consistency"] = None
        new_event.at[new_event.index[0], "model_version"] = None
        new_event.at[new_event.index[0], "magnitude"] = 4.7

        result = deduplicate_events(existing, new_event)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["alert_level"] == 1
        assert row["earthquake_consistency"] == 0.8
        assert row["model_version"] == "1.0.0"
        assert row["magnitude"] == 4.7


class TestIsExplosionEventType:
    def test_explosion_types(self):
        assert is_explosion_event_type("chemical explosion")
        assert is_explosion_event_type("nuclear explosion")
        assert is_explosion_event_type("quarry blast")
        assert is_explosion_event_type("experimental explosion")

    def test_non_explosion_types(self):
        assert not is_explosion_event_type("earthquake")
        assert not is_explosion_event_type("rockburst")
        assert not is_explosion_event_type(None)
        assert not is_explosion_event_type(float("nan"))


class TestCrossProviderReconciliation:
    def _make(self, sample_events_df, **overrides):
        row = sample_events_df.iloc[[0]].copy()
        for key, val in overrides.items():
            row.at[row.index[0], key] = val
        return row

    def test_drops_usgs_duplicate_of_emsc_event(self, sample_events_df):
        usgs = self._make(sample_events_df, event_id="us1", provider="usgs",
                          event_type="earthquake")
        emsc = self._make(sample_events_df, event_id="em1", provider="emsc",
                          event_type="earthquake")
        merged = pd.concat([usgs, emsc], ignore_index=True)

        result = reconcile_cross_provider_events(merged)
        assert len(result) == 1
        # USGS preferred for ordinary earthquakes
        assert result.iloc[0]["provider"] == "usgs"

    def test_explosion_label_wins_over_usgs_earthquake(self, sample_events_df):
        usgs = self._make(sample_events_df, event_id="us1", provider="usgs",
                          event_type="earthquake")
        emsc = self._make(sample_events_df, event_id="em1", provider="emsc",
                          event_type="chemical explosion")
        merged = pd.concat([usgs, emsc], ignore_index=True)

        result = reconcile_cross_provider_events(merged)
        assert len(result) == 1
        assert result.iloc[0]["provider"] == "emsc"
        assert result.iloc[0]["event_type"] == "chemical explosion"

    def test_distinct_events_kept(self, sample_events_df):
        a = self._make(sample_events_df, event_id="us1", provider="usgs",
                       latitude=35.0, longitude=51.0)
        b = self._make(sample_events_df, event_id="em1", provider="emsc",
                       latitude=30.0, longitude=45.0)
        merged = pd.concat([a, b], ignore_index=True)

        result = reconcile_cross_provider_events(merged)
        assert len(result) == 2

    def test_same_provider_never_merged(self, sample_events_df):
        a = self._make(sample_events_df, event_id="us1", provider="usgs")
        b = self._make(sample_events_df, event_id="us2", provider="usgs")
        merged = pd.concat([a, b], ignore_index=True)

        result = reconcile_cross_provider_events(merged)
        assert len(result) == 2

    def test_time_tolerance_respected(self, sample_events_df):
        a = self._make(sample_events_df, event_id="us1", provider="usgs")
        b = self._make(sample_events_df, event_id="em1", provider="emsc",
                       origin_time=pd.Timestamp("2024-01-15 13:30:00", tz="UTC"))
        merged = pd.concat([a, b], ignore_index=True)

        result = reconcile_cross_provider_events(merged)
        assert len(result) == 2

    def test_empty_input(self):
        assert reconcile_cross_provider_events(get_empty_events_df()).empty

    def test_order_independent(self, sample_events_df):
        rows = [
            self._make(sample_events_df, event_id="us1", provider="usgs"),
            self._make(sample_events_df, event_id="em1", provider="emsc"),
            self._make(sample_events_df, event_id="em2", provider="emsc",
                       latitude=10.0, longitude=10.0,
                       origin_time=pd.Timestamp("2025-06-01", tz="UTC")),
        ]
        merged = pd.concat(rows, ignore_index=True)
        ordered = reconcile_cross_provider_events(merged)
        shuffled = reconcile_cross_provider_events(
            merged.sample(frac=1, random_state=7).reset_index(drop=True)
        )
        key = lambda d: sorted(zip(d["event_id"], d["provider"]))
        assert key(ordered) == key(shuffled)
