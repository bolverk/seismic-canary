"""Tests for the scheduled event-update script.

These exercise the ``run_update`` pipeline (fetch → dedupe → score → save)
fully offline by stubbing out the USGS provider.
"""
import importlib.util
import os
import pathlib
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.config import Config

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update_events = _load_script("update_events")


def _event_dict(event_id: str, origin_time=None, provider="usgs", **overrides):
    base = {
        "event_id": event_id,
        "provider": provider,
        "origin_time": origin_time or datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc),
        "latitude": 35.7,
        "longitude": 51.4,
        "depth_km": 10.0,
        "magnitude": 4.2,
        "magnitude_type": "mb",
        "event_type": "earthquake",
        "place": "Test location",
        "source_url": f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}",
        "first_seen": datetime(2024, 1, 15, 12, 35, 0, tzinfo=timezone.utc),
        "last_updated": datetime(2024, 1, 15, 12, 35, 0, tzinfo=timezone.utc),
        "p_s_ratio": None,
        "mb_ms": None,
        "corner_frequency": None,
        "spectral_slope": None,
        "dominant_frequency": None,
        "snr": None,
        "station_count": None,
        "waveform_quality": None,
        "source_type_score": None,
        "earthquake_consistency": None,
        "explosion_consistency": None,
        "alert_level": None,
        "model_version": None,
        "processing_version": None,
    }
    base.update(overrides)
    return base


def _make_df(events):
    df = pd.DataFrame(events)
    for col in ["origin_time", "first_seen", "last_updated"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


class _FakeProvider:
    """Returns a fixed set of events, recording the requested window."""

    def __init__(self, events_df):
        self._events = events_df
        self.calls = []

    def fetch_events(self, start_time, end_time, region=None):
        self.calls.append((start_time, end_time))
        if self._events is None:
            from src.processing.events import get_empty_events_df
            return get_empty_events_df()
        return self._events


@pytest.fixture
def data_path(tmp_path, monkeypatch):
    path = str(tmp_path / "events.parquet")
    monkeypatch.setattr(Config, "EVENTS_PARQUET", path)
    monkeypatch.setattr(Config, "STATIONS_CACHE", str(tmp_path / "stations.json"))
    return path


def test_run_update_first_run_scores_and_saves(data_path, monkeypatch):
    """First run (no existing data) fetches, scores and persists events."""
    new_events = _make_df([
        _event_dict("e1", depth_km=1.5, alert_level=None),
        _event_dict("e2", depth_km=25.0, alert_level=None),
    ])
    monkeypatch.setattr(
        update_events, "USGSProvider", lambda *a, **k: _FakeProvider(new_events)
    )

    stats = update_events.run_update(days=30, dry_run=False)

    assert stats["fetched_count"] == 2
    assert stats["new_count"] == 2
    assert stats["final_count"] == 2

    # File written and events scored
    saved = update_events.load_events(data_path)
    assert len(saved) == 2
    # Shallow event scored as not-insufficient
    assert saved["alert_level"].notna().sum() == 2
    assert saved["model_version"].iloc[0] == "baseline-001"


def test_run_update_dry_run_does_not_write(data_path, monkeypatch):
    new_events = _make_df([_event_dict("e1", depth_km=10.0)])
    monkeypatch.setattr(
        update_events, "USGSProvider", lambda *a, **k: _FakeProvider(new_events)
    )
    stats = update_events.run_update(days=30, dry_run=True)
    assert stats["final_count"] == 1
    # Nothing persisted on dry run
    assert not os.path.exists(data_path)


def test_deduplicate_same_provider_updates_in_place(data_path, monkeypatch):
    """Re-fetching an existing event updates it rather than duplicating."""
    existing = _make_df([_event_dict("e1", depth_km=10.0, magnitude=4.2)])
    save_events = update_events.save_events
    save_events(existing, data_path)

    # Re-fetch the same event with updated magnitude
    new_events = _make_df([_event_dict("e1", depth_km=10.0, magnitude=4.5)])
    monkeypatch.setattr(
        update_events, "USGSProvider", lambda *a, **k: _FakeProvider(new_events)
    )
    stats = update_events.run_update(days=30, dry_run=False)

    assert stats["new_count"] == 0
    saved = update_events.load_events(data_path)
    assert len(saved) == 1
    assert float(saved["magnitude"].iloc[0]) == 4.5


def test_api_error_keeps_existing_data(data_path, monkeypatch):
    """A USGS error must not clobber the existing dataset."""
    existing = _make_df([_event_dict("e1", depth_km=10.0)])
    update_events.save_events(existing, data_path)

    class BrokenProvider(_FakeProvider):
        def fetch_events(self, start_time, end_time, region=None):
            raise update_events.USGSAPIError("boom")

    monkeypatch.setattr(
        update_events, "USGSProvider", lambda *a, **k: BrokenProvider(None)
    )
    stats = update_events.run_update(days=30, dry_run=False)

    assert stats["errors"], "expected an error to be recorded"
    assert stats["final_count"] == 1
