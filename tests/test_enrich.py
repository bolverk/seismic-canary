"""Tests for waveform feature enrichment orchestration."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.processing import enrich
from src.processing.enrich import (
    select_events_for_enrichment,
    enrich_event,
    obspy_available,
)
from src.processing.waveforms import PickResult, WaveformResult


def _event(**overrides):
    base = {
        "event_id": "em1",
        "provider": "emsc",
        "origin_time": pd.Timestamp("2026-09-10 18:08:47", tz="UTC"),
        "latitude": 33.3,
        "longitude": 35.3,
        "depth_km": 1.0,
        "magnitude": 4.1,
        "magnitude_type": "mb",
        "event_type": "earthquake",
        "alert_level": 2,
        "explosion_consistency": 0.7,
        "waveform_quality": None,
    }
    base.update(overrides)
    return base


class FakeDiscovery:
    """Station discovery stub that records calls and returns canned stations."""

    def __init__(self, cached=None, found=None):
        self._cached = cached
        self._found = found if found is not None else []
        self.find_calls = 0
        self.cache_calls = 0

    def get_cached_stations(self, lat, lon, tolerance_deg=0.5):
        return self._cached

    def find_stations(self, lat, lon, event_time=None, max_stations=50):
        self.find_calls += 1
        return list(self._found)

    def cache_stations(self, lat, lon, stations):
        self.cache_calls += 1


class TestObspyAvailable:
    def test_returns_bool(self):
        assert isinstance(obspy_available(), bool)


class TestSelectEventsForEnrichment:
    def _df(self, rows):
        return pd.DataFrame(rows)

    def test_excludes_attempted(self):
        df = self._df([
            _event(event_id="a", waveform_quality=None),
            _event(event_id="b", waveform_quality="unavailable"),
        ])
        assert select_events_for_enrichment(df) == [0]

    def test_excludes_ineligible(self):
        # magnitude < 3 and depth >= 5 km -> not eligible
        df = self._df([
            _event(event_id="a", magnitude=2.0, depth_km=15.0),
            _event(event_id="b", magnitude=4.0, depth_km=15.0),
        ])
        assert select_events_for_enrichment(df) == [1]

    def test_excludes_missing_coordinates(self):
        df = self._df([
            _event(event_id="a", latitude=None),
            _event(event_id="b", latitude=35.0, longitude=51.0),
        ])
        assert select_events_for_enrichment(df) == [1]

    def test_prioritizes_alert_then_explosion_then_recent(self):
        df = self._df([
            _event(event_id="lvl0", alert_level=0, explosion_consistency=0.9,
                   origin_time=pd.Timestamp("2026-09-01", tz="UTC")),
            _event(event_id="lvl2_old", alert_level=2, explosion_consistency=0.5,
                   origin_time=pd.Timestamp("2026-09-01", tz="UTC")),
            _event(event_id="lvl2_new", alert_level=2, explosion_consistency=0.5,
                   origin_time=pd.Timestamp("2026-09-05", tz="UTC")),
            _event(event_id="lvl2_highex", alert_level=2, explosion_consistency=0.8,
                   origin_time=pd.Timestamp("2026-09-01", tz="UTC")),
        ])
        result = select_events_for_enrichment(df, limit=10)
        ids = [df.loc[i, "event_id"] for i in result]
        assert ids == ["lvl2_highex", "lvl2_new", "lvl2_old", "lvl0"]

    def test_respects_limit(self):
        df = self._df([_event(event_id=f"e{i}") for i in range(10)])
        assert len(select_events_for_enrichment(df, limit=3)) == 3

    def test_empty_dataframe(self):
        assert select_events_for_enrichment(pd.DataFrame()) == []


class TestEnrichEvent:
    def test_returns_none_when_obspy_missing(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: False)
        assert enrich_event(_event()) is None

    def test_returns_none_when_ineligible(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: True)
        assert enrich_event(_event(magnitude=2.0, depth_km=20.0)) is None

    def test_returns_none_when_coordinates_missing(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: True)
        assert enrich_event(_event(latitude=None)) is None

    def test_marks_unavailable_when_no_stations(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: True)
        discovery = FakeDiscovery(cached=None, found=[])
        updates = enrich_event(_event(), station_discovery=discovery)
        assert updates["waveform_quality"] == "unavailable"
        assert updates["station_count"] == 0
        assert updates["p_s_ratio"] is None
        assert updates["alert_level"] is None  # forces rescore

    def test_populates_features_from_waveform_result(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: True)

        stations = [object()]
        discovery = FakeDiscovery(cached=stations)

        result = WaveformResult(
            event_id="em1",
            station_count=6,
            successful_stations=4,
            median_p_s_ratio=0.35,
            quality="marginal",
            picks=[
                PickResult(station="A", network="X", channel="BHZ", snr=8.0),
                PickResult(station="B", network="X", channel="BHZ", snr=12.0),
                PickResult(station="C", network="X", channel="BHZ", snr=None),
            ],
        )
        monkeypatch.setattr(enrich, "analyze_event_waveforms", lambda **kw: result)

        updates = enrich_event(_event(), station_discovery=discovery)
        assert updates["p_s_ratio"] == 0.35
        assert updates["snr"] == 10.0  # median of [8, 12]
        assert updates["station_count"] == 4
        assert updates["waveform_quality"] == "marginal"
        assert updates["alert_level"] is None

    def test_uses_cache_and_skips_station_query(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: True)
        discovery = FakeDiscovery(cached=[object()])
        monkeypatch.setattr(
            enrich, "analyze_event_waveforms",
            lambda **kw: WaveformResult(event_id="em1", quality="unavailable"),
        )
        enrich_event(_event(), station_discovery=discovery)
        assert discovery.find_calls == 0

    def test_queries_and_caches_when_no_cache(self, monkeypatch):
        monkeypatch.setattr(enrich, "obspy_available", lambda: True)
        discovery = FakeDiscovery(cached=None, found=[object(), object()])
        monkeypatch.setattr(
            enrich, "analyze_event_waveforms",
            lambda **kw: WaveformResult(event_id="em1", quality="good"),
        )
        enrich_event(_event(), station_discovery=discovery)
        assert discovery.find_calls == 1
        assert discovery.cache_calls == 1
