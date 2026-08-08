"""Shared test fixtures for Seismic Canary."""
import pytest
import pandas as pd
from datetime import datetime, timezone


@pytest.fixture
def sample_event_dict():
    """A single valid event as a dictionary."""
    return {
        "event_id": "us7000test",
        "provider": "usgs",
        "origin_time": datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc),
        "latitude": 35.7,
        "longitude": 51.4,
        "depth_km": 10.0,
        "magnitude": 4.2,
        "magnitude_type": "mb",
        "event_type": "earthquake",
        "place": "15km NW of Tehran, Iran",
        "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000test",
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


@pytest.fixture
def sample_events_df(sample_event_dict):
    """A DataFrame with a few sample events."""
    events = [
        sample_event_dict,
        {
            **sample_event_dict,
            "event_id": "us7000test2",
            "latitude": 32.5,
            "longitude": 48.7,
            "depth_km": 2.0,
            "magnitude": 3.7,
            "place": "20km S of Dezful, Iran",
            "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000test2",
        },
        {
            **sample_event_dict,
            "event_id": "us7000test3",
            "provider": "emsc",
            "latitude": 38.0,
            "longitude": 46.3,
            "depth_km": 15.0,
            "magnitude": 5.1,
            "magnitude_type": "Mw",
            "place": "Eastern Turkey",
            "source_url": "https://www.emsc-csem.org/Earthquake/test3",
        },
    ]
    df = pd.DataFrame(events)
    df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True)
    df["first_seen"] = pd.to_datetime(df["first_seen"], utc=True)
    df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True)
    return df


@pytest.fixture
def shallow_suspicious_event_dict():
    """An event with characteristics suspicious for an explosion."""
    return {
        "event_id": "us7000suspicious",
        "provider": "usgs",
        "origin_time": datetime(2024, 3, 10, 8, 15, 22, tzinfo=timezone.utc),
        "latitude": 34.5,
        "longitude": 55.2,
        "depth_km": 1.5,
        "magnitude": 3.8,
        "magnitude_type": "mb",
        "event_type": "earthquake",
        "place": "Central Iran",
        "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000suspicious",
        "first_seen": datetime(2024, 3, 10, 8, 20, 0, tzinfo=timezone.utc),
        "last_updated": datetime(2024, 3, 10, 8, 20, 0, tzinfo=timezone.utc),
        "p_s_ratio": 0.7,  # log10(P/S) - high
        "mb_ms": 1.3,
        "corner_frequency": None,
        "spectral_slope": None,
        "dominant_frequency": None,
        "snr": 15.0,
        "station_count": 8,
        "waveform_quality": "good",
        "source_type_score": None,
        "earthquake_consistency": None,
        "explosion_consistency": None,
        "alert_level": None,
        "model_version": None,
        "processing_version": None,
    }
