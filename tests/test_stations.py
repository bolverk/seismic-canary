"""Tests for station discovery and metadata module."""
import os
import tempfile

import pytest

from src.ingestion.stations import (
    StationInfo,
    StationDiscovery,
    haversine_distance,
    degrees_distance,
    compute_azimuth,
)


class TestHaversineDistance:
    def test_same_point(self):
        assert haversine_distance(35.0, 51.0, 35.0, 51.0) == 0.0

    def test_known_distance(self):
        # Tehran to Baghdad: approximately 695 km
        dist = haversine_distance(35.7, 51.4, 33.3, 44.4)
        assert 680 < dist < 710

    def test_symmetric(self):
        d1 = haversine_distance(35.0, 51.0, 33.0, 44.0)
        d2 = haversine_distance(33.0, 44.0, 35.0, 51.0)
        assert abs(d1 - d2) < 0.01


class TestDegreesDistance:
    def test_same_point(self):
        assert degrees_distance(35.0, 51.0, 35.0, 51.0) == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # 1 degree latitude ≈ 111 km, so 10° ≈ 1111 km
        dist_deg = degrees_distance(30.0, 50.0, 40.0, 50.0)
        assert 9.5 < dist_deg < 10.5

    def test_consistency_with_haversine(self):
        dist_deg = degrees_distance(35.0, 51.0, 33.0, 44.0)
        dist_km = haversine_distance(35.0, 51.0, 33.0, 44.0)
        # 1 degree ≈ 111.2 km
        assert abs(dist_km / dist_deg - 111.2) < 5


class TestComputeAzimuth:
    def test_due_north(self):
        az = compute_azimuth(30.0, 50.0, 35.0, 50.0)
        assert abs(az - 0.0) < 1.0

    def test_due_east(self):
        az = compute_azimuth(30.0, 50.0, 30.0, 55.0)
        assert 85 < az < 95

    def test_due_south(self):
        az = compute_azimuth(35.0, 50.0, 30.0, 50.0)
        assert 175 < az < 185

    def test_due_west(self):
        az = compute_azimuth(30.0, 50.0, 30.0, 45.0)
        assert 265 < az < 275

    def test_range_0_360(self):
        az = compute_azimuth(35.0, 51.0, 33.0, 44.0)
        assert 0 <= az < 360


class TestStationInfo:
    def test_creation(self):
        station = StationInfo(
            network="II",
            station="ABKT",
            latitude=37.93,
            longitude=58.12,
            elevation_m=678.0,
            description="Alibek, Turkmenistan",
        )
        assert station.network == "II"
        assert station.station == "ABKT"
        assert station.distance_km is None


class TestStationDiscoveryCaching:
    def test_cache_miss_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name

        try:
            os.unlink(cache_path)  # ensure file doesn't exist
            discovery = StationDiscovery(cache_path=cache_path)
            result = discovery.get_cached_stations(35.0, 51.0)
            assert result is None
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    def test_cache_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name

        try:
            discovery = StationDiscovery(cache_path=cache_path)
            stations = [
                StationInfo(
                    network="II",
                    station="ABKT",
                    latitude=37.93,
                    longitude=58.12,
                    elevation_m=678.0,
                    description="Alibek",
                    distance_km=500.0,
                    distance_deg=4.5,
                    azimuth=45.0,
                    back_azimuth=225.0,
                )
            ]

            discovery.cache_stations(35.0, 51.0, stations)
            result = discovery.get_cached_stations(35.0, 51.0)

            assert result is not None
            assert len(result) == 1
            assert result[0].network == "II"
            assert result[0].station == "ABKT"
            assert result[0].distance_km == 500.0
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    def test_no_stations_found(self):
        """When providers list is empty, return empty list."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name

        try:
            discovery = StationDiscovery(
                providers=[],  # no providers to query
                cache_path=cache_path,
            )
            stations = discovery.find_stations(35.0, 51.0)
            assert stations == []
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
