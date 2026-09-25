"""Tests for USGS event ingestion module."""
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

from src.ingestion.seismic import USGSProvider, USGSAPIError, SeismicProvider
from src.config import RegionBounds
from src.processing.events import EVENT_SCHEMA_COLUMNS


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def usgs_response_data():
    """Load the USGS GeoJSON fixture."""
    with open(os.path.join(FIXTURES_DIR, "usgs_response.json")) as f:
        return json.load(f)


@pytest.fixture
def provider():
    """Create a USGSProvider instance."""
    return USGSProvider()


@pytest.fixture
def mock_successful_response(usgs_response_data):
    """Mock a successful HTTP response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = usgs_response_data
    return mock_resp


class TestProviderInterface:
    def test_is_abstract(self):
        """SeismicProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SeismicProvider()

    def test_usgs_implements_interface(self, provider):
        assert provider.provider_name == "usgs"


class TestFetchEvents:
    @patch("src.ingestion.seismic.requests.get")
    def test_successful_fetch(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)

        assert len(result) == 4
        assert "event_id" in result.columns
        assert result.iloc[0]["event_id"] == "us7000abc1"
        assert result.iloc[0]["magnitude"] == 4.2

    @patch("src.ingestion.seismic.requests.get")
    def test_uses_default_region(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        provider.fetch_events(start, end)

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params["minlatitude"] == 24.0
        assert params["maxlatitude"] == 42.0
        assert params["minlongitude"] == 43.0
        assert params["maxlongitude"] == 64.0

    @patch("src.ingestion.seismic.requests.get")
    def test_custom_region(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        region = RegionBounds(30.0, 38.0, 50.0, 60.0)
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        provider.fetch_events(start, end, region=region)

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params["minlatitude"] == 30.0
        assert params["maxlatitude"] == 38.0

    @patch("src.ingestion.seismic.requests.get")
    def test_empty_response(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"type": "FeatureCollection", "features": []}
        mock_get.return_value = mock_resp

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert len(result) == 0
        assert list(result.columns) == EVENT_SCHEMA_COLUMNS

    @patch("src.ingestion.seismic.requests.get")
    def test_204_no_content(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_get.return_value = mock_resp

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert len(result) == 0


class TestErrorHandling:
    @patch("src.ingestion.seismic.requests.get")
    def test_400_bad_request_no_retry(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad request parameters"
        mock_get.return_value = mock_resp

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        with pytest.raises(USGSAPIError, match="Bad request"):
            provider.fetch_events(start, end)

        # Should not retry on 400
        assert mock_get.call_count == 1

    @patch("src.ingestion.seismic.requests.get")
    def test_timeout_retries(self, mock_get, provider):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        provider.retry_delay = 0.01  # Speed up test

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        with pytest.raises(USGSAPIError, match="timed out"):
            provider.fetch_events(start, end)

        assert mock_get.call_count == provider.max_retries

    @patch("src.ingestion.seismic.requests.get")
    def test_connection_error_retries(self, mock_get, provider):
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        provider.retry_delay = 0.01

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        with pytest.raises(USGSAPIError, match="Connection error"):
            provider.fetch_events(start, end)

        assert mock_get.call_count == provider.max_retries

    @patch("src.ingestion.seismic.requests.get")
    def test_500_server_error_retries(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        mock_get.return_value = mock_resp
        provider.retry_delay = 0.01

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        with pytest.raises(USGSAPIError, match="Server error"):
            provider.fetch_events(start, end)

        assert mock_get.call_count == provider.max_retries

    @patch("src.ingestion.seismic.requests.get")
    def test_malformed_json(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_resp

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        with pytest.raises(USGSAPIError, match="parse response"):
            provider.fetch_events(start, end)


class TestNormalization:
    @patch("src.ingestion.seismic.requests.get")
    def test_all_fields_populated(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        row = result.iloc[0]

        assert row["event_id"] == "us7000abc1"
        assert row["provider"] == "usgs"
        assert row["latitude"] == 35.8
        assert row["longitude"] == 51.2
        assert row["depth_km"] == 10.0
        assert row["magnitude"] == 4.2
        assert row["magnitude_type"] == "mb"
        assert row["event_type"] == "earthquake"
        assert row["place"] == "15km NW of Tehran, Iran"
        assert "earthquake.usgs.gov" in row["source_url"]

    @patch("src.ingestion.seismic.requests.get")
    def test_null_depth_handled(self, mock_get, provider, usgs_response_data):
        """Event us7000abc4 has null depth in coordinates."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = usgs_response_data
        mock_get.return_value = mock_resp

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        quarry = result[result["event_id"] == "us7000abc4"]
        # Depth should be None/NaN for null coordinates[2]
        assert quarry.iloc[0]["depth_km"] is None or pd.isna(quarry.iloc[0]["depth_km"])

    @patch("src.ingestion.seismic.requests.get")
    def test_event_type_preserved(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        quarry = result[result["event_id"] == "us7000abc4"]
        assert quarry.iloc[0]["event_type"] == "quarry blast"

    @patch("src.ingestion.seismic.requests.get")
    def test_origin_time_is_utc(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert result["origin_time"].dt.tz is not None

    @patch("src.ingestion.seismic.requests.get")
    def test_feature_fields_are_null(self, mock_get, provider, mock_successful_response):
        """Processing feature fields should be null after ingestion."""
        mock_get.return_value = mock_successful_response
        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert result["p_s_ratio"].isna().all()
        assert result["mb_ms"].isna().all()
        assert result["station_count"].isna().all()

    @patch("src.ingestion.seismic.requests.get")
    def test_malformed_feature_skipped(self, mock_get, provider):
        """A feature without coordinates should be skipped."""
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"mag": 3.0, "time": 1705312245000,
                                   "place": "Test", "type": "earthquake",
                                   "magType": "mb", "url": "http://test"},
                    "geometry": {"type": "Point", "coordinates": []},
                    "id": "bad_event"
                },
                {
                    "type": "Feature",
                    "properties": {"mag": 4.0, "time": 1705312245000,
                                   "place": "Good", "type": "earthquake",
                                   "magType": "mb", "url": "http://test"},
                    "geometry": {"type": "Point", "coordinates": [51.0, 35.0, 10.0]},
                    "id": "good_event"
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        mock_get.return_value = mock_resp

        start = datetime(2024, 1, 14, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert len(result) == 1
        assert result.iloc[0]["event_id"] == "good_event"
