"""Tests for the EMSC event ingestion module."""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests
from unittest.mock import patch, MagicMock

from src.ingestion.emsc import EMSCProvider, EMSCAPIError, map_emsc_event_type
from src.ingestion.seismic import SeismicProvider, SeismicAPIError
from src.config import Config
from src.processing.events import EVENT_SCHEMA_COLUMNS


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# The shared FDSN request logic lives in src.ingestion.seismic, so HTTP calls
# are patched there (the global ``requests`` module is the same object).
REQUESTS_GET = "src.ingestion.seismic.requests.get"


@pytest.fixture
def emsc_response_data():
    with open(os.path.join(FIXTURES_DIR, "emsc_response.json")) as f:
        return json.load(f)


@pytest.fixture
def provider():
    return EMSCProvider()


@pytest.fixture
def mock_successful_response(emsc_response_data):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = emsc_response_data
    return mock_resp


class TestProviderInterface:
    def test_implements_interface(self, provider):
        assert isinstance(provider, SeismicProvider)
        assert provider.provider_name == "emsc"

    def test_error_is_seismic_error(self):
        assert issubclass(EMSCAPIError, SeismicAPIError)


class TestEventTypeMapping:
    def test_earthquake_codes(self):
        for code in ("ke", "se", "fe", "de"):
            assert map_emsc_event_type(code) == "earthquake"

    def test_explosion_codes(self):
        assert map_emsc_event_type("kh") == "chemical explosion"
        assert map_emsc_event_type("sh") == "chemical explosion"
        assert map_emsc_event_type("km") == "mining explosion"
        assert map_emsc_event_type("sm") == "mining explosion"
        assert map_emsc_event_type("kn") == "nuclear explosion"
        assert map_emsc_event_type("sn") == "nuclear explosion"
        assert map_emsc_event_type("kx") == "explosion"
        assert map_emsc_event_type("sx") == "explosion"

    def test_other_codes(self):
        assert map_emsc_event_type("kr") == "rockburst"
        assert map_emsc_event_type("ki") == "induced event"
        assert map_emsc_event_type("uk") == "unknown"
        assert map_emsc_event_type("ue") == "unknown"

    def test_unknown_code_passthrough(self):
        assert map_emsc_event_type("zz") == "zz"
        assert map_emsc_event_type(None) is None


class TestFetchEvents:
    @patch(REQUESTS_GET)
    def test_successful_fetch(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)

        # 4 features, one (malformed_no_time) dropped
        assert len(result) == 3
        assert list(result.columns) == EVENT_SCHEMA_COLUMNS
        assert result.iloc[0]["event_id"] == "20260910_0000294"
        assert result.iloc[0]["provider"] == "emsc"

    @patch(REQUESTS_GET)
    def test_params_include_region_and_min_magnitude(
        self, mock_get, provider, mock_successful_response
    ):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        provider.fetch_events(start, end)

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params["format"] == "json"
        assert params["minlatitude"] == Config.REGION_BOUNDS.min_latitude
        assert params["maxlongitude"] == Config.REGION_BOUNDS.max_longitude
        assert params["minmagnitude"] == Config.EMSC_MIN_MAGNITUDE
        assert params["orderby"] == "time"

    @patch(REQUESTS_GET)
    def test_empty_response(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"type": "FeatureCollection", "features": []}
        mock_get.return_value = mock_resp

        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert len(result) == 0
        assert list(result.columns) == EVENT_SCHEMA_COLUMNS

    @patch(REQUESTS_GET)
    def test_204_no_content(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_get.return_value = mock_resp

        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert len(result) == 0


class TestNormalization:
    @patch(REQUESTS_GET)
    def test_all_fields_populated(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        row = result.iloc[0]

        assert row["event_id"] == "20260910_0000294"
        assert row["provider"] == "emsc"
        assert row["latitude"] == 33.3298
        assert row["longitude"] == 35.3298
        assert row["depth_km"] == 10.0
        assert row["magnitude"] == 4.1
        assert row["magnitude_type"] == "mb"
        assert row["event_type"] == "chemical explosion"
        assert row["place"] == "LEBANON - SYRIA REGION"
        assert "seismicportal.eu" in row["source_url"]

    @patch(REQUESTS_GET)
    def test_negative_depth_normalized(self, mock_get, provider, mock_successful_response):
        """EMSC reports depth as positive in properties; keep it positive."""
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert (result["depth_km"].dropna() >= 0).all()
        assert result.iloc[0]["depth_km"] == 10.0

    @patch(REQUESTS_GET)
    def test_null_depth_handled(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        nuclear = result[result["event_id"] == "20260912_0000001"]
        assert len(nuclear) == 1
        assert pd.isna(nuclear.iloc[0]["depth_km"])

    @patch(REQUESTS_GET)
    def test_nuclear_explosion_mapping(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        nuclear = result[result["event_id"] == "20260912_0000001"]
        assert nuclear.iloc[0]["event_type"] == "nuclear explosion"

    @patch(REQUESTS_GET)
    def test_origin_time_is_utc(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert result["origin_time"].dt.tz is not None

    @patch(REQUESTS_GET)
    def test_feature_fields_are_null(self, mock_get, provider, mock_successful_response):
        mock_get.return_value = mock_successful_response
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        end = datetime(2026, 9, 12, tzinfo=timezone.utc)

        result = provider.fetch_events(start, end)
        assert result["p_s_ratio"].isna().all()
        assert result["mb_ms"].isna().all()
        assert result["alert_level"].isna().all()


class TestErrorHandling:
    @patch(REQUESTS_GET)
    def test_400_bad_request_no_retry(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad request parameters"
        mock_get.return_value = mock_resp

        with pytest.raises(EMSCAPIError, match="Bad request"):
            provider.fetch_events(
                datetime(2026, 9, 9, tzinfo=timezone.utc),
                datetime(2026, 9, 12, tzinfo=timezone.utc),
            )

        assert mock_get.call_count == 1

    @patch(REQUESTS_GET)
    def test_timeout_retries(self, mock_get, provider):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")
        provider.retry_delay = 0.01

        with pytest.raises(EMSCAPIError, match="timed out"):
            provider.fetch_events(
                datetime(2026, 9, 9, tzinfo=timezone.utc),
                datetime(2026, 9, 12, tzinfo=timezone.utc),
            )

        assert mock_get.call_count == provider.max_retries
