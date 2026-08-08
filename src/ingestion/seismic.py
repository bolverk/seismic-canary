"""USGS FDSN event ingestion module.

Fetches seismic events from the USGS Earthquake Catalog API,
normalizes them to the internal event schema, and provides
a provider interface for future multi-source support.
"""
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import pandas as pd
import requests

from src.config import Config, RegionBounds

logger = logging.getLogger(__name__)


class SeismicProvider(ABC):
    """Abstract base class for seismic event data providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider."""
        ...

    @abstractmethod
    def fetch_events(
        self,
        start_time: datetime,
        end_time: datetime,
        region: Optional[RegionBounds] = None,
    ) -> pd.DataFrame:
        """Fetch events from the provider.

        Args:
            start_time: Start of time window (UTC).
            end_time: End of time window (UTC).
            region: Geographic bounding box. If None, uses Config default.

        Returns:
            DataFrame conforming to the event schema.
        """
        ...


class USGSProvider(SeismicProvider):
    """USGS FDSN Event API provider.

    Fetches events from:
    https://earthquake.usgs.gov/fdsnws/event/1/query

    Supports GeoJSON format with bounding-box geographic filtering.
    """

    def __init__(
        self,
        base_url: str = Config.USGS_API_BASE,
        timeout: int = Config.USGS_TIMEOUT_SECONDS,
        max_retries: int = Config.USGS_MAX_RETRIES,
        retry_delay: float = Config.USGS_RETRY_DELAY_SECONDS,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def provider_name(self) -> str:
        return "usgs"

    def fetch_events(
        self,
        start_time: datetime,
        end_time: datetime,
        region: Optional[RegionBounds] = None,
    ) -> pd.DataFrame:
        """Fetch events from USGS FDSN API.

        Args:
            start_time: Start of time window (UTC).
            end_time: End of time window (UTC).
            region: Geographic bounding box.

        Returns:
            DataFrame with normalized event records.

        Raises:
            USGSAPIError: On unrecoverable API errors.
        """
        if region is None:
            region = Config.REGION_BOUNDS

        params = self._build_params(start_time, end_time, region)
        data = self._make_request(params)

        if data is None:
            return self._empty_df()

        return self._normalize_response(data)

    def _build_params(
        self,
        start_time: datetime,
        end_time: datetime,
        region: RegionBounds,
    ) -> Dict[str, Any]:
        """Build query parameters for the USGS API."""
        return {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minlatitude": region.min_latitude,
            "maxlatitude": region.max_latitude,
            "minlongitude": region.min_longitude,
            "maxlongitude": region.max_longitude,
            "orderby": "time",
        }

    def _make_request(self, params: Dict[str, Any]) -> Optional[Dict]:
        """Make HTTP request with retries and error handling.

        Returns:
            Parsed JSON response or None if no data.

        Raises:
            USGSAPIError: On unrecoverable errors after retries.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 204:
                    # No content - empty result
                    logger.info("USGS API returned no events (204).")
                    return None

                if response.status_code == 200:
                    data = response.json()
                    return data

                if response.status_code == 400:
                    # Bad request - don't retry
                    raise USGSAPIError(
                        f"Bad request (400): {response.text[:200]}"
                    )

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limited (429). Waiting {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    # Server error - retry
                    last_error = USGSAPIError(
                        f"Server error ({response.status_code}): "
                        f"{response.text[:200]}"
                    )
                    logger.warning(
                        f"Server error {response.status_code}, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

                # Other error codes
                raise USGSAPIError(
                    f"Unexpected status {response.status_code}: "
                    f"{response.text[:200]}"
                )

            except requests.exceptions.Timeout:
                last_error = USGSAPIError(
                    f"Request timed out after {self.timeout}s"
                )
                logger.warning(
                    f"Timeout, attempt {attempt + 1}/{self.max_retries}"
                )
                time.sleep(self.retry_delay)

            except requests.exceptions.ConnectionError as e:
                last_error = USGSAPIError(f"Connection error: {e}")
                logger.warning(
                    f"Connection error, attempt {attempt + 1}/{self.max_retries}"
                )
                time.sleep(self.retry_delay)

            except (ValueError, KeyError) as e:
                # JSON parse error or unexpected response format
                raise USGSAPIError(f"Failed to parse response: {e}")

        # All retries exhausted
        if last_error:
            raise last_error
        raise USGSAPIError("All retries exhausted with no specific error")

    def _normalize_response(self, data: Dict) -> pd.DataFrame:
        """Normalize USGS GeoJSON response to internal schema.

        Args:
            data: Parsed GeoJSON response.

        Returns:
            DataFrame conforming to event schema.
        """
        features = data.get("features", [])
        if not features:
            logger.info("USGS response contained no features.")
            return self._empty_df()

        now = datetime.now(timezone.utc)
        records = []

        for feature in features:
            try:
                record = self._normalize_feature(feature, now)
                if record is not None:
                    records.append(record)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"Skipping malformed feature: {e}. "
                    f"Feature ID: {feature.get('id', 'unknown')}"
                )
                continue

        if not records:
            return self._empty_df()

        df = pd.DataFrame(records)
        df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True)
        df["first_seen"] = pd.to_datetime(df["first_seen"], utc=True)
        df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True)
        return df

    def _normalize_feature(
        self, feature: Dict, now: datetime
    ) -> Optional[Dict]:
        """Normalize a single GeoJSON feature to an event record.

        Args:
            feature: A single GeoJSON feature from USGS response.
            now: Current timestamp for first_seen/last_updated.

        Returns:
            Dict matching event schema, or None if invalid.
        """
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])

        if not coordinates or len(coordinates) < 2:
            logger.warning(
                f"Feature {feature.get('id', 'unknown')} has no coordinates."
            )
            return None

        # USGS GeoJSON: [longitude, latitude, depth]
        longitude = float(coordinates[0])
        latitude = float(coordinates[1])
        depth_km = None
        if len(coordinates) > 2 and coordinates[2] is not None:
            try:
                depth_km = float(coordinates[2])
            except (TypeError, ValueError):
                depth_km = None

        # Origin time is in milliseconds since epoch
        time_ms = props.get("time")
        if time_ms is None:
            return None
        origin_time = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)

        # Event ID - use USGS feature id
        event_id = feature.get("id", props.get("code", ""))
        if not event_id:
            return None

        # Magnitude
        magnitude = props.get("mag")
        if magnitude is not None:
            try:
                magnitude = float(magnitude)
            except (TypeError, ValueError):
                magnitude = None

        # Source URL
        source_url = props.get("url", "")
        if source_url and not source_url.startswith("http"):
            source_url = f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}"

        return {
            "event_id": str(event_id),
            "provider": self.provider_name,
            "origin_time": origin_time,
            "latitude": latitude,
            "longitude": longitude,
            "depth_km": depth_km,
            "magnitude": magnitude,
            "magnitude_type": props.get("magType"),
            "event_type": props.get("type"),
            "place": props.get("place"),
            "source_url": source_url,
            "first_seen": now,
            "last_updated": now,
            # Feature fields - null until processing
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

    def _empty_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with the event schema."""
        from src.processing.events import get_empty_events_df
        return get_empty_events_df()


class USGSAPIError(Exception):
    """Raised when the USGS API returns an unrecoverable error."""
    pass


# CLI entrypoint for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider = USGSProvider()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)

    print(f"Fetching events from {start.isoformat()} to {end.isoformat()}")
    print(f"Region: {Config.REGION_BOUNDS}")

    try:
        events = provider.fetch_events(start, end)
        print(f"\nFound {len(events)} events:")
        if not events.empty:
            for _, row in events.head(5).iterrows():
                print(
                    f"  {row['origin_time']} | M{row['magnitude']:.1f} | "
                    f"{row['depth_km']:.1f}km | {row['place']}"
                )
            if len(events) > 5:
                print(f"  ... and {len(events) - 5} more")
    except USGSAPIError as e:
        print(f"ERROR: {e}")
