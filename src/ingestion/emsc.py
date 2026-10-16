"""EMSC (Euro-Mediterranean Seismological Centre) event ingestion module.

Fetches seismic events from the EMSC SeismicPortal FDSN Event API:
https://www.seismicportal.eu/fdsnws/event/1/query

EMSC aggregates real-time solutions from dozens of regional networks
(AFAD, KOERI, GSI, GRAL, ...). Compared with USGS it has far better
completeness for small events (down to ~M2) in the eastern Mediterranean,
and — crucially — it publishes ISF event-type codes (``evtype``) that
distinguish chemical / mining / nuclear / experimental explosions from
earthquakes. USGS omits most sub-M4.5 events and never labels a blast as
an explosion, so these events are invisible to a USGS-only pipeline.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any

import pandas as pd

from src.config import Config, RegionBounds
from src.ingestion.seismic import FDSNEventProvider, SeismicAPIError

logger = logging.getLogger(__name__)


class EMSCAPIError(SeismicAPIError):
    """Raised when the EMSC API returns an unrecoverable error."""
    pass


# ISF (IASPEI Seismic Format) event-type codes used by EMSC's ``evtype`` field.
# Leading character = confidence (k=known, s=suspected); trailing character =
# event type (e=earthquake, h=chemical explosion, m=mine explosion,
# x=experimental explosion, n=nuclear explosion, r=rockburst, i=induced,
# l=landslide, c=meteorite). We map these onto the event_type vocabulary the
# anomaly model understands (see RuleBasedModel._assess_source_mechanism).
_EMSC_EVENT_TYPE_MAP: Dict[str, str] = {
    "ke": "earthquake",
    "se": "earthquake",
    "fe": "earthquake",          # felt earthquake
    "de": "earthquake",          # damaging earthquake
    "uk": "unknown",
    "ue": "unknown",             # non-standard EMSC "unknown" variant
    "kr": "rockburst",
    "sr": "rockburst",
    "ki": "induced event",
    "si": "induced event",
    # Explosions — mapped to the exact strings the model recognises.
    "km": "mining explosion",
    "sm": "mining explosion",
    "kh": "chemical explosion",
    "sh": "chemical explosion",
    "kx": "explosion",
    "sx": "explosion",
    "kn": "nuclear explosion",
    "sn": "nuclear explosion",
    "lc": "meteorite",
    "ls": "landslide",
}


def map_emsc_event_type(evtype: Optional[str]) -> Optional[str]:
    """Translate an EMSC/ISF event-type code to the internal event_type.

    Unknown codes are passed through unchanged so they remain visible
    rather than being silently dropped.
    """
    if evtype is None:
        return None
    return _EMSC_EVENT_TYPE_MAP.get(evtype.lower(), evtype)


class EMSCProvider(FDSNEventProvider):
    """EMSC SeismicPortal FDSN Event API provider.

    Fetches events from:
    https://www.seismicportal.eu/fdsnws/event/1/query

    Uses the ``json`` (GeoJSON) output format, which shares the same
    ``FeatureCollection`` shape as USGS but uses ISO-8601 timestamps and an
    EMSC-specific ``properties`` block.
    """

    def __init__(
        self,
        base_url: str = Config.EMSC_API_BASE,
        timeout: int = Config.EMSC_TIMEOUT_SECONDS,
        max_retries: int = Config.EMSC_MAX_RETRIES,
        retry_delay: float = Config.EMSC_RETRY_DELAY_SECONDS,
        min_magnitude: float = Config.EMSC_MIN_MAGNITUDE,
        max_results: int = Config.EMSC_MAX_RESULTS,
    ):
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            error_class=EMSCAPIError,
        )
        self.min_magnitude = min_magnitude
        self.max_results = max_results

    @property
    def provider_name(self) -> str:
        return "emsc"

    def _build_params(
        self,
        start_time: datetime,
        end_time: datetime,
        region: RegionBounds,
    ) -> Dict[str, Any]:
        """Build query parameters for the EMSC API."""
        return {
            "format": "json",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minlatitude": region.min_latitude,
            "maxlatitude": region.max_latitude,
            "minlongitude": region.min_longitude,
            "maxlongitude": region.max_longitude,
            "minmagnitude": self.min_magnitude,
            "orderby": "time",
            "limit": self.max_results,
        }

    def _normalize_response(self, data: Dict) -> pd.DataFrame:
        """Normalize an EMSC GeoJSON response to the internal schema."""
        features = data.get("features", [])
        if not features:
            logger.info("EMSC response contained no features.")
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
                    "Skipping malformed EMSC feature: %s. Feature ID: %s",
                    e, feature.get("id", "unknown"),
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
        """Normalize a single EMSC GeoJSON feature to an event record."""
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])

        # EMSC properties carry authoritative (positive) lat/lon/depth.
        latitude = props.get("lat")
        longitude = props.get("lon")
        if latitude is None and len(coordinates) >= 2:
            longitude, latitude = coordinates[0], coordinates[1]
        if latitude is None or longitude is None:
            logger.warning(
                "EMSC feature %s has no coordinates.", feature.get("id", "unknown")
            )
            return None

        depth_km = props.get("depth")
        if depth_km is not None:
            try:
                depth_km = abs(float(depth_km))
            except (TypeError, ValueError):
                depth_km = None

        # Origin time is an ISO-8601 string (unlike USGS epoch milliseconds).
        time_str = props.get("time")
        if not time_str:
            return None
        origin_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        if origin_time.tzinfo is None:
            origin_time = origin_time.replace(tzinfo=timezone.utc)

        event_id = feature.get("id") or props.get("unid")
        if not event_id:
            return None

        magnitude = props.get("mag")
        if magnitude is not None:
            try:
                magnitude = float(magnitude)
            except (TypeError, ValueError):
                magnitude = None

        return {
            "event_id": str(event_id),
            "provider": self.provider_name,
            "origin_time": origin_time,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "depth_km": depth_km,
            "magnitude": magnitude,
            "magnitude_type": props.get("magtype"),
            "event_type": map_emsc_event_type(props.get("evtype")),
            "place": props.get("flynn_region"),
            "source_url": (
                "https://www.seismicportal.eu/eventdetails.html"
                f"?unid={event_id}"
            ),
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


# CLI entrypoint for testing
if __name__ == "__main__":
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)
    provider = EMSCProvider()
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
                    f"{row['depth_km']}km | {row['event_type']} | {row['place']}"
                )
            if len(events) > 5:
                print(f"  ... and {len(events) - 5} more")
    except EMSCAPIError as e:
        print(f"ERROR: {e}")
