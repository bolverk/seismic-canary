"""Centralized configuration for Seismic Canary."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RegionBounds:
    """Geographic bounding box for the monitored region."""
    min_latitude: float = 24.0
    max_latitude: float = 42.0
    min_longitude: float = 43.0
    max_longitude: float = 64.0

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center of the bounding box (lat, lon)."""
        return (
            (self.min_latitude + self.max_latitude) / 2,
            (self.min_longitude + self.max_longitude) / 2,
        )


class Config:
    """Application configuration.

    All configuration is centralized here. Do not scatter
    geographic coordinates or API endpoints throughout the code.
    """

    # Project metadata
    PROJECT_NAME = "Seismic Canary"
    VERSION = "0.2.0"
    PROCESSING_VERSION = "0.1.0"
    MODEL_VERSION = "baseline-001"

    # Geographic scope: Iran + neighbors
    REGION_BOUNDS = RegionBounds()

    # Monitored countries (for display purposes)
    MONITORED_REGION_DESCRIPTION = (
        "Iran, Iraq, Turkey, Armenia, Azerbaijan, Turkmenistan, "
        "and the Persian Gulf region"
    )

    # USGS FDSN Event API
    USGS_API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    USGS_TIMEOUT_SECONDS = 30
    USGS_MAX_RETRIES = 3
    USGS_RETRY_DELAY_SECONDS = 5

    # FDSN Waveform/Station services
    FDSN_PROVIDERS = ["IRIS", "ORFEUS", "GFZ"]
    FDSN_STATION_RADIUS_DEG = 10.0  # degrees from event
    FDSN_WAVEFORM_PRE_SECONDS = 30.0
    FDSN_WAVEFORM_POST_SECONDS = 120.0

    # Event processing thresholds
    WAVEFORM_MIN_MAGNITUDE = 3.0
    WAVEFORM_SHALLOW_DEPTH_KM = 5.0
    DEFAULT_FETCH_DAYS = 30  # How far back to fetch on first run
    INCREMENTAL_FETCH_HOURS = 2  # Overlap for incremental fetches

    # Monitored sites of interest (nuclear / military facilities).
    # Centralized here so the dashboard, anomaly location rule, and any
    # future tooling all reference the same set of coordinates.
    SITES_OF_INTEREST = [
        {"name": "Natanz", "lat": 33.717, "lon": 51.717,
         "desc": "Uranium enrichment facility (underground)"},
        {"name": "Fordow", "lat": 34.885, "lon": 50.996,
         "desc": "Underground uranium enrichment plant, near Qom"},
        {"name": "Parchin", "lat": 35.520, "lon": 51.780,
         "desc": "Military complex, suspected weapons-related experiments"},
        {"name": "Semnan", "lat": 35.235, "lon": 53.921,
         "desc": "Space/missile center; Project Midan test area nearby"},
        {"name": "Lut Desert (Nayband area)", "lat": 33.5, "lon": 57.5,
         "desc": "Remote desert; identified as potential test site in IAEA archive"},
        {"name": "Project Midan (SE of Semnan)", "lat": 34.8, "lon": 54.5,
         "desc": "Underground nuclear test site development (per IAEA archive)"},
    ]

    # Distance threshold for the location rule to consider an event
    # "co-located" with a monitored site of interest.
    SITE_PROXIMITY_KM = 30.0

    # Anomaly model thresholds
    ALERT_LEVEL_THRESHOLDS = {
        0: 0.0,    # ordinary
        1: 0.3,    # unusual
        2: 0.6,    # probable explosion
    }

    # Alert generation criteria
    ALERT_MIN_STATION_COUNT = 3
    ALERT_MIN_CONFIDENCE = 0.5

    # Data paths
    DATA_DIR = "data"
    EVENTS_PARQUET = "data/events.parquet"
    ALERTS_PARQUET = "data/alerts.parquet"
    STATIONS_CACHE = "data/stations_cache.json"
    TIMELINES_DIR = "data/timelines"

    # Waveform processing
    BANDPASS_FREQMIN = 1.0  # Hz
    BANDPASS_FREQMAX = 10.0  # Hz
    P_WINDOW_SECONDS = 3.0
    S_WINDOW_SECONDS = 5.0

    # Display
    MAP_ZOOM_START = 5
    EVENT_COLORS = {
        0: "blue",       # ordinary
        1: "orange",     # unusual
        2: "red",        # probable explosion
        -1: "gray",      # insufficient data
    }

    EVENT_LEVEL_LABELS = {
        0: "Ordinary",
        1: "Unusual",
        2: "Probable Explosion",
        -1: "Insufficient Data",
    }
