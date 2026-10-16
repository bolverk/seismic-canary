"""Centralized configuration for Seismic Canary."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RegionBounds:
    """Geographic bounding box for the monitored region."""
    min_latitude: float = 24.0
    max_latitude: float = 42.0
    min_longitude: float = 34.0
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
    VERSION = "0.1.0"
    PROCESSING_VERSION = "0.1.0"
    MODEL_VERSION = "baseline-001"
    
    # Geographic scope: Iran + neighbors, including Israel and Lebanon
    REGION_BOUNDS = RegionBounds()
    
    # Monitored countries (for display purposes)
    MONITORED_REGION_DESCRIPTION = (
        "Iran, Iraq, Turkey, Armenia, Azerbaijan, Turkmenistan, "
        "Israel, Lebanon, and the Persian Gulf region"
    )
    
    # USGS FDSN Event API (global, authoritative for M4.5+)
    USGS_API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    USGS_TIMEOUT_SECONDS = 30
    USGS_MAX_RETRIES = 3
    USGS_RETRY_DELAY_SECONDS = 5

    # EMSC FDSN Event API (regional aggregator; far better M2-4 completeness
    # in the eastern Mediterranean, and carries ISF event-type codes that
    # distinguish explosions from earthquakes)
    EMSC_API_BASE = "https://www.seismicportal.eu/fdsnws/event/1/query"
    EMSC_TIMEOUT_SECONDS = 30
    EMSC_MAX_RETRIES = 3
    EMSC_RETRY_DELAY_SECONDS = 5
    EMSC_MIN_MAGNITUDE = 3.0  # keeps regional microseismicity manageable
    EMSC_MAX_RESULTS = 20000

    # Cross-provider reconciliation: collapse the same physical event reported
    # by more than one provider into a single record.
    CROSS_PROVIDER_TIME_TOLERANCE_SECONDS = 60.0
    CROSS_PROVIDER_DISTANCE_TOLERANCE_KM = 100.0
    
    # FDSN Waveform/Station services
    FDSN_PROVIDERS = ["IRIS", "ORFEUS", "GFZ"]
    FDSN_STATION_RADIUS_DEG = 15.0  # degrees from event (~1660 km)
    FDSN_WAVEFORM_PRE_SECONDS = 30.0
    FDSN_WAVEFORM_POST_SECONDS = 120.0
    
    # Event processing thresholds
    WAVEFORM_MIN_MAGNITUDE = 3.0
    WAVEFORM_SHALLOW_DEPTH_KM = 5.0
    DEFAULT_FETCH_DAYS = 30  # How far back to fetch on first run
    INCREMENTAL_FETCH_HOURS = 2  # Overlap for incremental fetches

    # Waveform feature enrichment (P/S, SNR). Bounded because download cost
    # grows with events x stations; a single update run only enriches a small
    # batch, prioritizing current high-alert events.
    WAVEFORM_MAX_EVENTS_PER_RUN = 20  # events enriched per update run
    # Candidate stations considered per event. The nearest stations are often
    # restricted regional networks (e.g. Iran's MP network) whose data is not
    # openly distributed, so the pool must extend far enough out to reach open
    # networks. Retrieval walks the pool nearest-first and stops at
    # WAVEFORM_TARGET_TRACES, so a well-covered event still only queries a
    # handful of nearby stations.
    WAVEFORM_MAX_STATIONS = 40
    WAVEFORM_TARGET_TRACES = 8        # stop downloading once this many have data
    WAVEFORM_MAX_ATTEMPTS = 60        # hard cap on waveform requests per event
    WAVEFORM_MIN_STATIONS = 3         # min agreeing stations to trust a P/S ratio
    
    # Anomaly model thresholds
    ALERT_LEVEL_THRESHOLDS = {
        0: 0.0,    # ordinary
        1: 0.3,    # unusual
        2: 0.6,    # probable explosion
    }

    # Level 2 is a positive claim ("probable explosion"), so it needs more
    # than the catalog depth, which is the least reliable input we have and
    # is often a near-surface default outside dense networks. When the only
    # explosion-leaning signal is depth, report the event at most as
    # "unusual" (level 1) until another discriminant corroborates it.
    ALERT_LEVEL2_REQUIRE_CORROBORATION = True

    # Curated watchlist of sites with independently confirmed explosion
    # activity (see src/models/calibration.py for the sources). An event
    # within KNOWN_EXPLOSION_RADIUS_KM of one of these sites gets an
    # explosion-leaning location contribution, which is the discriminator
    # between the south-Lebanon demolition campaign and ordinary shallow
    # regional seismicity (e.g. Iran) that is 80-200 km from any such site.
    KNOWN_EXPLOSION_SITES = (
        ("Beirut port", 33.901, 35.519),
        ("Ali al-Taher ridge", 33.3321, 35.3544),
        ("South Lebanon (IDF demolition)", 33.30, 35.30),
        ("Beaufort Castle", 33.36, 35.53),
    )
    KNOWN_EXPLOSION_RADIUS_KM = 20.0
    
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

    # Phase picking. "auto" prefers a pretrained SeisBench model (PhaseNet)
    # for P/S onset times and falls back to the classic STA/LTA picker when
    # SeisBench/torch is unavailable. "sta_lta" forces the legacy picker.
    WAVEFORM_PICKER = "auto"          # auto | seisbench | sta_lta
    SEISBENCH_MODEL = "EQTransformer"  # e.g. PhaseNet, EQTransformer
    SEISBENCH_CHECKPOINT = "instance"  # pretrained weight set
    # Optional override of the per-phase detection threshold. None (default)
    # uses the checkpoint's own calibrated default_args — model probabilities
    # are poorly calibrated for raw scores, so overriding usually loses
    # valid picks (e.g. 0.3 rejects S onsets peaking near 0.06).
    SEISBENCH_PICK_THRESHOLD = None
    
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
