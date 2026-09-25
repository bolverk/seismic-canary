"""FDSN station discovery and metadata module.

Discovers seismic stations near events using FDSN web services,
caches metadata, and computes station-to-event geometry.
"""
import json
import logging
import math
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class StationInfo:
    """Metadata for a single seismic station."""
    network: str
    station: str
    latitude: float
    longitude: float
    elevation_m: float
    description: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # Computed relative to an event
    distance_km: Optional[float] = None
    distance_deg: Optional[float] = None
    azimuth: Optional[float] = None
    back_azimuth: Optional[float] = None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance between two points in km using haversine formula.

    Args:
        lat1, lon1: First point (degrees).
        lat2, lon2: Second point (degrees).

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def degrees_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute angular distance in degrees.

    Uses the spherical law of cosines.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    cos_dist = (
        math.sin(phi1) * math.sin(phi2)
        + math.cos(phi1) * math.cos(phi2) * math.cos(dlambda)
    )
    # Clamp for numerical safety
    cos_dist = max(-1.0, min(1.0, cos_dist))
    return math.degrees(math.acos(cos_dist))


def compute_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute forward azimuth from point 1 to point 2.

    Args:
        lat1, lon1: Source point (degrees).
        lat2, lon2: Target point (degrees).

    Returns:
        Azimuth in degrees (0-360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


class StationDiscovery:
    """Discover and manage seismic station metadata.

    Uses FDSN web services (via ObsPy) to find stations near events.
    Caches results to minimize API calls.
    """

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        search_radius_deg: float = Config.FDSN_STATION_RADIUS_DEG,
        cache_path: str = Config.STATIONS_CACHE,
    ):
        self.providers = providers if providers is not None else Config.FDSN_PROVIDERS
        self.search_radius_deg = search_radius_deg
        self.cache_path = cache_path
        self._cache: Optional[Dict] = None

    def find_stations(
        self,
        event_lat: float,
        event_lon: float,
        event_time: Optional[datetime] = None,
        max_stations: int = 50,
    ) -> List[StationInfo]:
        """Find seismic stations near an event location.

        Args:
            event_lat: Event latitude (degrees).
            event_lon: Event longitude (degrees).
            event_time: Event origin time (for checking station operation).
            max_stations: Maximum number of stations to return.

        Returns:
            List of StationInfo sorted by distance.
        """
        stations = []

        for provider_name in self.providers:
            try:
                provider_stations = self._query_provider(
                    provider_name, event_lat, event_lon, event_time
                )
                stations.extend(provider_stations)
            except Exception as e:
                logger.warning(
                    f"Failed to query {provider_name}: {e}"
                )
                continue

        # Compute distances and sort
        for station in stations:
            station.distance_km = haversine_distance(
                event_lat, event_lon, station.latitude, station.longitude
            )
            station.distance_deg = degrees_distance(
                event_lat, event_lon, station.latitude, station.longitude
            )
            station.azimuth = compute_azimuth(
                event_lat, event_lon, station.latitude, station.longitude
            )
            station.back_azimuth = compute_azimuth(
                station.latitude, station.longitude, event_lat, event_lon
            )

        # Sort by distance and limit
        stations.sort(key=lambda s: s.distance_km or float("inf"))
        return stations[:max_stations]

    def _query_provider(
        self,
        provider_name: str,
        event_lat: float,
        event_lon: float,
        event_time: Optional[datetime],
    ) -> List[StationInfo]:
        """Query a single FDSN provider for stations.

        Uses ObsPy's FDSN client.
        """
        try:
            from obspy.clients.fdsn import Client
            from obspy import UTCDateTime
        except ImportError:
            logger.warning("ObsPy not available. Cannot query FDSN stations.")
            return []

        try:
            client = Client(provider_name)
        except Exception as e:
            logger.warning(f"Cannot create FDSN client for {provider_name}: {e}")
            return []

        kwargs = {
            "latitude": event_lat,
            "longitude": event_lon,
            "maxradius": self.search_radius_deg,
            "level": "station",
            "channel": "BH*,HH*",  # broadband channels
        }

        if event_time:
            utc_time = UTCDateTime(event_time)
            kwargs["startbefore"] = utc_time
            kwargs["endafter"] = utc_time

        try:
            inventory = client.get_stations(**kwargs)
        except Exception as e:
            logger.warning(f"get_stations failed for {provider_name}: {e}")
            return []

        stations = []
        for network in inventory:
            for station in network:
                info = StationInfo(
                    network=network.code,
                    station=station.code,
                    latitude=station.latitude,
                    longitude=station.longitude,
                    elevation_m=station.elevation,
                    description=station.site.name if station.site else "",
                    start_date=str(station.start_date) if station.start_date else None,
                    end_date=str(station.end_date) if station.end_date else None,
                )
                stations.append(info)

        logger.info(f"Found {len(stations)} stations from {provider_name}")
        return stations

    def load_cache(self) -> Dict:
        """Load station cache from disk."""
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self.cache_path):
            self._cache = {}
            return self._cache

        try:
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._cache = {}

        return self._cache

    def save_cache(self) -> None:
        """Save station cache to disk."""
        if self._cache is None:
            return

        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2, default=str)

    def get_cached_stations(
        self, event_lat: float, event_lon: float, tolerance_deg: float = 0.5
    ) -> Optional[List[StationInfo]]:
        """Check cache for stations near a location.

        Args:
            event_lat: Event latitude.
            event_lon: Event longitude.
            tolerance_deg: How close a cached query must be to reuse.

        Returns:
            List of StationInfo if cache hit, None if miss.
        """
        cache = self.load_cache()
        cache_key = f"{event_lat:.1f}_{event_lon:.1f}"

        if cache_key in cache:
            entries = cache[cache_key]
            return [StationInfo(**entry) for entry in entries]

        return None

    def cache_stations(
        self, event_lat: float, event_lon: float, stations: List[StationInfo]
    ) -> None:
        """Store stations in cache."""
        cache = self.load_cache()
        cache_key = f"{event_lat:.1f}_{event_lon:.1f}"
        cache[cache_key] = [asdict(s) for s in stations]
        self._cache = cache
        self.save_cache()


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Find stations near a seismic event")
    parser.add_argument("--event-lat", type=float, required=True)
    parser.add_argument("--event-lon", type=float, required=True)
    parser.add_argument("--radius", type=float, default=Config.FDSN_STATION_RADIUS_DEG)
    args = parser.parse_args()

    discovery = StationDiscovery(search_radius_deg=args.radius)

    print(f"Searching for stations within {args.radius}° of ({args.event_lat}, {args.event_lon})...")
    stations = discovery.find_stations(args.event_lat, args.event_lon)

    if not stations:
        print("No stations found.")
    else:
        print(f"\nFound {len(stations)} stations:\n")
        print(f"{'Network':<8} {'Station':<8} {'Dist (km)':<12} {'Dist (°)':<10} {'Azimuth':<10} {'Description'}")
        print("-" * 80)
        for s in stations[:20]:
            print(
                f"{s.network:<8} {s.station:<8} "
                f"{s.distance_km:>8.1f}    {s.distance_deg:>6.2f}     "
                f"{s.azimuth:>6.1f}    {s.description}"
            )
