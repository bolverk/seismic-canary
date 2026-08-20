"""Geographic helpers for site-of-interest proximity and event location.

These helpers let the dashboard and the anomaly model's location rule
share a single notion of "how far is this event from a monitored site
of interest".

Note: proximity to a monitored site of interest is a *monitoring* signal
(flag events co-located with known nuclear/military facilities for human
review) — it does not by itself establish that an event was related to
that site.
"""
import logging
from typing import Dict, List, Optional, Tuple

from src.config import Config
from src.ingestion.stations import haversine_distance

logger = logging.getLogger(__name__)


def nearest_site_of_interest(
    latitude: Optional[float],
    longitude: Optional[float],
    sites: Optional[List[Dict]] = None,
) -> Optional[Tuple[Dict, float]]:
    """Return the nearest monitored site of interest and its distance in km.

    Args:
        latitude: Event latitude (degrees) or None.
        longitude: Event longitude (degrees) or None.
        sites: Site list to search (defaults to Config.SITES_OF_INTEREST).

    Returns:
        Tuple of (site dict, distance_km) if coordinates are provided,
        otherwise None.
    """
    if latitude is None or longitude is None:
        return None

    if sites is None:
        sites = Config.SITES_OF_INTEREST

    best_site = None
    best_distance = float("inf")
    for site in sites:
        distance = haversine_distance(
            latitude, longitude, float(site["lat"]), float(site["lon"])
        )
        if distance < best_distance:
            best_distance = distance
            best_site = site

    if best_site is None:
        return None
    return best_site, best_distance


def is_near_site_of_interest(
    latitude: Optional[float],
    longitude: Optional[float],
    radius_km: float = Config.SITE_PROXIMITY_KM,
    sites: Optional[List[Dict]] = None,
) -> Tuple[bool, Optional[Dict], Optional[float]]:
    """Check whether a location is within `radius_km` of a site of interest.

    Args:
        latitude: Event latitude (degrees) or None.
        longitude: Event longitude (degrees) or None.
        radius_km: Proximity threshold in km.
        sites: Site list to search (defaults to Config.SITES_OF_INTEREST).

    Returns:
        Tuple of (is_near, nearest_site, distance_km). When coordinates are
        missing, returns (False, None, None).
    """
    nearest = nearest_site_of_interest(latitude, longitude, sites=sites)
    if nearest is None:
        return False, None, None
    site, distance = nearest
    return distance <= radius_km, site, distance
