"""Tests for the geographic site-of-interest helpers."""
from src.geography import nearest_site_of_interest, is_near_site_of_interest
from src.config import Config


class TestNearestSiteOfInterest:
    def test_missing_coordinates_returns_none(self):
        assert nearest_site_of_interest(None, 51.7) is None
        assert nearest_site_of_interest(33.7, None) is None

    def test_at_site_returns_that_site(self):
        # Natanz is at (33.717, 51.717)
        nearest = nearest_site_of_interest(33.717, 51.717)
        assert nearest is not None
        site, distance = nearest
        assert site["name"] == "Natanz"
        assert distance < 2.0  # ~km

    def test_custom_site_list(self):
        sites = [{"name": "A", "lat": 0.0, "lon": 0.0}]
        nearest = nearest_site_of_interest(0.1, 0.1, sites=sites)
        assert nearest[0]["name"] == "A"

    def test_config_exposes_sites(self):
        assert len(Config.SITES_OF_INTEREST) >= 6


class TestIsNearSiteOfInterest:
    def test_within_radius(self):
        near, site, distance = is_near_site_of_interest(33.72, 51.72)
        assert near is True
        assert site["name"] == "Natanz"
        assert distance < 30.0

    def test_outside_radius(self):
        near, _, _ = is_near_site_of_interest(36.0, 53.0)  # ~300 km from sites
        assert near is False

    def test_custom_radius(self):
        # 33.735 is ~2 km north of Natanz (33.717) → outside a 0.5 km radius
        near, _, _ = is_near_site_of_interest(33.735, 51.72, radius_km=0.5)
        assert near is False

    def test_missing_coordinates(self):
        near, site, distance = is_near_site_of_interest(None, None)
        assert near is False
        assert site is None
        assert distance is None
