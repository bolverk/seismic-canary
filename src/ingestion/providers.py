"""Abstract sensor provider interfaces for future modalities.

Defines the contract that each sensor modality must implement
to contribute observations to the evidence fusion system.

Currently unimplemented — these are architectural placeholders
to ensure the system can be extended without redesign.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from src.models.evidence import Observation


class SensorProvider(ABC):
    """Abstract base class for all sensor data providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique name for this provider."""
        ...

    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """Sensor modality type (seismic, infrasound, satellite, etc.)."""
        ...

    @abstractmethod
    def fetch(
        self,
        latitude: float,
        longitude: float,
        event_time: datetime,
        **kwargs,
    ) -> List[Observation]:
        """Fetch observations for a given event location and time.

        Args:
            latitude: Event latitude.
            longitude: Event longitude.
            event_time: Event origin time.

        Returns:
            List of Observations from this sensor.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is currently operational."""
        ...

    @abstractmethod
    def get_status(self) -> str:
        """Get human-readable status of this provider."""
        ...


class InfrasoundProvider(SensorProvider):
    """Infrasound detection provider (placeholder).

    Future implementation will:
    - Calculate predicted infrasound arrival times
    - Query available infrasound observations
    - Estimate back-azimuth
    - Assess whether source region overlaps seismic source
    """

    @property
    def provider_name(self) -> str:
        return "infrasound_ctbto"

    @property
    def sensor_type(self) -> str:
        return "infrasound"

    def fetch(self, latitude: float, longitude: float,
              event_time: datetime, **kwargs) -> List[Observation]:
        raise NotImplementedError(
            "Infrasound provider not yet implemented. "
            "Requires access to IMS infrasound data."
        )

    def is_available(self) -> bool:
        return False

    def get_status(self) -> str:
        return "Not implemented. Planned for future version."


class SatelliteProvider(SensorProvider):
    """Satellite imagery provider (placeholder).

    Future implementation will:
    - Query Copernicus/Sentinel-1 SAR products
    - Query Sentinel-2 optical products
    - Perform change detection around event locations
    - Report surface disturbance indicators
    """

    @property
    def provider_name(self) -> str:
        return "sentinel_copernicus"

    @property
    def sensor_type(self) -> str:
        return "satellite"

    def fetch(self, latitude: float, longitude: float,
              event_time: datetime, **kwargs) -> List[Observation]:
        raise NotImplementedError(
            "Satellite provider not yet implemented. "
            "Requires Copernicus Data Space API integration."
        )

    def is_available(self) -> bool:
        return False

    def get_status(self) -> str:
        return "Not implemented. Planned for future version."


class GNSSProvider(SensorProvider):
    """GNSS/GPS displacement provider (placeholder).

    Future implementation will:
    - Query nearby GNSS stations
    - Measure surface displacement
    - Compare with seismic source model
    """

    @property
    def provider_name(self) -> str:
        return "gnss_unavco"

    @property
    def sensor_type(self) -> str:
        return "gnss"

    def fetch(self, latitude: float, longitude: float,
              event_time: datetime, **kwargs) -> List[Observation]:
        raise NotImplementedError(
            "GNSS provider not yet implemented. "
            "Requires UNAVCO/IGS data access."
        )

    def is_available(self) -> bool:
        return False

    def get_status(self) -> str:
        return "Not implemented. Planned for future version."


class RadionuclideProvider(SensorProvider):
    """Radionuclide measurement provider (placeholder).

    Future implementation will:
    - Ingest officially released radionuclide measurements
    - NOT attempt to infer from unrelated radiation data
    - Report "no publicly available data" when none exists

    The CTBTO describes radionuclide observations as the component
    capable of confirming the nuclear nature of an event.
    """

    @property
    def provider_name(self) -> str:
        return "radionuclide_public"

    @property
    def sensor_type(self) -> str:
        return "radionuclide"

    def fetch(self, latitude: float, longitude: float,
              event_time: datetime, **kwargs) -> List[Observation]:
        # By default, return empty — no public radionuclide data
        # This is intentionally NOT raising NotImplementedError
        # because "no data" is a valid observation state
        return []

    def is_available(self) -> bool:
        # Always "available" in the sense that we can report absence
        return True

    def get_status(self) -> str:
        return (
            "No publicly available real-time radionuclide data. "
            "IMS data requires CTBTO member state access or vDEC research approval."
        )
