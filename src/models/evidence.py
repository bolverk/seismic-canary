"""Evidence model for multi-sensor event characterization.

Defines a generic observation/evidence framework that allows
different sensor modalities (seismic, infrasound, satellite, GNSS,
radionuclide) to contribute evidence to the same event assessment.

The eventual architecture:
    seismic evidence + infrasound evidence + satellite evidence
    + GNSS evidence + radionuclide evidence → evidence fusion layer
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """A single observation from any sensor modality.

    This is the atomic unit of evidence. Each observation records
    what was measured, where, when, and how.
    """
    source: str  # e.g., "usgs", "iris", "sentinel-1"
    sensor_type: str  # e.g., "seismic", "infrasound", "satellite"
    timestamp: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    measurement_type: str = ""  # e.g., "p_s_ratio", "back_azimuth"
    measurement_value: Optional[Any] = None  # float, dict, etc.
    uncertainty: Optional[float] = None
    quality: str = "unknown"  # good, marginal, poor, unknown
    provenance: Dict[str, Any] = field(default_factory=dict)
    processing_version: str = ""

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "Observation":
        """Deserialize from dictionary."""
        data = data.copy()
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class TimelineEntry:
    """A single entry in an event's processing timeline."""
    timestamp: datetime
    entry_type: str  # detected, processed, assessed, updated, alerted
    description: str
    data_reference: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "TimelineEntry":
        data = data.copy()
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class EvidenceRecord:
    """Complete evidence record for a single event.

    Aggregates observations from all sensor modalities,
    derived features, model assessments, and the processing timeline.
    """
    event_id: str
    observations: List[Observation] = field(default_factory=list)
    derived_features: Dict[str, Any] = field(default_factory=dict)
    model_assessments: Dict[str, Any] = field(default_factory=dict)
    overall_assessment: Dict[str, float] = field(default_factory=dict)
    timeline: List[TimelineEntry] = field(default_factory=list)
    created: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    def add_observation(self, obs: Observation) -> None:
        """Add an observation to the evidence record."""
        self.observations.append(obs)
        self.last_updated = datetime.now(timezone.utc)

    def add_timeline_entry(
        self, entry_type: str, description: str, data_ref: Optional[str] = None
    ) -> None:
        """Add a timeline entry."""
        entry = TimelineEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type=entry_type,
            description=description,
            data_reference=data_ref,
        )
        self.timeline.append(entry)

    def get_observations_by_sensor(self, sensor_type: str) -> List[Observation]:
        """Get all observations of a specific sensor type."""
        return [o for o in self.observations if o.sensor_type == sensor_type]

    def get_sensor_status(self) -> Dict[str, str]:
        """Get data availability status for each sensor type."""
        sensor_types = ["seismic", "infrasound", "satellite", "gnss", "radionuclide"]
        status = {}
        for sensor in sensor_types:
            obs = self.get_observations_by_sensor(sensor)
            if obs:
                status[sensor] = f"{len(obs)} observation(s) available"
            else:
                status[sensor] = "No data available"
        return status

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "observations": [o.to_dict() for o in self.observations],
            "derived_features": self.derived_features,
            "model_assessments": self.model_assessments,
            "overall_assessment": self.overall_assessment,
            "timeline": [t.to_dict() for t in self.timeline],
            "created": self.created.isoformat() if self.created else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict) -> "EvidenceRecord":
        """Deserialize from dictionary."""
        record = cls(event_id=data["event_id"])
        record.observations = [Observation.from_dict(o) for o in data.get("observations", [])]
        record.derived_features = data.get("derived_features", {})
        record.model_assessments = data.get("model_assessments", {})
        record.overall_assessment = data.get("overall_assessment", {})
        record.timeline = [TimelineEntry.from_dict(t) for t in data.get("timeline", [])]
        if data.get("created"):
            record.created = datetime.fromisoformat(data["created"])
        if data.get("last_updated"):
            record.last_updated = datetime.fromisoformat(data["last_updated"])
        return record

    @classmethod
    def from_json(cls, json_str: str) -> "EvidenceRecord":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
