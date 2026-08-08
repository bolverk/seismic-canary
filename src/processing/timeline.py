"""Event timeline tracking.

Records the processing lifecycle of each event for transparency
and reproducibility.
"""
import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict

from src.config import Config
from src.models.evidence import TimelineEntry

logger = logging.getLogger(__name__)


class EventTimeline:
    """Manages the processing timeline for a single event."""

    def __init__(self, event_id: str):
        self.event_id = event_id
        self.entries: List[TimelineEntry] = []

    def add(
        self,
        entry_type: str,
        description: str,
        data_reference: Optional[str] = None,
    ) -> None:
        """Add a timeline entry.

        Args:
            entry_type: One of: detected, processed, assessed, updated, alerted
            description: Human-readable description.
            data_reference: Optional reference to data source/file.
        """
        entry = TimelineEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type=entry_type,
            description=description,
            data_reference=data_reference,
        )
        self.entries.append(entry)

    def record_detection(self, source: str) -> None:
        """Record when an event was first detected."""
        self.add("detected", f"Event first detected from {source}", source)

    def record_processing(self, processing_type: str, version: str) -> None:
        """Record when features were computed."""
        self.add("processed", f"{processing_type} (v{version})")

    def record_assessment(self, model_version: str, alert_level: int) -> None:
        """Record when anomaly assessment was run."""
        self.add(
            "assessed",
            f"Anomaly model {model_version} → alert level {alert_level}",
        )

    def record_alert(self, alert_id: str) -> None:
        """Record when an alert was generated."""
        self.add("alerted", f"Alert generated: {alert_id}", alert_id)

    def record_update(self, description: str) -> None:
        """Record when event data was updated."""
        self.add("updated", description)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict) -> "EventTimeline":
        """Deserialize from dictionary."""
        timeline = cls(data["event_id"])
        timeline.entries = [
            TimelineEntry.from_dict(e) for e in data.get("entries", [])
        ]
        return timeline

    @classmethod
    def from_json(cls, json_str: str) -> "EventTimeline":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def save_timeline(timeline: EventTimeline, base_dir: str = Config.TIMELINES_DIR) -> str:
    """Save a timeline to disk.

    Args:
        timeline: The timeline to save.
        base_dir: Directory to store timeline files.

    Returns:
        Path to the saved file.
    """
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{timeline.event_id}.json")

    with open(path, "w") as f:
        f.write(timeline.to_json())

    return path


def load_timeline(event_id: str, base_dir: str = Config.TIMELINES_DIR) -> Optional[EventTimeline]:
    """Load a timeline from disk.

    Args:
        event_id: Event identifier.
        base_dir: Directory where timelines are stored.

    Returns:
        EventTimeline if found, None otherwise.
    """
    path = os.path.join(base_dir, f"{event_id}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            return EventTimeline.from_json(f.read())
    except (json.JSONDecodeError, IOError, KeyError) as e:
        logger.warning(f"Failed to load timeline for {event_id}: {e}")
        return None
