"""Alert system for Seismic Canary.

Generates and manages alerts for anomalous seismic events.
Provides an abstract notification interface for future channels.
"""
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import pandas as pd

from src.config import Config
from src.models.anomaly import Assessment

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """An alert for an anomalous seismic event."""
    alert_id: str
    event_id: str
    alert_level: int
    timestamp: datetime
    summary_text: str
    contributing_evidence: List[str] = field(default_factory=list)
    data_links: List[str] = field(default_factory=list)
    acknowledged: bool = False
    model_version: str = Config.MODEL_VERSION


def generate_alert(
    event_id: str,
    assessment: Assessment,
    event_data: Dict[str, Any],
) -> Optional[Alert]:
    """Generate an alert if criteria are met.

    Criteria:
    - alert_level >= 1
    - confidence >= ALERT_MIN_CONFIDENCE
    - station_count >= ALERT_MIN_STATION_COUNT (if waveform data available)

    Args:
        event_id: Event identifier.
        assessment: Assessment from anomaly model.
        event_data: Event metadata dict.

    Returns:
        Alert if criteria met, None otherwise.
    """
    # Check alert level
    if assessment.alert_level < 1:
        return None

    # Check confidence
    if assessment.confidence < Config.ALERT_MIN_CONFIDENCE:
        return None

    # Check station count (only if waveform features are present)
    station_count = event_data.get("station_count")
    has_waveforms = event_data.get("p_s_ratio") is not None
    if has_waveforms and station_count is not None:
        if station_count < Config.ALERT_MIN_STATION_COUNT:
            return None

    # Generate alert
    mag = event_data.get("magnitude", "?")
    depth = event_data.get("depth_km", "?")
    place = event_data.get("place", "Unknown location")

    # Build neutral summary
    level_label = Config.EVENT_LEVEL_LABELS.get(assessment.alert_level, "Unknown")
    summary = (
        f"Unusual seismic event detected: M{mag} at {depth}km depth "
        f"near {place}. Assessment: {level_label}."
    )

    # Contributing evidence
    evidence = []
    for contrib in assessment.contributing_features:
        if contrib.confidence > 0 and contrib.contribution_explosion > 0.5:
            evidence.append(f"{contrib.feature_name}: {contrib.explanation}")

    # Data links
    links = []
    source_url = event_data.get("source_url")
    if source_url:
        links.append(source_url)

    alert_id = f"alert_{event_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    return Alert(
        alert_id=alert_id,
        event_id=event_id,
        alert_level=assessment.alert_level,
        timestamp=datetime.now(timezone.utc),
        summary_text=summary,
        contributing_evidence=evidence,
        data_links=links,
        model_version=assessment.model_version,
    )


def load_alerts(path: str = Config.ALERTS_PARQUET) -> pd.DataFrame:
    """Load alerts from Parquet storage."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "alert_id", "event_id", "alert_level", "timestamp",
            "summary_text", "contributing_evidence", "data_links",
            "acknowledged", "model_version",
        ])
    return pd.read_parquet(path)


def save_alerts(alerts: List[Alert], path: str = Config.ALERTS_PARQUET) -> None:
    """Save alerts to Parquet storage."""
    if not alerts:
        return

    records = []
    for alert in alerts:
        record = asdict(alert)
        # Convert lists to strings for Parquet storage
        record["contributing_evidence"] = "|".join(record["contributing_evidence"])
        record["data_links"] = "|".join(record["data_links"])
        record["timestamp"] = alert.timestamp
        records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


# Notification interface for future channels

class NotificationChannel(ABC):
    """Abstract interface for alert notification delivery."""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Human-readable channel name."""
        ...

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert notification.

        Args:
            alert: The alert to send.

        Returns:
            True if successfully sent, False otherwise.
        """
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if this channel is properly configured."""
        ...


class LogChannel(NotificationChannel):
    """Simple logging-based notification channel (always available)."""

    @property
    def channel_name(self) -> str:
        return "log"

    def send(self, alert: Alert) -> bool:
        logger.info(
            f"[ALERT Level {alert.alert_level}] {alert.summary_text} "
            f"(Event: {alert.event_id})"
        )
        return True

    def is_configured(self) -> bool:
        return True


class WebhookChannel(NotificationChannel):
    """Webhook notification channel (placeholder)."""

    def __init__(self, url: Optional[str] = None):
        self.url = url

    @property
    def channel_name(self) -> str:
        return "webhook"

    def send(self, alert: Alert) -> bool:
        raise NotImplementedError("Webhook notifications not yet implemented")

    def is_configured(self) -> bool:
        return self.url is not None


class EmailChannel(NotificationChannel):
    """Email notification channel (placeholder)."""

    @property
    def channel_name(self) -> str:
        return "email"

    def send(self, alert: Alert) -> bool:
        raise NotImplementedError("Email notifications not yet implemented")

    def is_configured(self) -> bool:
        return False


class TelegramChannel(NotificationChannel):
    """Telegram notification channel (placeholder)."""

    @property
    def channel_name(self) -> str:
        return "telegram"

    def send(self, alert: Alert) -> bool:
        raise NotImplementedError("Telegram notifications not yet implemented")

    def is_configured(self) -> bool:
        return False
