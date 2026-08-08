"""Tests for alert system."""
import os
import tempfile

import pytest
from datetime import datetime, timezone

from src.models.alerts import (
    Alert,
    generate_alert,
    load_alerts,
    save_alerts,
    LogChannel,
    WebhookChannel,
    EmailChannel,
    TelegramChannel,
    NotificationChannel,
)
from src.models.anomaly import RuleBasedModel


@pytest.fixture
def model():
    return RuleBasedModel()


@pytest.fixture
def suspicious_event():
    return {
        "event_id": "us_suspicious",
        "depth_km": 1.5,
        "p_s_ratio": 0.7,
        "mb_ms": 1.3,
        "latitude": 34.5,
        "longitude": 55.2,
        "magnitude": 3.8,
        "place": "Central Iran",
        "source_url": "https://earthquake.usgs.gov/test",
        "station_count": 5,
    }


@pytest.fixture
def ordinary_event():
    return {
        "event_id": "us_ordinary",
        "depth_km": 25.0,
        "p_s_ratio": -0.2,
        "mb_ms": 0.3,
        "latitude": 35.0,
        "longitude": 51.0,
        "magnitude": 4.0,
        "place": "Tehran, Iran",
        "source_url": "https://earthquake.usgs.gov/test2",
        "station_count": 10,
    }


class TestAlertGeneration:
    def test_suspicious_event_generates_alert(self, model, suspicious_event):
        assessment = model.assess(suspicious_event)
        alert = generate_alert("us_suspicious", assessment, suspicious_event)
        assert alert is not None
        assert alert.alert_level >= 1
        assert "Unusual" in alert.summary_text or "M3.8" in alert.summary_text

    def test_ordinary_event_no_alert(self, model, ordinary_event):
        assessment = model.assess(ordinary_event)
        alert = generate_alert("us_ordinary", assessment, ordinary_event)
        assert alert is None

    def test_alert_has_neutral_language(self, model, suspicious_event):
        assessment = model.assess(suspicious_event)
        alert = generate_alert("us_suspicious", assessment, suspicious_event)
        assert alert is not None
        assert "nuclear" not in alert.summary_text.lower()

    def test_alert_has_evidence(self, model, suspicious_event):
        assessment = model.assess(suspicious_event)
        alert = generate_alert("us_suspicious", assessment, suspicious_event)
        assert alert is not None
        assert len(alert.contributing_evidence) > 0

    def test_alert_has_data_links(self, model, suspicious_event):
        assessment = model.assess(suspicious_event)
        alert = generate_alert("us_suspicious", assessment, suspicious_event)
        assert alert is not None
        assert len(alert.data_links) > 0


class TestAlertPersistence:
    def test_save_and_load(self, model, suspicious_event):
        assessment = model.assess(suspicious_event)
        alert = generate_alert("us_suspicious", assessment, suspicious_event)
        assert alert is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alerts.parquet")
            save_alerts([alert], path)
            loaded = load_alerts(path)
            assert len(loaded) == 1
            assert loaded.iloc[0]["event_id"] == "us_suspicious"

    def test_load_nonexistent(self):
        loaded = load_alerts("/nonexistent/alerts.parquet")
        assert len(loaded) == 0


class TestNotificationChannels:
    def test_log_channel(self):
        channel = LogChannel()
        assert channel.is_configured() is True
        assert channel.channel_name == "log"

        alert = Alert(
            alert_id="test_alert",
            event_id="test_event",
            alert_level=1,
            timestamp=datetime.now(timezone.utc),
            summary_text="Test alert",
        )
        assert channel.send(alert) is True

    def test_webhook_not_implemented(self):
        channel = WebhookChannel(url="http://example.com")
        assert channel.is_configured() is True
        with pytest.raises(NotImplementedError):
            channel.send(Alert(
                alert_id="t", event_id="t", alert_level=1,
                timestamp=datetime.now(timezone.utc), summary_text="t"
            ))

    def test_email_not_configured(self):
        channel = EmailChannel()
        assert channel.is_configured() is False

    def test_telegram_not_configured(self):
        channel = TelegramChannel()
        assert channel.is_configured() is False

    def test_all_are_notification_channels(self):
        for cls in [LogChannel, WebhookChannel, EmailChannel, TelegramChannel]:
            instance = cls() if cls != WebhookChannel else cls(url=None)
            assert isinstance(instance, NotificationChannel)
