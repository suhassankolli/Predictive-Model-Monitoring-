"""Tests for AlertManager."""
import pytest
from unittest.mock import MagicMock, patch
from modelsentinel.alerting.alert_manager import AlertManager, _DEDUP_CACHE
from modelsentinel.components.base import MonitoringResult
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus

def make_cfg(ch_type="slack"):
    ch = MagicMock(); ch.type = ch_type
    ch.webhook_url = "https://hooks.slack.com/test"
    ch.routing_key = "key"; ch.recipients = ["t@example.com"]
    ch.severities = [AlertSeverity.WARNING, AlertSeverity.CRITICAL]
    cfg = MagicMock(); cfg.channels = [ch]; cfg.deduplication_window_minutes = 60
    return cfg

def critical_result():
    return MonitoringResult(component_name="run_test", status=MonitoringStatus.CRITICAL,
                             alert_triggered=True, alert_severity=AlertSeverity.CRITICAL,
                             alert_message="PSI RED")

def test_no_send_when_not_triggered():
    mgr = AlertManager(make_cfg())
    r = MonitoringResult(component_name="x", status=MonitoringStatus.OK, alert_triggered=False)
    with patch("modelsentinel.alerting.alert_manager.requests.post") as mock:
        mgr.send(r, "m"); mock.assert_not_called()

def test_slack_sent_on_critical():
    _DEDUP_CACHE.clear()
    mgr = AlertManager(make_cfg("slack"))
    with patch("modelsentinel.alerting.alert_manager.requests.post") as mock:
        mock.return_value.raise_for_status = MagicMock()
        mgr.send(critical_result(), "m"); mock.assert_called_once()

def test_dedup_suppresses():
    _DEDUP_CACHE.clear()
    mgr = AlertManager(make_cfg("slack"))
    with patch("modelsentinel.alerting.alert_manager.requests.post") as mock:
        mock.return_value.raise_for_status = MagicMock()
        mgr.send(critical_result(), "m"); mgr.send(critical_result(), "m")
        assert mock.call_count == 1
