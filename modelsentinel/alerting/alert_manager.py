"""
ModelSentinel alert manager — routes alerts to configured channels with deduplication.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import smtplib
import time
from email.mime.text import MIMEText
import requests
from modelsentinel.components.base import MonitoringResult
from modelsentinel.config.schema import AlertingConfig
from modelsentinel.utils.constants import AlertSeverity
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)
_DEDUP_CACHE: dict[str, float] = {}


def _dedup_key(result: MonitoringResult) -> str:
    return f"{result.component_name}:{result.alert_message}"


def _is_duplicate(key: str, window_minutes: int) -> bool:
    now = time.time()
    if key in _DEDUP_CACHE and now - _DEDUP_CACHE[key] < window_minutes * 60:
        return True
    _DEDUP_CACHE[key] = now
    return False


class AlertManager:
    def __init__(self, alerting_config: AlertingConfig) -> None:
        self.config = alerting_config

    def send(self, result: MonitoringResult, model_id: str) -> None:
        if not result.alert_triggered or result.alert_severity is None:
            return
        key = _dedup_key(result)
        if _is_duplicate(key, self.config.deduplication_window_minutes):
            logger.info(f"Alert suppressed (duplicate): {key}")
            return
        for channel in self.config.channels:
            if result.alert_severity not in channel.severities:
                continue
            try:
                if channel.type == "slack":
                    self._send_slack(channel.webhook_url, result, model_id)
                elif channel.type == "pagerduty":
                    self._send_pagerduty(channel.routing_key, result, model_id)
                elif channel.type == "email":
                    self._send_email(channel.recipients, result, model_id)
                elif channel.type == "webhook":
                    self._send_webhook(channel.webhook_url, result, model_id)
            except Exception as exc:
                logger.error(f"Failed to send alert via {channel.type}: {exc}")

    def _send_slack(self, webhook_url: str, result: MonitoringResult, model_id: str) -> None:
        emoji = "🔴" if result.alert_severity == AlertSeverity.CRITICAL else "🟡"
        payload = {"text": (f"{emoji} *ModelSentinel* — `{model_id}`\n"
                            f"*Component:* {result.component_name}\n"
                            f"*Severity:* {result.alert_severity}\n"
                            f"*Message:* {result.alert_message}")}
        requests.post(webhook_url, json=payload, timeout=10).raise_for_status()
        logger.info(f"Slack alert sent for {result.component_name}")

    def _send_pagerduty(self, routing_key: str, result: MonitoringResult, model_id: str) -> None:
        payload = {
            "routing_key": routing_key, "event_action": "trigger",
            "payload": {
                "summary": f"ModelSentinel {result.alert_severity}: {result.alert_message}",
                "source": f"modelsentinel/{model_id}",
                "severity": "critical" if result.alert_severity == AlertSeverity.CRITICAL else "warning",
                "custom_details": result.to_dict(),
            },
        }
        requests.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=10).raise_for_status()
        logger.info(f"PagerDuty alert triggered for {result.component_name}")

    def _send_email(self, recipients: list[str], result: MonitoringResult, model_id: str) -> None:
        body = (f"ModelSentinel Alert\nModel: {model_id}\nComponent: {result.component_name}\n"
                f"Severity: {result.alert_severity}\nMessage: {result.alert_message}\n\n"
                f"Recommendations:\n" + "\n".join(f"- {r}" for r in result.recommendations))
        msg = MIMEText(body)
        msg["Subject"] = f"[ModelSentinel] {result.alert_severity} — {model_id}"
        msg["From"] = "modelsentinel@noreply.internal"
        msg["To"] = ", ".join(recipients)
        with smtplib.SMTP("localhost") as server:
            server.sendmail(msg["From"], recipients, msg.as_string())
        logger.info(f"Email alert sent to {recipients}")

    def _send_webhook(self, webhook_url: str, result: MonitoringResult, model_id: str) -> None:
        requests.post(webhook_url, json={"model_id": model_id, **result.to_dict()}, timeout=10).raise_for_status()
        logger.info(f"Webhook alert sent to {webhook_url}")
