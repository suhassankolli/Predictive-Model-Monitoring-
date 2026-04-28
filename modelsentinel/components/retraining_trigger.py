"""
Retraining trigger monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus


@monitoring_component
class RetrainingTriggerComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_retraining_triggers"
    COMPONENT_DESCRIPTION = (
        "Evaluates rule-based retraining triggers, multi-signal urgency scoring, "
        "and scheduled periodic review. Returns structured retraining recommendation."
    )

    def __init__(self, config, current_metrics=None, last_retrain_date=None,
                 last_trigger_dates=None, spark_session=None):
        super().__init__(config, spark_session)
        self.current_metrics = current_metrics or {}
        self.last_retrain_date = last_retrain_date
        self.last_trigger_dates = last_trigger_dates or {}
        self.retrain_cfg = config.monitoring.retraining_triggers

    def run(self) -> MonitoringResult:
        if not self.retrain_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="disabled.")
        metrics, triggered_rules, recommendations = {}, [], []
        now = datetime.now(timezone.utc)

        for rule in self.retrain_cfg.rules:
            current_val = self.current_metrics.get(rule.metric)
            if current_val is None: continue
            breached = ((rule.direction == "above" and current_val > rule.threshold) or
                        (rule.direction == "below" and current_val < rule.threshold))
            last_trigger = self.last_trigger_dates.get(rule.metric)
            in_cooldown = (last_trigger is not None and
                           (now - last_trigger).days < rule.cooldown_days)
            triggered = breached and not in_cooldown
            metrics[f"rule_{rule.metric}"] = {
                "current_value": round(current_val, 6), "threshold": rule.threshold,
                "breached": breached, "in_cooldown": in_cooldown, "triggered": triggered,
            }
            if triggered:
                triggered_rules.append(rule.metric)
                recommendations.append(
                    f"Rule triggered: '{rule.metric}'={current_val:.4f} "
                    f"({rule.direction} {rule.threshold}). Initiate retraining review.")

        n = len(triggered_rules)
        urgency = "IMMEDIATE" if n >= 2 else "ADVISORY" if n == 1 else "NONE"
        metrics["urgency_score"] = {"level": urgency, "triggered_rule_count": n}

        scheduled_due = False
        if self.last_retrain_date is not None:
            months = ((now.year - self.last_retrain_date.year) * 12
                      + now.month - self.last_retrain_date.month)
            scheduled_due = months >= self.retrain_cfg.scheduled_review_months
            metrics["scheduled_review"] = {"months_since_retrain": months,
                                            "review_due": scheduled_due}
            if scheduled_due:
                recommendations.append(
                    f"Periodic review DUE ({months} months since retraining). "
                    "SR 11-7 requires this to be completed and documented.")

        alert_triggered = bool(triggered_rules) or scheduled_due
        worst_status = (MonitoringStatus.CRITICAL if urgency == "IMMEDIATE"
                        else (MonitoringStatus.WARNING if alert_triggered else MonitoringStatus.OK))
        alert_sev = (AlertSeverity.CRITICAL if worst_status == MonitoringStatus.CRITICAL
                     else (AlertSeverity.WARNING if worst_status == MonitoringStatus.WARNING else None))
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=alert_triggered,
                                alert_severity=alert_sev,
                                alert_message=(f"Retraining: {urgency}. {n} rule(s) triggered."
                                               + (" Periodic review overdue." if scheduled_due else ""))
                                if alert_triggered else None,
                                recommendations=recommendations)

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {"name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {
                    "current_metrics": {"type": "object"},
                }, "required": []}}
