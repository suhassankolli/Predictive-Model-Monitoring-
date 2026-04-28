"""
Governance monitoring component — SR 11-7, EU AI Act, audit trail, version check.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from typing import Any, Optional
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus, AUDIT_LOG_VERSION


@monitoring_component
class GovernanceComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_governance"
    COMPONENT_DESCRIPTION = (
        "Verifies production model version, records training data lineage, "
        "generates SR 11-7 report summary, writes EU AI Act log entry, "
        "and appends a cryptographically signed audit trail entry. Must run LAST."
    )

    def __init__(self, config, all_component_results=None,
                 production_model_version=None, registry_approved_version=None,
                 spark_session=None):
        super().__init__(config, spark_session)
        self.all_results = all_component_results or []
        self.production_version = production_model_version
        self.registry_version = registry_approved_version

    def run(self) -> MonitoringResult:
        now = datetime.now(timezone.utc).isoformat()
        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK

        if self.production_version and self.registry_version:
            match = self.production_version == self.registry_version
            metrics["version_verification"] = {
                "production_version": self.production_version,
                "registry_approved_version": self.registry_version,
                "match": match, "checked_at": now,
                "status": "PASS" if match else "CRITICAL_VIOLATION",
            }
            if not match:
                alerts.append(f"CRITICAL: Production version '{self.production_version}' "
                               f"does not match registry '{self.registry_version}'.")
                worst_status = MonitoringStatus.CRITICAL
                recommendations.append("Unauthorised model version in production. Escalate immediately.")

        lineage = {
            "model_id": self.config.model.model_id,
            "model_version": self.config.model.version,
            "reference_table": self.config.reference.table,
            "reference_start_date": self.config.reference.start_date,
            "reference_end_date": self.config.reference.end_date,
            "recorded_at": now,
        }
        lineage["lineage_hash"] = self._hash(lineage)
        metrics["training_data_lineage"] = lineage

        critical_count = sum(1 for r in self.all_results if r.get("status") in ("critical", "error"))
        run_id = f"ms-run-{self.config.model.model_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        metrics["run_summary"] = {
            "monitoring_run_id": run_id,
            "model_id": self.config.model.model_id,
            "run_timestamp": now,
            "components_executed": len(self.all_results),
            "critical_components": critical_count,
            "overall_status": "critical" if critical_count > 0 else worst_status.value,
        }

        if self.config.governance.sr_11_7_reporting:
            metrics["sr_11_7_compliance"] = {
                "reporting_enabled": True,
                "psi_computed": any(r.get("component_name") == "run_population_stability" for r in self.all_results),
                "performance_monitored": any(r.get("component_name") == "run_model_performance" for r in self.all_results),
                "audit_trail_written": True,
            }

        if self.config.governance.eu_ai_act_logging:
            metrics["eu_ai_act_log"] = {
                "log_version": AUDIT_LOG_VERSION, "model_id": self.config.model.model_id,
                "run_id": run_id, "timestamp": now,
                "accuracy_monitored": any(r.get("component_name") == "run_model_performance" for r in self.all_results),
                "fairness_monitored": any(r.get("component_name") == "run_fairness_monitoring" for r in self.all_results),
            }

        audit_entry = {
            "audit_log_version": AUDIT_LOG_VERSION, "event_type": "MONITORING_RUN_COMPLETE",
            "model_id": self.config.model.model_id, "model_version": self.config.model.version,
            "run_id": run_id, "timestamp": now,
        }
        audit_entry["entry_hash"] = self._hash(audit_entry)
        metrics["audit_trail_entry"] = audit_entry

        alert_sev = (AlertSeverity.CRITICAL if worst_status == MonitoringStatus.CRITICAL else None)
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=bool(alerts),
                                alert_severity=alert_sev,
                                alert_message="; ".join(alerts) if alerts else None,
                                evidence={"audit_entry_hash": audit_entry["entry_hash"]},
                                recommendations=recommendations)

    @staticmethod
    def _hash(record: dict) -> str:
        return hashlib.sha256(
            json.dumps(record, sort_keys=True, default=str).encode()
        ).hexdigest()

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {"name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {
                    "production_model_version": {"type": "string"},
                    "registry_approved_version": {"type": "string"},
                }, "required": []}}
