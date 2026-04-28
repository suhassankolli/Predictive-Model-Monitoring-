"""
Operational health monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus


@monitoring_component
class OperationalHealthComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_operational_health"
    COMPONENT_DESCRIPTION = (
        "Monitors inference latency (p50/p95/p99), serving error rates, "
        "Kubeflow pipeline health, and resource utilisation."
    )

    def __init__(self, config, latency_samples_ms=None, error_count=0,
                 total_requests=0, pipeline_step_results=None,
                 resource_metrics=None, spark_session=None):
        super().__init__(config, spark_session)
        self.latency_samples_ms = latency_samples_ms
        self.error_count = error_count
        self.total_requests = total_requests
        self.pipeline_step_results = pipeline_step_results or []
        self.resource_metrics = resource_metrics or {}
        self.ops_cfg = config.monitoring.operational_health

    def run(self) -> MonitoringResult:
        if not self.ops_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="disabled.")
        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK

        if self.latency_samples_ms is not None and len(self.latency_samples_ms) > 0:
            p50, p95, p99 = (float(np.percentile(self.latency_samples_ms, p)) for p in [50, 95, 99])
            metrics["latency"] = {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2),
                                   "p99_ms": round(p99, 2),
                                   "p95_status": "OK" if p95 <= self.ops_cfg.latency_p95_ms_max else "BREACH",
                                   "p99_status": "OK" if p99 <= self.ops_cfg.latency_p99_ms_max else "BREACH"}
            if p95 > self.ops_cfg.latency_p95_ms_max:
                alerts.append(f"p95 latency {p95:.0f}ms > SLA {self.ops_cfg.latency_p95_ms_max}ms")
                worst_status = MonitoringStatus.WARNING
                recommendations.append("p95 latency above SLA. Review serving capacity.")
            if p99 > self.ops_cfg.latency_p99_ms_max:
                alerts.append(f"p99 latency {p99:.0f}ms > SLA {self.ops_cfg.latency_p99_ms_max}ms")
                worst_status = MonitoringStatus.CRITICAL

        if self.total_requests > 0:
            error_rate = self.error_count / self.total_requests
            metrics["serving_errors"] = {"error_rate": round(error_rate, 6),
                                          "status": "OK" if error_rate <= self.ops_cfg.error_rate_max else "BREACH"}
            if error_rate > self.ops_cfg.error_rate_max:
                alerts.append(f"Error rate {error_rate:.4%} > {self.ops_cfg.error_rate_max:.4%}")
                worst_status = MonitoringStatus.CRITICAL
                recommendations.append("Serving error rate above threshold. Check endpoint logs.")

        if self.pipeline_step_results:
            failed = [s for s in self.pipeline_step_results if s.get("status") != "SUCCESS"]
            metrics["pipeline_health"] = {"total_steps": len(self.pipeline_step_results),
                                           "failed_steps": [s.get("name") for s in failed],
                                           "status": "OK" if not failed else "DEGRADED"}
            if failed:
                alerts.append(f"Pipeline step failures: {[s.get('name') for s in failed]}")
                worst_status = MonitoringStatus.CRITICAL

        alert_sev = (AlertSeverity.CRITICAL if worst_status == MonitoringStatus.CRITICAL
                     else (AlertSeverity.WARNING if worst_status == MonitoringStatus.WARNING else None))
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=bool(alerts),
                                alert_severity=alert_sev,
                                alert_message="; ".join(alerts) if alerts else None,
                                recommendations=recommendations)

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {"name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {
                    "latency_p95_ms_max": {"type": "integer"},
                    "latency_p99_ms_max": {"type": "integer"},
                    "error_rate_max": {"type": "number"},
                }, "required": []}}
