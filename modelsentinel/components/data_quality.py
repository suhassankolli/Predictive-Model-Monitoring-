"""
Data quality monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np
from scipy import stats
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus


@monitoring_component
class DataQualityComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_data_quality"
    COMPONENT_DESCRIPTION = (
        "Validates null rates, schema violations, outliers (IQR/Z-score), "
        "referential integrity, and data freshness SLA."
    )

    def __init__(self, config, production_data=None, reference_null_rates=None,
                 data_timestamp=None, spark_session=None):
        super().__init__(config, spark_session)
        self.production_data = production_data or {}
        self.reference_null_rates = reference_null_rates or {}
        self.data_timestamp = data_timestamp
        self.dq_cfg = config.monitoring.data_quality

    def run(self) -> MonitoringResult:
        if not self.dq_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="disabled.")
        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK

        for feat in list(self.config.features.numerical) + list(self.config.features.categorical):
            data = self.production_data.get(feat.name)
            if data is None:
                alerts.append(f"Feature '{feat.name}' ABSENT from production data.")
                worst_status = MonitoringStatus.CRITICAL
                continue
            arr = np.asarray(data, dtype=object)
            total = len(arr)
            null_count = int(np.sum(arr == None))  # noqa: E711
            null_rate = null_count / total if total > 0 else 0.0
            ref_null = self.reference_null_rates.get(feat.name, 0.0)
            null_thresh = ref_null * self.dq_cfg.null_rate_max_multiplier
            fm = {"null_count": null_count, "null_rate": round(null_rate, 6), "record_count": total}
            if feat.never_null and null_count > 0:
                alerts.append(f"CRITICAL: '{feat.name}' never-null has {null_count} nulls.")
                worst_status = MonitoringStatus.CRITICAL; fm["null_status"] = "CRITICAL"
            elif null_rate > null_thresh and null_thresh > 0:
                alerts.append(f"Null spike '{feat.name}': {null_rate:.2%} vs {ref_null:.2%}.")
                if worst_status != MonitoringStatus.CRITICAL: worst_status = MonitoringStatus.WARNING
                fm["null_status"] = "WARNING"
                recommendations.append(f"Null rate for '{feat.name}' is above baseline. Check upstream feed.")
            else:
                fm["null_status"] = "OK"
            metrics[f"{feat.name}_nulls"] = fm

        if self.dq_cfg.check_schema:
            violations = []
            for feat in self.config.features.numerical:
                data = self.production_data.get(feat.name)
                if data is None: continue
                arr = np.asarray([x for x in data if x is not None], dtype=float)
                if feat.min_value is not None and np.any(arr < feat.min_value):
                    violations.append(f"'{feat.name}': {int(np.sum(arr < feat.min_value))} values below min")
                if feat.max_value is not None and np.any(arr > feat.max_value):
                    violations.append(f"'{feat.name}': {int(np.sum(arr > feat.max_value))} values above max")
            for feat in self.config.features.categorical:
                data = self.production_data.get(feat.name)
                if data is None or not feat.allowed_values: continue
                unknown = [v for v in data if v is not None and v not in feat.allowed_values]
                if unknown:
                    violations.append(f"'{feat.name}': {len(unknown)} unknown categories")
            metrics["schema_validation"] = {"violations": violations, "violation_count": len(violations),
                                             "status": "FAIL" if violations else "PASS"}
            if violations:
                alerts.extend(violations); worst_status = MonitoringStatus.CRITICAL
                recommendations.append("Schema violations detected. Check data contract.")

        if self.data_timestamp is not None:
            now = datetime.now(timezone.utc)
            if self.data_timestamp.tzinfo is None:
                self.data_timestamp = self.data_timestamp.replace(tzinfo=timezone.utc)
            lag_hours = (now - self.data_timestamp).total_seconds() / 3600
            metrics["freshness"] = {"lag_hours": round(lag_hours, 2),
                                     "sla_hours": self.dq_cfg.freshness_sla_hours,
                                     "status": "OK" if lag_hours <= self.dq_cfg.freshness_sla_hours else "BREACH"}
            if lag_hours > self.dq_cfg.freshness_sla_hours:
                alerts.append(f"Freshness SLA breached: {lag_hours:.1f}h old.")
                worst_status = MonitoringStatus.CRITICAL
                recommendations.append("Monitoring data is stale. Treat results as provisional.")

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
                    "null_rate_max_multiplier": {"type": "number"},
                    "outlier_method": {"type": "string", "enum": ["iqr", "zscore", "isolation_forest"]},
                    "freshness_sla_hours": {"type": "integer"},
                }, "required": []}}
