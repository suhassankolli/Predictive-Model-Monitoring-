"""
Fairness and bias monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from sklearn.metrics import roc_auc_score
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus, DEFAULT_FAIRNESS_MIN_GROUP_SIZE


@monitoring_component
class FairnessComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_fairness_monitoring"
    COMPONENT_DESCRIPTION = (
        "Computes disparate impact ratio, equalised odds gap, and per-slice AUC/F1 "
        "across all configured protected demographic groups."
    )

    def __init__(self, config, y_true=None, y_score=None, y_pred=None,
                 group_labels=None, spark_session=None):
        super().__init__(config, spark_session)
        self.y_true = y_true; self.y_score = y_score
        self.y_pred = y_pred if y_pred is not None else (
            (y_score >= 0.5).astype(int) if y_score is not None else None)
        self.group_labels = group_labels or {}
        self.fairness_cfg = config.monitoring.fairness

    def run(self) -> MonitoringResult:
        if not self.fairness_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="disabled.")
        if not self.config.features.protected_attributes:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED,
                                    skipped_reason="No protected attributes defined.")
        if self.y_pred is None:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="No predictions.")

        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK

        for attr in self.config.features.protected_attributes:
            groups = self.group_labels.get(attr.name)
            if groups is None: continue
            attr_metrics, pos_rates, tpr_rates = {}, {}, {}
            for grp in np.unique(groups):
                mask = groups == grp
                if mask.sum() < DEFAULT_FAIRNESS_MIN_GROUP_SIZE: continue
                grp_pred = self.y_pred[mask]
                pos_rates[str(grp)] = float(grp_pred.mean())
                attr_metrics[str(grp)] = {"size": int(mask.sum()),
                                           "positive_rate": round(pos_rates[str(grp)], 4)}
                if self.y_true is not None:
                    gt = self.y_true[mask]
                    if gt.sum() > 0:
                        tpr = float(grp_pred[gt == 1].mean())
                        tpr_rates[str(grp)] = tpr; attr_metrics[str(grp)]["tpr"] = round(tpr, 4)
                    if len(np.unique(gt)) > 1 and self.y_score is not None:
                        attr_metrics[str(grp)]["auc_roc"] = round(
                            roc_auc_score(gt, self.y_score[mask]), 4)

            if len(pos_rates) >= 2:
                di = min(pos_rates.values()) / max(pos_rates.values())
                attr_metrics["disparate_impact_ratio"] = round(di, 4)
                attr_metrics["disparate_impact_status"] = "PASS" if di >= self.fairness_cfg.disparate_impact_min else "FAIL"
                if di < self.fairness_cfg.disparate_impact_min:
                    alerts.append(f"FAIRNESS VIOLATION '{attr.name}': DI={di:.4f}")
                    worst_status = MonitoringStatus.CRITICAL
                    low_g = min(pos_rates, key=pos_rates.get)
                    recommendations.append(
                        f"Group '{low_g}' in '{attr.name}' has low positive rate. "
                        "Review training data representation.")
                if len(tpr_rates) >= 2:
                    tpr_gap = max(tpr_rates.values()) - min(tpr_rates.values())
                    attr_metrics["tpr_gap"] = round(tpr_gap, 4)
                    if tpr_gap > self.fairness_cfg.equalised_odds_gap_max:
                        alerts.append(f"TPR gap '{attr.name}': {tpr_gap:.4f}")
                        worst_status = MonitoringStatus.CRITICAL
            metrics[attr.name] = attr_metrics

        alert_sev = self.fairness_cfg.alert_severity if alerts else None
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=bool(alerts),
                                alert_severity=alert_sev,
                                alert_message="; ".join(alerts) if alerts else None,
                                recommendations=recommendations)

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {"name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {
                    "disparate_impact_min": {"type": "number"},
                    "equalised_odds_gap_max": {"type": "number"},
                }, "required": []}}
