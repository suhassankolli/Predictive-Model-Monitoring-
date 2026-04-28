"""
Model performance monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, f1_score, precision_score, recall_score)
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus


def _ks_statistic(y_true, y_score):
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    pos_cdf = np.searchsorted(pos, np.sort(y_score)) / len(pos)
    neg_cdf = np.searchsorted(neg, np.sort(y_score)) / len(neg)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def _ece(y_true, y_prob, n_bins=10):
    bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bounds[i]) & (y_prob < bounds[i+1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / len(y_true))


@monitoring_component
class ModelPerformanceComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_model_performance"
    COMPONENT_DESCRIPTION = (
        "Computes AUC-ROC, KS, Gini, F1, Brier score, and ECE against ground-truth labels. "
        "Handles label lag join. Alerts on threshold breach."
    )

    def __init__(self, config, y_true=None, y_score=None, spark_session=None):
        super().__init__(config, spark_session)
        self.y_true = y_true
        self.y_score = y_score
        self.perf_cfg = config.monitoring.model_performance

    def run(self) -> MonitoringResult:
        if not self.perf_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="disabled.")
        if self.y_true is None or self.y_score is None:
            lag = self.config.actuals.label_lag_days if self.config.actuals else "N/A"
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED,
                                    skipped_reason=f"Labels not available (lag={lag} days).")
        if len(np.unique(self.y_true)) < 2:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED,
                                    skipped_reason="Only one class in window.")

        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK
        auc_roc = roc_auc_score(self.y_true, self.y_score)
        ks = _ks_statistic(self.y_true, self.y_score)
        ece_val = _ece(self.y_true, self.y_score)
        y_pred = (self.y_score >= 0.5).astype(int)
        metrics.update({
            "auc_roc": round(auc_roc, 6), "ks_statistic": round(ks, 6),
            "gini": round(2*auc_roc - 1, 6),
            "brier_score": round(brier_score_loss(self.y_true, self.y_score), 6),
            "ece": round(ece_val, 6),
            "f1": round(f1_score(self.y_true, y_pred, zero_division=0), 6),
            "precision": round(precision_score(self.y_true, y_pred, zero_division=0), 6),
            "recall": round(recall_score(self.y_true, y_pred, zero_division=0), 6),
        })
        if self.perf_cfg.auc_roc_min and auc_roc < self.perf_cfg.auc_roc_min:
            alerts.append(f"AUC-ROC {auc_roc:.4f} < {self.perf_cfg.auc_roc_min}")
            worst_status = MonitoringStatus.WARNING
            recommendations.append("AUC below threshold — review drift findings and consider retraining.")
        if ece_val > self.perf_cfg.calibration_ece_max:
            alerts.append(f"ECE {ece_val:.4f} > {self.perf_cfg.calibration_ece_max}")
            worst_status = MonitoringStatus.WARNING
            recommendations.append("Calibration drift — consider Platt scaling recalibration.")
        alert_sev = AlertSeverity.WARNING if alerts else None
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=bool(alerts),
                                alert_severity=alert_sev,
                                alert_message="; ".join(alerts) if alerts else None,
                                recommendations=recommendations)

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {"name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {
                    "auc_roc_min": {"type": "number"}, "ks_statistic_min": {"type": "number"},
                    "calibration_ece_max": {"type": "number"},
                }, "required": []}}
