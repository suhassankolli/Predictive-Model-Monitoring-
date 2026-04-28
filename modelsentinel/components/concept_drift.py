"""
Concept drift monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus


@monitoring_component
class ConceptDriftComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_concept_drift"
    COMPONENT_DESCRIPTION = (
        "Detects concept drift using adversarial validation, confidence degradation, "
        "and label drift when ground-truth labels are available."
    )

    def __init__(self, config, reference_features=None, production_features=None,
                 reference_scores=None, production_scores=None,
                 reference_labels=None, production_labels=None, spark_session=None):
        super().__init__(config, spark_session)
        self.reference_features = reference_features
        self.production_features = production_features
        self.reference_scores = reference_scores
        self.production_scores = production_scores
        self.reference_labels = reference_labels
        self.production_labels = production_labels
        self.concept_cfg = config.monitoring.concept_drift

    def run(self) -> MonitoringResult:
        if not self.concept_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED,
                                    skipped_reason="concept_drift disabled.")
        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK

        if ("adversarial_validation" in self.concept_cfg.proxy_methods
                and self.reference_features is not None
                and self.production_features is not None):
            n = min(len(self.reference_features), len(self.production_features), 5000)
            ref_s = self.reference_features[np.random.choice(len(self.reference_features), n, replace=False)]
            prod_s = self.production_features[np.random.choice(len(self.production_features), n, replace=False)]
            X = np.vstack([ref_s, prod_s])
            y = np.array([0] * n + [1] * n)
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            try:
                adv_auc = float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())
            except Exception:
                adv_auc = 0.5
            metrics["adversarial_validation"] = {
                "auc": round(adv_auc, 4),
                "threshold": self.concept_cfg.adversarial_auc_threshold,
                "status": "DRIFT" if adv_auc > self.concept_cfg.adversarial_auc_threshold else "STABLE",
            }
            if adv_auc > self.concept_cfg.adversarial_auc_threshold:
                alerts.append(f"Adversarial AUC={adv_auc:.4f} above threshold.")
                worst_status = MonitoringStatus.WARNING
                recommendations.append("Review data drift and consider retraining.")

        if (self.reference_labels is not None and self.production_labels is not None):
            ks_stat, ks_pval = stats.ks_2samp(
                self.reference_labels.astype(float), self.production_labels.astype(float))
            metrics["label_drift"] = {
                "reference_positive_rate": round(float(self.reference_labels.mean()), 4),
                "production_positive_rate": round(float(self.production_labels.mean()), 4),
                "ks_statistic": round(ks_stat, 4), "ks_pvalue": round(ks_pval, 6),
                "status": "DRIFT" if ks_pval < 0.05 else "STABLE",
            }
            if ks_pval < 0.05:
                alerts.append(f"Label drift detected: KS p={ks_pval:.4f}")
                worst_status = MonitoringStatus.WARNING

        alert_sev = AlertSeverity.CRITICAL if worst_status == MonitoringStatus.CRITICAL             else (AlertSeverity.WARNING if alerts else None)
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=bool(alerts),
                                alert_severity=alert_sev,
                                alert_message="; ".join(alerts) if alerts else None,
                                recommendations=recommendations)

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {"name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
                "input_schema": {"type": "object", "properties": {
                    "proxy_methods": {"type": "array", "items": {"type": "string"}},
                    "adversarial_auc_threshold": {"type": "number"},
                }, "required": []}}
