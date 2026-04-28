"""
Population Stability Index (PSI) and CSI monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.utils.constants import AlertSeverity, MonitoringStatus


def _compute_psi(reference, production, strategy="equal_frequency", n_bins=10):
    if strategy == "equal_frequency":
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    else:
        bins = np.linspace(reference.min(), reference.max(), n_bins + 1)
    bins[0] -= 1e-9; bins[-1] += 1e-9; bins = np.unique(bins)
    ref_pct = np.clip(np.histogram(reference, bins=bins)[0] / len(reference), 1e-9, None)
    prod_pct = np.clip(np.histogram(production, bins=bins)[0] / len(production), 1e-9, None)
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


@monitoring_component
class PopulationStabilityComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_population_stability"
    COMPONENT_DESCRIPTION = (
        "Computes PSI for all features and CSI for SR 11-7 compliance. "
        "Returns Green/Amber/Red traffic-light status per feature."
    )

    def __init__(self, config, reference_data=None, production_data=None,
                 reference_scores=None, production_scores=None, spark_session=None):
        super().__init__(config, spark_session)
        self.reference_data = reference_data or {}
        self.production_data = production_data or {}
        self.reference_scores = reference_scores
        self.production_scores = production_scores
        self.psi_cfg = config.monitoring.population_stability

    def run(self) -> MonitoringResult:
        if not self.psi_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED, skipped_reason="disabled.")
        metrics, alerts, recommendations, psi_vals = {}, [], [], []
        worst_status = MonitoringStatus.OK

        all_feats = ([f.name for f in self.config.features.numerical]
                     + [f.name for f in self.config.features.categorical])
        for feat_name in all_feats:
            ref = self.reference_data.get(feat_name)
            prod = self.production_data.get(feat_name)
            if ref is None or prod is None: continue
            psi_val = _compute_psi(np.asarray(ref, dtype=float), np.asarray(prod, dtype=float),
                                   strategy=self.psi_cfg.binning_strategy, n_bins=self.psi_cfg.n_bins)
            psi_vals.append(psi_val)
            tl = ("GREEN" if psi_val < self.psi_cfg.psi_green_threshold else
                  "AMBER" if psi_val < self.psi_cfg.psi_amber_threshold else "RED")
            fm = {"psi": round(psi_val, 6), "traffic_light": tl}
            if self.psi_cfg.compute_csi:
                fm["csi"] = round(psi_val, 6)
                fm["csi_sr117_status"] = "PASS" if psi_val < self.psi_cfg.psi_amber_threshold else "REVIEW_REQUIRED"
            metrics[f"{feat_name}_psi"] = fm
            if tl == "RED":
                alerts.append(f"PSI RED '{feat_name}': {psi_val:.4f}")
                worst_status = MonitoringStatus.CRITICAL
                recommendations.append(f"'{feat_name}' PSI RED ({psi_val:.4f}). SR 11-7 requires investigation.")
            elif tl == "AMBER":
                alerts.append(f"PSI AMBER '{feat_name}': {psi_val:.4f}")
                if worst_status != MonitoringStatus.CRITICAL: worst_status = MonitoringStatus.WARNING

        if self.reference_scores is not None and self.production_scores is not None:
            score_psi = _compute_psi(self.reference_scores, self.production_scores,
                                     self.psi_cfg.binning_strategy, self.psi_cfg.n_bins)
            psi_vals.append(score_psi)
            tl = ("GREEN" if score_psi < self.psi_cfg.psi_green_threshold else
                  "AMBER" if score_psi < self.psi_cfg.psi_amber_threshold else "RED")
            metrics["score_distribution_psi"] = {"psi": round(score_psi, 6), "traffic_light": tl}
            if tl in ("AMBER", "RED"):
                alerts.append(f"Score distribution PSI {tl}: {score_psi:.4f}")

        metrics["summary"] = {
            "max_psi": round(max(psi_vals), 6) if psi_vals else 0.0,
            "mean_psi": round(float(np.mean(psi_vals)), 6) if psi_vals else 0.0,
            "red_features": sum(1 for v in metrics.values() if isinstance(v, dict) and v.get("traffic_light") == "RED"),
            "amber_features": sum(1 for v in metrics.values() if isinstance(v, dict) and v.get("traffic_light") == "AMBER"),
        }
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
                    "psi_green_threshold": {"type": "number"},
                    "psi_amber_threshold": {"type": "number"},
                    "psi_red_threshold": {"type": "number"},
                    "binning_strategy": {"type": "string"},
                    "n_bins": {"type": "integer"},
                    "compute_csi": {"type": "boolean"},
                }, "required": []}}
