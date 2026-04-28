"""
Data drift monitoring component.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import monitoring_component
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.constants import AlertSeverity, DriftMethod, MonitoringStatus


def _psi(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-9; bins[-1] += 1e-9
    ref_pct = np.histogram(reference, bins=bins)[0] / len(reference)
    prod_pct = np.histogram(production, bins=bins)[0] / len(production)
    ref_pct = np.clip(ref_pct, 1e-9, None); prod_pct = np.clip(prod_pct, 1e-9, None)
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def _jsd(reference: np.ndarray, production: np.ndarray, n_bins: int = 50) -> float:
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    ref_hist = np.clip(np.histogram(reference, bins=bins, density=True)[0], 1e-9, None)
    prod_hist = np.clip(np.histogram(production, bins=bins, density=True)[0], 1e-9, None)
    return float(jensenshannon(ref_hist, prod_hist))


def _wasserstein(reference: np.ndarray, production: np.ndarray) -> float:
    return float(stats.wasserstein_distance(reference, production))


def _tvd(ref_counts: dict, prod_counts: dict) -> float:
    all_cats = set(ref_counts) | set(prod_counts)
    ref_total = sum(ref_counts.values()) or 1
    prod_total = sum(prod_counts.values()) or 1
    return 0.5 * sum(
        abs(ref_counts.get(c, 0) / ref_total - prod_counts.get(c, 0) / prod_total)
        for c in all_cats
    )


@monitoring_component
class DataDriftComponent(MonitoringComponentBase):
    COMPONENT_NAME = "run_data_drift"
    COMPONENT_DESCRIPTION = (
        "Computes KS, PSI, JSD, Wasserstein for numerical features and Chi-squared/TVD "
        "for categorical features between reference and production windows."
    )

    def __init__(self, config, reference_data=None, production_data=None, spark_session=None):
        super().__init__(config, spark_session)
        self.reference_data = reference_data or {}
        self.production_data = production_data or {}
        self.drift_cfg = config.monitoring.data_drift

    def run(self) -> MonitoringResult:
        if not self.drift_cfg.enabled:
            return MonitoringResult(component_name=self.COMPONENT_NAME,
                                    status=MonitoringStatus.SKIPPED,
                                    skipped_reason="data_drift disabled.")
        metrics, alerts, recommendations = {}, [], []
        worst_status = MonitoringStatus.OK

        for feat in self.config.features.numerical:
            ref = self.reference_data.get(feat.name)
            prod = self.production_data.get(feat.name)
            if ref is None or prod is None:
                alerts.append(f"Feature '{feat.name}' absent from data.")
                worst_status = MonitoringStatus.CRITICAL
                continue
            ref_arr, prod_arr = np.asarray(ref, dtype=float), np.asarray(prod, dtype=float)
            fm = {"type": "numerical"}
            if DriftMethod.KS in self.drift_cfg.methods:
                ks_stat, ks_pval = stats.ks_2samp(ref_arr, prod_arr)
                fm["ks_statistic"] = round(ks_stat, 6); fm["ks_pvalue"] = round(ks_pval, 6)
                if ks_pval < self.drift_cfg.numerical_threshold_ks_pvalue:
                    alerts.append(f"KS drift: '{feat.name}' p={ks_pval:.4f}")
                    worst_status = MonitoringStatus.WARNING
            if DriftMethod.PSI in self.drift_cfg.methods:
                psi_val = _psi(ref_arr, prod_arr)
                fm["psi"] = round(psi_val, 6)
                if psi_val > self.drift_cfg.numerical_threshold_psi:
                    sev = MonitoringStatus.CRITICAL if psi_val > 0.25 else MonitoringStatus.WARNING
                    alerts.append(f"PSI drift '{feat.name}': {psi_val:.4f}")
                    worst_status = sev
                    recommendations.append(f"Investigate upstream pipeline for '{feat.name}'.")
            if DriftMethod.JSD in self.drift_cfg.methods:
                fm["jsd"] = round(_jsd(ref_arr, prod_arr), 6)
            if DriftMethod.WASSERSTEIN in self.drift_cfg.methods:
                fm["wasserstein"] = round(_wasserstein(ref_arr, prod_arr), 6)
            metrics[feat.name] = fm

        for feat in self.config.features.categorical:
            rc = self.reference_data.get(feat.name)
            pc = self.production_data.get(feat.name)
            if rc is None or pc is None:
                continue
            fm = {"type": "categorical", "tvd": round(_tvd(rc, pc), 6)}
            tvd = fm["tvd"]
            if tvd > self.drift_cfg.categorical_threshold_tvd:
                alerts.append(f"Categorical drift '{feat.name}': TVD={tvd:.4f}")
                worst_status = MonitoringStatus.WARNING
            new_cats = set(pc) - set(rc)
            if new_cats:
                alerts.append(f"New categories in '{feat.name}': {new_cats}")
                worst_status = MonitoringStatus.CRITICAL
            metrics[feat.name] = fm

        alert_sev = AlertSeverity.CRITICAL if worst_status == MonitoringStatus.CRITICAL             else (self.drift_cfg.alert_severity if alerts else None)
        return MonitoringResult(component_name=self.COMPONENT_NAME, status=worst_status,
                                metrics=metrics, alert_triggered=bool(alerts),
                                alert_severity=alert_sev,
                                alert_message="; ".join(alerts) if alerts else None,
                                recommendations=recommendations)

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {
            "name": cls.COMPONENT_NAME, "description": cls.COMPONENT_DESCRIPTION,
            "input_schema": {"type": "object", "properties": {
                "methods": {"type": "array", "items": {"type": "string"}},
                "numerical_threshold_psi": {"type": "number"},
                "categorical_threshold_tvd": {"type": "number"},
            }, "required": []},
        }
