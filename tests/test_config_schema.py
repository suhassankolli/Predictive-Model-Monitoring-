"""Tests for config schema validation."""
import pytest
from modelsentinel.config.schema import MonitoringConfig

MINIMAL = {
    "modelsentinel_version": "1.0",
    "model": {"model_id": "test-model", "version": "1.0.0", "framework": "xgboost",
               "task_type": "binary_classification", "risk_tier": "medium",
               "owner": "test@example.com", "business_domain": "test", "deployment_date": "2024-01-01"},
    "reference": {"type": "bigquery", "table": "p.d.ref", "start_date": "2024-01-01", "end_date": "2024-06-01"},
    "production": {"type": "bigquery", "table": "p.d.prod", "window_days": 30,
                   "record_id_column": "id", "score_column": "score"},
    "platform": {"type": "vertex_ai", "pipeline_root": "gs://bucket/ms"},
}

def test_valid_config_loads():
    c = MonitoringConfig.model_validate(MINIMAL)
    assert c.model.model_id == "test-model"

def test_missing_model_raises():
    bad = {k: v for k, v in MINIMAL.items() if k != "model"}
    with pytest.raises(Exception): MonitoringConfig.model_validate(bad)

def test_fairness_without_protected_attrs_raises():
    cfg = dict(MINIMAL); cfg["monitoring"] = {"fairness": {"enabled": True}}
    cfg["features"] = {"numerical": [], "categorical": [], "protected_attributes": []}
    with pytest.raises(Exception, match="fairness"): MonitoringConfig.model_validate(cfg)

def test_high_risk_sr117_needs_psi():
    cfg = dict(MINIMAL); cfg["model"] = {**MINIMAL["model"], "risk_tier": "high"}
    cfg["governance"] = {"sr_11_7_reporting": True}
    cfg["monitoring"] = {"population_stability": {"enabled": False}}
    with pytest.raises(Exception, match="population_stability"): MonitoringConfig.model_validate(cfg)

def test_bigquery_requires_table():
    cfg = dict(MINIMAL)
    cfg["reference"] = {"type": "bigquery", "start_date": "2024-01-01", "end_date": "2024-06-01"}
    with pytest.raises(Exception): MonitoringConfig.model_validate(cfg)
