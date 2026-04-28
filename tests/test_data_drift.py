"""Tests for the DataDriftComponent."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from modelsentinel.components.data_drift import DataDriftComponent, _psi, _jsd, _tvd
from modelsentinel.utils.constants import MonitoringStatus

def make_config(enabled=True, psi_thresh=0.10):
    c = MagicMock()
    c.monitoring.data_drift.enabled = enabled
    c.monitoring.data_drift.methods = ["ks", "psi", "jsd"]
    c.monitoring.data_drift.numerical_threshold_psi = psi_thresh
    c.monitoring.data_drift.numerical_threshold_ks_pvalue = 0.05
    c.monitoring.data_drift.categorical_threshold_tvd = 0.05
    c.monitoring.data_drift.alert_severity = "warning"
    feat = MagicMock(); feat.name = "income"; feat.never_null = False
    c.features.numerical = [feat]
    c.features.categorical = []
    return c

def test_psi_identical(): assert _psi(np.random.normal(0,1,5000), np.random.normal(0,1,5000)) < 0.05
def test_psi_different(): assert _psi(np.random.normal(0,1,5000), np.random.normal(5,1,5000)) > 0.20
def test_jsd_symmetric():
    a = np.random.normal(0,1,3000); b = np.random.normal(1,1,3000)
    assert abs(_jsd(a, b) - _jsd(b, a)) < 0.01
def test_tvd_identical(): assert _tvd({"A":100,"B":200}, {"A":100,"B":200}) < 0.001

def test_disabled_skips():
    c = DataDriftComponent(config=make_config(enabled=False), reference_data={}, production_data={})
    assert c.run().status == MonitoringStatus.SKIPPED

def test_no_alert_identical():
    data = {"income": np.random.normal(50000,10000,3000).tolist()}
    c = DataDriftComponent(config=make_config(), reference_data=data, production_data=data)
    assert not c.run().alert_triggered

def test_alert_on_shift():
    ref = {"income": np.random.normal(50000,5000,3000).tolist()}
    prod = {"income": np.random.normal(80000,5000,3000).tolist()}
    c = DataDriftComponent(config=make_config(psi_thresh=0.01), reference_data=ref, production_data=prod)
    assert c.run().alert_triggered
