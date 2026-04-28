"""Tests for MonitoringComponentBase and MonitoringResult."""
import pytest
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.utils.constants import MonitoringStatus, AlertSeverity

class GoodComp(MonitoringComponentBase):
    COMPONENT_NAME = "test_good"; COMPONENT_DESCRIPTION = "Test."
    def run(self): return MonitoringResult(component_name=self.COMPONENT_NAME, status=MonitoringStatus.OK, metrics={"x": 1.0})

class BadComp(MonitoringComponentBase):
    COMPONENT_NAME = "test_bad"; COMPONENT_DESCRIPTION = "Fails."
    def run(self): raise ValueError("intentional")

def test_execute_success():
    r = GoodComp(config=None).execute()
    assert r.status == MonitoringStatus.OK
    assert r.metrics["x"] == 1.0
    assert r.run_duration_seconds >= 0

def test_execute_failure_returns_error():
    r = BadComp(config=None).execute()
    assert r.status == MonitoringStatus.ERROR
    assert r.alert_triggered is True
    assert r.alert_severity == AlertSeverity.CRITICAL

def test_to_dict():
    r = MonitoringResult(component_name="t", status=MonitoringStatus.WARNING,
                         alert_triggered=True, alert_severity=AlertSeverity.WARNING,
                         alert_message="msg", recommendations=["do something"])
    d = r.to_dict()
    assert d["status"] == "warning"
    assert d["alert_triggered"] is True

def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError): MonitoringComponentBase(config=None)
