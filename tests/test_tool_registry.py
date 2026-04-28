"""Tests for the tool registry."""
import pytest
from modelsentinel.components.base import MonitoringComponentBase, MonitoringResult
from modelsentinel.components.tool_registry import (
    monitoring_component, get_registry, get_component, get_all_tool_schemas)
from modelsentinel.utils.constants import MonitoringStatus
from modelsentinel.utils.exceptions import ComponentRegistrationError

def test_decorator_registers():
    @monitoring_component
    class _C(MonitoringComponentBase):
        COMPONENT_NAME = "test_reg_001"; COMPONENT_DESCRIPTION = "Test."
        def run(self): return MonitoringResult(component_name=self.COMPONENT_NAME, status=MonitoringStatus.OK)
    assert "test_reg_001" in get_registry()

def test_get_component_returns_class():
    @monitoring_component
    class _C2(MonitoringComponentBase):
        COMPONENT_NAME = "test_get_002"; COMPONENT_DESCRIPTION = "Test."
        def run(self): return MonitoringResult(component_name=self.COMPONENT_NAME, status=MonitoringStatus.OK)
    assert get_component("test_get_002") is _C2

def test_unknown_raises():
    with pytest.raises(KeyError): get_component("does_not_exist_xyz")

def test_no_component_name_raises():
    with pytest.raises(ComponentRegistrationError):
        @monitoring_component
        class _C3(MonitoringComponentBase):
            COMPONENT_NAME = ""; COMPONENT_DESCRIPTION = "Test."
            def run(self): pass

def test_data_drift_schema():
    import modelsentinel.components.data_drift
    s = get_component("run_data_drift").get_tool_schema()
    assert s["name"] == "run_data_drift"
    assert "input_schema" in s
