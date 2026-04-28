"""Tests for OrchestratorAgent with mocked Anthropic client."""
import json
import pytest
from unittest.mock import MagicMock, patch
from modelsentinel.agent.orchestrator import OrchestratorAgent

def make_config():
    c = MagicMock()
    c.model.model_id = "test-model"; c.model.version = "1.0.0"
    c.model.task_type = MagicMock(value="binary_classification")
    c.model.risk_tier = MagicMock(value="medium")
    c.governance.sr_11_7_reporting = False; c.governance.eu_ai_act_logging = False
    c.monitoring.model_dump.return_value = {"data_drift": {"enabled": True}}
    return c

def end_turn_response():
    r = MagicMock(); r.stop_reason = "end_turn"
    b = MagicMock(); b.type = "text"
    b.text = json.dumps({"components_selected": [], "pipeline_id": "test-001",
                          "critical_alerts": [], "recommendations": [],
                          "pipeline_spec": {"ordered_components": []}})
    r.content = [b]; return r

def test_end_turn_completes():
    client = MagicMock(); client.messages.create.return_value = end_turn_response()
    with patch("modelsentinel.agent.orchestrator.get_all_tool_schemas", return_value=[]):
        state = OrchestratorAgent(make_config(), anthropic_client=client, human_in_the_loop=False).run()
    assert state.completed

def test_api_called_once():
    client = MagicMock(); client.messages.create.return_value = end_turn_response()
    with patch("modelsentinel.agent.orchestrator.get_all_tool_schemas", return_value=[]):
        OrchestratorAgent(make_config(), anthropic_client=client, human_in_the_loop=False).run()
    assert client.messages.create.call_count == 1
