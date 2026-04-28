"""
ModelSentinel Orchestrator Agent.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Optional
import anthropic
from modelsentinel.agent.prompts import build_system_prompt, build_user_prompt
from modelsentinel.components.tool_registry import get_all_tool_schemas, get_component
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.constants import ANTHROPIC_MODEL, AGENT_MAX_RETRIES, AGENT_MAX_TOKENS
from modelsentinel.utils.exceptions import HumanCheckpointRequired
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    config: MonitoringConfig
    tool_results: list[dict] = field(default_factory=list)
    component_sequence: list[str] = field(default_factory=list)
    critical_alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    pipeline_spec: Optional[dict] = None
    completed: bool = False


class OrchestratorAgent:
    """LLM-powered orchestrator that selects and sequences monitoring components."""

    def __init__(self, config: MonitoringConfig,
                 anthropic_client: Optional[anthropic.Anthropic] = None,
                 human_in_the_loop: bool = True,
                 component_data: Optional[dict[str, Any]] = None) -> None:
        self.config = config
        self.client = anthropic_client or anthropic.Anthropic()
        self.human_in_the_loop = human_in_the_loop
        self.component_data = component_data or {}
        self.state = AgentState(config=config)

    def run(self) -> AgentState:
        tool_schemas = get_all_tool_schemas()
        system_prompt = build_system_prompt(tool_schemas, self.config)
        user_prompt = build_user_prompt(self.config)
        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        retry_count = 0
        logger.info(f"Orchestrator starting for '{self.config.model.model_id}' "
                    f"with {len(tool_schemas)} tools.")

        while not self.state.completed and retry_count < AGENT_MAX_RETRIES:
            try:
                response = self.client.messages.create(
                    model=ANTHROPIC_MODEL, max_tokens=AGENT_MAX_TOKENS,
                    system=system_prompt, tools=tool_schemas, messages=messages,
                )
            except anthropic.APIError as exc:
                logger.error(f"API error: {exc}. Retry {retry_count + 1}/{AGENT_MAX_RETRIES}")
                retry_count += 1
                continue

            logger.info(f"Agent stop_reason: {response.stop_reason}")
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                self._parse_final_summary(response)
                self.state.completed = True
                break

            if response.stop_reason == "tool_use":
                tool_results_content = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = self._invoke_tool(block.name, block.input, block.id)
                    tool_results_content.append(result)
                    if result.get("is_error") and self.human_in_the_loop:
                        self._emit_checkpoint(f"Tool '{block.name}' returned error.", result)
                messages.append({"role": "user", "content": tool_results_content})

        if not self.state.completed:
            logger.error("Orchestrator did not complete within max retries.")
        return self.state

    def _invoke_tool(self, tool_name: str, tool_input: dict, tool_use_id: str) -> dict:
        logger.info(f"Invoking tool: {tool_name}")
        self.state.component_sequence.append(tool_name)
        try:
            component_cls = get_component(tool_name)
            data = self.component_data.get(tool_name, {})
            component = component_cls(config=self.config, **data)
            result = component.execute()
            result_dict = result.to_dict()
            self.state.tool_results.append(result_dict)
            if result.alert_triggered:
                self.state.critical_alerts.append(result.alert_message or "")
            self.state.recommendations.extend(result.recommendations)
            return {"type": "tool_result", "tool_use_id": tool_use_id,
                    "content": json.dumps(result_dict)}
        except KeyError:
            return {"type": "tool_result", "tool_use_id": tool_use_id,
                    "content": f"Tool '{tool_name}' not registered.", "is_error": True}
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' failed: {exc}")
            return {"type": "tool_result", "tool_use_id": tool_use_id,
                    "content": f"Execution error: {exc}", "is_error": True}

    def _emit_checkpoint(self, reason: str, context: dict) -> None:
        logger.warning(f"HUMAN CHECKPOINT: {reason}")
        if self.human_in_the_loop:
            print(f"\n[ModelSentinel] Human checkpoint: {reason}")
            print(f"Context: {json.dumps(context, indent=2)}")
            resp = input("Type 'proceed' to continue or 'abort' to stop: ").strip().lower()
            if resp != "proceed":
                raise HumanCheckpointRequired(reason=reason, context=context)

    def _parse_final_summary(self, response: Any) -> None:
        for block in response.content:
            if hasattr(block, "text"):
                try:
                    summary = json.loads(block.text)
                    self.state.pipeline_spec = summary.get("pipeline_spec")
                except (json.JSONDecodeError, AttributeError):
                    pass
