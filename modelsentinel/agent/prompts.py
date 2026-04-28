"""
Prompt templates for the ModelSentinel orchestrator agent and component generator.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import json
from modelsentinel.config.schema import MonitoringConfig


def build_system_prompt(tool_schemas: list[dict], config: MonitoringConfig) -> str:
    enabled = [
        k for k, v in config.monitoring.model_dump().items()
        if isinstance(v, dict) and v.get("enabled", False)
    ]
    return f"""You are ModelSentinel Orchestrator — an expert ML monitoring pipeline architect.

Your job is to analyse a validated monitoring configuration for a production ML model,
then select, sequence, and invoke the correct set of monitoring components as tools
to build a complete, ordered monitoring pipeline.

AVAILABLE TOOLS: {json.dumps([t['name'] for t in tool_schemas], indent=2)}
ENABLED MONITORING TYPES: {json.dumps(enabled, indent=2)}

SEQUENCING RULES (enforce strictly):
1. run_data_quality FIRST — bad data invalidates all subsequent results.
2. run_data_drift SECOND — establishes distributional context.
3. run_concept_drift THIRD — proxy signals before performance evidence.
4. run_model_performance FOURTH — only if ground-truth labels available.
5. run_fairness_monitoring FIFTH — requires predictions.
6. run_population_stability SIXTH — PSI/CSI for regulatory evidence.
7. run_feature_attribution SEVENTH — computationally expensive, run after critical checks.
8. run_output_distribution EIGHTH — score-level monitoring.
9. run_operational_health — can run parallel with data_quality (no data dependency).
10. run_retraining_triggers SECOND-TO-LAST — aggregates all prior signals.
11. run_governance LAST — summarises results and writes audit entry.

TOOL INVOCATION RULES:
- Only invoke tools for monitoring types that are ENABLED in the config.
- After each tool call, evaluate the MonitoringResult status field.
- If status is CRITICAL, note it in your response before calling the next tool.
- If a required type has no matching tool, note it in your summary.

OUTPUT: After all tools are called, output JSON:
{{
  "components_selected": [...],
  "pipeline_id": "<model_id>-<timestamp>",
  "critical_alerts": [...],
  "recommendations": [...],
  "pipeline_spec": {{ "ordered_components": [...] }}
}}"""


def build_user_prompt(config: MonitoringConfig) -> str:
    return f"""Build the monitoring pipeline for this model:

Model: {config.model.model_id} v{config.model.version}
Task type: {config.model.task_type}
Risk tier: {config.model.risk_tier}
SR 11-7 reporting: {config.governance.sr_11_7_reporting}
EU AI Act logging: {config.governance.eu_ai_act_logging}

Select and invoke all appropriate monitoring components based on the enabled monitoring
types in the configuration. Follow sequencing rules. Return final JSON summary."""


COMPONENT_GENERATION_PROMPT = """You are an expert PySpark and ML monitoring engineer.
Generate a complete Python class implementing a new ModelSentinel monitoring component.

REQUIREMENT: {requirement}

INTERFACE CONTRACT:
1. Inherit from MonitoringComponentBase
2. Define COMPONENT_NAME and COMPONENT_DESCRIPTION as class attributes
3. Be decorated with @monitoring_component
4. Implement run() returning a MonitoringResult
5. Use PySpark DataFrames for all data processing
6. Allowed imports: pyspark.sql, pyspark.sql.functions, scipy.stats, numpy,
   modelsentinel.components.base, modelsentinel.components.tool_registry,
   modelsentinel.utils.constants

Generate ONLY the complete Python class, no explanations, no markdown fences."""
