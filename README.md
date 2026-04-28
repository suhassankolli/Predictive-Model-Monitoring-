# ModelSentinel — Configuration-Driven ML Model Monitoring

ModelSentinel is an enterprise-grade, configuration-driven Python framework that
dynamically builds and executes Kubeflow Pipelines for predictive model monitoring.

## Features
- 11 reusable monitoring components (drift, performance, fairness, PSI, SHAP, governance, and more)
- LLM-powered agentic orchestrator (Claude) that selects and sequences components via tool use
- Dynamic component generation from natural language at runtime
- Configuration-driven via a single YAML file — no pipeline code required
- Runs on Google Vertex AI Pipelines (primary), with stubs for Azure ML and IBM CP4D

## Installation

```bash
pip install "modelsentinel[vertex]"
```

Set required environment variables:
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/your-webhook"
```

## Quickstart

**Step 1 — Collect monitoring requirements (CLI)**
```bash
modelsentinel collect --output monitoring-config.yaml
```

**Step 2 — Validate configuration**
```bash
modelsentinel validate --config monitoring-config.yaml
```

**Step 3 — Run end-to-end in Python**
```python
from modelsentinel.config.loader import load_config
from modelsentinel.agent.orchestrator import OrchestratorAgent
from modelsentinel.pipeline.builder import PipelineBuilder
from modelsentinel.pipeline.platform_factory import get_platform_runner

config        = load_config("monitoring-config.yaml")
state         = OrchestratorAgent(config=config).run()
pipeline_path = PipelineBuilder(config).build(state.component_sequence)
run_id        = get_platform_runner(config).submit(pipeline_path)
print(f"Pipeline running: {run_id}")
```

**Or use the Jupyter notebook widget:**
```python
from modelsentinel.ui.notebook_widget import MonitoringConfigWidget
widget = MonitoringConfigWidget()
widget.display()
```

## Module structure

```
modelsentinel/
├── config/         # Pydantic v2 schema, YAML loader, defaults
├── agent/          # Orchestrator agent, tool registry, component generator, prompts
├── components/     # 11 monitoring components + base class
├── pipeline/       # kfp v2 builder, Vertex AI / Azure ML / CP4D runners
├── spark/          # PySpark session factory, reusable transformations
├── alerting/       # Alert manager (Slack, PagerDuty, email, webhook)
├── reporting/      # SR 11-7, EU AI Act, fairness report builder
├── ui/             # CLI collector, Jupyter notebook widget
└── utils/          # Logging, exceptions, constants
tests/              # pytest suite (schema, components, orchestrator, alerts, pipeline)
```

## Running tests

```bash
pytest
pytest --cov-report=html   # HTML coverage report
```

## Platform support

| Platform | Status |
|---|---|
| Google Vertex AI Pipelines | Full implementation |
| Azure ML Pipelines | Interface-complete stub |
| IBM CP4D on OpenShift | Interface-complete stub |

## Licence
Apache 2.0
