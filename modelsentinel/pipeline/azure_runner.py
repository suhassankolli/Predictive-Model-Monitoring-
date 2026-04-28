"""
Azure ML Pipelines runner — interface-complete stub.
TODO: Complete using azure-ai-ml SDK.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import json
from typing import Optional
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)


class AzureMLRunner:
    """
    STUB — Azure ML Pipelines runner.
    TODO: Authenticate with azure.identity.DefaultAzureCredential
    TODO: Instantiate azure.ai.ml.MLClient
    TODO: Translate kfp v2 YAML to Azure ML Component YAML
    TODO: Submit via ml_client.jobs.create_or_update(pipeline_job)
    TODO: Track via ml_client.jobs.get(job_name).status
    TODO: Route artifacts to Azure Blob Storage
    TODO: Publish metrics to Azure Monitor
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        logger.info("[AzureMLRunner STUB] Initialised.")

    def submit(self, pipeline_yaml_path: str, job_name: Optional[str] = None) -> str:
        mock_run_id = f"mock-azure-{self.config.model.model_id}-001"
        logger.info(f"[AzureMLRunner STUB] Would submit: {pipeline_yaml_path}")
        print(json.dumps({"platform": "azure_ml", "mock_run_id": mock_run_id, "status": "Submitted (mock)"}, indent=2))
        return mock_run_id

    def get_status(self, resource_name: str) -> str:
        return "Running (mock)"
