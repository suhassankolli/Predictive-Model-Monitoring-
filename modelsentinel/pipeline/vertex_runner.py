"""
Vertex AI Pipelines runner — full implementation.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import datetime
from typing import Optional
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.exceptions import PlatformAdapterError
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False


class VertexAIRunner:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.platform = config.platform
        if VERTEX_AVAILABLE:
            aiplatform.init(project=self.platform.project, location=self.platform.location)

    def submit(self, pipeline_yaml_path: str, job_name: Optional[str] = None) -> str:
        if not VERTEX_AVAILABLE:
            raise PlatformAdapterError(
                "google-cloud-aiplatform not installed. Run: pip install modelsentinel[vertex]")
        ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        display_name = job_name or f"modelsentinel-{self.config.model.model_id}-{ts}"
        logger.info(f"Submitting pipeline to Vertex AI: {display_name}")
        try:
            job = aiplatform.PipelineJob(
                display_name=display_name, template_path=pipeline_yaml_path,
                pipeline_root=self.platform.pipeline_root, enable_caching=False)
            job.submit(service_account=self.platform.service_account)
            logger.info(f"Pipeline submitted: {job.resource_name}")
            return job.resource_name
        except Exception as exc:
            raise PlatformAdapterError(f"Vertex AI submission failed: {exc}") from exc

    def get_status(self, resource_name: str) -> str:
        if not VERTEX_AVAILABLE:
            raise PlatformAdapterError("google-cloud-aiplatform not installed.")
        return aiplatform.PipelineJob.get(resource_name=resource_name).state.name
