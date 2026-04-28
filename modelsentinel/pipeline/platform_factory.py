"""
Platform adapter factory for ModelSentinel.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.constants import PlatformType
from modelsentinel.utils.exceptions import PlatformAdapterError


def get_platform_runner(config: MonitoringConfig):
    if config.platform.type == PlatformType.VERTEX_AI:
        from modelsentinel.pipeline.vertex_runner import VertexAIRunner
        return VertexAIRunner(config)
    if config.platform.type == PlatformType.AZURE_ML:
        from modelsentinel.pipeline.azure_runner import AzureMLRunner
        return AzureMLRunner(config)
    if config.platform.type == PlatformType.CP4D:
        from modelsentinel.pipeline.cp4d_runner import CP4DRunner
        return CP4DRunner(config)
    raise PlatformAdapterError(f"Unsupported platform: {config.platform.type}")
