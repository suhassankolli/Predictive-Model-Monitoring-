"""
ModelSentinel custom exception hierarchy.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations


class ModelSentinelError(Exception):
    """Base exception for all ModelSentinel errors."""


class ConfigValidationError(ModelSentinelError):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed with {len(errors)} error(s).")


class ComponentRegistrationError(ModelSentinelError):
    pass


class ComponentExecutionError(ModelSentinelError):
    def __init__(self, component_name: str, cause: Exception) -> None:
        self.component_name = component_name
        self.cause = cause
        super().__init__(f"Component '{component_name}' failed: {cause}")


class ComponentGenerationError(ModelSentinelError):
    pass


class PipelineBuildError(ModelSentinelError):
    pass


class PlatformAdapterError(ModelSentinelError):
    pass


class HumanCheckpointRequired(ModelSentinelError):
    def __init__(self, reason: str, context: dict) -> None:
        self.reason = reason
        self.context = context
        super().__init__(f"Human checkpoint required: {reason}")


class AlertDeliveryError(ModelSentinelError):
    pass


class AuditLogError(ModelSentinelError):
    pass


class BaselineError(ModelSentinelError):
    pass
