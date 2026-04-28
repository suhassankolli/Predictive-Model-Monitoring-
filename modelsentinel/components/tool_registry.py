"""
Auto-registration tool registry for ModelSentinel monitoring components.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Type
from modelsentinel.components.base import MonitoringComponentBase
from modelsentinel.utils.exceptions import ComponentRegistrationError
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)
_REGISTRY: dict[str, Type[MonitoringComponentBase]] = {}


def monitoring_component(cls: Type[MonitoringComponentBase]) -> Type[MonitoringComponentBase]:
    """Class decorator that registers a MonitoringComponentBase subclass."""
    if not issubclass(cls, MonitoringComponentBase):
        raise ComponentRegistrationError(f"{cls.__name__} must subclass MonitoringComponentBase.")
    if not cls.COMPONENT_NAME:
        raise ComponentRegistrationError(f"{cls.__name__} must define COMPONENT_NAME.")
    if cls.COMPONENT_NAME in _REGISTRY:
        logger.warning(f"Component '{cls.COMPONENT_NAME}' already registered. Overwriting.")
    _REGISTRY[cls.COMPONENT_NAME] = cls
    logger.info(f"Registered monitoring component: {cls.COMPONENT_NAME}")
    return cls


def get_registry() -> dict[str, Type[MonitoringComponentBase]]:
    return dict(_REGISTRY)


def get_component(name: str) -> Type[MonitoringComponentBase]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Component '{name}' not found in registry. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_all_tool_schemas() -> list[dict]:
    return [cls.get_tool_schema() for cls in _REGISTRY.values()]


def register_dynamic_component(cls: Type[MonitoringComponentBase]) -> None:
    monitoring_component(cls)
