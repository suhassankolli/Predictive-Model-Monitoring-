"""
YAML configuration loader and validator for ModelSentinel.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml
from pydantic import ValidationError
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.exceptions import ConfigValidationError
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)


def _expand_env_vars(obj: object) -> object:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(i) for i in obj]
    return obj


def load_config(path: str | Path) -> MonitoringConfig:
    """Load, validate, and return a MonitoringConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.info(f"Loading monitoring config from: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    raw = _expand_env_vars(raw)
    try:
        config = MonitoringConfig.model_validate(raw)
    except ValidationError as exc:
        errors = exc.errors()
        logger.error(f"Config validation failed with {len(errors)} error(s).")
        raise ConfigValidationError(errors) from exc
    logger.info(f"Config loaded: {config.model.model_id} v{config.model.version}")
    return config


def save_config(config: MonitoringConfig, path: str | Path) -> None:
    """Serialise a MonitoringConfig back to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to: {path}")
