"""
Abstract base class for all ModelSentinel monitoring components.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from modelsentinel.utils.constants import MonitoringStatus, AlertSeverity
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MonitoringResult:
    """Standardised output from every monitoring component."""
    component_name: str
    status: MonitoringStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    alert_triggered: bool = False
    alert_severity: Optional[AlertSeverity] = None
    alert_message: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    run_duration_seconds: float = 0.0
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "metrics": self.metrics,
            "alert_triggered": self.alert_triggered,
            "alert_severity": self.alert_severity.value if self.alert_severity else None,
            "alert_message": self.alert_message,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "run_duration_seconds": self.run_duration_seconds,
            "skipped_reason": self.skipped_reason,
        }


class MonitoringComponentBase(ABC):
    """Abstract base class for all monitoring components."""
    COMPONENT_NAME: str = ""
    COMPONENT_DESCRIPTION: str = ""

    def __init__(self, config: Any, spark_session: Any = None) -> None:
        self.config = config
        self.spark = spark_session
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self) -> MonitoringResult:
        """Execute the monitoring component and return a standardised result."""

    def execute(self) -> MonitoringResult:
        """Wraps run() with timing, error handling, and logging."""
        self.logger.info(f"Starting component: {self.COMPONENT_NAME}")
        start = time.perf_counter()
        try:
            result = self.run()
            result.run_duration_seconds = time.perf_counter() - start
            self.logger.info(
                f"Component {self.COMPONENT_NAME} completed in "
                f"{result.run_duration_seconds:.2f}s — status: {result.status.value}"
            )
            if result.alert_triggered:
                self.logger.warning(f"ALERT [{result.alert_severity}]: {result.alert_message}")
            return result
        except Exception as exc:
            duration = time.perf_counter() - start
            self.logger.error(f"Component {self.COMPONENT_NAME} failed: {exc}")
            return MonitoringResult(
                component_name=self.COMPONENT_NAME,
                status=MonitoringStatus.ERROR,
                alert_triggered=True,
                alert_severity=AlertSeverity.CRITICAL,
                alert_message=f"Component execution failed: {exc}",
                run_duration_seconds=duration,
            )

    @classmethod
    def get_tool_schema(cls) -> dict:
        return {
            "name": cls.COMPONENT_NAME,
            "description": cls.COMPONENT_DESCRIPTION,
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
