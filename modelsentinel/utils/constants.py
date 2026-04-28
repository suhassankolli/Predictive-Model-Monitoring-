"""
ModelSentinel framework-wide constants.
Author: ModelSentinel Team
Version: 1.0.0
"""
from enum import Enum


class MonitoringStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    SKIPPED = "skipped"
    ERROR = "error"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftMethod(str, Enum):
    KS = "ks"
    PSI = "psi"
    JSD = "jsd"
    WASSERSTEIN = "wasserstein"
    MMD = "mmd"
    CHI_SQUARED = "chi_squared"
    TVD = "tvd"


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    RANKING = "ranking"


class PlatformType(str, Enum):
    VERTEX_AI = "vertex_ai"
    AZURE_ML = "azure_ml"
    CP4D = "cp4d"


class RiskTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# PSI traffic-light thresholds
PSI_GREEN_MAX = 0.10
PSI_AMBER_MAX = 0.20
PSI_RED_MIN = 0.20

# Default sampling rates
DEFAULT_SHAP_SAMPLE_RATE = 0.10
DEFAULT_MIN_SAMPLE_SIZE = 500
DEFAULT_FAIRNESS_MIN_GROUP_SIZE = 30

# Anthropic model
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
AGENT_MAX_RETRIES = 3
AGENT_MAX_TOKENS = 4096

# Audit log
AUDIT_LOG_VERSION = "1.0"
