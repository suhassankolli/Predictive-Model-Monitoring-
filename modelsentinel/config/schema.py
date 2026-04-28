"""
Pydantic v2 models for the full ModelSentinel monitoring configuration schema.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator
from modelsentinel.utils.constants import DriftMethod, TaskType, PlatformType, RiskTier, AlertSeverity


class NumericalFeature(BaseModel):
    name: str
    never_null: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class CategoricalFeature(BaseModel):
    name: str
    allowed_values: list[str] = Field(default_factory=list)
    never_null: bool = False


class ProtectedAttribute(BaseModel):
    name: str
    reference_group: str


class FeatureSchema(BaseModel):
    numerical: list[NumericalFeature] = Field(default_factory=list)
    categorical: list[CategoricalFeature] = Field(default_factory=list)
    protected_attributes: list[ProtectedAttribute] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_id: str
    version: str
    framework: str
    task_type: TaskType
    risk_tier: RiskTier = RiskTier.MEDIUM
    owner: str
    business_domain: str
    deployment_date: str
    registry_uri: Optional[str] = None


class DataSourceConfig(BaseModel):
    type: str = Field(..., pattern="^(bigquery|gcs_parquet|feature_store)$")
    table: Optional[str] = None
    path: Optional[str] = None
    date_column: str = "date"
    min_sample_size: int = 500

    @model_validator(mode="after")
    def validate_source(self) -> "DataSourceConfig":
        if self.type == "bigquery" and not self.table:
            raise ValueError("BigQuery source requires 'table' field.")
        if self.type == "gcs_parquet" and not self.path:
            raise ValueError("GCS Parquet source requires 'path' field.")
        return self


class ReferenceDataConfig(DataSourceConfig):
    start_date: str
    end_date: str


class ProductionDataConfig(DataSourceConfig):
    window_days: int = 30
    record_id_column: str
    score_column: str
    prediction_column: Optional[str] = None


class ActualsConfig(BaseModel):
    type: str = "bigquery"
    table: str
    label_column: str
    join_key: str
    label_lag_days: int = 90


class DataDriftConfig(BaseModel):
    enabled: bool = True
    methods: list[DriftMethod] = Field(default_factory=lambda: [DriftMethod.KS, DriftMethod.PSI])
    numerical_threshold_psi: float = 0.10
    numerical_threshold_ks_pvalue: float = 0.05
    categorical_threshold_tvd: float = 0.05
    alert_severity: AlertSeverity = AlertSeverity.WARNING


class ConceptDriftConfig(BaseModel):
    enabled: bool = True
    proxy_methods: list[str] = Field(default_factory=lambda: ["adversarial_validation"])
    adversarial_auc_threshold: float = 0.65


class ModelPerformanceConfig(BaseModel):
    enabled: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: ["auc_roc", "ks_statistic", "gini"]
    )
    auc_roc_min: Optional[float] = None
    ks_statistic_min: Optional[float] = None
    gini_min: Optional[float] = None
    calibration_ece_max: float = 0.05


class DataQualityConfig(BaseModel):
    enabled: bool = True
    null_rate_max_multiplier: float = 2.0
    outlier_method: str = "iqr"
    outlier_rate_max: float = 0.02
    check_schema: bool = True
    check_referential_integrity: bool = True
    freshness_sla_hours: int = 24


class FeatureAttributionConfig(BaseModel):
    enabled: bool = False
    sample_rate: float = Field(default=0.10, ge=0.01, le=1.0)
    shap_drift_threshold_psi: float = 0.15
    track_importance_rank: bool = True
    top_k_features: int = 10


class FairnessConfig(BaseModel):
    enabled: bool = False
    disparate_impact_min: float = 0.80
    equalised_odds_gap_max: float = 0.05
    alert_severity: AlertSeverity = AlertSeverity.CRITICAL


class PopulationStabilityConfig(BaseModel):
    enabled: bool = True
    psi_green_threshold: float = 0.10
    psi_amber_threshold: float = 0.20
    psi_red_threshold: float = 0.25
    binning_strategy: str = "equal_frequency"
    n_bins: int = 10
    compute_csi: bool = True


class OutputDistributionConfig(BaseModel):
    enabled: bool = True
    methods: list[DriftMethod] = Field(
        default_factory=lambda: [DriftMethod.KS, DriftMethod.WASSERSTEIN]
    )
    alert_threshold_ks_pvalue: float = 0.01


class OperationalHealthConfig(BaseModel):
    enabled: bool = True
    latency_p95_ms_max: int = 200
    latency_p99_ms_max: int = 500
    error_rate_max: float = 0.001
    prediction_volume_anomaly_std: float = 3.0


class ChampionChallengerConfig(BaseModel):
    enabled: bool = False
    challenger_model_id: Optional[str] = None
    challenger_score_table: Optional[str] = None


class RetrainingTriggerRule(BaseModel):
    metric: str
    threshold: float
    direction: str = Field(..., pattern="^(above|below)$")
    cooldown_days: int = 30


class RetrainingTriggersConfig(BaseModel):
    enabled: bool = True
    rules: list[RetrainingTriggerRule] = Field(default_factory=list)
    scheduled_review_months: int = 3


class MonitoringObjectivesConfig(BaseModel):
    data_drift: DataDriftConfig = Field(default_factory=DataDriftConfig)
    concept_drift: ConceptDriftConfig = Field(default_factory=ConceptDriftConfig)
    model_performance: ModelPerformanceConfig = Field(default_factory=ModelPerformanceConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    feature_attribution: FeatureAttributionConfig = Field(default_factory=FeatureAttributionConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    population_stability: PopulationStabilityConfig = Field(default_factory=PopulationStabilityConfig)
    output_distribution: OutputDistributionConfig = Field(default_factory=OutputDistributionConfig)
    operational_health: OperationalHealthConfig = Field(default_factory=OperationalHealthConfig)
    champion_challenger: ChampionChallengerConfig = Field(default_factory=ChampionChallengerConfig)
    retraining_triggers: RetrainingTriggersConfig = Field(default_factory=RetrainingTriggersConfig)


class AlertChannelConfig(BaseModel):
    type: str = Field(..., pattern="^(slack|email|pagerduty|webhook)$")
    webhook_url: Optional[str] = None
    routing_key: Optional[str] = None
    recipients: list[str] = Field(default_factory=list)
    severities: list[AlertSeverity] = Field(
        default_factory=lambda: [AlertSeverity.WARNING, AlertSeverity.CRITICAL]
    )


class AlertingConfig(BaseModel):
    channels: list[AlertChannelConfig] = Field(default_factory=list)
    deduplication_window_minutes: int = 60


class ScheduleConfig(BaseModel):
    cron: str = "0 6 * * *"
    timezone: str = "UTC"
    max_run_duration_minutes: int = 120


class GovernanceConfig(BaseModel):
    sr_11_7_reporting: bool = False
    report_frequency: str = "quarterly"
    eu_ai_act_logging: bool = False
    audit_retention_years: int = 7
    compliance_report_recipients: list[str] = Field(default_factory=list)


class PlatformConfig(BaseModel):
    type: PlatformType = PlatformType.VERTEX_AI
    project: Optional[str] = None
    location: str = "us-central1"
    pipeline_root: str
    service_account: Optional[str] = None
    dataproc_region: Optional[str] = None
    dataproc_subnet: Optional[str] = None
    machine_type: str = "n2-standard-8"


class MonitoringConfig(BaseModel):
    """Root configuration model for ModelSentinel."""
    modelsentinel_version: str = "1.0"
    model: ModelConfig
    features: FeatureSchema = Field(default_factory=FeatureSchema)
    reference: ReferenceDataConfig
    production: ProductionDataConfig
    actuals: Optional[ActualsConfig] = None
    monitoring: MonitoringObjectivesConfig = Field(default_factory=MonitoringObjectivesConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    platform: PlatformConfig

    @model_validator(mode="after")
    def validate_governance_requirements(self) -> "MonitoringConfig":
        if (
            self.model.risk_tier == RiskTier.HIGH
            and self.governance.sr_11_7_reporting
            and not self.monitoring.population_stability.enabled
        ):
            raise ValueError(
                "SR 11-7 high-risk models require population_stability monitoring to be enabled."
            )
        if self.monitoring.fairness.enabled and not self.features.protected_attributes:
            raise ValueError(
                "Fairness monitoring is enabled but no protected_attributes are defined in features."
            )
        return self
