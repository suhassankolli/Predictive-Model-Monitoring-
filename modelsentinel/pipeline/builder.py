"""
Kubeflow Pipeline DAG builder for ModelSentinel.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.exceptions import PipelineBuildError
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import kfp
    from kfp import dsl
    KFP_AVAILABLE = True
except ImportError:
    KFP_AVAILABLE = False

COMPONENT_IMAGE = "us-central1-docker.pkg.dev/{project}/modelsentinel/runner:1.0.0"

_DEPENDENCY_ORDER = [
    "run_data_quality", "run_data_drift", "run_concept_drift", "run_model_performance",
    "run_fairness_monitoring", "run_population_stability", "run_feature_attribution",
    "run_output_distribution", "run_operational_health", "run_champion_challenger",
    "run_retraining_triggers", "run_governance",
]


def _resolve_order(component_sequence: list[str]) -> list[str]:
    ordered = [c for c in _DEPENDENCY_ORDER if c in component_sequence]
    extras = [c for c in component_sequence if c not in ordered]
    return ordered + extras


class PipelineBuilder:
    """Builds a Kubeflow Pipeline from an ordered list of monitoring component names."""

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.image = COMPONENT_IMAGE.format(project=config.platform.project or "my-project")

    def build(self, component_sequence: list[str]) -> str:
        if not KFP_AVAILABLE:
            raise PipelineBuildError("kfp package is not installed.")
        ordered = _resolve_order(component_sequence)
        logger.info(f"Building pipeline with {len(ordered)} components: {ordered}")
        config_json = self.config.model_dump_json()
        image = self.image

        @dsl.pipeline(
            name=f"modelsentinel-{self.config.model.model_id}",
            description=f"ModelSentinel monitoring pipeline for {self.config.model.model_id}",
        )
        def monitoring_pipeline(monitoring_config_json: str = config_json) -> None:
            prev_task = None
            for comp_name in ordered:
                task = (
                    dsl.ContainerSpec(image=image,
                                      command=["python", "-m", "modelsentinel.runner"],
                                      args=["--component", comp_name,
                                            "--config-json", monitoring_config_json])
                    .set_display_name(comp_name)
                    .set_cpu_limit("4")
                    .set_memory_limit("16G")
                )
                if prev_task is not None:
                    task.after(prev_task)
                prev_task = task

        import tempfile, os
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"modelsentinel_{self.config.model.model_id}_pipeline.yaml",
        )
        try:
            kfp.compiler.Compiler().compile(
                pipeline_func=monitoring_pipeline, package_path=output_path)
            logger.info(f"Pipeline compiled to: {output_path}")
            return output_path
        except Exception as exc:
            raise PipelineBuildError(f"Pipeline compilation failed: {exc}") from exc
