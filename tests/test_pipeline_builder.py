"""Tests for PipelineBuilder DAG ordering."""
import pytest
from unittest.mock import MagicMock, patch
from modelsentinel.pipeline.builder import _resolve_order, PipelineBuilder

def test_dependency_order():
    seq = ["run_governance", "run_data_drift", "run_data_quality"]
    ordered = _resolve_order(seq)
    assert ordered.index("run_data_quality") < ordered.index("run_data_drift")
    assert ordered.index("run_data_drift") < ordered.index("run_governance")

def test_all_included():
    seq = ["run_model_performance", "run_data_drift", "run_fairness_monitoring"]
    assert set(_resolve_order(seq)) == set(seq)

def test_unknown_appended():
    seq = ["run_data_drift", "run_custom_xyz"]
    ordered = _resolve_order(seq)
    assert "run_custom_xyz" in ordered
    assert ordered.index("run_data_drift") < ordered.index("run_custom_xyz")

def test_raises_without_kfp():
    cfg = MagicMock(); cfg.model.model_id = "t"; cfg.platform.project = "p"
    builder = PipelineBuilder(cfg)
    with patch("modelsentinel.pipeline.builder.KFP_AVAILABLE", False):
        from modelsentinel.utils.exceptions import PipelineBuildError
        with pytest.raises(PipelineBuildError): builder.build(["run_data_drift"])
