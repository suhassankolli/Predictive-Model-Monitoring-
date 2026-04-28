"""
Reusable PySpark transformation utilities for ModelSentinel.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)


def compute_null_rates(df) -> dict[str, float]:
    """Compute null rate per column for a Spark DataFrame."""
    total = df.count()
    if total == 0:
        return {}
    from pyspark.sql import functions as F
    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    return {col: count / total for col, count in null_counts.items()}


def sample_to_pandas(df, sample_rate: float = 0.10, seed: int = 42):
    """Sample a Spark DataFrame and return as Pandas. Used for SHAP computation."""
    return df.sample(fraction=min(sample_rate, 1.0), seed=seed).toPandas()


def compute_feature_stats(df, column: str) -> dict:
    """Compute descriptive statistics for a numerical column."""
    from pyspark.sql import functions as F
    stats = df.select(
        F.min(column).alias("min"), F.max(column).alias("max"),
        F.mean(column).alias("mean"), F.stddev(column).alias("stddev"),
        F.percentile_approx(column, 0.25).alias("p25"),
        F.percentile_approx(column, 0.50).alias("p50"),
        F.percentile_approx(column, 0.75).alias("p75"),
        F.percentile_approx(column, 0.95).alias("p95"),
        F.percentile_approx(column, 0.99).alias("p99"),
    ).collect()[0].asDict()
    return {k: round(float(v), 6) if v is not None else None for k, v in stats.items()}


def value_counts(df, column: str) -> dict[str, int]:
    """Return value counts for a categorical column."""
    from pyspark.sql import functions as F
    rows = df.groupBy(column).agg(F.count("*").alias("count")).collect()
    return {str(row[column]): row["count"] for row in rows}
