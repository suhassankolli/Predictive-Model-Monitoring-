"""
PySpark session factory for ModelSentinel — Dataproc Serverless configuration.
Author: ModelSentinel Team
Version: 1.0.0
"""
from __future__ import annotations
from typing import Optional
from modelsentinel.config.schema import MonitoringConfig
from modelsentinel.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def create_spark_session(config: MonitoringConfig, app_name: Optional[str] = None,
                          local_mode: bool = False) -> "SparkSession":
    if not SPARK_AVAILABLE:
        raise ImportError("pyspark not installed. Run: pip install modelsentinel")
    name = app_name or f"modelsentinel-{config.model.model_id}"
    if local_mode:
        logger.info(f"Creating local Spark session: {name}")
        return (SparkSession.builder.master("local[*]").appName(name)
                .config("spark.sql.shuffle.partitions", "8")
                .config("spark.driver.memory", "4g").getOrCreate())
    logger.info(f"Configuring Spark session for Dataproc Serverless: {name}")
    builder = (SparkSession.builder.appName(name)
               .config("spark.sql.shuffle.partitions", "200")
               .config("spark.sql.adaptive.enabled", "true")
               .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
               .config("spark.jars.packages",
                       "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1"))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created.")
    return session


def read_bigquery_table(spark, table: str, project: Optional[str] = None,
                         date_filter: Optional[str] = None):
    reader = spark.read.format("bigquery").option("table", table)
    if project:
        reader = reader.option("project", project)
    if date_filter:
        reader = reader.option("filter", date_filter)
    logger.info(f"Reading BigQuery table: {table}")
    return reader.load()
