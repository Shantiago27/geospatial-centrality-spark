"""Shared SparkSession construction.

Pins the worker Python interpreter to the one currently running (so a venv
doesn't silently fall back to a system `python` that may not have pyspark
installed) and forces the driver to bind to the IPv4 loopback address. The
latter works around a known Windows issue where the JVM driver listens on
the IPv6 loopback by default and the Python worker's connect-back times out
trying to reach it on IPv4 -- harmless to set on Linux/macOS too.
"""
import os
import sys

from pyspark.sql import SparkSession

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def build_spark_session(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
