"""Reading the source dataset into Spark and writing results out, with an
explicit schema on read -- no schema inference on a file this project's
correctness depends on.
"""
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

CENTERS_SCHEMA = StructType(
    [
        StructField("center_id", StringType(), False),
        StructField("center_name", StringType(), False),
        StructField("center_type", StringType(), True),
        StructField("ownership", StringType(), False),
        StructField("municipality", StringType(), True),
        StructField("education_area", StringType(), True),
        StructField("utm_x", DoubleType(), False),
        StructField("utm_y", DoubleType(), False),
        StructField("longitude", DoubleType(), False),
        StructField("latitude", DoubleType(), False),
    ]
)


def load_centers(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.schema(CENTERS_SCHEMA).option("multiLine", True).json(path)


def write_result(df: DataFrame, path: str, fmt: str = "json") -> None:
    """Write a centrality result to `path`.

    Centrality results are one row per group -- a few groups at most, never
    the full dataset -- so they're collected to the driver and written with
    pandas rather than Spark's distributed writer. That sidesteps Spark's
    Hadoop-based FileOutputCommitter, which on Windows requires winutils.exe
    native binaries that aren't part of a standard PySpark install. This
    would be the wrong call for a large output (defeats the point of
    distributed writes), but is the right one here.
    """
    if fmt not in ("json", "csv", "parquet"):
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json', 'csv', or 'parquet'.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = df.toPandas()

    if fmt == "json":
        pdf.to_json(output_path, orient="records", force_ascii=False, indent=2)
    elif fmt == "csv":
        pdf.to_csv(output_path, index=False)
    elif fmt == "parquet":
        pdf.to_parquet(output_path, index=False)
