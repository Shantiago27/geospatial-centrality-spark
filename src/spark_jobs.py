"""Three Spark implementations of the same computation -- distance from
each center to its group's centroid -- to make the performance difference
between them concrete instead of theoretical. See README.md for measured
timings and an explanation of why they differ.

- python_udf: a plain Python function wrapped with `udf()`. Spark serializes
  each row, ships it to a Python worker process, runs the function one row
  at a time, and serializes the result back. That round-trip -- plus no
  vectorization at all -- is the whole cost.
- pandas_udf: the same Python code, but Spark hands it a whole Arrow-backed
  pandas Series per batch instead of one row at a time. Serialization is
  columnar and vectorized instead of row-by-row, and pandas/numpy operations
  run in C, not the Python interpreter loop.
- native: no Python execution at all during the query. The distance formula
  is built entirely out of pyspark.sql.functions primitives, so it compiles
  into the JVM's Catalyst/Tungsten execution plan alongside everything else.
"""
import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.distance import euclidean_degrees, haversine, utm_projected

EARTH_RADIUS_METERS = 6_371_000.0

_ROW_DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_degrees,
    "haversine": haversine,
    "utm": utm_projected,
}


def _centroid_columns(df: DataFrame, group_col: str, x_col: str, y_col: str) -> DataFrame:
    centroids = df.groupBy(group_col).agg(
        F.avg(x_col).alias("centroid_x"), F.avg(y_col).alias("centroid_y")
    )
    return df.join(centroids, group_col)


def distance_to_centroid_python_udf(
    df: DataFrame, group_col: str, x_col: str, y_col: str, method: str
) -> DataFrame:
    distance_fn = _ROW_DISTANCE_FUNCTIONS[method]
    distance_udf = F.udf(distance_fn, DoubleType())
    joined = _centroid_columns(df, group_col, x_col, y_col)
    return joined.withColumn(
        "distance", distance_udf(F.col(x_col), F.col(y_col), F.col("centroid_x"), F.col("centroid_y"))
    )


def distance_to_centroid_pandas_udf(
    df: DataFrame, group_col: str, x_col: str, y_col: str, method: str
) -> DataFrame:
    distance_fn = _ROW_DISTANCE_FUNCTIONS[method]

    @F.pandas_udf(DoubleType())
    def distance_series(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series) -> pd.Series:
        return pd.Series(
            [distance_fn(a, b, c, d) for a, b, c, d in zip(x1, y1, x2, y2)], index=x1.index
        )

    joined = _centroid_columns(df, group_col, x_col, y_col)
    return joined.withColumn(
        "distance", distance_series(F.col(x_col), F.col(y_col), F.col("centroid_x"), F.col("centroid_y"))
    )


def distance_to_centroid_native(
    df: DataFrame, group_col: str, x_col: str, y_col: str, method: str
) -> DataFrame:
    joined = _centroid_columns(df, group_col, x_col, y_col)
    x1, y1, x2, y2 = F.col(x_col), F.col(y_col), F.col("centroid_x"), F.col("centroid_y")

    if method in ("euclidean", "utm"):
        distance_expr = F.sqrt(F.pow(x2 - x1, 2) + F.pow(y2 - y1, 2))
    elif method == "haversine":
        phi1, phi2 = F.radians(y1), F.radians(y2)
        dphi, dlambda = F.radians(y2 - y1), F.radians(x2 - x1)
        a = F.pow(F.sin(dphi / 2), 2) + F.cos(phi1) * F.cos(phi2) * F.pow(F.sin(dlambda / 2), 2)
        distance_expr = 2 * F.lit(EARTH_RADIUS_METERS) * F.atan2(F.sqrt(a), F.sqrt(1 - a))
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return joined.withColumn("distance", distance_expr)


IMPLEMENTATIONS = {
    "udf": distance_to_centroid_python_udf,
    "pandas_udf": distance_to_centroid_pandas_udf,
    "native": distance_to_centroid_native,
}


def most_central_by_centroid(
    df: DataFrame, group_col: str, x_col: str, y_col: str, method: str, implementation: str
) -> DataFrame:
    with_distance = IMPLEMENTATIONS[implementation](df, group_col, x_col, y_col, method)
    window_spec = Window.partitionBy(group_col).orderBy("distance")
    ranked = with_distance.withColumn("rank", F.rank().over(window_spec))
    return ranked.filter(F.col("rank") == 1).drop("rank", "centroid_x", "centroid_y")


def most_central_by_medoid(df: DataFrame, group_col: str, x_col: str, y_col: str, method: str) -> DataFrame:
    """Self-joins every point in a group to every other point in that same
    group, sums the pairwise distances per point, and keeps the point with
    the smallest sum. O(n^2) within each group -- fine at this dataset's
    scale (a few thousand rows per group at most), but the first thing to
    revisit if this needs to run on a much larger set of groups.
    """
    left = df.select(group_col, F.col("center_id").alias("id_a"), F.col(x_col).alias("xa"), F.col(y_col).alias("ya"))
    right = df.select(group_col, F.col("center_id").alias("id_b"), F.col(x_col).alias("xb"), F.col(y_col).alias("yb"))
    pairs = left.join(right, on=group_col)

    xa, ya, xb, yb = F.col("xa"), F.col("ya"), F.col("xb"), F.col("yb")
    if method in ("euclidean", "utm"):
        distance_expr = F.sqrt(F.pow(xb - xa, 2) + F.pow(yb - ya, 2))
    elif method == "haversine":
        phi1, phi2 = F.radians(ya), F.radians(yb)
        dphi, dlambda = F.radians(yb - ya), F.radians(xb - xa)
        a = F.pow(F.sin(dphi / 2), 2) + F.cos(phi1) * F.cos(phi2) * F.pow(F.sin(dlambda / 2), 2)
        distance_expr = 2 * F.lit(EARTH_RADIUS_METERS) * F.atan2(F.sqrt(a), F.sqrt(1 - a))
    else:
        raise ValueError(f"Unknown method: {method!r}")

    total_distances = (
        pairs.withColumn("pair_distance", distance_expr)
        .groupBy(group_col, "id_a")
        .agg(F.sum("pair_distance").alias("total_distance"))
    )
    window_spec = Window.partitionBy(group_col).orderBy("total_distance")
    ranked = total_distances.withColumn("rank", F.rank().over(window_spec))
    medoid_ids = ranked.filter(F.col("rank") == 1).select(group_col, F.col("id_a").alias("center_id"))
    return medoid_ids.join(df, on=[group_col, "center_id"])
