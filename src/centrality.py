"""Two different, both legitimate, definitions of "the most central point
in a group":

- nearest_to_centroid: the point closest to the group's average position.
  Cheap (O(n) per group) but sensitive to outliers, since the centroid
  itself shifts toward them.
- medoid: the point that minimizes the *sum* of distances to every other
  point in the group. More expensive (O(n^2) per group -- every point's
  distance to every other point) but more robust: it's always an actual
  member of the group, and outliers can't drag it away from the bulk of
  the points the way they drag a centroid.

Both are implemented here in pure Python/pandas for use outside Spark (the
notebook, quick local checks); src/spark_jobs.py has the Spark-native
versions used for the benchmark.
"""
from typing import Callable

import pandas as pd

DistanceFn = Callable[[float, float, float, float], float]


def nearest_to_centroid(
    group: pd.DataFrame, x_col: str, y_col: str, distance_fn: DistanceFn
) -> pd.Series:
    centroid_x, centroid_y = group[x_col].mean(), group[y_col].mean()
    distances = group.apply(
        lambda row: distance_fn(row[x_col], row[y_col], centroid_x, centroid_y), axis=1
    )
    return group.loc[distances.idxmin()]


def medoid(group: pd.DataFrame, x_col: str, y_col: str, distance_fn: DistanceFn) -> pd.Series:
    coords = list(zip(group[x_col], group[y_col]))
    total_distances = []
    for x1, y1 in coords:
        total_distances.append(sum(distance_fn(x1, y1, x2, y2) for x2, y2 in coords))
    best_idx = total_distances.index(min(total_distances))
    return group.iloc[best_idx]


def most_central_per_group(
    df: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    distance_fn: DistanceFn,
    method: str,
) -> pd.DataFrame:
    if method not in ("centroid", "medoid"):
        raise ValueError(f"Unknown method: {method!r}. Use 'centroid' or 'medoid'.")
    selector = nearest_to_centroid if method == "centroid" else medoid
    results = [
        selector(group, x_col, y_col, distance_fn) for _, group in df.groupby(group_col)
    ]
    return pd.DataFrame(results).reset_index(drop=True)
