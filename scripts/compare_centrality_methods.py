"""Run both centrality definitions (centroid, medoid) under all three
distance metrics (euclidean, haversine, utm), print a comparison table,
and report the centroid-vs-medoid gap in km for each group -- plus whether
that gap holds regardless of which distance metric picked the centers.

Pure pandas, no Spark session needed -- same approach as
compare_distance_methods.py, reusing src.centrality and src.distance
instead of recomputing any of their formulas.
"""
import json
from pathlib import Path

import pandas as pd

from src.centrality import most_central_per_group
from src.distance import DISTANCE_FUNCTIONS, haversine

DATA_PATH = Path(__file__).parent.parent / "data" / "centros_educativos_madrid.json"

COORDINATE_COLUMNS = {
    "euclidean": ("longitude", "latitude"),
    "haversine": ("longitude", "latitude"),
    "utm": ("utm_x", "utm_y"),
}


def main() -> None:
    df = pd.DataFrame(json.loads(DATA_PATH.read_text(encoding="utf-8")))
    groups = sorted(df["ownership"].unique())

    picks = {}  # (method, metric) -> {group: row}
    for metric, distance_fn in DISTANCE_FUNCTIONS.items():
        x_col, y_col = COORDINATE_COLUMNS[metric]
        for method in ("centroid", "medoid"):
            result = most_central_per_group(df, "ownership", x_col, y_col, distance_fn, method)
            picks[(method, metric)] = {row["ownership"]: row for _, row in result.iterrows()}

    print("\n=== Comparative table: method x metric x group ===")
    print(f"{'method':<9} {'metric':<10} {'group':<30} {'center':<50} {'lon':>10} {'lat':>10}")
    for metric in DISTANCE_FUNCTIONS:
        for method in ("centroid", "medoid"):
            for group in groups:
                row = picks[(method, metric)][group]
                print(
                    f"{method:<9} {metric:<10} {group:<30} {row['center_name']:<50} "
                    f"{row['longitude']:>10.5f} {row['latitude']:>10.5f}"
                )

    print("\n=== Centroid vs medoid gap (km), by group ===")
    print(f"{'group':<30} {'centroid pick':<40} {'medoid pick':<40} {'gap (km)':>10} {'same across metrics?':>22}")
    for group in groups:
        picks_by_metric = {
            metric: (picks[("centroid", metric)][group]["center_id"], picks[("medoid", metric)][group]["center_id"])
            for metric in DISTANCE_FUNCTIONS
        }
        invariant = len(set(picks_by_metric.values())) == 1

        c = picks[("centroid", "utm")][group]
        m = picks[("medoid", "utm")][group]
        gap_km = haversine(c["longitude"], c["latitude"], m["longitude"], m["latitude"]) / 1000.0
        print(f"{group:<30} {c['center_name']:<40} {m['center_name']:<40} {gap_km:>10.3f} {'yes' if invariant else 'NO':>22}")


if __name__ == "__main__":
    main()
