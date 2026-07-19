"""Quantify how far the three distance methods actually diverge, using
pandas directly (no Spark needed -- this is a one-off analysis, not part of
the pipeline). Produces the numbers cited in README.md's methodology
section.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.centrality import most_central_per_group
from src.distance import euclidean_degrees, haversine, utm_projected

DATA_PATH = Path(__file__).parent.parent / "data" / "centros_educativos_madrid.json"
DEGREES_TO_METERS_NAIVE = 111_320  # what someone would use if they (wrongly) treated 1 degree as a fixed distance


def main() -> None:
    df = pd.DataFrame(json.loads(DATA_PATH.read_text(encoding="utf-8")))

    utm_picks = most_central_per_group(df, "ownership", "utm_x", "utm_y", utm_projected, "centroid")
    hav_picks = most_central_per_group(df, "ownership", "longitude", "latitude", haversine, "centroid")
    euc_picks = most_central_per_group(df, "ownership", "longitude", "latitude", euclidean_degrees, "centroid")

    print("## UTM vs. haversine cross-validation (both should agree closely, in meters)\n")
    print("| ownership | UTM pick | UTM distance (m) | haversine pick | haversine distance (m) | same pick? |")
    print("|---|---|---|---|---|---|")
    for group in sorted(df["ownership"].unique()):
        u = utm_picks[utm_picks["ownership"] == group].iloc[0]
        h = hav_picks[hav_picks["ownership"] == group].iloc[0]
        u_dist = utm_projected(u["utm_x"], u["utm_y"], df[df.ownership == group]["utm_x"].mean(), df[df.ownership == group]["utm_y"].mean())
        h_dist = haversine(h["longitude"], h["latitude"], df[df.ownership == group]["longitude"].mean(), df[df.ownership == group]["latitude"].mean())
        same = "yes" if u["center_id"] == h["center_id"] else "NO"
        print(f"| {group} | {u['center_name']} | {u_dist:.1f} | {h['center_name']} | {h_dist:.1f} | {same} |")

    print("\n## Naive euclidean-on-degrees: what its number would mean if misread as meters\n")
    print("| ownership | euclidean pick | raw degree-distance | naively-scaled 'meters' | true haversine distance (m) | error |")
    print("|---|---|---|---|---|---|")
    for group in sorted(df["ownership"].unique()):
        e = euc_picks[euc_picks["ownership"] == group].iloc[0]
        group_df = df[df.ownership == group]
        centroid_lon, centroid_lat = group_df["longitude"].mean(), group_df["latitude"].mean()
        degree_dist = euclidean_degrees(e["longitude"], e["latitude"], centroid_lon, centroid_lat)
        naive_meters = degree_dist * DEGREES_TO_METERS_NAIVE
        true_meters = haversine(e["longitude"], e["latitude"], centroid_lon, centroid_lat)
        error_pct = (naive_meters - true_meters) / true_meters * 100
        print(f"| {group} | {e['center_name']} | {degree_dist:.6f} | {naive_meters:.1f} | {true_meters:.1f} | {error_pct:+.1f}% |")

    print("\n## Does the choice of metric change WHICH center is picked?\n")
    for group in sorted(df["ownership"].unique()):
        u = utm_picks[utm_picks["ownership"] == group].iloc[0]["center_id"]
        h = hav_picks[hav_picks["ownership"] == group].iloc[0]["center_id"]
        e = euc_picks[euc_picks["ownership"] == group].iloc[0]["center_id"]
        print(f"{group}: utm={u} haversine={h} euclidean_degrees={e} all_agree={u == h == e}")

    print("\n## Centroid vs. medoid: how far apart are the two definitions of \"most central\"?\n")
    medoid_picks = most_central_per_group(df, "ownership", "utm_x", "utm_y", utm_projected, "medoid")
    print("| ownership | group size | centroid pick | medoid pick | same? | distance apart (m) |")
    print("|---|---|---|---|---|---|")
    for group in sorted(df["ownership"].unique()):
        c = utm_picks[utm_picks["ownership"] == group].iloc[0]
        m = medoid_picks[medoid_picks["ownership"] == group].iloc[0]
        group_size = len(df[df.ownership == group])
        same = "yes" if c["center_id"] == m["center_id"] else "NO"
        apart_m = utm_projected(c["utm_x"], c["utm_y"], m["utm_x"], m["utm_y"])
        print(f"| {group} | {group_size} | {c['center_name']} | {m['center_name']} | {same} | {apart_m:.0f} |")

    print(
        "\n## Why centroid and medoid diverge: mean-vs-median offset and rural tail\n"
    )
    print(
        "| ownership | mean-to-median offset (m) | centers >20km from median | "
        "centroid pick dist. to median (m) | medoid pick dist. to median (m) | "
        "centroid shift after dropping single farthest point (m) |"
    )
    print("|---|---|---|---|---|---|")
    for group in sorted(df["ownership"].unique()):
        g = df[df.ownership == group]
        mean_x, mean_y = g["utm_x"].mean(), g["utm_y"].mean()
        median_x, median_y = g["utm_x"].median(), g["utm_y"].median()
        mean_to_median = utm_projected(mean_x, mean_y, median_x, median_y)

        dist_from_median = np.sqrt((g["utm_x"] - median_x) ** 2 + (g["utm_y"] - median_y) ** 2)
        pct_far = (dist_from_median > 20_000).mean() * 100

        c = utm_picks[utm_picks["ownership"] == group].iloc[0]
        m = medoid_picks[medoid_picks["ownership"] == group].iloc[0]
        c_to_median = utm_projected(c["utm_x"], c["utm_y"], median_x, median_y)
        m_to_median = utm_projected(m["utm_x"], m["utm_y"], median_x, median_y)

        # Single-outlier-removal check: does dropping just the farthest
        # point (from the centroid) move the centroid anywhere near the
        # full centroid-medoid gap? If not, a lone outlier isn't the story.
        dist_from_mean = np.sqrt((g["utm_x"] - mean_x) ** 2 + (g["utm_y"] - mean_y) ** 2)
        farthest_id = g.loc[dist_from_mean.idxmax(), "center_id"]
        g_no_outlier = g[g["center_id"] != farthest_id]
        mean_x2, mean_y2 = g_no_outlier["utm_x"].mean(), g_no_outlier["utm_y"].mean()
        outlier_shift = utm_projected(mean_x, mean_y, mean_x2, mean_y2)

        print(
            f"| {group} | {mean_to_median:.0f} | {pct_far:.0f}% | "
            f"{c_to_median:.0f} | {m_to_median:.0f} | {outlier_shift:.1f} |"
        )


if __name__ == "__main__":
    main()
