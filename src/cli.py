"""Command-line entry point: find the most central educational center per
ownership group and write the result to a caller-chosen path.

Example:
    python -m src.cli --input data/centros_educativos_madrid.json \\
        --output out/most_central --output-format json \\
        --distance utm --method centroid --implementation native
"""
import argparse

from src.io import load_centers, write_result
from src.spark_jobs import most_central_by_centroid, most_central_by_medoid
from src.spark_session import build_spark_session

COORDINATE_COLUMNS = {
    "euclidean": ("longitude", "latitude"),
    "haversine": ("longitude", "latitude"),
    "utm": ("utm_x", "utm_y"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the centers JSON dataset")
    parser.add_argument("--output", required=True, help="Output path for the result (no default -- must be set explicitly)")
    parser.add_argument("--output-format", choices=["json", "csv", "parquet"], default="json")
    parser.add_argument("--distance", choices=["euclidean", "haversine", "utm"], default="utm")
    parser.add_argument(
        "--method",
        choices=["centroid", "medoid"],
        default="centroid",
        help=(
            "'centroid': the real center nearest to the group's mean position "
            "(not the mean position itself). 'medoid': the real center minimizing "
            "total distance to every other member of its group."
        ),
    )
    parser.add_argument(
        "--implementation",
        choices=["udf", "pandas_udf", "native"],
        default="native",
        help="Only applies to --method centroid; medoid always uses the native Spark SQL implementation.",
    )
    parser.add_argument("--group-col", default="ownership")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_col, y_col = COORDINATE_COLUMNS[args.distance]

    spark = build_spark_session("geospatial-centrality")
    try:
        df = load_centers(spark, args.input)

        if args.method == "centroid":
            result = most_central_by_centroid(
                df, args.group_col, x_col, y_col, args.distance, args.implementation
            )
        else:
            result = most_central_by_medoid(df, args.group_col, x_col, y_col, args.distance)

        write_result(result, args.output, args.output_format)
        result.select(args.group_col, "center_name", "distance" if args.method == "centroid" else "center_id").show(
            truncate=False
        )
        print(f"Wrote result to {args.output} ({args.output_format})")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
