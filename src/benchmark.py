"""Time the three centroid-distance implementations against the real
dataset and print a markdown table. Each implementation is forced to
materialize its result with .count() so lazy evaluation doesn't let one
run's work leak into the next timer.

Usage:
    python -m src.benchmark --input data/centros_educativos_madrid.json --runs 5
"""
import argparse
import time

from src.io import load_centers
from src.spark_jobs import IMPLEMENTATIONS
from src.spark_session import build_spark_session


def time_implementation(df, group_col, x_col, y_col, method, implementation, runs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        IMPLEMENTATIONS[implementation](df, group_col, x_col, y_col, method).count()
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-col", default="ownership")
    parser.add_argument("--distance", choices=["euclidean", "haversine", "utm"], default="utm")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Union the dataset with itself this many times, to see how the gap between "
        "implementations scales with row count beyond this project's ~3.8k-row source data.",
    )
    args = parser.parse_args()

    x_col, y_col = ("utm_x", "utm_y") if args.distance == "utm" else ("longitude", "latitude")

    spark = build_spark_session("benchmark")
    try:
        base_df = load_centers(spark, args.input)
        df = base_df
        for _ in range(args.replicate - 1):
            df = df.union(base_df)
        # Keep partition count independent of --replicate: a handful of
        # large partitions isolates per-row cost, instead of per-task
        # scheduling overhead from many small ones.
        default_parallelism = spark.sparkContext.defaultParallelism
        df = df.repartition(default_parallelism).cache()
        row_count = df.count()
        print(f"Dataset: {row_count} rows, grouped by `{args.group_col}`\n")

        results = {}
        for implementation in ("udf", "pandas_udf", "native"):
            # one untimed warm-up run so JIT/codegen isn't charged to run 1
            IMPLEMENTATIONS[implementation](df, args.group_col, x_col, y_col, args.distance).count()
            times = time_implementation(
                df, args.group_col, x_col, y_col, args.distance, implementation, args.runs
            )
            results[implementation] = times

        print("| implementation | runs | mean (s) | min (s) | max (s) |")
        print("|---|---|---|---|---|")
        for implementation, times in results.items():
            mean_t = sum(times) / len(times)
            print(
                f"| {implementation} | {len(times)} | {mean_t:.3f} | {min(times):.3f} | {max(times):.3f} |"
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
