# Geospatial Centrality with PySpark

Finds the "most central" educational center within each ownership group in
the Comunidad de Madrid, using PySpark, and treats both "which distance
metric" and "which definition of central" as decisions to be measured, not
assumed.

## Problem

Given ~3,850 educational centers grouped by ownership type (public,
private, publicly-subsidized), find the one center per group best
positioned to host meetings for that group. That sounds like a one-line
`groupBy` + `min(distance)`, but it hides two real decisions: what "central"
means (closest to the average position? or the point that minimizes total
travel for everyone?), and what "distance" means once you're working with
geographic coordinates instead of a flat plane. This project answers both
explicitly, with the disagreement between the options quantified rather
than glossed over -- and separately benchmarks three ways of computing a
per-row distance in Spark, since that choice has real cost implications at
scale.

## Data

[Centros educativos de la Comunidad de Madrid](https://datos.comunidad.madrid/catalogo/dataset/centros_educativos)
-- Comunidad de Madrid open data portal, license
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.es).

The source publishes 7,757 records with coordinates in **ETRS89 UTM zone
30N (EPSG:25830)**, in meters. [`scripts/prepare_data.py`](scripts/prepare_data.py)
filters to the 3,851 active centers with usable coordinates, derives
WGS84 longitude/latitude from the UTM values via `pyproj` (so both a
projected and a geographic representation are available), and writes the
committed snapshot at [`data/centros_educativos_madrid.json`](data/centros_educativos_madrid.json).
The `TITULARIDAD` field (renamed `ownership`: `PRIVADO`, `PRIVADO
CONCERTADO`, `PUBLICO`, `PUBLICO-TITULARIDAD PRIVADA`) is the grouping
column throughout.

## Approach

- **Two definitions of "most central"** ([`src/centrality.py`](src/centrality.py), [`src/spark_jobs.py`](src/spark_jobs.py)):
  - *nearest-to-centroid* -- the center closest to the group's average position. O(n) per group, but a centroid can be pulled off-center by outliers.
  - *medoid* -- the center minimizing the sum of distances to every other center in its group. O(n²) per group (implemented as a self-join), but it's always an actual member of the group and more robust to outliers.
- **Three distance functions** ([`src/distance.py`](src/distance.py)):
  - `utm` -- Euclidean distance on the source's native EPSG:25830 meters. Accurate to sub-meter precision across this region and used as the pipeline's default.
  - `haversine` -- great-circle distance on WGS84 lon/lat. Used to cross-validate the UTM result independently.
  - `euclidean` -- the flat-plane formula applied directly to lon/lat *degrees*. Included as a methodological cautionary case: a degree of longitude is not a fixed distance, and quantifying exactly how wrong this gets is more instructive than just asserting it.
- **Three Spark implementations of the same per-row computation** ([`src/spark_jobs.py`](src/spark_jobs.py), benchmarked in [`src/benchmark.py`](src/benchmark.py)): a plain Python UDF, a `pandas_udf`, and a distance expression built entirely from `pyspark.sql.functions` (no Python execution at all).

## Key findings

**Verify your CRS before picking a distance metric.** UTM and haversine
independently picked the *same* center in all four ownership groups, and
their distance estimates agree within a few meters:

| ownership | UTM distance (m) | haversine distance (m) | difference |
|---|---|---|---|
| PRIVADO | 302.7 | 306.8 | 4.1 m |
| PRIVADO CONCERTADO | 410.7 | 406.4 | 4.3 m |
| PUBLICO | 403.0 | 385.6 | 17.4 m |
| PUBLICO-TITULARIDAD PRIVADA | 868.5 | 866.0 | 2.5 m |

That agreement is the point: two independently-implemented, unit-correct
methods converging is what should give you confidence in a geospatial
result. The naive-degrees `euclidean` metric, by contrast, returns a
number with no physical unit -- and if someone mistakenly rescaled it by
a flat "1 degree = 111.32 km" factor (a real mistake, since that figure is
only true for latitude), it overstates the true distance by **+7.5% to
+29.6%** across the four groups, consistent with the ~24% foreshortening
of a degree of longitude at Madrid's ~40.4N latitude (`cos(40.4 deg) ~=
0.76` -- verified in [`tests/test_distance.py`](tests/test_distance.py)).
In this particular dataset the *center picked* didn't change under the
naive metric, because these ownership groups form fairly compact clusters
-- but the magnitude it reports would still mislead anyone reading it as
meters, and a more elongated group could easily flip the pick.

**Centroid vs. medoid is the choice that actually moves the answer.**
All four groups picked a *different* center under the two definitions,
by 1.2-1.6 km each time:

| ownership | group size | centroid pick | medoid pick | apart |
|---|---|---|---|---|
| PRIVADO | 1,221 | Instituto Vox | Teo Bretón | 1,555 m |
| PRIVADO CONCERTADO | 552 | Teide II | Real Colegio Santa Isabel-La Asunción | 1,260 m |
| PUBLICO | 2,056 | San Isidro | San Eugenio y San Isidro | 1,221 m |
| PUBLICO-TITULARIDAD PRIVADA | 22 | Escuela Infantil Complejo Cuzco | Centro Infantil Ministerio de Fomento | 1,479 m |

Which definition is "right" depends entirely on what the result is for: a
centroid pick minimizes distance from the *average* position (better if
you can't weight by who's actually attending); a medoid pick minimizes
total travel for everyone in the group (better if attendance is roughly
uniform across members) and is guaranteed to be a real, reachable
location rather than wherever the average happens to fall.

**Row-at-a-time Python UDFs are slower because every row leaves the JVM.**
A plain `udf()` serializes each row, ships it across a socket to a
separate Python process, runs the function on one row, and serializes the
result back -- for a two-argument distance calculation, that round-trip
costs more than the arithmetic itself. `pandas_udf` closes most of that
gap by batching many rows into an Arrow-backed pandas Series per call, so
serialization is columnar and the actual computation runs vectorized in
C. The native implementation avoids Python entirely: the formula is built
from `pyspark.sql.functions` primitives and compiles straight into
Spark's JVM-native Catalyst/Tungsten execution plan. Measured on this
project's dataset (local `[*]` Spark, 8 cores, mean of 5 runs each,
`python -m src.benchmark`):

| implementation | n=3,851 (project dataset) | n=770,200 (200x replicated) |
|---|---|---|
| Python UDF | 0.524 s | 1.252 s |
| pandas_udf | 0.372 s | 0.959 s |
| native (Spark SQL functions) | 0.303 s | 0.937 s |

The gap is real but modest here because the per-row computation is a few
FLOPs and this runs on a single machine -- fixed per-task overhead still
makes up a large share of the total. On a genuine cluster, with a more
expensive per-row function, or across many more partitions, the same
mechanism produces a much larger gap; rerun `python -m src.benchmark
--replicate N` to see the trend on your own hardware rather than trusting
numbers measured on someone else's laptop.

## How to run

Requires Python 3.11+ and a JDK (Spark 3.5 needs Java 8/11/17, not 21+).

```bash
pip install -r requirements.txt

# Rebuild the dataset snapshot from the official source (optional -- a snapshot is already committed)
python scripts/prepare_data.py

# Find the most central center per group; --output is required, never defaults to /tmp
python -m src.cli --input data/centros_educativos_madrid.json \
    --output out/most_central.json --distance utm --method centroid --implementation native

# Compare all three distance methods and both centrality definitions
python scripts/compare_distance_methods.py

# Benchmark the three Spark implementations
python -m src.benchmark --input data/centros_educativos_madrid.json --runs 5

pytest tests/
jupyter notebook notebooks/demo.ipynb
```

## Tech stack

Python 3.11, PySpark 3.5, pandas, PyArrow, pyproj, pytest, matplotlib,
Jupyter.
