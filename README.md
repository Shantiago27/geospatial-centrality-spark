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

**That metric-invariance is a robustness result at this dataset's urban
scale, not a claim that distance metric never matters.** We checked all
three metrics (`euclidean`, `haversine`, `utm`) against *both* centrality
definitions -- not just centroid -- and every metric agreed on the same
picked center, for every group, for both centroid and medoid:

```
uv run python scripts/compare_centrality_methods.py
```

The reason this holds is specific to the input: the Comunidad de Madrid
spans roughly 50-80 km edge to edge, small enough that the naive
degrees-as-meters distortion (up to +29.6%, see above) doesn't reorder
which candidate is closest within a group. That margin shrinks as the
area covered grows -- at national or continental scale, where longitude
foreshortening varies enormously across latitudes and a naive metric's
error is no longer a near-constant local factor, the same naive `euclidean`
metric would be expected to flip picks, not just mis-scale a distance. Rerun
the command above on a differently-shaped or larger-scale dataset before
assuming this invariance transfers.

**Centroid vs. medoid is the choice that actually moves the answer.**
Both picks are real dataset rows -- verified by joining the picked
`center_id` back against the source file, not just trusting the pipeline
-- not synthetic averaged points; only the intermediate centroid location
used to rank candidates is a computed average. All four groups picked a
*different* center under the two definitions, and -- per the
metric-invariance check above -- by the same margin regardless of which
distance metric selected the centers:

| ownership | group size | centroid pick | medoid pick | apart (km, great-circle) |
|---|---|---|---|---|
| PRIVADO | 1,221 | Instituto Vox | Teo Bretón | 1.552 |
| PRIVADO CONCERTADO | 552 | Teide II | Real Colegio Santa Isabel-La Asunción | 1.258 |
| PUBLICO | 2,056 | San Isidro | San Eugenio y San Isidro | 1.223 |
| PUBLICO-TITULARIDAD PRIVADA | 22 | Escuela Infantil Complejo Cuzco | Centro Infantil Ministerio de Fomento | 1.481 |

Reproduce this table with the same command as above:
`uv run python scripts/compare_centrality_methods.py`.

*Why they diverge, checked against the data rather than assumed:* these
ownership groups span the entire Comunidad de Madrid region -- a dense
cluster of centers in Madrid city plus a long tail of centers in outlying
towns tens of kilometers away. That's a skewed spatial distribution, and
an arithmetic mean is not robust to skew: in every group, the mean
(centroid) position sits 700 m-1.9 km away from the group's coordinate
*median* (a skew-robust measure of "typical location"), because the
distant, sparser tail pulls the average toward it even though most centers
are nowhere near there:

| ownership | mean-to-median offset | centers >20 km from the median point |
|---|---|---|
| PRIVADO | 1,104 m | 21% |
| PRIVADO CONCERTADO | 719 m | 17% |
| PUBLICO | 1,647 m | 36% |
| PUBLICO-TITULARIDAD PRIVADA | 1,877 m | 9% |

The medoid, which picks the point minimizing total distance to every other
point, isn't pulled the same way: being near the *dense* cluster minimizes
a sum far more effectively than sitting at the skewed geometric average
does, so it stays anchored close to where centers actually concentrate. We
checked this directly -- in 3 of the 4 groups, the medoid pick sits
noticeably closer to the group's median position than the centroid pick
does (e.g. PUBLICO: 111 m vs. 1,333 m from the median; PRIVADO: 375 m vs.
1,336 m). We also checked the simpler "one extreme point" version of this
hypothesis and it doesn't hold: removing only the single farthest-out
center shifts the centroid by just 35-116 m in the three larger groups --
nowhere near enough to explain a 1.2-1.6 km gap. It only becomes the
dominant effect in the smallest group (n=22), where one point among 22
has real leverage over the mean (removing it shifts the centroid 1,276 m,
comparable to that group's full centroid-medoid gap). So the real driver
is the region's overall urban-core-plus-rural-tail shape, not any single
outlier -- see `scripts/compare_distance_methods.py` for the full
per-group breakdown behind these numbers.

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
Spark's JVM-native Catalyst/Tungsten execution plan.

**What was actually measured:** both columns below come from the same
single Windows machine, running Spark locally (`local[*]`, 8 cores) --
there is no cluster involved anywhere in this benchmark. The "n=770,200"
column is *not* a larger real dataset: it's the same 3,851-row file
unioned with itself 200 times (`python -m src.benchmark --replicate 200`),
repartitioned to 8 partitions, so it has the same 4 groups and the same
duplicated coordinates repeated many times over. It's useful for seeing
how the gap between implementations trends as row count grows, but it is
*not* a substitute for real data volume or diversity, and these are
wall-clock numbers from one machine, not a controlled benchmark
environment -- treat the trend as indicative, not the absolute seconds.
Each cell is the mean of 5 timed runs (after one untimed warm-up run):

| implementation | n=3,851 (project dataset, single run of real rows) | n=770,200 (same rows, unioned x200) |
|---|---|---|
| Python UDF | 0.524 s | 1.252 s |
| pandas_udf | 0.372 s | 0.959 s |
| native (Spark SQL functions) | 0.303 s | 0.937 s |

The gap is real but modest here because the per-row computation is a few
FLOPs and this runs on a single machine -- fixed per-task overhead still
makes up a large share of the total. On a genuine cluster, with a more
expensive per-row function, or across many more partitions, the same
mechanism would be expected to produce a much larger gap; rerun `python -m
src.benchmark --replicate N` on your own hardware to check that trend
rather than trusting numbers measured on someone else's laptop.

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
