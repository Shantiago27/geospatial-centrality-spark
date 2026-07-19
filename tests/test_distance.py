"""Reference values and where they come from:

- Exact values (3-4-5 triangles): pure arithmetic, true by construction --
  no external source needed, these just confirm the formulas aren't broken.
- Identity/symmetry properties (distance to self is 0, distance(a,b) ==
  distance(b,a)): mathematical properties of any correct distance function,
  not empirical facts -- no external source needed.
- The cos(latitude) ratio: a direct trigonometric consequence of the
  sphere's geometry (computed with Python's own math.cos, independent of
  this project's haversine code), not a number this project invented.
- Every distance in kilometers/meters below (Madrid-Barcelona, the two
  Madrid landmarks, one-degree-of-lat/lon near Madrid) is checked against
  geopy's `geodesic` distance -- a third-party implementation of Karney's
  algorithm on the WGS84 ellipsoid, independent of this project's haversine
  (which assumes a sphere). The tolerances below are set to comfortably
  cover the sphere-vs-ellipsoid gap actually observed (typically
  0.2-0.5%), not tuned to make our own numbers pass.
"""
import math

import pytest
from geopy.distance import geodesic

from src.distance import euclidean_degrees, haversine, utm_projected

MADRID = (-3.7038, 40.4168)  # (lon, lat)
BARCELONA = (2.1734, 41.3851)

# Two adjacent, well-known Madrid landmarks -- a small-scale check at the
# same distance order of magnitude (hundreds of meters) this project
# actually computes, rather than only a long cross-country distance.
PUERTA_DEL_SOL = (-3.7035, 40.4169)
PLAZA_MAYOR = (-3.7074, 40.4155)


def _geopy_km(point_a_lonlat, point_b_lonlat) -> float:
    """geopy takes (lat, lon); this project uses (lon, lat) throughout."""
    lon1, lat1 = point_a_lonlat
    lon2, lat2 = point_b_lonlat
    return geodesic((lat1, lon1), (lat2, lon2)).km


def test_haversine_zero_distance_to_self():
    lon, lat = MADRID
    assert haversine(lon, lat, lon, lat) == pytest.approx(0.0, abs=1e-6)


def test_haversine_is_symmetric():
    a, b = MADRID, BARCELONA
    assert haversine(*a, *b) == pytest.approx(haversine(*b, *a))


def test_haversine_madrid_to_barcelona_matches_geopy_geodesic():
    lon1, lat1 = MADRID
    lon2, lat2 = BARCELONA
    ours_km = haversine(lon1, lat1, lon2, lat2) / 1000
    reference_km = _geopy_km(MADRID, BARCELONA)
    assert ours_km == pytest.approx(reference_km, abs=2.0)


def test_haversine_puerta_del_sol_to_plaza_mayor_matches_geopy_geodesic():
    """Same check at ~350m -- the scale this project actually operates at,
    rather than only a long cross-country distance.
    """
    lon1, lat1 = PUERTA_DEL_SOL
    lon2, lat2 = PLAZA_MAYOR
    ours_m = haversine(lon1, lat1, lon2, lat2)
    reference_m = _geopy_km(PUERTA_DEL_SOL, PLAZA_MAYOR) * 1000
    assert ours_m == pytest.approx(reference_m, abs=2.0)


def test_utm_projected_exact_3_4_5_triangle():
    assert utm_projected(0, 0, 3, 4) == pytest.approx(5.0)


def test_euclidean_degrees_exact_3_4_5_triangle():
    assert euclidean_degrees(0, 0, 3, 4) == pytest.approx(5.0)


def test_haversine_one_degree_latitude_near_madrid_matches_geopy():
    lon, lat = MADRID
    ours_km = haversine(lon, lat, lon, lat + 1) / 1000
    reference_km = _geopy_km((lon, lat), (lon, lat + 1))
    assert ours_km == pytest.approx(reference_km, abs=0.3)


def test_haversine_one_degree_longitude_at_madrid_latitude_matches_geopy():
    """At ~40.4N, a degree of longitude covers meaningfully less ground
    than a degree of latitude -- this is exactly the distortion
    euclidean_degrees ignores by treating both axes as equal.
    """
    lon, lat = MADRID
    ours_km = haversine(lon, lat, lon + 1, lat) / 1000
    reference_km = _geopy_km((lon, lat), (lon + 1, lat))
    assert ours_km == pytest.approx(reference_km, abs=0.5)
    assert ours_km < haversine(lon, lat, lon, lat + 1) / 1000 * 0.8


def test_longitude_to_latitude_degree_ratio_matches_cos_of_latitude():
    """The ratio of a longitude-degree's length to a latitude-degree's
    length at a given latitude approaches cos(latitude) on a sphere -- a
    trigonometric fact (computed here with math.cos, not our own code)
    that our haversine implementation should reproduce.
    """
    lon, lat = MADRID
    lat_degree_km = haversine(lon, lat, lon, lat + 1) / 1000
    lon_degree_km = haversine(lon, lat, lon + 1, lat) / 1000
    expected_ratio = math.cos(math.radians(lat))
    assert lon_degree_km / lat_degree_km == pytest.approx(expected_ratio, abs=0.02)


def test_euclidean_degrees_treats_both_axes_as_equal_unlike_haversine():
    """euclidean_degrees gives identical 'distance' for a 1-degree move in
    either axis; haversine (correctly) does not. This is a direct
    consequence of euclidean_degrees's own definition (sqrt of squared
    differences is symmetric under swapping axes of equal magnitude), not
    an externally-sourced value -- it documents the flaw quantified in the
    README.
    """
    lon, lat = MADRID
    lat_move = euclidean_degrees(lon, lat, lon, lat + 1)
    lon_move = euclidean_degrees(lon, lat, lon + 1, lat)
    assert lat_move == pytest.approx(lon_move)
