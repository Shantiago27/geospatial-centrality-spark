import math

import pytest

from src.distance import euclidean_degrees, haversine, utm_projected

MADRID = (-3.7038, 40.4168)  # (lon, lat)
BARCELONA = (2.1734, 41.3851)
MADRID_BARCELONA_KM_REFERENCE = 504.6  # widely-published great-circle figure


def test_haversine_zero_distance_to_self():
    lon, lat = MADRID
    assert haversine(lon, lat, lon, lat) == pytest.approx(0.0, abs=1e-6)


def test_haversine_is_symmetric():
    a, b = MADRID, BARCELONA
    assert haversine(*a, *b) == pytest.approx(haversine(*b, *a))


def test_haversine_madrid_to_barcelona_matches_known_distance():
    lon1, lat1 = MADRID
    lon2, lat2 = BARCELONA
    distance_km = haversine(lon1, lat1, lon2, lat2) / 1000
    assert distance_km == pytest.approx(MADRID_BARCELONA_KM_REFERENCE, abs=3.0)


def test_utm_projected_exact_3_4_5_triangle():
    assert utm_projected(0, 0, 3, 4) == pytest.approx(5.0)


def test_euclidean_degrees_exact_3_4_5_triangle():
    assert euclidean_degrees(0, 0, 3, 4) == pytest.approx(5.0)


def test_haversine_one_degree_latitude_is_about_111_km():
    lon, lat = MADRID
    distance_km = haversine(lon, lat, lon, lat + 1) / 1000
    assert distance_km == pytest.approx(111.32, abs=0.5)


def test_haversine_one_degree_longitude_at_madrid_latitude_is_shorter():
    """At ~40.4N, a degree of longitude covers ~24% less ground than a
    degree of latitude (cos(40.4 deg) ~= 0.76) -- this is exactly the
    distortion euclidean_degrees ignores by treating both axes as equal.
    """
    lon, lat = MADRID
    lat_degree_km = haversine(lon, lat, lon, lat + 1) / 1000
    lon_degree_km = haversine(lon, lat, lon + 1, lat) / 1000
    expected_ratio = math.cos(math.radians(lat))
    assert lon_degree_km / lat_degree_km == pytest.approx(expected_ratio, abs=0.02)
    assert lon_degree_km < lat_degree_km * 0.8


def test_euclidean_degrees_treats_both_axes_as_equal_unlike_haversine():
    """euclidean_degrees gives identical 'distance' for a 1-degree move in
    either axis; haversine (correctly) does not. This is the flaw
    quantified in the README.
    """
    lon, lat = MADRID
    lat_move = euclidean_degrees(lon, lat, lon, lat + 1)
    lon_move = euclidean_degrees(lon, lat, lon + 1, lat)
    assert lat_move == pytest.approx(lon_move)
