"""Three ways to measure distance between two points, and why they don't
agree.

`euclidean_degrees` is included deliberately as a cautionary example: it
applies the flat-plane distance formula to longitude/latitude *degrees*,
which is only valid on a true Cartesian plane. A degree of longitude is not
a fixed distance -- it shrinks by a factor of cos(latitude) as you move away
from the equator -- so at Madrid's ~40.4N latitude, one degree of longitude
covers roughly 24% less ground than one degree of latitude. Treating the two
axes as interchangeable, as `euclidean_degrees` does, silently distorts
every distance computed from it.

`haversine` and `utm_projected` don't have this problem: `haversine` works
on the sphere directly, and `utm_projected` works in meters on a projection
built for this exact purpose (ETRS89 UTM zone 30N, valid to sub-meter
accuracy across the Madrid region). See README.md for how much this
divergence actually moves the "most central" result.
"""
import math

EARTH_RADIUS_METERS = 6_371_000.0


def euclidean_degrees(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Naive planar distance applied directly to longitude/latitude degrees.

    Returns a value in *degrees*, not meters -- it is geometrically invalid
    to treat this as a physical distance. Included only to demonstrate and
    quantify that invalidity; see the module docstring.
    """
    return math.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two WGS84 lon/lat points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def utm_projected(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in meters between two points already projected
    into a metric CRS (this project uses ETRS89 UTM zone 30N, EPSG:25830).
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_degrees,
    "haversine": haversine,
    "utm": utm_projected,
}
