"""Build the committed dataset snapshot from the official Comunidad de
Madrid "Centros educativos" open dataset.

The source publishes coordinates as UTM_X/UTM_Y in ETRS89 UTM zone 30N
(EPSG:25830) meters, plus a TITULARIDAD field (PUBLICO / PRIVADO / PRIVADO
CONCERTADO / PUBLICO-TITULARIDAD PRIVADA) that is exactly the ownership
grouping this project computes centrality within. WGS84 longitude/latitude
is derived from the UTM coordinates via pyproj so the dataset can drive all
three distance methods (see src/distance.py).

Source: https://datos.comunidad.madrid/catalogo/dataset/centros_educativos
License: Creative Commons Attribution 4.0 (CC BY 4.0)
"""
import argparse
import json
import urllib.request
from pathlib import Path

from pyproj import Transformer

SOURCE_URL = (
    "https://datos.comunidad.madrid/dataset/"
    "c750856d-3166-4dac-8e80-d1b824c968b5/resource/"
    "be2264df-c720-4619-ab79-aebad9b248e0/download/centros_educativos.json"
)

# EPSG:25830 = ETRS89 / UTM zone 30N, the CRS the source publishes coordinates in.
UTM_TO_WGS84 = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)


def download_raw(destination: Path) -> None:
    print(f"Downloading source dataset from {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, destination)


def clean_record(raw: dict) -> dict | None:
    if raw.get("SITUACIÓN") != "ALTA":
        return None
    utm_x_raw, utm_y_raw = raw.get("UTM_X", ""), raw.get("UTM_Y", "")
    center_name = (raw.get("CENTRO") or "").strip()
    if not utm_x_raw or not utm_y_raw or not center_name:
        return None

    utm_x, utm_y = float(utm_x_raw), float(utm_y_raw)
    longitude, latitude = UTM_TO_WGS84.transform(utm_x, utm_y)

    return {
        "center_id": raw.get("CODIGO", ""),
        "center_name": center_name,
        "center_type": (raw.get("TIPO_EXT") or "").strip(),
        "ownership": (raw.get("TITULARIDAD") or "").strip(),
        "municipality": (raw.get("MUNICIPIO") or "").strip(),
        "education_area": (raw.get("DAT") or "").strip(),
        "utm_x": utm_x,
        "utm_y": utm_y,
        "longitude": longitude,
        "latitude": latitude,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=None,
        help="Path to an already-downloaded raw source JSON. If omitted, downloads it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "centros_educativos_madrid.json",
    )
    args = parser.parse_args()

    raw_path = args.raw_input
    if raw_path is None:
        raw_path = Path(__file__).parent / "_raw_centros_educativos.json"
        download_raw(raw_path)

    raw_records = json.loads(raw_path.read_text(encoding="utf-8"))["data"]
    cleaned = [r for r in (clean_record(rec) for rec in raw_records) if r is not None]
    cleaned.sort(key=lambda r: r["center_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=None), encoding="utf-8"
    )
    print(f"Wrote {len(cleaned)} active centers to {args.output}")


if __name__ == "__main__":
    main()
