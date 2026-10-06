"""Roads and rivers from an OSM extract, distance to the nearest line, and absence weights.

Reports of landslides cluster near roads, so uniformly drawn absences teach a
model "far from a road = no landslide". `presence_matched_weights` draws
absences with the presences' own road-distance profile instead, which removes
road distance as a shortcut. The sampling itself stays in `src/sampling.py`.

Which ways count is a choice, made here and nowhere else. Roads: motor roads
only; footpaths, tracks, steps and cycleways are excluded because in Nepal they
are numerous and say little about where a report can come from. Rivers:
waterway=river only. Streams are left out because how well they are mapped
probably follows mapping effort, and canals, ditches and drains are man-made.

Reading the .pbf needs `osmium` (pyosmium); extract once, then reuse the gpkg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MOTOR = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
RIVERS = {"river"}
# Distance bins in km for matching presences to background (last bin is open-ended).
BIN_EDGES_KM = np.array([0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, np.inf])


def _extract_ways(pbf: Path, out_gpkg: Path, crs: str, key: str, keep: set[str]) -> int:
    """Write LineStrings of ways whose `key` tag is in `keep` (projected to `crs`) to a GeoPackage."""
    import geopandas as gpd
    import osmium
    import shapely.wkb

    factory = osmium.geom.WKBFactory()
    geoms, kinds = [], []
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.KeyFilter(key))
    for obj in fp:
        if not obj.is_way():
            continue
        kind = obj.tags.get(key)
        if kind not in keep:
            continue
        try:
            wkb = factory.create_linestring(obj)
        except RuntimeError:  # a node is missing at the edge of the extract
            continue
        geoms.append(shapely.wkb.loads(wkb, hex=True))
        kinds.append(kind)
    gdf = gpd.GeoDataFrame({key: kinds}, geometry=geoms, crs="EPSG:4326").to_crs(crs)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    return len(gdf)


def extract_motor_roads(pbf: Path, out_gpkg: Path, crs: str) -> int:
    """Motor-road LineStrings to a GeoPackage; returns the way count."""
    return _extract_ways(pbf, out_gpkg, crs, "highway", MOTOR)


def extract_rivers(pbf: Path, out_gpkg: Path, crs: str) -> int:
    """River LineStrings to a GeoPackage; returns the way count."""
    return _extract_ways(pbf, out_gpkg, crs, "waterway", RIVERS)


def distance_to_lines(x: np.ndarray, y: np.ndarray, lines, chunk: int = 200_000) -> np.ndarray:
    """Exact distance in metres from each (x, y) to the nearest line. x, y and lines share a metric CRS."""
    import shapely
    from shapely import STRtree

    tree = STRtree(np.asarray(lines))
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        pts = shapely.points(x[i : i + chunk], y[i : i + chunk])
        _, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
        out[i : i + chunk] = d
    return out


distance_to_roads = distance_to_lines  # the first caller was roads


def presence_matched_weights(
    bg_dist_m: np.ndarray, pres_dist_m: np.ndarray, edges_km: np.ndarray = BIN_EDGES_KM
) -> np.ndarray:
    """Per-cell sampling weight so drawn absences follow the presences' road-distance profile.

    weight(bin) = share of presences in the bin / share of background cells in the bin.
    A bin with no presences gets weight 0. Only the road-distance MARGINAL is matched.
    """
    b_bg = np.digitize(bg_dist_m / 1000.0, edges_km[1:-1])
    b_pr = np.digitize(pres_dist_m / 1000.0, edges_km[1:-1])
    n_bins = len(edges_km) - 1
    p_bg = np.bincount(b_bg, minlength=n_bins) / len(b_bg)
    p_pr = np.bincount(b_pr, minlength=n_bins) / len(b_pr)
    w_bin = np.divide(p_pr, p_bg, out=np.zeros(n_bins), where=p_bg > 0)
    return w_bin[b_bg]
