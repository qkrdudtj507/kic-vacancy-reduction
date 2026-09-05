"""
Aggregate grid-level features into a buffer (circular radius) around
each knowledge-industry-center, and compute the distance from each
center to its nearest bus stop / subway station.

Original notebook: 코드정리_분석.ipynb ("Buffer" / "교통 - 가장 가까운 거리" sections)

Note on the nearest-distance step: the original notebook computed this
with an O(n * m) Python double loop over a Haversine formula. Here the
same Haversine geometry is kept (so results match), but the search is
vectorized with a `sklearn.neighbors.BallTree`, which is both much
faster and avoids the loop entirely.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import BallTree

from src.config import COMMERCIAL_BUFFER_M, CRS_METRIC_KOREA, CRS_WGS84, POPULATION_BUFFER_M, SUBWAY_BUFFER_M

EARTH_RADIUS_M = 6_371_000.0


def buffer_aggregate(
    centers_df: pd.DataFrame,
    grid_features: pd.DataFrame,
    buffer_radius_m: float,
    mean_columns: tuple[str, ...] = ("official_land_price", "market_price"),
) -> pd.DataFrame:
    """For each center, sum every numeric grid feature that intersects
    a `buffer_radius_m` circle around it (mean instead of sum for the
    columns listed in `mean_columns`, e.g. price fields which shouldn't
    be summed across cells).

    `grid_features` must contain a `geometry` column (grid cell polygons,
    WGS84) plus a `lon`/`lat` per row is not required - centers supply
    their own coordinates.
    """
    grid_gdf = grid_features.copy()
    grid_gdf["geometry"] = gpd.GeoSeries(grid_gdf["geometry"], crs=CRS_WGS84).values
    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry="geometry", crs=CRS_WGS84).to_crs(CRS_METRIC_KOREA)

    results = []
    for _, center in centers_df.iterrows():
        center_point = gpd.GeoSeries([Point(center["lon"], center["lat"])], crs=CRS_WGS84).to_crs(CRS_METRIC_KOREA)
        buffer_geom = center_point.buffer(buffer_radius_m).iloc[0]

        within_buffer = grid_gdf[grid_gdf.geometry.intersects(buffer_geom)]

        numeric_cols = within_buffer.select_dtypes(include="number").columns
        summary = within_buffer[numeric_cols].sum().to_frame().T

        for col in mean_columns:
            if col in within_buffer.columns:
                summary[col] = within_buffer[col].mean()

        summary["lon"] = center["lon"]
        summary["lat"] = center["lat"]
        results.append(summary)

    aggregated = pd.concat(results, ignore_index=True)
    return centers_df.merge(aggregated, on=["lon", "lat"], how="left", suffixes=("", "_buffer"))


def nearest_distance(
    centers_df: pd.DataFrame,
    target_df: pd.DataFrame,
    distance_col_name: str,
    target_lon_col: str = "lon",
    target_lat_col: str = "lat",
) -> pd.DataFrame:
    """Add the great-circle distance (meters) from each center to its
    nearest point in `target_df` (e.g. nearest bus stop / subway station).
    """
    target_radians = np.radians(target_df[[target_lat_col, target_lon_col]].values)
    tree = BallTree(target_radians, metric="haversine")

    center_radians = np.radians(centers_df[["lat", "lon"]].values)
    distances_rad, _ = tree.query(center_radians, k=1)

    df = centers_df.copy()
    df[distance_col_name] = distances_rad[:, 0] * EARTH_RADIUS_M
    return df


def build_buffer_dataset(
    centers_df: pd.DataFrame,
    commercial_grid: pd.DataFrame,
    population_grid: pd.DataFrame,
    subway_grid: pd.DataFrame,
    bus_stops_df: pd.DataFrame,
    subway_stations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the full buffer-feature pipeline: commercial-district
    aggregation (500m), population aggregation (100m), subway-station
    count aggregation (2000m), plus nearest bus-stop / subway-station
    distance. Returns the fully-assembled per-center feature table.
    """
    commercial = buffer_aggregate(centers_df, commercial_grid, COMMERCIAL_BUFFER_M)
    population = buffer_aggregate(centers_df, population_grid, POPULATION_BUFFER_M)
    subway = buffer_aggregate(centers_df, subway_grid, SUBWAY_BUFFER_M)

    join_keys = [c for c in centers_df.columns if c in commercial.columns and c in population.columns and c in subway.columns]

    buffer_df = commercial.merge(population, on=join_keys, how="outer").merge(subway, on=join_keys, how="outer")

    # bus stop lon/lat columns are swapped in the source file
    bus_stops_df = bus_stops_df.rename(columns={"lon": "lat_raw", "lat": "lon_raw"}).rename(
        columns={"lat_raw": "lat", "lon_raw": "lon"}
    )

    buffer_df = nearest_distance(buffer_df, bus_stops_df, "nearest_bus_stop_m")
    buffer_df = nearest_distance(buffer_df, subway_stations_df, "nearest_subway_station_m")

    return buffer_df
