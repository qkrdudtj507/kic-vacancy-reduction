"""
Build a 100m x 100m grid-level feature dataset by spatially joining
floating population, resident population, card sales, commercial
district composition, transit stop counts, and land price onto a
pre-built grid (GeoJSON, e.g. 15.성남시_격자(100M).geojson).

Original notebook: 공공공_격자생성.ipynb, plus the `Update_colums`
category-consolidation step from 코드정리_분석.ipynb.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.config import CRS_WGS84, GRID_COLUMN_RENAME


def load_grid(geojson_path: str) -> gpd.GeoDataFrame:
    """Load a pre-built grid GeoJSON and standardize the CRS/column names."""
    grid = gpd.read_file(geojson_path)
    grid = grid.set_crs(CRS_WGS84, allow_override=True)
    return grid.rename(columns=GRID_COLUMN_RENAME)


def _to_point_gdf(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat") -> gpd.GeoDataFrame:
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=CRS_WGS84)


def join_floating_population(grid: gpd.GeoDataFrame, floating_pop_df: pd.DataFrame) -> pd.DataFrame:
    """Assign each floating-population sample point to a grid cell and
    average the numeric columns per cell (time-of-day / age / weekday
    breakdown columns already present in `floating_pop_df`).
    """
    points = _to_point_gdf(floating_pop_df)
    joined = gpd.sjoin(points, grid, how="right", predicate="within")

    value_cols = [c for c in floating_pop_df.columns if c not in ("lon", "lat")]
    return joined.groupby("grid_id")[value_cols].mean().reset_index()


def join_card_sales(grid_features: pd.DataFrame, card_sales_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-aggregated per-grid card sales figures (already keyed by grid_id)."""
    return grid_features.merge(card_sales_df.rename(columns=GRID_COLUMN_RENAME), how="left", on="grid_id")


def join_resident_population(grid_features: pd.DataFrame, resident_pop_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Merge resident population for a single reference year."""
    yearly = resident_pop_df[resident_pop_df["year"] == year]
    return grid_features.merge(yearly.rename(columns=GRID_COLUMN_RENAME), how="left", on="grid_id")


def join_commercial_district(grid: gpd.GeoDataFrame, grid_features: pd.DataFrame, district_df: pd.DataFrame) -> pd.DataFrame:
    """Count commercial-district businesses per grid cell, one column
    per broad category (com_lc_nm), and merge onto `grid_features`.
    """
    points = _to_point_gdf(district_df)
    joined = gpd.sjoin(points, grid, how="right", predicate="within")

    category_counts = pd.crosstab(joined["grid_id"], joined["com_lc_nm"]).reset_index()
    category_counts.columns = [c.replace("\u00b7", "/") for c in category_counts.columns]

    return grid_features.merge(category_counts, how="left", on="grid_id")


def join_transit_counts(
    grid: gpd.GeoDataFrame,
    grid_features: pd.DataFrame,
    subway_df: pd.DataFrame,
    bus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count subway stations and bus stops per grid cell."""
    subway_points = _to_point_gdf(subway_df)
    subway_joined = gpd.sjoin(subway_points, grid, how="left", predicate="within")
    subway_count = subway_joined.groupby("grid_id").size().reset_index(name="subway_stop_count")

    bus_points = _to_point_gdf(bus_df, lon_col="경도", lat_col="위도")
    bus_joined = gpd.sjoin(bus_points, grid, how="left", predicate="within")
    bus_count = bus_joined.groupby("grid_id").size().reset_index(name="bus_stop_count")

    grid_features = grid_features.merge(subway_count, how="left", on="grid_id")
    grid_features = grid_features.merge(bus_count, how="left", on="grid_id")
    return grid_features


def join_land_price(grid: gpd.GeoDataFrame, grid_features: pd.DataFrame, land_price_df: pd.DataFrame) -> pd.DataFrame:
    """Average official land price (PNILP) and market price
    (ESTT_CURPRC_SMNT_AMT) per grid cell.
    """
    points = _to_point_gdf(land_price_df, lon_col="LO", lat_col="LA")
    joined = gpd.sjoin(points, grid, how="left", predicate="within")

    agg = joined.groupby("grid_id").agg(
        official_land_price=("PNILP", "mean"),
        market_price=("ESTT_CURPRC_SMNT_AMT", "mean"),
    ).reset_index()

    return grid_features.merge(agg, how="left", on="grid_id")


def consolidate_commercial_categories(grid_features: pd.DataFrame) -> pd.DataFrame:
    """Roll the fine-grained commercial-district categories up into
    the five broad groups used by the model (science_tech, real_estate,
    service, retail, education), matching `Update_colums` in the
    original analysis notebook.
    """
    df = grid_features.copy()

    df["service_count"] = (
        df.get("생활서비스", 0) + df.get("보건의료", 0) + df.get("수리/개인", 0)
        + df.get("숙박", 0) + df.get("시설관리/임대", 0)
    )
    df["retail_count"] = df.get("소매", 0) + df.get("음식", 0) + df.get("관광/여가/오락", 0)
    df["education_count"] = df.get("교육", 0) + df.get("학문/교육", 0)
    df["culture_sports_count"] = df.get("스포츠", 0) + df.get("예술/스포츠", 0)

    df = df.rename(columns={"과학/기술": "science_tech_count", "부동산": "real_estate_count"})

    raw_cols = [
        "생활서비스", "보건의료", "수리/개인", "숙박", "시설관리/임대",
        "소매", "음식", "관광/여가/오락", "교육", "학문/교육", "스포츠", "예술/스포츠",
    ]
    return df.drop(columns=[c for c in raw_cols if c in df.columns])


def consolidate_resident_age_groups(grid_features: pd.DataFrame) -> pd.DataFrame:
    """Roll detailed male/female-by-decade resident population columns
    up into three broad age bands used by the model.
    """
    df = grid_features.copy()

    df["resident_pop_20_30"] = (
        df.get("m_20g_pop", 0) + df.get("w_20g_pop", 0) + df.get("m_30g_pop", 0) + df.get("w_30g_pop", 0)
    )
    df["resident_pop_40_50"] = (
        df.get("m_40g_pop", 0) + df.get("w_40g_pop", 0) + df.get("m_50g_pop", 0) + df.get("w_50g_pop", 0)
    )
    df["resident_pop_60_80"] = (
        df.get("m_60g_pop", 0) + df.get("w_60g_pop", 0) + df.get("m_70g_pop", 0)
        + df.get("w_70g_pop", 0) + df.get("m_80g_pop", 0) + df.get("w_80g_pop", 0)
    )

    detail_cols = [
        "m_20g_pop", "w_20g_pop", "m_30g_pop", "w_30g_pop", "m_40g_pop", "w_40g_pop",
        "m_50g_pop", "w_50g_pop", "m_60g_pop", "w_60g_pop", "m_70g_pop", "w_70g_pop",
        "m_80g_pop", "w_80g_pop", "m_90g_pop", "w_90g_pop", "m_100g_pop", "w_100g_pop",
    ]
    return df.drop(columns=[c for c in detail_cols if c in df.columns])


def build_grid_dataset(
    grid: gpd.GeoDataFrame,
    floating_pop_df: pd.DataFrame,
    card_sales_df: pd.DataFrame,
    resident_pop_df: pd.DataFrame,
    district_df: pd.DataFrame,
    subway_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    land_price_df: pd.DataFrame,
    resident_pop_year: int = 2023,
) -> pd.DataFrame:
    """Run the full grid-feature pipeline end to end and return a single
    grid_id-indexed feature table ready for buffer aggregation.
    """
    features = join_floating_population(grid, floating_pop_df)
    features = join_card_sales(features, card_sales_df)
    features = join_resident_population(features, resident_pop_df, year=resident_pop_year)
    features = join_commercial_district(grid, features, district_df)
    features = join_transit_counts(grid, features, subway_df, bus_df)
    features = join_land_price(grid, features, land_price_df)
    features = consolidate_commercial_categories(features)
    features = consolidate_resident_age_groups(features)
    features = features.merge(grid[["grid_id", "geometry"]], how="left", on="grid_id")

    return features
