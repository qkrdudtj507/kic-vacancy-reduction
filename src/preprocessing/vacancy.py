"""
Vacancy-rate (공실률) computation for knowledge-industry-centers.

Two vacancy measures are used across the project:

1. `compute_quarterly_occupancy_rate` - a quick, center-level ratio
   (occupied unit count / total unit count) computed straight from
   the raw industrial-center registry. Used only for early EDA.
   Source: 공실전처리.ipynb

2. `compute_vacancy_rate` (+ helpers) - the measure actually fed into
   the final model. It works at the individual-store level: every
   occupied store's floor area (plc_area) is summed per center and
   compared against the center's total floor area, which is far more
   accurate than a raw unit count because unit sizes vary a lot.
   Source: 코드정리_분석.ipynb

   vacancy_rate = 1 - (sum of occupied store area / total floor area)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUARTER_COLUMNS = ["cpn_in_2101", "cpn_in_2201", "cpn_in_2302", "cpn_in_2402", "cpn_in_2406"]


def compute_quarterly_occupancy_rate(centers_df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-quarter occupancy ratio and a multi-year average.

    Expects `centers_df` to contain `tot_cpn` (total unit count) and
    the quarterly occupied-unit-count columns in QUARTER_COLUMNS.
    """
    df = centers_df.copy()
    quarter_labels = [col.replace("cpn_in_", "") for col in QUARTER_COLUMNS]

    for label, col in zip(quarter_labels, QUARTER_COLUMNS):
        df[f"{label}_occupancy_rate"] = df[col] / df["tot_cpn"]

    rate_cols = [f"{label}_occupancy_rate" for label in quarter_labels]
    df["avg_occupancy_rate"] = df[rate_cols].mean(axis=1)
    return df


def compute_avg_unit_area(centers_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the average occupiable unit area per center.

    avg_unit_area = total_floor_area / total_unit_count

    This is later used to impute missing store-level floor areas.
    """
    df = centers_df.copy()
    df["avg_unit_area"] = df["total_floor_area"] / df["total_unit_count"]
    return df


def drop_centers_missing_unit_count(centers_df: pd.DataFrame, extra_drop: list[str] | None = None) -> pd.DataFrame:
    """Remove centers with no `total_unit_count`, plus any explicit
    outliers (e.g. mis-recorded / unrepresentative centers).
    """
    extra_drop = extra_drop or []
    missing = centers_df.loc[centers_df["total_unit_count"].isna(), "center_name"].tolist()
    drop_names = set(missing) | set(extra_drop)
    return centers_df[~centers_df["center_name"].isin(drop_names)].copy()


def interpolate_missing_store_area(
    stores_df: pd.DataFrame,
    centers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fill in missing individual store floor area (`plc_area`).

    Two cases are handled, matching the source analysis:

    * A center has *some* stores with a known area -> missing rows for
      that center are filled with the center's `avg_unit_area`.
    * A center has *no* stores with a known area at all -> the whole
      center's occupied area is estimated as
      `avg_unit_area * known_occupied_unit_count` (falling back
      through the most recent available quarterly count).
    """
    stores_df = stores_df.copy()
    center_names = stores_df["center_name"].str.replace(" ", "", regex=False).unique()

    # Case 1: some rows have data, some don't -> fill with the center average
    for name in center_names:
        mask = stores_df["center_name"].str.replace(" ", "", regex=False) == name
        center_row = centers_df[centers_df["center_name"].str.replace(" ", "", regex=False) == name]
        if center_row.empty:
            continue
        avg_area = center_row["avg_unit_area"].iloc[0]
        stores_df.loc[mask & stores_df["plc_area"].isna(), "plc_area"] = avg_area

    # Case 2: centers with zero known store-area rows -> estimate from
    # avg_unit_area * most-recent known occupied unit count
    fully_missing = [
        name for name in center_names
        if stores_df.loc[stores_df["center_name"].str.replace(" ", "", regex=False) == name, "plc_area"].count() == 0
    ]

    for name in fully_missing:
        center_row = centers_df[centers_df["center_name"].str.replace(" ", "", regex=False) == name]
        if center_row.empty:
            continue
        avg_area = center_row["avg_unit_area"].iloc[0]
        occupied_units = None
        for col in ["cpn_in_2302", "cpn_in_2402", "cpn_in_2406"]:
            if col in center_row and pd.notna(center_row[col].iloc[0]):
                occupied_units = center_row[col].iloc[0]
                break
        if occupied_units is not None:
            mask = stores_df["center_name"].str.replace(" ", "", regex=False) == name
            stores_df.loc[mask & stores_df["plc_area"].isna(), "plc_area"] = avg_area * occupied_units

    return stores_df


def compute_vacancy_rate(stores_df: pd.DataFrame, centers_df: pd.DataFrame) -> pd.DataFrame:
    """Compute center-level vacancy rate from (imputed) store areas.

    vacancy_rate = 1 - (sum of occupied store area / total floor area)

    Returns `centers_df` with a new `vacancy_rate` column.
    """
    centers_df = centers_df.copy()
    centers_df["vacancy_rate"] = np.nan

    for name in stores_df["center_name"].str.replace(" ", "", regex=False).unique():
        occupied_area = stores_df.loc[
            stores_df["center_name"].str.replace(" ", "", regex=False) == name, "plc_area"
        ].sum()

        center_mask = centers_df["center_name"].str.replace(" ", "", regex=False) == name
        total_area_series = centers_df.loc[center_mask, "total_floor_area"]
        if total_area_series.empty or total_area_series.iloc[0] in (0, None) or pd.isna(total_area_series.iloc[0]):
            continue

        total_area = total_area_series.iloc[0]
        centers_df.loc[center_mask, "vacancy_rate"] = 1 - (occupied_area / total_area)

    return centers_df
