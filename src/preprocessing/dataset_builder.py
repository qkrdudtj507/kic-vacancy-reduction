"""
Merge auxiliary building-registry (표제부) and transaction-price data
into the center-level dataset.

Original notebook: 공공공데이터셋.ipynb
"""

from __future__ import annotations

import pandas as pd


def merge_elevator_and_parking(
    centers_df: pd.DataFrame,
    building_ledger_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach elevator count and indoor parking capacity to each center.

    `building_ledger_df` (표제부) is address-level and may contain
    multiple rows per building (e.g. per wing); counts are summed per
    address before joining on `centers_df.address`.
    """
    ledger = building_ledger_df[["plat_plc", "ride_use_elvt_cnt", "indr_auto_utcnt"]].copy()
    ledger["plat_plc"] = ledger["plat_plc"].str.replace("번지", "", regex=True).str.strip()

    ledger_agg = ledger.groupby("plat_plc", as_index=False).agg(
        elevator_count=("ride_use_elvt_cnt", "sum"),
        indoor_parking_capacity=("indr_auto_utcnt", "sum"),
    )

    merged = centers_df.merge(
        ledger_agg, left_on="address", right_on="plat_plc", how="left"
    ).drop(columns=["plat_plc"])

    return merged.dropna(axis=1, how="all")


def _clean_krw_amount(series: pd.Series) -> pd.Series:
    """'1,234,000원' -> 1234000.0 (numeric, NaN on parse failure)."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=True)
        .str.replace("원", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )


def _clean_area(series: pd.Series) -> pd.Series:
    """'84.98㎡' -> 84.98 (numeric, NaN on parse failure)."""
    return series.astype(str).str.replace("㎡", "", regex=True).pipe(pd.to_numeric, errors="coerce")


def merge_transaction_price(centers_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent transaction-price fields to each center
    and clean the Korean-formatted numeric columns (won amounts, ㎡).
    """
    merged = centers_df.merge(price_df, left_on="address", right_on="주소", how="left")

    for col, cleaner in [
        ("3.3㎡기준", _clean_krw_amount),
        ("거래금액", _clean_krw_amount),
        ("전용면적", _clean_area),
        ("공급면적", _clean_area),
    ]:
        if col in merged.columns:
            merged[col] = cleaner(merged[col])

    drop_cols = [
        "approval_date", "use_zone", "total_floor_area", "Unnamed: 0",
        "거래일자", "지식산업센터명", "주소", "건축년도",
    ]
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")
