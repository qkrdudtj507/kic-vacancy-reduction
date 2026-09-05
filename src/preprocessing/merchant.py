"""
Storefront (상가개폐업) preprocessing: filter active businesses, derive
store age, and compute a per-center commercial-category density
feature used to enrich the industrial-center dataset.

Original notebooks: 공실전처리.ipynb, 공공공데이터셋.ipynb
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ACTIVE_STATUS_CODE = 1  # biz_stts_cd == 1 -> business currently operating

# service_nm -> broad commercial category, used for the 250m density feature
CATEGORY_MAPPING: dict[str, list[str]] = {
    "food": [
        "일반음식점", "휴게음식점", "식품제조가공업", "제과점영업", "즉석판매제조가공업",
        "식품소분업", "위탁급식영업", "집단급식소", "식품운반업", "식품냉동냉장업",
        "축산가공업", "축산물운반업", "축산판매업", "식육포장처리업",
        "건강기능식품유통전문판매업", "건강기능식품일반판매업",
    ],
    "retail_distribution": [
        "유통전문판매업", "방문판매업", "통신판매업", "전화권유판매업", "후원방문판매업체",
        "대규모점포", "용기포장지제조업", "대중문화예술기획업", "안전상비의약품판매업소",
        "축산물보관업", "축산물운반업", "담배소매업", "담배도매업", "담배수입판매업체",
    ],
    "medical_health": [
        "약국", "의료기기판매임대업", "의료기기수리업", "의원", "부속의료기관",
        "동물병원", "동물미용업", "동물용의료용구판매업", "동물용의약품도매상",
        "안경업", "치과기공소", "의료법인", "요양보호사교육기관",
    ],
    "personal_service": ["세탁업", "목욕장업", "이용업", "미용업", "소독업", "건물위생관리업", "저수조청소업", "환경관리대행기관"],
    "leisure_culture": [
        "노래연습장업", "단란주점영업", "PC방", "당구장업", "체육도장업", "골프연습장업",
        "공연장", "인터넷컴퓨터게임시설제공업", "게임물제작업", "게임물배급업",
        "온라인음악서비스제공업", "음반음악영상물제작업", "음반음악영상물배급업",
        "영화제작업", "영화배급업", "영화수입업", "출판사",
    ],
    "manufacturing": [
        "전력기술설계업체", "전력기술감리업체", "승강기제조및수입업체", "특정고압가스업",
        "고압가스업", "석유판매업", "목재수입유통업", "계량기제조업", "계량기수입업",
        "계량기수리업", "계량기증명업", "축산가공업", "환경측정대행업", "환경전문공사업",
        "환경컨설팅회사", "환경측정업체",
    ],
    "education_planning": ["국내여행업", "국내외여행업", "종합여행업", "국제회의기획업"],
    "logistics": ["물류창고업체", "유료직업소개소", "무료직업소개소", "동물운송업"],
}

_SERVICE_TO_CATEGORY = {
    service: category for category, services in CATEGORY_MAPPING.items() for service in services
}


def filter_active_stores(stores_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only currently-operating businesses (biz_stts_cd == 1)."""
    return stores_df[stores_df["biz_stts_cd"] == ACTIVE_STATUS_CODE].copy()


def add_store_age(stores_df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add a `store_age_years` column derived from the registration date."""
    df = stores_df.copy()
    df["lcpmt_dt"] = pd.to_datetime(df["lcpmt_dt"])
    ref = reference_date or pd.Timestamp.today()
    df["store_age_years"] = (ref - df["lcpmt_dt"]).dt.days // 365
    return df


def map_store_category(service_name: str) -> str:
    return _SERVICE_TO_CATEGORY.get(service_name, "other")


def compute_category_density(
    centers_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    radius_m: float = 250.0,
) -> pd.DataFrame:
    """For each center, count nearby stores (within `radius_m`) per
    commercial category and express it as a density (count / buffer area).

    Uses a KDTree over raw lon/lat degrees (converted to an approximate
    metric radius) for speed - matches the original notebook's method.
    """
    stores_df = stores_df.copy()
    stores_df["category"] = stores_df["service_nm"].apply(map_store_category)

    tree = cKDTree(stores_df[["lon", "lat"]].values)
    deg_radius = radius_m / 111_320  # 1 degree ~= 111.32km at the equator
    buffer_area_m2 = np.pi * (radius_m ** 2)

    records = []
    for _, center in centers_df.iterrows():
        idx = tree.query_ball_point([center["lon"], center["lat"]], deg_radius)
        nearby = stores_df.iloc[idx]

        counts = nearby["category"].value_counts()
        density = {f"{cat}_density": count / buffer_area_m2 for cat, count in counts.items()}
        records.append({"center_name": center["center_name"], **density})

    return pd.DataFrame(records).fillna(0)
