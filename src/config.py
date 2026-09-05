"""
Project-wide configuration: file paths, spatial constants, and the
Korean -> English column mapping used to keep the rest of the codebase
readable.

Raw data files are not included in this repository (see data/README.md
for the expected schema of each file). Update RAW_DATA_DIR to point at
your local copy of the source CSV/GeoJSON files before running the
pipeline.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --------------------------------------------------------------------------- #
# Spatial constants
# --------------------------------------------------------------------------- #
GRID_CELL_SIZE_M = 100          # side length of the analysis grid (100m x 100m)
COMMERCIAL_BUFFER_M = 500       # buffer radius used to aggregate commercial-district features
POPULATION_BUFFER_M = 100       # buffer radius used to aggregate resident/floating population
SUBWAY_BUFFER_M = 2000          # buffer radius used to aggregate subway accessibility
CRS_WGS84 = "EPSG:4326"         # lon/lat degrees
CRS_METRIC_KOREA = "EPSG:32652" # UTM 52N (meters) - used for buffer/distance calculations

# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
PCA_N_COMPONENTS = 4
TEST_SIZE = 0.3

# Features that go into the PCA + RandomForest vacancy-rate model.
# These are the (already English-renamed) columns produced by the
# preprocessing / grid pipeline - see src/grid/grid_features.py
MODEL_FEATURE_COLUMNS = [
    "science_tech_count",
    "real_estate_count",
    "service_count",
    "retail_count",
    "education_count",
    "official_land_price",
    "market_price",
    "card_sales",
    "estimated_sales",
    "bus_stop_count",
    "subway_stop_count",
    "resident_pop_20_30",
    "resident_pop_40_50",
    "resident_pop_60_80",
    "midday_floating_pop",  # 12~17시 유동인구
]

TARGET_COLUMN = "vacancy_rate"

# --------------------------------------------------------------------------- #
# Column name mapping (Korean source columns -> English working names)
# --------------------------------------------------------------------------- #
# Applied right after each raw file is loaded so every downstream module
# only ever has to deal with English identifiers.
COMMERCIAL_CATEGORY_RENAME = {
    "과학/기술": "science_tech_count",
    "부동산": "real_estate_count",
    "서비스업": "service_count",
    "소매업": "retail_count",
    "교육/학문": "education_count",
}

CENTER_COLUMN_RENAME = {
    "gbn": "district_type",
    "klg_ids_ct_nm": "center_name",
    "lon": "lon",
    "lat": "lat",
    "addr": "address",
    "rd_addr": "road_address",
    "useapr_day": "approval_date",
    "use_area": "use_zone",
    "arch_area": "total_floor_area",
    "tot_cpn": "total_unit_count",
}

GRID_COLUMN_RENAME = {
    "gid": "grid_id",
    "공시지가": "official_land_price",
    "시세": "market_price",
    "CARD_SALES": "card_sales",
    "EST_SALES": "estimated_sales",
    "버스_stop_count": "bus_stop_count",
    "지하철_stop_count": "subway_stop_count",
    "12~17": "midday_floating_pop",
    "weekdays": "weekday_floating_pop",
    "weekends": "weekend_floating_pop",
    "total_pop": "total_floating_pop",
}
