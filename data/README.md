# Data

Raw data files are **not included** in this repository (LH COMPAS 공모전
제공 데이터 + 직접 수집한 외부 데이터로, 공모전 규정상 재배포하지 않습니다).
To reproduce the pipeline, place the following files under `data/raw/`:

| File | Description | Source |
|---|---|---|
| `1.성남시_거주인구.csv` | Resident population by grid cell, by year/age/gender | LH COMPAS |
| `2.성남시_상권정보.csv` | Commercial district point data (소상공인시장진흥공단) | LH COMPAS |
| `3.성남시_상가개폐업.csv` | Storefront open/close registry | LH COMPAS |
| `4.성남시_표제부.csv` | Building registry (elevator / parking capacity) | LH COMPAS |
| `5.성남시_층별개요.csv` | Building floor-level overview | LH COMPAS |
| `6.성남시_연속지적도.geojson` | Cadastral map (parcel polygons) | 국토교통부 |
| `7.성남시_지식산업센터.csv` | Knowledge-industry-center registry (unit counts, occupancy by quarter) | LH COMPAS |
| `8.성남시_개별공시지가.csv` | Official individual land price | 국토교통부 |
| `9.성남시_버스정류장.csv` | Bus stop locations | 경기도 교통정보센터 |
| `10.성남시_지하철역.csv` | Subway station locations | 경기도 교통정보센터 |
| `15.성남시_격자(100M).geojson` | Pre-built 100m analysis grid, Seongnam | direct QGIS output |
| `20.하남시_격자(100M).geojson` | Pre-built 100m analysis grid, Hanam (incl. Gyosan) | direct QGIS output |
| `성남_지산세_상가1.csv` | Storefronts spatially joined to each center's building polygon (QGIS) | direct QGIS output |
| `SN_지식산업센터_매매가.csv` | Transaction price listings | `src/crawling/crawl_transaction_price.py` |
| `하남교산_개별공시지가.csv` | Official land price, Hanam-Gyosan candidate area | `src/crawling/crawl_land_price.py` |

`data/interim/` and `data/processed/` are left empty (gitignored) and are
populated as you run the pipeline in `notebooks/00_full_pipeline.ipynb`.
