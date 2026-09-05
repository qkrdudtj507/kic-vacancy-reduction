"""
Crawl knowledge-industry-center (지식산업센터) transaction price listings
from kic114.kr for a given region code.

Original notebook: 0129_지식산업센터_데이터_크롤링.ipynb
"""

from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.kic114.kr/transactionList.do"


def _fetch_page(page_index: int, area_cd: str, sido_cd: str) -> tuple[list[str], list[list[str]]]:
    """Fetch and parse a single result page.

    Returns (header_columns, row_values) for that page.
    """
    params = {
        "BD_ID": "",
        "PAGE_INDEX": page_index,
        "PAGE_CURRENT": "",
        "ORDER_BY_CD_1": "",
        "ORDER_BY_SC_1": "",
        "BD_AREA_CD": area_cd,
        "BD_SIDO_CD": sido_cd,
        "PRICE_PER_PY_CD": "",
        "SEARCH_NM": "",
    }
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    header_selector = "#container > div > div.contWrap > div.tblArea.type01 > table > thead"
    body_row_selector = "#container > div > div.contWrap > div.tblArea.type01 > table > tbody > tr"

    header_block = soup.select(header_selector)
    columns = [h.strip() for h in header_block[0].text.splitlines() if h.strip()] if header_block else []

    rows: list[list[str]] = []
    for row in soup.select(body_row_selector):
        cell_values = [cell.text.strip() for cell in row.select("td")]
        if cell_values:
            rows.append(cell_values)

    return columns, rows


def crawl_transaction_prices(
    num_pages: int = 3,
    area_cd: str = "H100",
    sido_cd: str = "H102",
) -> pd.DataFrame:
    """Crawl `num_pages` of transaction-price listings for the given region.

    Parameters
    ----------
    num_pages : number of result pages to crawl (site is paginated).
    area_cd / sido_cd : KIC114 region codes (defaults target Seongnam-si).

    Returns
    -------
    DataFrame of raw transaction listings (one row per listing).
    """
    columns: list[str] = []
    all_rows: list[list[str]] = []

    for page_index in range(1, num_pages + 1):
        page_columns, page_rows = _fetch_page(page_index, area_cd, sido_cd)
        if page_columns:
            columns = page_columns
        all_rows.extend(page_rows)

    return pd.DataFrame(all_rows, columns=columns)


if __name__ == "__main__":
    from src.config import RAW_DATA_DIR

    df = crawl_transaction_prices(num_pages=3)
    out_path = RAW_DATA_DIR / "SN_지식산업센터_매매가.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows -> {out_path}")
