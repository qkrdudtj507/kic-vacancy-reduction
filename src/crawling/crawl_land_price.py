"""
Crawl official individual land price (개별공시지가) listings from the
Korea Ministry of Land, Infrastructure and Transport's realtyprice.kr
site for a given city / district set.

Requires a local Chrome + chromedriver installation.
Original notebook: 공공공_하남교산_공시지가.ipynb

Note: the target site paginates per 동(neighbourhood) and requires
selecting dropdowns (시/도 -> 시/군/구 -> 동) before results load, so
this crawler drives a real browser rather than issuing plain HTTP
requests.
"""

from __future__ import annotations

import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

SEARCH_URL = "https://www.realtyprice.kr/notice/gsstandard/search.htm"
WAIT_TIMEOUT_S = 30

# 하남시 동별 select-box value 목록 (교산지구 인접 행정동 기준)
DEFAULT_DONG_CODES = [
    "11700", "10800", "10600", "10300",
    "10100", "11800", "11900", "12100", "12000",
]


def _build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def _select_region(driver: webdriver.Chrome, sido: str, sigungu: str) -> None:
    WebDriverWait(driver, WAIT_TIMEOUT_S).until(
        EC.presence_of_element_located((By.ID, "sido_list"))
    )
    Select(driver.find_element(By.ID, "sido_list")).select_by_visible_text(sido)

    WebDriverWait(driver, WAIT_TIMEOUT_S).until(
        EC.presence_of_element_located((By.ID, "sgg_list"))
    )
    Select(driver.find_element(By.ID, "sgg_list")).select_by_visible_text(sigungu)


def _count_pages(driver: webdriver.Chrome) -> int:
    page_buttons = driver.find_elements(
        By.XPATH, "//div[@id='pagination']//a[contains(@href, 'javascript:goPage')]"
    )
    return len(page_buttons) + 1 if page_buttons else 1


def _parse_current_page(driver: webdriver.Chrome, dong_value: str) -> list[dict]:
    rows = driver.find_elements(By.XPATH, "//table[@id='dataList']//tr")
    records = []
    for row in rows:
        cells = [c.text.strip() for c in row.find_elements(By.TAG_NAME, "td")]
        if cells:
            records.append({"dong_code": dong_value, "cells": cells})
    return records


def crawl_official_land_price(
    sido: str = "경기도",
    sigungu: str = "하남시",
    dong_codes: list[str] | None = None,
    headless: bool = True,
) -> pd.DataFrame:
    """Crawl official individual land price rows for every 동 in `dong_codes`.

    Returns a long-format DataFrame with one row per listing; `cells`
    holds the raw table-cell text (parse further per your target
    columns depending on the site layout at crawl time).
    """
    dong_codes = dong_codes or DEFAULT_DONG_CODES
    driver = _build_driver(headless=headless)
    all_records: list[dict] = []

    try:
        driver.get(SEARCH_URL)
        _select_region(driver, sido, sigungu)

        for dong_value in dong_codes:
            WebDriverWait(driver, WAIT_TIMEOUT_S).until(
                EC.presence_of_element_located((By.ID, "eub_list"))
            )
            Select(driver.find_element(By.ID, "eub_list")).select_by_value(dong_value)
            driver.find_element(By.XPATH, "//input[@alt='검색']").click()

            WebDriverWait(driver, WAIT_TIMEOUT_S).until(
                EC.presence_of_element_located((By.ID, "dataList"))
            )
            time.sleep(2)  # allow dynamic content to settle

            total_pages = _count_pages(driver)
            print(f"[{dong_value}] {total_pages} page(s) found")

            for page in range(1, total_pages + 1):
                if page > 1:
                    driver.execute_script(f"goPage({page})")
                    time.sleep(1.5)
                all_records.extend(_parse_current_page(driver, dong_value))
    finally:
        driver.quit()

    return pd.DataFrame(all_records)


if __name__ == "__main__":
    from src.config import RAW_DATA_DIR

    df = crawl_official_land_price()
    out_path = RAW_DATA_DIR / "하남교산_개별공시지가.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows -> {out_path}")
