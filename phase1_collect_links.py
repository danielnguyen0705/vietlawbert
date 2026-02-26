# phase1_collect_links.py
from __future__ import annotations

import json
import time
from typing import List, Dict
from urllib.parse import urlencode

import requests

from law_dataset.utils.extractor import extract_links_from_list_html

# =========================
# ====== CONFIG ===========
# =========================

BASE_URL = "https://vbpl.vn"
OUT_PATH = "law_dataset/data/raw_links.json"

TYPE_VALUE = "luat_giao_thong"
MINISTRY_TAG = "GTVT"

# Endpoint danh sách kết quả tìm kiếm (có phân trang)
RESULT_ENDPOINT = f"{BASE_URL}/VBQPPL_UserControls/Publishing/TimKiem/pKetQuaTimKiem.aspx"

# Filter cơ bản (bạn mở dần sau)
BASE_PARAMS = {
    "dvid": "315",
    "IsVietNamese": "True",
    # Nếu muốn chỉ “còn hiệu lực” thì mở comment:
    # "TrangThaiHieuLuc": "2",
    # Nếu muốn chỉ Luật/Thông tư thì mở comment:
    # "LoaiVanBan": "17",  # Luật
    # "LoaiVanBan": "22",  # Thông tư
}

# Thử các key phân trang hay gặp
PAGE_KEYS_CANDIDATES = ["PageIndex", "page", "Page", "CurrentPage", "p"]

MAX_PAGES = 500
SLEEP_EACH_PAGE = 0.4
STOP_AFTER_EMPTY_PAGES = 3  # dừng khi 3 trang liên tiếp không có link mới

# =========================


def build_url(params: dict) -> str:
    return RESULT_ENDPOINT + "?" + urlencode(params, doseq=True)


def fetch_html(session: requests.Session, params: dict) -> str:
    url = build_url(params)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def detect_pagination_key(session: requests.Session) -> str:
    """
    Thử từng key phân trang, key nào trả ra listLaw (có link) thì dùng.
    """
    for key in PAGE_KEYS_CANDIDATES:
        try:
            params = dict(BASE_PARAMS)
            params[key] = "1"
            html = fetch_html(session, params)
            items = extract_links_from_list_html(
                html=html,
                base_url=BASE_URL,
                type_value=TYPE_VALUE,
                ministry=MINISTRY_TAG,
                include_most_viewed=False,
            )
            if items:
                print(f"✅ Detected pagination key: {key} (page1 items={len(items)})")
                return key
        except Exception:
            continue

    raise RuntimeError(
        "❌ Không detect được param phân trang. "
        "Hãy mở F12 → Network, bấm chuyển trang, tìm request pKetQuaTimKiem.aspx "
        "rồi xem nó dùng param gì (vd PageIndex=2...)."
    )


def main():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "vi,en;q=0.9",
        }
    )

    page_key = detect_pagination_key(session)

    all_items: List[Dict] = []
    seen_urls = set()
    empty_streak = 0

    for page in range(1, MAX_PAGES + 1):
        params = dict(BASE_PARAMS)
        params[page_key] = str(page)

        html = fetch_html(session, params)

        items = extract_links_from_list_html(
            html=html,
            base_url=BASE_URL,
            type_value=TYPE_VALUE,
            ministry=MINISTRY_TAG,
            include_most_viewed=False,
        )

        new_count = 0
        for it in items:
            if it["url"] in seen_urls:
                continue
            seen_urls.add(it["url"])
            all_items.append(it)
            new_count += 1

        print(f"[PAGE {page:03d}] found={len(items)} new={new_count} total={len(all_items)}")

        if new_count == 0:
            empty_streak += 1
        else:
            empty_streak = 0

        if empty_streak >= STOP_AFTER_EMPTY_PAGES:
            print(f"[STOP] {STOP_AFTER_EMPTY_PAGES} pages in a row have no new links.")
            break

        time.sleep(SLEEP_EACH_PAGE)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(all_items)} links -> {OUT_PATH}")


if __name__ == "__main__":
    main()