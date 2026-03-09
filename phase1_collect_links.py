# phase1_collect_links.py
from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests

from law_dataset.utils.extractor import extract_links_from_result_html


# =========================
# ====== CONFIG ===========
# =========================

# ✅ Dán đúng URL bạn lấy từ Network (mình để y nguyên theo bạn gửi)
SEARCH_URL = (
    "https://vbpl.vn/VBQPPL_UserControls/Publishing/TimKiem/pKetQuaTimKiem.aspx"
    "?dvid=315&IsVietNamese=True&&type=1&stemp=1&TimTrong1=VBPQFulltext&TimTrong1=Title"
    "&order=VBPQNgayBanHanh&TypeOfOrder=False"
    "&LoaiVanBan=15,16,17,19,2,18,3,20,21,22,23,24"
    "&CoQuanBanHanh=274"
    "&TrangThaiHieuLuc=7,6,5,4,3,2,1"
    "&Page"
)

BASE_URL = "https://vbpl.vn"

TYPE_VALUE = "luat_giao_thong"
MINISTRY_TAG = "GTVT"

OUT_PATH = "law_dataset/data/raw_links.json"

MAX_PAGES = 500
SLEEP_EACH_PAGE = 0.35
STOP_AFTER_EMPTY_PAGES = 3  # 3 trang liên tiếp không có link mới thì stop

# VBPL có thể dùng các key phân trang khác nhau -> thử lần lượt
PAGE_KEYS_CANDIDATES = ["Page", "PageIndex", "page", "CurrentPage", "p"]

# =========================


def parse_search_url(url: str) -> Tuple[str, Dict[str, str], List[Tuple[str, str]]]:
    """
    Parse SEARCH_URL thành:
      - base_endpoint (scheme+host+path)
      - params dict (giữ param cuối cùng nếu trùng key)
      - params_list (giữ thứ tự & giữ cả key trùng như TimTrong1=... nhiều lần)
    """
    u = urlparse(url)

    # giữ nguyên tất cả query pairs kể cả key trùng
    pairs = parse_qsl(u.query, keep_blank_values=True)

    # VBPL đôi khi có '&Page' (không '=') => parse_qsl sẽ ra ('Page','')
    # mình giữ y như vậy để lát override thành Page=1,2,3...

    endpoint = urlunparse((u.scheme, u.netloc, u.path, "", "", ""))

    # dict chỉ để tiện xem, không dùng để build url (vì trùng key)
    params_dict = {}
    for k, v in pairs:
        params_dict[k] = v

    return endpoint, params_dict, pairs


def build_url(endpoint: str, pairs: List[Tuple[str, str]]) -> str:
    query = urlencode(pairs, doseq=True)
    return endpoint + "?" + query


def override_page_param(
    pairs: List[Tuple[str, str]],
    page_key: str,
    page_value: int,
) -> List[Tuple[str, str]]:
    """
    Trả về list pairs mới:
      - xóa hết các page keys candidate cũ
      - set page_key=page_value
    """
    filtered = [(k, v) for (k, v) in pairs if k not in PAGE_KEYS_CANDIDATES]
    filtered.append((page_key, str(page_value)))
    return filtered


def fetch_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def detect_page_key(
    session: requests.Session,
    endpoint: str,
    base_pairs: List[Tuple[str, str]],
) -> str:
    """
    Thử từng page_key xem cái nào cho ra listLaw.
    """
    for key in PAGE_KEYS_CANDIDATES:
        try:
            pairs = override_page_param(base_pairs, key, 1)
            url = build_url(endpoint, pairs)
            html = fetch_html(session, url)
            items = extract_links_from_result_html(
                html=html,
                base_url=BASE_URL,
                type_value=TYPE_VALUE,
                ministry=MINISTRY_TAG,
            )
            if items:
                print(f"✅ Detected page param: {key} (page1 items={len(items)})")
                return key
        except Exception:
            continue

    # Nếu không detect được thì vẫn thử dùng Page (hay gặp nhất)
    print("⚠️ Could not auto-detect page key, fallback to 'Page'")
    return "Page"


def main():
    endpoint, _, base_pairs = parse_search_url(SEARCH_URL)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "vi,en;q=0.9",
        }
    )

    page_key = detect_page_key(session, endpoint, base_pairs)

    all_items: List[dict] = []
    seen_urls = set()
    empty_streak = 0

    for page in range(1, MAX_PAGES + 1):
        pairs = override_page_param(base_pairs, page_key, page)
        url = build_url(endpoint, pairs)

        html = fetch_html(session, url)
        items = extract_links_from_result_html(
            html=html,
            base_url=BASE_URL,
            type_value=TYPE_VALUE,
            ministry=MINISTRY_TAG,
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