from __future__ import annotations

import json
import re
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "law_dataset" / "data"
OUT_PATH = DATA_DIR / "raw_links.json"
DEBUG_DIR = DATA_DIR / "debug"

BASE_URL = "https://vbpl.vn/"
SEARCH_KEYWORD = "luật giao thông đường bộ"

TYPE_VALUE = "traffic_law"
SOURCE_SITE = "vbpl.vn"

MAX_RESULT_PAGES = 3
MAX_ITEMS_PER_PAGE = 10


def clean_text(text: str) -> str:
    text = str(text or "").replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return text


def make_safe_id(text: str, index: int) -> str:
    """
    Tạm tạo ID logic từ title vì giao diện mới không expose detail URL trong DOM.
    """
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s/-]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    text = text[:120].strip("-")
    return f"vbpl_card_{index:04d}_{text}" if text else f"vbpl_card_{index:04d}"


def fill_search_keyword(page, keyword: str) -> None:
    inputs = page.locator("input")
    count = inputs.count()

    for i in range(count):
        item = inputs.nth(i)
        try:
            if not item.is_visible(timeout=1000):
                continue

            item.click(timeout=3000)
            item.fill(keyword, timeout=3000)
            return
        except Exception:
            continue

    raise RuntimeError("Cannot find visible search input")


def choose_title_search_if_possible(page) -> None:
    candidates = [
        "text=Tiêu đề văn bản",
        "label:has-text('Tiêu đề văn bản')",
    ]

    for selector in candidates:
        try:
            page.locator(selector).first.click(timeout=2000)
            page.wait_for_timeout(500)
            return
        except Exception:
            continue


def click_search(page) -> None:
    candidates = [
        "button:has-text('Tìm kiếm')",
        "text=Tìm kiếm",
    ]

    for selector in candidates:
        try:
            page.locator(selector).first.click(timeout=3000)
            return
        except Exception:
            continue

    raise RuntimeError("Cannot find search button")


def wait_for_results(page) -> None:
    try:
        page.wait_for_selector("[class*='DocumentCard_documentTitle']", timeout=20000)
        return
    except PlaywrightTimeoutError:
        pass

    raise RuntimeError("Cannot find document result cards after search.")


def get_card_locators(page):
    return page.locator("li.ant-list-item")


def extract_label_value(card, label: str) -> str:
    """
    Card text có dạng:
    Trạng thái: Còn hiệu lực
    Ngày ban hành: 12/02/2026
    Ngày hiệu lực: 28/02/2026
    """
    try:
        text = clean_text(card.inner_text(timeout=2000))
    except Exception:
        return ""

    pattern = rf"{re.escape(label)}\s*:\s*([^:]+?)(?=\s+Trạng thái:|\s+Ngày ban hành:|\s+Ngày hiệu lực:|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        return clean_text(match.group(1))

    return ""


def extract_card_title(card) -> str:
    try:
        title = card.locator("[class*='DocumentCard_documentTitle']").first.inner_text(timeout=2000)
        return normalize_title(title)
    except Exception:
        pass

    try:
        text = clean_text(card.inner_text(timeout=2000))
    except Exception:
        return ""

    # Cắt trước các nút và metadata
    cut_markers = ["PDF", "Lược đồ", "Tải về", "Trạng thái:", "Ngày ban hành:", "Ngày hiệu lực:"]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker)[0]

    return normalize_title(text)


def collect_cards_from_current_page(page, global_start_index: int) -> list[dict]:
    cards = get_card_locators(page)
    count = min(cards.count(), MAX_ITEMS_PER_PAGE)

    print(f"Detected {cards.count()} cards. Will collect {count} cards.")

    items = []

    for i in range(count):
        card = cards.nth(i)
        title = extract_card_title(card)

        if not title:
            continue

        status = extract_label_value(card, "Trạng thái")
        issued_date = extract_label_value(card, "Ngày ban hành")
        effective_date = extract_label_value(card, "Ngày hiệu lực")

        record_index = global_start_index + i + 1

        items.append(
            {
                "type": TYPE_VALUE,
                "doc_id": title,
                "document_id": make_safe_id(title, record_index),
                "url": "",
                "source_site": SOURCE_SITE,
                "url_schema": "vbpl_search_card",
                "search_keyword": SEARCH_KEYWORD,
                "status": status,
                "issued_date": issued_date,
                "effective_date": effective_date,
                "ministry": "",
            }
        )

        print(f"[CARD {record_index}] {title[:100]}")

    return items


def go_to_next_page(page) -> bool:
    candidates = [
        "text=Sau",
        "button:has-text('Sau')",
        "a:has-text('Sau')",
        "li.ant-pagination-next",
        ".ant-pagination-next",
    ]

    before_titles = []
    try:
        before_titles = [
            clean_text(t)
            for t in page.locator("[class*='DocumentCard_documentTitle']").all_inner_texts()
        ]
    except Exception:
        pass

    before_first = before_titles[0] if before_titles else ""

    for selector in candidates:
        try:
            locator = page.locator(selector).last

            if locator.count() == 0:
                continue

            if not locator.is_visible(timeout=1000):
                continue

            locator.click(timeout=5000)
            page.wait_for_timeout(3000)
            wait_for_results(page)

            after_titles = [
                clean_text(t)
                for t in page.locator("[class*='DocumentCard_documentTitle']").all_inner_texts()
            ]
            after_first = after_titles[0] if after_titles else ""

            if after_first and after_first != before_first:
                return True

            return True

        except Exception:
            continue

    return False


def dump_debug(page, name: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    html_path = DEBUG_DIR / f"{name}.html"
    png_path = DEBUG_DIR / f"{name}.png"

    try:
        html_path.write_text(page.content(), encoding="utf-8")
        print(f"[DEBUG] Saved HTML -> {html_path}")
    except Exception as exc:
        print(f"[DEBUG] Cannot save HTML: {exc}")

    try:
        page.screenshot(path=str(png_path), full_page=True)
        print(f"[DEBUG] Saved screenshot -> {png_path}")
    except Exception as exc:
        print(f"[DEBUG] Cannot save screenshot: {exc}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []
    seen_titles: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            viewport={"width": 1600, "height": 1200},
            locale="vi-VN",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )

        page = context.new_page()

        print(f"Opening {BASE_URL}")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        print(f"Searching keyword: {SEARCH_KEYWORD}")
        fill_search_keyword(page, SEARCH_KEYWORD)
        choose_title_search_if_possible(page)
        click_search(page)

        try:
            page.wait_for_load_state("networkidle", timeout=30000)
        except PlaywrightTimeoutError:
            pass

        page.wait_for_timeout(3000)
        wait_for_results(page)

        for result_page in range(1, MAX_RESULT_PAGES + 1):
            print(f"\n=== Result page {result_page} ===")

            page_items = collect_cards_from_current_page(page, len(all_items))

            new_items = []
            for item in page_items:
                key = item["doc_id"]
                if key in seen_titles:
                    continue

                seen_titles.add(key)
                new_items.append(item)

            all_items.extend(new_items)

            print(f"Collected on page {result_page}: {len(new_items)}")
            print(f"Total collected: {len(all_items)}")

            if result_page >= MAX_RESULT_PAGES:
                break

            if not go_to_next_page(page):
                print("No next page found. Stop pagination.")
                break

        if not all_items:
            dump_debug(page, "phase1_no_cards")

        context.close()
        browser.close()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(all_items)} records -> {OUT_PATH}")

    if not all_items:
        raise RuntimeError("No search cards collected.")


if __name__ == "__main__":
    main()