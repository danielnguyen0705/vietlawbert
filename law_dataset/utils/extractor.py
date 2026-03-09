# law_dataset/utils/extractor.py
from __future__ import annotations

import re
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def extract_links_from_result_html(
    html: str,
    base_url: str,
    type_value: str,
    ministry: str,
) -> List[Dict]:
    """
    VBPL kết quả trả về có dạng:
      <ul class="listLaw">
        <li> ... <p class="title"><a href="/.../vbpq-toanvan.aspx?ItemID=...">Số hiệu</a> ...

    Return:
      [{type, doc_id, url, ministry}, ...]  (dedup theo url)
    """
    soup = BeautifulSoup(html, "lxml")

    items: List[Dict] = []
    for a in soup.select("ul.listLaw p.title a[href*='vbpq-toanvan.aspx']"):
        href = (a.get("href") or "").strip()
        doc_id = _norm_text(a.get_text(" ", strip=True))
        if not href:
            continue

        items.append(
            {
                "type": type_value,
                "doc_id": doc_id,
                "url": urljoin(base_url, href),
                "ministry": ministry,
            }
        )

    # Dedup theo url
    seen = set()
    dedup: List[Dict] = []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        dedup.append(it)

    return dedup


def fetch_html(url: str, timeout: int = 30) -> str:
    """
    Tải HTML từ URL.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def extract_item_id(url: str) -> Optional[str]:
    """
    Lấy ItemID từ URL, ví dụ:
    ...vbpq-toanvan.aspx?ItemID=51216&Keyword= -> 51216
    """
    m = re.search(r"ItemID=(\d+)", url)
    return m.group(1) if m else None


def _extract_effective_date(text: str) -> str:
    """
    Tìm 'Ngày có hiệu lực' trong text.
    Trả về chuỗi dd/mm/yyyy nếu có, ngược lại trả về "".
    """
    patterns = [
        r"Ngày có hiệu lực\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4})",
        r"Hiệu lực\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4})",
        r"Có hiệu lực từ ngày\s*(\d{1,2}/\d{1,2}/\d{4})",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    return ""


def _extract_title(soup: BeautifulSoup) -> str:
    """
    Thử lấy title của văn bản từ các selector phổ biến.
    Nếu không có thì fallback sang <title>.
    """
    selectors = [
        "div#toanvancontent .title",
        "div.toanvan .title",
        "div.title",
        "h1",
        "h2",
    ]

    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            txt = _norm_text(tag.get_text(" ", strip=True))
            if txt:
                return txt

    if soup.title:
        txt = _norm_text(soup.title.get_text(" ", strip=True))
        if txt:
            return txt

    return ""


def _extract_raw_text(soup: BeautifulSoup) -> str:
    """
    Lấy raw_text từ trang toàn văn.
    Ưu tiên các vùng nội dung có khả năng là thân văn bản,
    nếu không có thì fallback sang toàn bộ body.
    """
    candidate_selectors = [
        "div#toanvancontent",
        "div.toanvan",
        "div#contentdetail",
        "div.content1",
        "div.content",
        "td.ToanVan",
        "body",
    ]

    for sel in candidate_selectors:
        tag = soup.select_one(sel)
        if not tag:
            continue

        text = tag.get_text("\n", strip=True)
        text = _norm_preserve_newlines(text)
        if text and len(text) > 200:
            return text

    body_text = soup.get_text("\n", strip=True)
    return _norm_preserve_newlines(body_text)


def _norm_preserve_newlines(text: str) -> str:
    """
    Chuẩn hóa text nhưng vẫn giữ xuống dòng.
    """
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = _norm_text(line)
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document_from_html(
    html: str,
    url: str,
    fallback_doc_id: str = "",
) -> Dict:
    """
    Parse trang toàn văn văn bản.

    Return:
    {
        "doc_id": ...,
        "title": ...,
        "effective_date": ...,
        "raw_text": ...,
    }
    """
    soup = BeautifulSoup(html, "lxml")

    title = _extract_title(soup)
    raw_text = _extract_raw_text(soup)
    effective_date = _extract_effective_date(raw_text)

    doc_id = fallback_doc_id.strip() if fallback_doc_id else ""
    if not doc_id:
        doc_id = title

    return {
        "doc_id": doc_id,
        "title": title,
        "effective_date": effective_date,
        "raw_text": raw_text,
        "url": url,
    }