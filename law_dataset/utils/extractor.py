# law_dataset/utils/extractor.py
from __future__ import annotations

from typing import List, Dict
from urllib.parse import urljoin
from bs4 import BeautifulSoup


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def extract_links_from_list_html(
    html: str,
    base_url: str,
    type_value: str = "luat_giao_thong",
    ministry: str = "GTVT",
    include_most_viewed: bool = False,
) -> List[Dict]:
    """
    Parse HTML dạng danh sách văn bản (đúng theo HTML bạn đưa):
      - Khu chính: ul.listLaw p.title a[href*='vbpq-toanvan.aspx']
      - (Optional) Sidebar: a[href*='vbpq-toanvan.aspx'] (văn bản xem nhiều)
    Return: list[{type, doc_id, url, ministry}] đã dedup theo url.
    """
    soup = BeautifulSoup(html, "lxml")  # nhanh + ổn

    items: List[Dict] = []

    # 1) Khu danh sách văn bản (chính)
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

    # 2) (Optional) Sidebar văn bản xem nhiều
    if include_most_viewed:
        for a in soup.select("a[href*='vbpq-toanvan.aspx']"):
            href = (a.get("href") or "").strip()
            doc_id = _norm_text(a.get_text(" ", strip=True))
            if not href or not doc_id:
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