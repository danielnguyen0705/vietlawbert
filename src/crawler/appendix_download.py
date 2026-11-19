# -*- coding: utf-8 -*-
"""
crawler/appendix_download.py
============================
TẢI ZIP PHỤ LỤC từ danh sách file mà server-action vbpl.vn trả về (đã có presignedUrl tươi).
Tách riêng khỏi luồng Playwright để KIỂM THỬ được: `fetch` được tiêm vào (test dùng fetch giả),
thực tế mặc định tải qua HTTP. presignedUrl là link lưu trữ trực tiếp (không qua WAF), nên chỉ
khâu LẤY danh sách file cần trình duyệt; tải thì HTTP thường là đủ.

Đặt `item['appendix_zip_path']` → ingest (hook có guard) tự bóc + khử trùng + nhúng chunk con.
"""
from __future__ import annotations
import os
import re
import urllib.request
from typing import Any, Callable, Dict, List, Optional


def appendix_zip_candidates(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """File phụ lục dạng .zip có presignedUrl (giữ thứ tự fileOrder nếu có)."""
    cands = [
        f for f in (files or [])
        if str(f.get("fileName") or "").lower().endswith(".zip") and f.get("presignedUrl")
    ]
    return sorted(cands, key=lambda f: f.get("fileOrder", 0))


def _http_fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def save_appendix_zip(
    files: List[Dict[str, Any]],
    item: Dict[str, Any],
    dest_dir: str,
    fetch: Optional[Callable[[str], bytes]] = None,
) -> Optional[str]:
    """Tải zip phụ lục đầu tiên (nếu có) về dest_dir và set item['appendix_zip_path'].
    Trả đường dẫn đã lưu, hoặc None nếu không có phụ lục / tải rỗng. `fetch` tiêm được để test."""
    cands = appendix_zip_candidates(files)
    if not cands:
        return None
    f = cands[0]
    fetch = fetch or _http_fetch
    data = fetch(f["presignedUrl"])
    if not data or len(data) < 100:          # zip rỗng/hỏng → bỏ qua, không set path
        return None
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w.-]", "_", str(f.get("fileName") or "appendix.zip"))
    path = os.path.join(dest_dir, f"{item.get('item_id', 'doc')}__{safe_name}")
    with open(path, "wb") as out:
        out.write(data)
    item["appendix_zip_path"] = path
    item["appendix_file_name"] = f.get("fileName")
    return path
