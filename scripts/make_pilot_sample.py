# -*- coding: utf-8 -*-
"""
scripts/make_pilot_sample.py
============================
Lấy mẫu NGẪU NHIÊN PHÂN TẦNG ~5.000 văn bản từ kho crawl (chỉ đọc) phục vụ chạy
thử cục bộ (CLAUDE.md §IV: Tầng 1, ~5.000 mẫu). seed=42 (tất định).

Phân tầng theo (content_source × doc_type_proxy). Vì doc_type ~100% None ở tầng
crawl, doc_type_proxy được SUY ra từ số hiệu bằng hàm CÓ SẴN của dự án (KHÔNG viết
regex mới): configs.global_id.classify_doc_number + doc_type_and_rank.
  - NO_NUMBER   → tầng riêng "NO_NUMBER"
  - CONTAMINATED→ tầng riêng "CONTAMINATED"
  - VALID       → doc_type_and_rank(...)[0] (bucket cấp ban hành, hoặc "Khác")
doc_type_proxy CHỈ dùng để phân tầng lấy mẫu, KHÔNG dùng cho bất kỳ kết luận đánh giá nào.

KHÔNG làm: tầng "văn bản khó" (sửa đổi/Phụ lục/không Chương) — các dấu hiệu này cần
AST parser (phát hiện Chương) chưa chạy ở tầng crawl; nhận diện bây giờ là ĐOÁN, nên
bỏ qua và ghi chú thay vì bịa (đúng quy tắc "không nhận diện được thì ghi chú").
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.global_id import classify_doc_number, doc_type_and_rank  # noqa: E402 (nhẹ)


def doc_type_proxy(doc_number: str, title: str = "") -> str:
    """Nhãn tầng SUY RA (proxy) — chỉ để lấy mẫu, không phải doc_type thật."""
    status, _ = classify_doc_number(doc_number or "")
    if status == "NO_NUMBER":
        return "NO_NUMBER"
    if status == "CONTAMINATED":
        return "CONTAMINATED"
    return doc_type_and_rank(doc_number or "", title or "")[0]


def allocate(sizes: dict, target: int, min_per: int = 5) -> dict:
    """Phân bổ số mẫu cho từng tầng: TỈ LỆ với kích thước tầng, mỗi tầng không rỗng
    tối thiểu min(min_per, size). Clamp ≤ size. Tổng XẤP XỈ target (mẫu thử, không
    cần đúng tuyệt đối). Tất định."""
    sizes = {k: v for k, v in sizes.items() if v > 0}
    total = sum(sizes.values())
    if total == 0:
        return {}
    if target >= total:
        return dict(sizes)  # lấy tất
    out = {}
    for k, v in sizes.items():
        prop = round(v / total * target)
        out[k] = min(v, max(min(min_per, v), prop))
    return out


def build_sample(items: list[dict], target: int = 5000, min_per: int = 5,
                 seed: int = 42) -> list[dict]:
    """items: list dict có khóa 'stratum' (+ các trường để xuất). Trả danh sách đã chọn.
    Mỗi tầng chọn random.sample theo allocate; tất định với seed."""
    rng = random.Random(seed)
    by_str: dict[str, list[dict]] = {}
    for it in items:
        by_str.setdefault(it["stratum"], []).append(it)
    sizes = {k: len(v) for k, v in by_str.items()}
    alloc = allocate(sizes, target, min_per)
    chosen = []
    for k in sorted(by_str):                       # sort → thứ tự tất định
        kk = min(alloc.get(k, 0), len(by_str[k]))
        chosen.extend(rng.sample(by_str[k], kk) if kk else [])
    return chosen


# ---------------------------------------------------------------------------
# Lớp nạp mỏng (I/O) — chỉ đọc shard, giữ item nhẹ (không giữ nội dung)
# ---------------------------------------------------------------------------
def load_items(shards_dir: str, limit=None) -> list[dict]:
    from quality.crawl_audit import safe_read_jsonl
    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    if limit:
        shards = shards[:limit]
    items = []
    for sh in shards:
        for raw in safe_read_jsonl(sh):
            dn = raw.get("doc_number") or ""
            cs = raw.get("content_source") or "(null)"
            proxy = doc_type_proxy(dn, raw.get("title") or "")
            items.append({
                "doc_id": raw.get("doc_id") or raw.get("item_id"),
                "doc_number": dn,
                "content_source": cs,
                "doc_type_proxy": proxy,
                "stratum": f"{cs}|{proxy}",
            })
    return items


def run(shards_dir: str, out_path: str, target=5000, min_per=5, seed=42, limit=None):
    items = load_items(shards_dir, limit)
    sizes = Counter(it["stratum"] for it in items)
    chosen = build_sample(items, target, min_per, seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in chosen:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    picked = Counter(it["stratum"] for it in chosen)
    print(f"Tổng bản ghi: {len(items)} | số tầng: {len(sizes)} | đã lấy: {len(chosen)} "
          f"(đích {target}, seed {seed}, min/tầng {min_per})")
    print(f"{'Tầng (content_source|doc_type_proxy)':<48}{'cỡ tầng':>9}{'lấy':>7}{'%tầng':>8}")
    print("-" * 72)
    for k, sz in sizes.most_common():
        pk = picked.get(k, 0)
        print(f"{k:<48}{sz:>9}{pk:>7}{(100*pk/sz):>7.1f}%")
    print(f"\nĐầu ra: {out_path}  (chỉ doc_id/doc_number/doc_type_proxy/content_source/stratum)")
    print("Ghi chú: BỎ tầng 'văn bản khó' (sửa đổi/Phụ lục/không Chương) — cần AST parser, "
          "chưa có ở tầng crawl nên không nhận diện được, không đoán.")
    return chosen


if __name__ == "__main__":
    from configs.paths import RAW_SHARDS_DIR, REPORTS_DIR
    ap = argparse.ArgumentParser(description="Lấy mẫu phân tầng ~5000 (chỉ đọc).")
    ap.add_argument("--shards-dir", default=str(RAW_SHARDS_DIR))
    ap.add_argument("--out", default=str(REPORTS_DIR / "pilot_sample_5000_ids.jsonl"))
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--min-per", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="số shard tối đa (ước lượng nhanh)")
    a = ap.parse_args()
    run(a.shards_dir, a.out, a.target, a.min_per, a.seed, a.limit)
