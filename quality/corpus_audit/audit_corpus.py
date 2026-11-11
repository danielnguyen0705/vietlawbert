# -*- coding: utf-8 -*-
"""
corpus_audit/audit_corpus.py
============================
Driver quét ngữ liệu, chạy 7 detector (detectors.py), tổng hợp tỉ lệ từng lỗi kèm
KHOẢNG TIN CẬY 95% (Wilson), và xuất mẫu vi phạm ra CSV để soi tay.

Hai nguồn đầu vào:
  --source csv   <path>   : file có cột chunk_id,doc_number,doc_type,content
                            (đúng định dạng mẫu Cochran đã xuất) — dùng để chạy NHANH
                            trên mẫu, ra số ước lượng + CI.
  --source jsonl <path>   : dump toàn corpus từ Mongo, mỗi dòng 1 JSON có các khóa
                            chunk_id, doc_number, doc_type, content (hoặc meta/hierarchy/body).
                            Dùng để ra SỐ CHÍNH XÁC toàn corpus trên cloud.

Ví dụ kết xuất Mongo (chạy ở service storage, không kèm ở đây):
    mongoexport --db vietlawbert --collection chunks \
        --fields chunk_id,doc_number,doc_type,content \
        --type json --out chunks_dump.jsonl
rồi:  python -m corpus_audit.audit_corpus --source jsonl chunks_dump.jsonl --out report/
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

# cho phép chạy đứng một mình (python audit_corpus.py) lẫn dạng module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import Chunk, audit_chunk, DETECTORS, is_structural_node  # noqa: E402

DEFECT_NAMES = [name for name, _ in DETECTORS]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Khoảng tin cậy Wilson cho tỉ lệ (ổn định hơn chuẩn khi p nhỏ). Trả (p, lo, hi) theo %."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return round(p * 100, 2), round(max(0.0, center - half) * 100, 2), round(min(1.0, center + half) * 100, 2)


def iter_csv(path: str):
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            yield Chunk.from_flat(
                row.get("chunk_id", ""), row.get("doc_number", ""),
                row.get("doc_type", ""), row.get("content", ""))


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "content" in d:
                yield Chunk.from_flat(
                    d.get("chunk_id", ""), d.get("doc_number", ""),
                    d.get("doc_type", ""), d.get("content", ""))
            else:  # đã tách sẵn meta/hierarchy/body
                yield Chunk(d.get("chunk_id", ""), d.get("doc_number", ""),
                            d.get("doc_type", ""), d.get("meta", ""),
                            d.get("hierarchy", ""), d.get("body", ""))


def run(source: str, path: str, out_dir: str, sample_per_defect: int = 30) -> dict:
    it = iter_csv(path) if source == "csv" else iter_jsonl(path)
    os.makedirs(out_dir, exist_ok=True)
    counts = Counter()
    samples: dict[str, list] = {name: [] for name in DEFECT_NAMES}
    empty_leaf = empty_struct = 0   # tách node cụt: lá (mất nội dung thật) vs cha cấu trúc
    n = 0
    for c in it:
        n += 1
        res = audit_chunk(c)
        for name in DEFECT_NAMES:
            if res[name]:
                counts[name] += 1
                if len(samples[name]) < sample_per_defect:
                    samples[name].append((c.chunk_id, c.doc_number, res[name + "_info"]))
        if res["empty_node"]:
            if is_structural_node(c.chunk_id):
                empty_struct += 1
            else:
                empty_leaf += 1

    # xuất mẫu vi phạm từng lỗi
    for name in DEFECT_NAMES:
        with open(os.path.join(out_dir, f"defect_{name}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chunk_id", "doc_number", "info"])
            w.writerows(samples[name])

    # summary
    summary = {"total_chunks": n, "defects": {}}
    for name in DEFECT_NAMES:
        p, lo, hi = wilson_ci(counts[name], n)
        summary["defects"][name] = {
            "count": counts[name], "rate_pct": p, "ci95_lo_pct": lo, "ci95_hi_pct": hi}
    # tách riêng node cụt lá vs cấu trúc
    pl, llo, lhi = wilson_ci(empty_leaf, n)
    summary["empty_node_breakdown"] = {
        "empty_leaf": {"count": empty_leaf, "rate_pct": pl, "ci95_lo_pct": llo, "ci95_hi_pct": lhi},
        "empty_structural_root_preamble": {"count": empty_struct},
        "note": "empty_leaf = mất nội dung thật; structural = node cha (_root/_preamble), "
                "cần kiểm có chunk con trên toàn corpus mới kết luận.",
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _print(summary: dict):
    n = summary["total_chunks"]
    print(f"\nTổng số chunk quét: {n}\n")
    print(f"{'Lỗi':<22}{'Số ca':>7}{'Tỉ lệ %':>9}{'  CI95% (Wilson)':>22}")
    print("-" * 60)
    for name, d in summary["defects"].items():
        ci = f"[{d['ci95_lo_pct']:.2f}, {d['ci95_hi_pct']:.2f}]"
        print(f"{name:<22}{d['count']:>7}{d['rate_pct']:>9.2f}{ci:>22}")
    eb = summary.get("empty_node_breakdown", {})
    if eb:
        el = eb["empty_leaf"]; es = eb["empty_structural_root_preamble"]
        print(f"\n  ↳ empty_node tách: node LÁ (mất nội dung) = {el['count']} "
              f"({el['rate_pct']:.2f}%, CI[{el['ci95_lo_pct']:.2f},{el['ci95_hi_pct']:.2f}]) "
              f"| node CHA _root/_preamble = {es['count']} (cần kiểm có con)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "jsonl"], required=True)
    ap.add_argument("path")
    ap.add_argument("--out", default="corpus_audit_report")
    ap.add_argument("--sample-per-defect", type=int, default=30)
    a = ap.parse_args()
    s = run(a.source, a.path, a.out, a.sample_per_defect)
    _print(s)
    print(f"Chi tiết: {a.out}/summary.json + defect_*.csv")
