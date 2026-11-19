# -*- coding: utf-8 -*-
"""
corpus_audit/profile_sources.py
===============================
Lập HỒ SƠ NGUỒN NGỮ LIỆU (chỉ đọc) ở TẦNG CRAWL: đếm theo nguồn trích
(content_source), trạng thái HTML/OCR/rescue, phân loại số hiệu
(VALID/NO_NUMBER/CONTAMINATED), tỷ lệ ký tự tiếng Việt (theo đúng cổng TextGate
của crawler) và phân bố độ dài văn bản. Dùng để quyết định có cần cải tiến khâu
trích văn bản hay không.

THỰC TẾ MÁY LOCAL (đo 17/11): MongoDB vietlawbert_db RỖNG; dữ liệu thật nằm ở
các shard JSONL nén /mnt/data/vietlawbert_data/raw_shards/crawl_pages_*.jsonl.gz
(+ .quarantine.jsonl). Vì vậy loader đọc TRỰC TIẾP các shard (chỉ đọc), không
qua Mongo. Ở tầng crawl: doc_type ~100% None, ocr_status chỉ có trên bản HTML
rỗng, extraction_method KHÔNG tồn tại ở cấp văn bản (chỉ ở cạnh quan hệ) — các
chiều này được báo cáo trung thực là "không có / rỗng", không bịa.

Tách 2 lớp để kiểm thử được:
  - profile_records(reduced): HÀM THUẦN, chỉ tổng hợp trên dict đã rút gọn
    (không đụng DB, không đọc file). Test bơm dữ liệu tổng hợp vào đây.
  - reduce_raw / iter_reduced_shards: lớp nạp mỏng, streaming O(1) bản ghi, TÁI
    DÙNG evaluate_linguistic_quality (TextGate) + safe_read_jsonl của crawl_audit.

Tái sử dụng (không viết lại): configs.global_id.classify_doc_number,
quality.crawl_audit.evaluate_linguistic_quality / extract_plain_text /
safe_read_jsonl.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import date
from typing import Iterable, Iterator, Optional

# repo root lên sys.path để import configs.* / quality.* khi chạy đứng một mình
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs.global_id import classify_doc_number  # noqa: E402  (nhẹ: chỉ re/hashlib/unicodedata)

_PCTS = (1, 5, 25, 50, 75, 95)
# Các trường rút gọn mà profile_records cần (loader tạo ra; test tự dựng).
_CONTENT_SOURCE_NULL = "(null)"


def percentiles(values: list[float], ps: tuple[int, ...] = _PCTS) -> dict:
    """Phân vị theo nearest-rank (tất định, dễ kiểm). Trả {p: giá trị}. Rỗng → tất cả 0."""
    if not values:
        return {p: 0 for p in ps}
    s = sorted(values)
    n = len(s)
    out = {}
    for p in ps:
        idx = min(n - 1, max(0, (p * n) // 100))  # nearest-rank, 0-based
        out[p] = s[idx]
    return out


def profile_records(reduced: Iterable[dict]) -> dict:
    """HÀM THUẦN. Nhận các bản ghi ĐÃ RÚT GỌN, mỗi dict có khóa:
        doc_id, doc_number, content_source, html_status, ocr_status,
        rescue_status, doc_type, vi_ratio (float|None), length (int)
    Trả bảng đếm tổng hợp (toàn bộ + bổ theo content_source).

    Trường thiếu không làm sập (dùng .get). vi_ratio=None (ví dụ HTML rỗng) được
    đếm riêng là 'no_text', không tính vào phân vị tỷ lệ.
    """
    total = 0
    uniq_doc_number: set[str] = set()
    cls_count = Counter()          # VALID / NO_NUMBER / CONTAMINATED
    cs_count = Counter()           # content_source
    html_count = Counter()
    ocr_count = Counter()
    rescue_count = Counter()
    doctype_count = Counter()
    cross_cs_ocr = Counter()       # (content_source, ocr_status)

    vi_overall: list[float] = []
    len_overall: list[int] = []
    vi_by_cs: dict[str, list[float]] = {}
    len_by_cs: dict[str, list[int]] = {}
    below_vi_overall = 0
    below_vi_by_cs = Counter()
    empty_text = 0                 # length == 0
    short_text = 0                 # 0 < length < 200
    no_text = 0                    # vi_ratio is None (không có text để đo)

    samples_by_cs: dict[str, list[str]] = {}
    samples_by_cls: dict[str, list[str]] = {}

    def _key(v):
        return _CONTENT_SOURCE_NULL if v in (None, "") else str(v)

    for r in reduced:
        total += 1
        dn = r.get("doc_number")
        if dn:
            uniq_doc_number.add(str(dn))
        cls = classify_doc_number(dn or "")[0]
        cls_count[cls] += 1

        cs = _key(r.get("content_source"))
        cs_count[cs] += 1
        html_count[_key(r.get("html_status"))] += 1
        ocr = _key(r.get("ocr_status"))
        ocr_count[ocr] += 1
        rescue_count[_key(r.get("rescue_status"))] += 1
        doctype_count[_key(r.get("doc_type"))] += 1
        cross_cs_ocr[(cs, ocr)] += 1

        length = int(r.get("length") or 0)
        len_overall.append(length)
        len_by_cs.setdefault(cs, []).append(length)
        if length == 0:
            empty_text += 1
        elif length < 200:
            short_text += 1

        vi = r.get("vi_ratio")
        if vi is None:
            no_text += 1
        else:
            vi_overall.append(vi)
            vi_by_cs.setdefault(cs, []).append(vi)
            if vi < 0.05:
                below_vi_overall += 1
                below_vi_by_cs[cs] += 1

        did = str(r.get("doc_id") or "")
        if did:
            if len(samples_by_cs.setdefault(cs, [])) < 20:
                samples_by_cs[cs].append(did)
            if len(samples_by_cls.setdefault(cls, [])) < 20:
                samples_by_cls[cls].append(did)

    by_source = {}
    for cs in cs_count:
        by_source[cs] = {
            "count": cs_count[cs],
            "vi_ratio_pct": percentiles(vi_by_cs.get(cs, [])),
            "length_pct": percentiles(len_by_cs.get(cs, [])),
            "below_vi_0_05": below_vi_by_cs[cs],
            "sample_doc_ids": samples_by_cs.get(cs, []),
        }

    return {
        "total": total,
        "unique_doc_number": len(uniq_doc_number),
        "doc_number_class": dict(cls_count),
        "content_source": dict(cs_count),
        "html_status": dict(html_count),
        "ocr_status": dict(ocr_count),
        "rescue_status": dict(rescue_count),
        "doc_type": dict(doctype_count),
        "cross_content_source_x_ocr": {f"{a}|{b}": n for (a, b), n in cross_cs_ocr.items()},
        "vi_ratio_pct_overall": percentiles(vi_overall),
        "below_vi_0_05_overall": below_vi_overall,
        "length_pct_overall": percentiles(len_overall),
        "empty_text": empty_text,
        "short_text_lt_200": short_text,
        "no_text_records": no_text,
        "by_content_source": by_source,
        "sample_doc_ids_by_class": samples_by_cls,
    }


# ---------------------------------------------------------------------------
# Lớp nạp mỏng (I/O) — TÁI DÙNG TextGate + safe_read_jsonl, streaming O(1)
# ---------------------------------------------------------------------------
def reduce_raw(raw: dict) -> dict:
    """Rút một bản ghi crawl thô về các trường cần + vi_ratio (TextGate) + length.
    vi_ratio=None khi không có html_raw/plain-text để đo."""
    from quality.crawl_audit import evaluate_linguistic_quality, extract_plain_text  # lazy: kéo configs.config
    html_raw = raw.get("html_raw") or ""
    full_text = raw.get("full_text") or raw.get("text") or ""
    if html_raw:
        lq = evaluate_linguistic_quality(html_raw)
        vi = lq.get("vietnamese_ratio")
        length = len(full_text) if full_text else len(extract_plain_text(html_raw))
    else:
        vi = None
        length = len(full_text)
    return {
        "doc_id": raw.get("doc_id") or raw.get("item_id"),
        "doc_number": raw.get("doc_number"),
        "content_source": raw.get("content_source"),
        "html_status": raw.get("html_status"),
        "ocr_status": raw.get("ocr_status"),
        "rescue_status": raw.get("rescue_status"),
        "doc_type": raw.get("doc_type"),
        "vi_ratio": vi,
        "length": length,
    }


def iter_reduced_shards(shards_dir: str, limit: Optional[int] = None) -> Iterator[dict]:
    """Duyệt các shard crawl_pages_*.jsonl.gz theo thứ tự, yield bản ghi đã rút gọn.
    limit = số SHARD tối đa (để ước lượng nhanh)."""
    from quality.crawl_audit import safe_read_jsonl
    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    if limit:
        shards = shards[:limit]
    for sh in shards:
        for raw in safe_read_jsonl(sh):
            yield reduce_raw(raw)


def profile_quarantine(shards_dir: str, limit: Optional[int] = None) -> dict:
    """Đếm bản ghi .quarantine.jsonl theo ocr_status và rescue_status (chỉ đọc)."""
    from quality.crawl_audit import safe_read_jsonl
    qfiles = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.quarantine.jsonl")))
    if limit:
        qfiles = qfiles[:limit]
    n = 0
    ocr = Counter()
    rescue = Counter()
    for qf in qfiles:
        for rec in safe_read_jsonl(qf):
            n += 1
            ocr[rec.get("ocr_status") or "(null)"] += 1
            rescue[rec.get("rescue_status") or "(null)"] += 1
    return {"files": len(qfiles), "records": n,
            "ocr_status": dict(ocr), "rescue_status": dict(rescue)}


def _md(summary: dict, quarantine: dict, env: dict, commands: list[str]) -> str:
    L = []
    a = L.append
    a(f"# Hồ sơ nguồn ngữ liệu VietLawBERT (tầng crawl) — {env['date']}\n")
    a("## Môi trường")
    a(f"- Nguồn: shard JSONL `{env['shards_dir']}` (đọc trực tiếp, Mongo rỗng).")
    a(f"- Số shard quét: {env['shards_scanned']}" + (f" (giới hạn {env['limit']})" if env.get("limit") else " (toàn bộ)"))
    a(f"- Git HEAD: {env.get('git_head','?')}\n")
    a("## Tổng quan")
    a(f"- Tổng bản ghi: **{summary['total']}**")
    a(f"- doc_number khác nhau: {summary['unique_doc_number']}")
    a(f"- Phân loại số hiệu: {summary['doc_number_class']}")
    a(f"- doc_type (tầng crawl): {summary['doc_type']}  "
      "← ~toàn bộ None: doc_type được gán ở bước parse/ingest sau, KHÔNG có ở đây.")
    a(f"- extraction_method: KHÔNG tồn tại ở cấp văn bản (chỉ là trường của cạnh quan hệ, `crawler/pipelines.py`).\n")
    a("## Theo nguồn (content_source)")
    a("| content_source | số | vi_ratio p50 | độ dài p50 | < ngưỡng vi 0.05 |")
    a("|---|---|---|---|---|")
    for cs, d in sorted(summary["by_content_source"].items(), key=lambda kv: -kv[1]["count"]):
        a(f"| {cs} | {d['count']} | {d['vi_ratio_pct'][50]} | {d['length_pct'][50]} | {d['below_vi_0_05']} |")
    a("")
    a("## OCR / quarantine")
    a(f"- html_status: {summary['html_status']}")
    a(f"- ocr_status (bản ghi chính): {summary['ocr_status']}")
    a(f"- rescue_status (bản ghi chính): {summary['rescue_status']}")
    a(f"- Quarantine: {quarantine['records']} bản ghi / {quarantine['files']} file; "
      f"ocr={quarantine['ocr_status']}; rescue={quarantine['rescue_status']}\n")
    a("## Tiếng Việt & độ dài")
    a(f"- vi_ratio phân vị (toàn bộ): {summary['vi_ratio_pct_overall']}")
    a(f"- Số bản < ngưỡng vi 0.05: {summary['below_vi_0_05_overall']}")
    a(f"- Độ dài ký tự phân vị (toàn bộ): {summary['length_pct_overall']}")
    a(f"- Rỗng (0 ký tự): {summary['empty_text']}; <200 ký tự: {summary['short_text_lt_200']}; "
      f"không có text để đo: {summary['no_text_records']}\n")
    a("## Bất thường (chỉ số đo được, không kết luận thay người giao việc)")
    worst = sorted(summary["by_content_source"].items(),
                   key=lambda kv: -(kv[1]["below_vi_0_05"] / max(1, kv[1]["count"])))[:3]
    for cs, d in worst:
        pct = 100 * d["below_vi_0_05"] / max(1, d["count"])
        a(f"- Nguồn `{cs}`: {d['below_vi_0_05']}/{d['count']} ({pct:.2f}%) dưới ngưỡng tiếng Việt.")
    a(f"- OCR_PENDING còn tồn (quarantine): {quarantine['ocr_status'].get('OCR_PENDING', 0)}. "
      "Đề cương ước ~1% PDF scan — số đo thực tế ở trên để đối chiếu.\n")
    a("## Lệnh đã chạy")
    for c in commands:
        a(f"- `{c}`")
    return "\n".join(L) + "\n"


def run(shards_dir: str, out_dir: str, limit: Optional[int] = None,
        git_head: str = "?") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    scanned = min(len(shards), limit) if limit else len(shards)
    summary = profile_records(iter_reduced_shards(shards_dir, limit))
    quarantine = profile_quarantine(shards_dir, limit)
    stamp = date.today().strftime("%Y%m%d")
    env = {"date": date.today().isoformat(), "shards_dir": shards_dir,
           "shards_scanned": scanned, "limit": limit, "git_head": git_head}
    cmds = [f"python -m quality.corpus_audit.profile_sources --shards-dir {shards_dir}"
            + (f" --limit {limit}" if limit else "") + f" --out {out_dir}"]
    payload = {"summary": summary, "quarantine": quarantine, "env": env}
    with open(os.path.join(out_dir, f"corpus_profile_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"corpus_profile_{stamp}.md"), "w", encoding="utf-8") as f:
        f.write(_md(summary, quarantine, env, cmds))
    return payload


if __name__ == "__main__":
    from configs.paths import RAW_SHARDS_DIR, REPORTS_DIR
    ap = argparse.ArgumentParser(description="Hồ sơ nguồn ngữ liệu (chỉ đọc) từ shard crawl.")
    ap.add_argument("--shards-dir", default=str(RAW_SHARDS_DIR))
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument("--limit", type=int, default=None, help="số shard tối đa (ước lượng nhanh)")
    a = ap.parse_args()
    head = "?"
    try:
        import subprocess
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:
        pass
    p = run(a.shards_dir, a.out, a.limit, head)
    s = p["summary"]
    print(f"Tổng bản ghi: {s['total']} | nguồn: {s['content_source']} | "
          f"số hiệu: {s['doc_number_class']}")
    print(f"Báo cáo: {a.out}/corpus_profile_*.{{json,md}}")
