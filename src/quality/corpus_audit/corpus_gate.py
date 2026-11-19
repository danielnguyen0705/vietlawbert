# -*- coding: utf-8 -*-
"""
quality/corpus_audit/corpus_gate.py
===================================
BA CỔNG CHẶN RÁC BẮT BUỘC (3 MUST GATES) & CỔNG KIỂM ĐỊNH PHA 1 (p1_gate_corpus)
trước khi thuê Cloud GPU nhúng toàn bộ ngữ liệu VietLawBERT (CLAUDE.md §C.6-7, §D.2-4, §E, §H.2).

Ba cổng chặn rác ở đầu vào:
  1) CỔNG ĐỊNH DANH (Gate 1 - Identity & Deduplication):
     - Phân loại doc_number qua classify_doc_number (VALID / NO_NUMBER / CONTAMINATED).
     - Chặn cứng bản ghi CONTAMINATED (rò rỉ tiêu đề / không chứa số hiệu hợp lệ).
     - Gộp định danh idempotent qua compute_record_doc_id (tất định từ doc_number; riêng
       VBHN phân tách phiên bản theo ngày ký + cơ quan theo TN-3; NO_NUMBER giữ khóa riêng).
     - Khi nhiều bản ghi cào trùng cùng một văn bản (~90k bản tái-crawl), ưu tiên chọn
       bản ĐẠT cổng nội dung có chất lượng nguồn cao nhất (detail_html > fallback_docx >
       fallback_html > fallback_pdf) và độ dài thực chất lớn nhất.
  2) CỔNG RỖNG & OCR_PENDING (Gate 2 - Empty & Pending OCR):
     - Loại bỏ bản ghi HTML rỗng (html_status == 'EMPTY', content_source rỗng, hoặc
       độ dài văn bản thực chất < min_chars = 50).
     - Loại bỏ bản ghi đang chờ OCR ngoại tuyến (ocr_status == 'OCR_PENDING' hoặc
       rescue_status == 'OCR_PENDING' mà chưa có văn bản hợp lệ).
  3) CỔNG TIẾNG VIỆT TEXTGATE & TÁCH FALLBACK_PDF (Gate 3 - Vietnamese Ratio >= 0.05):
     - Loại bỏ bản ghi có tỷ lệ nguyên âm tiếng Việt có dấu < 0.05 (Đề cương v3.1 mục 5.1).
     - Tách riêng danh sách fallback_pdf dưới ngưỡng tiếng Việt (~1.4k bản PDF scan/vỡ font)
       ra hàng đợi manifest JSONL để xử lý ngoại tuyến bằng AI/Vision OCR (không đốt GPU
       nhúng rác và không gọi LLM tràn lan toàn kho).

Hỗ trợ chế độ cổng Pha 1 (p1_gate_corpus):
  - GATE_SOFT=1 (mặc định ban đầu theo CLAUDE.md §E.4): chỉ báo cáo cảnh báo, trả exit code 0.
  - GATE_SOFT=0 (chặn cứng trên Cloud): trả exit code 1 nếu bất kỳ bất biến/ngưỡng nào ĐỎ.
"""
from __future__ import annotations

import argparse
import glob
import html as html_lib
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AUDIT_DIR not in sys.path:
    sys.path.insert(0, _AUDIT_DIR)

from configs.global_id import (  # noqa: E402
    classify_doc_number,
    compute_record_doc_id,
    is_collision_prone,
    normalize_doc_number,
)
from audit_corpus import wilson_ci  # noqa: E402

# Tập 68 ký tự tiếng Việt có dấu (đồng nhất tuyệt đối với quality/crawl_audit.py)
VIETNAMESE_CHARS = frozenset(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

_RE_SCRIPT_STYLE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")

# Thứ tự ưu tiên nguồn khi gộp nhiều bản ghi trùng cùng một effective_doc_id
_SOURCE_PRIORITY = {
    "detail_html": 4,
    "fallback_docx": 3,
    "fallback_html": 2,
    "fallback_pdf": 1,
    "(null)": 0,
    "": 0,
}

# Ngưỡng mặc định cho cổng p1_gate_corpus (chốt từ số đo thực nghiệm trên kho 160.380 bản ghi)
DEFAULT_P1_THRESHOLDS = {
    "min_distinct_valid_docs": 50000,
    "min_doc_pass_rate_pct": 85.0,
    "min_article_match_pct": 95.0,
    "min_coverage_pct": 90.0,
    "min_monotonic_pct": 94.0,
    "min_no_empty_chunk_pct": 98.0,
}


def extract_plain_fast(html_content: str) -> str:
    """Bóc tách văn bản thuần từ HTML (tương đương crawl_audit.extract_plain_text nhưng không kéo torch)."""
    if not html_content:
        return ""
    text = _RE_SCRIPT_STYLE.sub(" ", html_content)
    text = _RE_HTML_TAG.sub(" ", text)
    text = html_lib.unescape(text)
    return _RE_WS.sub(" ", text).strip()


def compute_vi_ratio(text: str) -> Optional[float]:
    """Tính tỷ lệ ký tự tiếng Việt có dấu trên tổng số chữ cái (TextGate, Đề cương v3.1 mục 5.1).
    Trả None nếu chuỗi rỗng, 0.0 nếu không có chữ cái nào."""
    if not text:
        return None
    alpha_total = 0
    vn_count = 0
    for ch in text:
        if ch.isalpha():
            alpha_total += 1
            if ch in VIETNAMESE_CHARS:
                vn_count += 1
    if alpha_total == 0:
        return 0.0
    return round(vn_count / alpha_total, 4)


def evaluate_record_gate(
    record: Dict[str, Any],
    min_vi_ratio: float = 0.05,
    min_chars: int = 50,
) -> Dict[str, Any]:
    """HÀM THUẦN. Kiểm định một bản ghi thông qua 3 cổng MUST:
      - Cổng 1 (Identity): phân loại số hiệu + tính khóa canonical_doc_id (có version cho VBHN).
      - Cổng 2 (Empty / OCR_PENDING): chặn HTML rỗng, văn bản quá ngắn (< min_chars), hoặc OCR_PENDING.
      - Cổng 3 (TextGate vi_ratio >= 0.05): chặn văn bản lỗi mã hóa / PDF scan dưới ngưỡng tiếng Việt.
    Hỗ trợ cả bản ghi crawl thô (có full_text/html_raw) lẫn bản ghi rút gọn (đã có vi_ratio, length).
    """
    raw_id = str(record.get("doc_id") or record.get("item_id") or "").strip()
    md = record.get("metadata_detail") or {}
    ma = record.get("metadata_api") or {}
    doc_number = str(
        record.get("doc_number") or md.get("docNum") or ma.get("docNum") or ""
    ).strip()

    doc_cls, cleaned_num = classify_doc_number(doc_number)
    norm_num = normalize_doc_number(cleaned_num) if doc_cls == "VALID" else ""
    canon_id = compute_record_doc_id(record) if doc_cls == "VALID" else ""
    is_vbhn = bool(norm_num and is_collision_prone(norm_num))

    if doc_cls == "VALID" and canon_id:
        eff_id = canon_id
    elif doc_cls == "NO_NUMBER":
        eff_id = f"NO-NUMBER-{raw_id}" if raw_id else ""
    else:
        eff_id = ""

    cs_raw = record.get("content_source")
    content_source = "(null)" if cs_raw in (None, "") else str(cs_raw)
    html_status = str(record.get("html_status") or "(null)")
    ocr_status = str(record.get("ocr_status") or "(null)")
    rescue_status = str(record.get("rescue_status") or "(null)")

    # Hỗ trợ cả dict đã rút gọn (có sẵn length, vi_ratio) lẫn bản ghi thô
    if "length" in record and ("vi_ratio" in record or record.get("html_raw") is None):
        length = int(record.get("length") or 0)
        vi_ratio = record.get("vi_ratio")
    else:
        full_text = str(record.get("full_text") or record.get("text") or "").strip()
        html_raw = str(record.get("html_raw") or "")
        if full_text and "<" not in full_text:
            length = len(full_text)
            vi_ratio = compute_vi_ratio(full_text) if (html_raw or full_text) else None
        elif html_raw:
            plain = extract_plain_fast(html_raw)
            length = len(full_text) if full_text else len(plain)
            vi_ratio = compute_vi_ratio(plain)
        else:
            length = len(full_text)
            vi_ratio = compute_vi_ratio(full_text) if full_text else None

    is_ocr_pending = (ocr_status == "OCR_PENDING") or (rescue_status == "OCR_PENDING")
    is_empty = (
        html_status == "EMPTY"
        or content_source == "(null)"
        or length < min_chars
        or vi_ratio is None
    )
    is_low_vi = (vi_ratio is not None) and (vi_ratio < min_vi_ratio)

    # Xác định lý do loại (ưu tiên báo cáo đúng bản chất nguồn rác để thống kê không chồng lấn)
    reject_reason: Optional[str] = None
    if is_ocr_pending and (is_empty or is_low_vi):
        reject_reason = "ocr_pending"
    elif is_empty:
        reject_reason = "empty_html"
    elif is_low_vi:
        reject_reason = (
            "fallback_pdf_low_vi" if content_source == "fallback_pdf" else "below_vi_threshold"
        )
    elif doc_cls == "CONTAMINATED" or not eff_id:
        reject_reason = "contaminated_doc_number"

    passed = reject_reason is None
    is_fallback_pdf_low_vi = (content_source == "fallback_pdf") and (
        is_low_vi or is_ocr_pending or is_empty
    )

    return {
        "passed": passed,
        "reject_reason": reject_reason,
        "raw_doc_id": raw_id,
        "effective_doc_id": eff_id,
        "canonical_doc_id": canon_id,
        "doc_number": doc_number,
        "doc_number_norm": norm_num,
        "doc_number_class": doc_cls,
        "is_vbhn": is_vbhn,
        "content_source": content_source,
        "html_status": html_status,
        "ocr_status": ocr_status,
        "rescue_status": rescue_status,
        "vi_ratio": vi_ratio,
        "length": length,
        "is_fallback_pdf_low_vi": is_fallback_pdf_low_vi,
    }


def _record_quality_key(eval_res: Dict[str, Any]) -> Tuple[int, int, int]:
    """Khóa so sánh để chọn bản đại diện tốt nhất khi nhiều bản ghi trùng effective_doc_id:
    (1) passed cổng = True > False; (2) độ ưu tiên nguồn; (3) độ dài văn bản."""
    return (
        1 if eval_res["passed"] else 0,
        _SOURCE_PRIORITY.get(eval_res["content_source"], 0),
        int(eval_res["length"] or 0),
    )


def gate_and_dedup_records(
    records: Iterable[Dict[str, Any]],
    min_vi_ratio: float = 0.05,
    min_chars: int = 50,
) -> Dict[str, Any]:
    """HÀM THUẦN. Duyệt luồng bản ghi, áp dụng 3 cổng MUST và gộp theo effective_doc_id.
    Trả về thống kê đầy đủ cả ở cấp BẢN GHI THÔ (raw records) lẫn cấp VĂN BẢN ĐỊNH DANH
    DUY NHẤT (distinct canonical/effective documents), kèm danh sách fallback_pdf cần OCR."""
    total_records = 0
    raw_passed = 0
    raw_reject_counts: Counter = Counter()
    raw_by_source: Counter = Counter()
    raw_passed_by_source: Counter = Counter()
    doc_cls_counts: Counter = Counter()

    # Tập hợp đếm distinct ở tầng định danh
    uniq_raw_doc_numbers: set[str] = set()
    uniq_norm_doc_numbers: set[str] = set()
    uniq_canonical_doc_ids: set[str] = set()
    vbhn_norm_numbers: set[str] = set()
    vbhn_canonical_ids: set[str] = set()

    # Lưu bản đại diện tốt nhất cho mỗi effective_doc_id:
    # eff_id -> (best_eval_dict, count_copies, had_rejected_copy)
    best_by_eff_id: Dict[str, List[Any]] = {}

    # Hàng đợi fallback_pdf dưới ngưỡng tiếng Việt (PDF scan/vỡ font cần Vision/OCR)
    fallback_pdf_ocr_queue: List[Dict[str, Any]] = []

    for rec in records:
        total_records += 1
        ev = evaluate_record_gate(rec, min_vi_ratio=min_vi_ratio, min_chars=min_chars)
        cs = ev["content_source"]
        raw_by_source[cs] += 1
        doc_cls_counts[ev["doc_number_class"]] += 1

        if ev["doc_number"]:
            uniq_raw_doc_numbers.add(ev["doc_number"])
        if ev["doc_number_norm"]:
            uniq_norm_doc_numbers.add(ev["doc_number_norm"])
        if ev["canonical_doc_id"]:
            uniq_canonical_doc_ids.add(ev["canonical_doc_id"])
        if ev["is_vbhn"]:
            vbhn_norm_numbers.add(ev["doc_number_norm"])
            if ev["canonical_doc_id"]:
                vbhn_canonical_ids.add(ev["canonical_doc_id"])

        if ev["passed"]:
            raw_passed += 1
            raw_passed_by_source[cs] += 1
        else:
            raw_reject_counts[ev["reject_reason"] or "unknown"] += 1

        if ev["is_fallback_pdf_low_vi"]:
            fallback_pdf_ocr_queue.append({
                "doc_id": ev["raw_doc_id"],
                "effective_doc_id": ev["effective_doc_id"],
                "doc_number": ev["doc_number"],
                "content_source": cs,
                "ocr_status": ev["ocr_status"],
                "rescue_status": ev["rescue_status"],
                "vi_ratio": ev["vi_ratio"],
                "length": ev["length"],
                "reject_reason": ev["reject_reason"],
            })

        eff_id = ev["effective_doc_id"]
        if not eff_id:
            continue

        entry = best_by_eff_id.get(eff_id)
        if entry is None:
            best_by_eff_id[eff_id] = [ev, 1, not ev["passed"]]
        else:
            entry[1] += 1
            if not ev["passed"]:
                entry[2] = True
            if _record_quality_key(ev) > _record_quality_key(entry[0]):
                entry[0] = ev

    # Tổng hợp sau khi gộp định danh (deduplicated document level)
    distinct_all_eff_docs = len(best_by_eff_id)
    distinct_passed_docs = 0
    distinct_rejected_docs = 0
    distinct_reject_reasons: Counter = Counter()
    distinct_passed_by_source: Counter = Counter()
    rescued_by_dedup = 0
    total_chars_passed_distinct = 0
    multi_copy_docs = 0

    for eff_id, (best_ev, copies, had_fail) in best_by_eff_id.items():
        if copies > 1:
            multi_copy_docs += 1
        if best_ev["passed"]:
            distinct_passed_docs += 1
            distinct_passed_by_source[best_ev["content_source"]] += 1
            total_chars_passed_distinct += int(best_ev["length"] or 0)
            if had_fail:
                rescued_by_dedup += 1
        else:
            distinct_rejected_docs += 1
            distinct_reject_reasons[best_ev["reject_reason"] or "unknown"] += 1

    duplicate_raw_passed_skipped = max(0, raw_passed - distinct_passed_docs)

    # Trong hàng đợi fallback_pdf lỗi, có bao nhiêu văn bản thực sự CHƯA có bản sạch thay thế?
    passed_eff_ids = {
        eff_id for eff_id, (best_ev, _, _) in best_by_eff_id.items() if best_ev["passed"]
    }
    unrescued_pdf_queue = [
        item for item in fallback_pdf_ocr_queue
        if not item["effective_doc_id"] or item["effective_doc_id"] not in passed_eff_ids
    ]
    distinct_unrescued_pdf_docs = len({
        item["effective_doc_id"] or item["doc_id"] for item in unrescued_pdf_queue
    })

    # Ước lượng chi phí nhúng trên Cloud GPU (trung bình ~1.500 ký tự / chunk theo Đề cương)
    est_chunks_raw = round(total_records * 21.2) if total_records else 0
    est_chunks_dedup_clean = round(total_chars_passed_distinct / 1500) if total_chars_passed_distinct else 0
    records_saved = total_records - distinct_passed_docs
    savings_pct = round(100.0 * records_saved / total_records, 2) if total_records else 0.0

    raw_pass_rate, raw_ci_lo, raw_ci_hi = wilson_ci(raw_passed, total_records)
    doc_pass_rate, doc_ci_lo, doc_ci_hi = wilson_ci(distinct_passed_docs, distinct_all_eff_docs)

    return {
        "raw_records": {
            "total": total_records,
            "passed": raw_passed,
            "rejected": total_records - raw_passed,
            "pass_rate_pct": raw_pass_rate,
            "pass_ci95_pct": [raw_ci_lo, raw_ci_hi],
            "reject_reasons": dict(raw_reject_counts),
            "by_content_source": dict(raw_by_source),
            "passed_by_content_source": dict(raw_passed_by_source),
            "doc_number_class": dict(doc_cls_counts),
        },
        "identity_deduplication": {
            "distinct_raw_doc_number": len(uniq_raw_doc_numbers),
            "distinct_normalized_doc_number": len(uniq_norm_doc_numbers),
            "distinct_canonical_doc_id_valid": len(uniq_canonical_doc_ids),
            "vbhn_distinct_doc_numbers": len(vbhn_norm_numbers),
            "vbhn_distinct_version_ids": len(vbhn_canonical_ids),
            "vbhn_extra_versions_preserved": max(0, len(vbhn_canonical_ids) - len(vbhn_norm_numbers)),
            "distinct_effective_docs_total": distinct_all_eff_docs,
            "distinct_effective_docs_passed": distinct_passed_docs,
            "distinct_effective_docs_rejected": distinct_rejected_docs,
            "distinct_pass_rate_pct": doc_pass_rate,
            "distinct_pass_ci95_pct": [doc_ci_lo, doc_ci_hi],
            "distinct_reject_reasons": dict(distinct_reject_reasons),
            "distinct_passed_by_source": dict(distinct_passed_by_source),
            "docs_with_multiple_crawl_copies": multi_copy_docs,
            "duplicate_valid_records_skipped": duplicate_raw_passed_skipped,
            "docs_rescued_by_healthy_duplicate": rescued_by_dedup,
        },
        "fallback_pdf_vision_queue": {
            "raw_low_vi_or_pending_records": len(fallback_pdf_ocr_queue),
            "already_rescued_by_duplicate_html": len(fallback_pdf_ocr_queue) - len(unrescued_pdf_queue),
            "unrescued_records_needing_ocr": len(unrescued_pdf_queue),
            "distinct_docs_needing_vision_ocr": distinct_unrescued_pdf_docs,
        },
        "cloud_cost_estimate": {
            "raw_records_if_ungated": total_records,
            "distinct_clean_docs_to_embed": distinct_passed_docs,
            "records_eliminated_total": records_saved,
            "compute_savings_pct": savings_pct,
            "total_clean_chars": total_chars_passed_distinct,
            "estimated_clean_chunks": est_chunks_dedup_clean,
            "estimated_raw_chunks_if_ungated": est_chunks_raw,
        },
        "_ocr_queue_items": unrescued_pdf_queue,
    }


def evaluate_p1_gate(
    gate_summary: Dict[str, Any],
    parser_summary: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    soft_mode: bool = True,
) -> Dict[str, Any]:
    """HÀM THUẦN. Kiểm định cổng Pha 1 (p1_gate_corpus) kết hợp thống kê 3 cổng MUST và
    bất biến Lớp 1 của AST parser (nếu có).
    - soft_mode=True (GATE_SOFT=1): chỉ báo cáo, exit_code luôn bằng 0.
    - soft_mode=False (GATE_SOFT=0): nếu có bất kỳ check nào thất bại, exit_code = 1."""
    th = dict(DEFAULT_P1_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    dedup = gate_summary["identity_deduplication"]
    checks: Dict[str, Dict[str, Any]] = {}

    distinct_passed = dedup["distinct_effective_docs_passed"]
    checks["min_distinct_valid_docs"] = {
        "actual": distinct_passed,
        "threshold": th["min_distinct_valid_docs"],
        "passed": distinct_passed >= th["min_distinct_valid_docs"],
    }

    doc_pass_rate = dedup["distinct_pass_rate_pct"]
    checks["min_doc_pass_rate_pct"] = {
        "actual": doc_pass_rate,
        "threshold": th["min_doc_pass_rate_pct"],
        "passed": doc_pass_rate >= th["min_doc_pass_rate_pct"],
    }

    if parser_summary and "invariants" in parser_summary:
        invs = parser_summary["invariants"]
        mapping = [
            ("parser_article_match_pct", "article_match", "min_article_match_pct"),
            ("parser_coverage_pct", "coverage", "min_coverage_pct"),
            ("parser_monotonic_pct", "monotonic", "min_monotonic_pct"),
            ("parser_no_empty_chunk_pct", "no_empty_chunk", "min_no_empty_chunk_pct"),
        ]
        for chk_name, inv_key, th_key in mapping:
            if inv_key in invs:
                actual_val = invs[inv_key]["rate_pct"]
                checks[chk_name] = {
                    "actual": actual_val,
                    "ci95_pct": [invs[inv_key]["ci95_lo_pct"], invs[inv_key]["ci95_hi_pct"]],
                    "threshold": th[th_key],
                    "passed": actual_val >= th[th_key],
                }

    all_passed = all(c["passed"] for c in checks.values())
    exit_code = 0 if (all_passed or soft_mode) else 1
    return {
        "gate_passed": all_passed,
        "soft_mode": soft_mode,
        "exit_code": exit_code,
        "checks": checks,
    }


class CorpusIngestGate:
    """Bộ lọc trạng thái gọn nhẹ gắn trực tiếp vào IngestPipelineWorker (pipeline/ingest_pipeline.py).
    Giúp chặn rác 3 cổng MUST + khử trùng effective_doc_id ngay trong RAM trước khi gọi
    Qdrant/SentenceTransformer, đồng thời đếm đầy đủ số bản ghi bị bỏ theo từng nguyên nhân."""

    def __init__(self, min_vi_ratio: float = 0.05, min_chars: int = 50):
        self.min_vi_ratio = min_vi_ratio
        self.min_chars = min_chars
        self.seen_doc_ids: set[str] = set()
        self.counts: Counter = Counter()
        # Khi primed: điểm chất lượng CAO NHẤT trong các bản sao ĐẠT của mỗi effective_doc_id
        # (tiền-quét toàn kho, chỉ text). Nhờ đó bản ĐƯỢC nhúng là bản nguồn tốt nhất, không
        # phải bản xuất hiện trước — đồng bộ với gate offline gate_and_dedup_records.
        self.primed: bool = False
        self.winner_quality: Dict[str, Tuple[int, int, int]] = {}

    def prime_best_copies(self, records: Iterable[Dict[str, Any]]) -> int:
        """Tiền-quét (chỉ đọc, KHÔNG nhúng) để chọn bản sao tốt nhất mỗi effective_doc_id.
        Gọi MỘT LẦN trước vòng ingest. Trả số văn bản định danh duy nhất đã ghi điểm."""
        for rec in records:
            ev = evaluate_record_gate(rec, min_vi_ratio=self.min_vi_ratio, min_chars=self.min_chars)
            if not ev["passed"]:
                continue
            eff_id = ev["effective_doc_id"]
            if not eff_id:
                continue
            q = _record_quality_key(ev)
            if q > self.winner_quality.get(eff_id, (-1, -1, -1)):
                self.winner_quality[eff_id] = q
        self.primed = True
        return len(self.winner_quality)

    def check_and_register(self, record: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Kiểm tra bản ghi và đăng ký effective_doc_id nếu ĐẠT.
        Trả về (should_ingest: bool, effective_doc_id: str, reason: str)."""
        self.counts["total_seen"] += 1
        ev = evaluate_record_gate(record, min_vi_ratio=self.min_vi_ratio, min_chars=self.min_chars)
        eff_id = ev["effective_doc_id"]
        if not ev["passed"]:
            reason = ev["reject_reason"] or "gate_rejected"
            self.counts[f"rejected:{reason}"] += 1
            return False, eff_id, reason
        # Nếu đã primed: bỏ qua bản ĐẠT nhưng kém hơn bản sao tốt nhất cùng văn bản.
        if self.primed and eff_id in self.winner_quality \
                and _record_quality_key(ev) < self.winner_quality[eff_id]:
            self.counts["skipped:superseded_by_better_copy"] += 1
            return False, eff_id, "superseded_by_better_copy"
        if eff_id in self.seen_doc_ids:
            self.counts["skipped:duplicate_in_run"] += 1
            return False, eff_id, "duplicate_in_run"
        self.seen_doc_ids.add(eff_id)
        self.counts["accepted"] += 1
        return True, eff_id, "ok"


# ---------------------------------------------------------------------------
# Lớp nạp I/O (chỉ đọc) trên các shard crawl_pages_*.jsonl.gz
# ---------------------------------------------------------------------------
def iter_raw_shards(shards_dir: str, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    from quality.crawl_audit import safe_read_jsonl

    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    if limit:
        shards = shards[:limit]
    for sh in shards:
        for raw in safe_read_jsonl(sh):
            yield raw


def _md(payload: Dict[str, Any]) -> str:
    env = payload["env"]
    g = payload["gate_summary"]
    raw = g["raw_records"]
    dedup = g["identity_deduplication"]
    pdf_q = g["fallback_pdf_vision_queue"]
    cost = g["cloud_cost_estimate"]
    p1 = payload["p1_gate"]

    status_str = "XANH (PASS)" if p1["gate_passed"] else (
        "CẢNH BÁO MỀM (SOFT WARN - GATE_SOFT=1)" if p1["soft_mode"] else "ĐỎ (BLOCKED - GATE_SOFT=0)"
    )

    L = [
        f"# Báo cáo Cổng Ngữ liệu & Gộp Định danh (p1_gate_corpus) — {env['date']}\n",
        f"**Trạng thái cổng Pha 1:** `{status_str}` (Mã thoát: `{p1['exit_code']}`)\n",
        "## 1. Môi trường",
        f"- Nguồn shard (chỉ đọc): `{env['shards_dir']}`",
        f"- Số shard quét: {env['shards_scanned']}" + (f" (giới hạn {env['limit']})" if env.get("limit") else " (toàn bộ)"),
        f"- Git HEAD: `{env.get('git_head', '?')}`\n",
        "## 2. Kết quả Gộp Định danh (Cổng MUST #1 — Quyết định chi phí Cloud)",
        "| Chỉ số định danh | Số lượng | Ghi chú |",
        "|---|---:|---|",
        f"| Tổng bản ghi thô (raw crawl records) | **{raw['total']:,}** | Chứa ~90k bản tái-crawl/trùng |",
        f"| Số `doc_number` thô khác nhau | {dedup['distinct_raw_doc_number']:,} | Chưa chuẩn hóa khoảng trắng/tiền tố |",
        f"| Số `doc_number` chuẩn hóa (`VALID`) | {dedup['distinct_normalized_doc_number']:,} | Đã qua `_preclean` + `normalize_doc_number` |",
        f"| Số `canonical_doc_id` (`VALID`, có tách version VBHN) | **{dedup['distinct_canonical_doc_id_valid']:,}** | Gồm +{dedup['vbhn_extra_versions_preserved']:,} phiên bản VBHN phân biệt theo ngày ký (TN-3) |",
        f"| Tổng văn bản định danh duy nhất (`effective_doc_id`) | **{dedup['distinct_effective_docs_total']:,}** | Cộng thêm các bản `NO_NUMBER` giữ khóa riêng |",
        f"| **Văn bản duy nhất ĐẠT cả 3 cổng (cần nhúng Cloud)** | **{dedup['distinct_effective_docs_passed']:,}** | **{dedup['distinct_pass_rate_pct']}%** (CI95% `{dedup['distinct_pass_ci95_pct']}`) |",
        f"| Văn bản duy nhất bị loại sau gộp | {dedup['distinct_effective_docs_rejected']:,} | `{dedup['distinct_reject_reasons']}` |",
        f"| Bản ghi hợp lệ nhưng trùng lặp bị bỏ qua | {dedup['duplicate_valid_records_skipped']:,} | Không phải nhúng lại trên GPU |",
        f"| Văn bản được cứu nhờ có bản sao tái-crawl sạch | {dedup['docs_rescued_by_healthy_duplicate']:,} | Có ≥1 bản lỗi nhưng được bản sạch cùng `doc_id` bù đắp |\n",
        "## 3. Thống kê Chặn Rác ở cấp Bản ghi thô (Cổng MUST #2 & #3)",
        f"- Bản ghi ĐẠT cổng: **{raw['passed']:,} / {raw['total']:,}** ({raw['pass_rate_pct']}%, CI95% `{raw['pass_ci95_pct']}`)",
        f"- Bản ghi BỊ CHẶN: **{raw['rejected']:,}** — phân rã theo lý do: `{raw['reject_reasons']}`",
        f"- Phân bố văn bản duy nhất ĐẠT cổng theo nguồn: `{dedup['distinct_passed_by_source']}`\n",
        "## 4. Hàng đợi `fallback_pdf` cho AI/Vision OCR ngoại tuyến (Cổng MUST #3)",
        f"- Tổng bản ghi `fallback_pdf` dưới ngưỡng tiếng Việt / chờ OCR: **{pdf_q['raw_low_vi_or_pending_records']:,}**",
        f"- Đã có bản HTML sạch trùng `canonical_doc_id` cứu tự động: **{pdf_q['already_rescued_by_duplicate_html']:,}**",
        f"- Số bản ghi thực sự còn thiếu nội dung: **{pdf_q['unrescued_records_needing_ocr']:,}** "
        f"(tương ứng **{pdf_q['distinct_docs_needing_vision_ocr']:,}** văn bản định danh duy nhất)",
        f"- Danh sách chi tiết đã xuất tại: `{env.get('ocr_queue_path', 'reports/fallback_pdf_ocr_queue_*.jsonl')}`\n",
        "## 5. Ước tính Tiết kiệm Chi phí Cloud GPU",
        f"- Nếu nạp thô không cổng: **{cost['raw_records_if_ungated']:,}** bản ghi (~{cost['estimated_raw_chunks_if_ungated']:,} chunks)",
        f"- Sau 3 cổng MUST + gộp định danh: **{cost['distinct_clean_docs_to_embed']:,}** văn bản sạch "
        f"(~{cost['estimated_clean_chunks']:,} chunks, tổng {cost['total_clean_chars']:,} ký tự)",
        f"- **Tỷ lệ cắt giảm khối lượng nhúng:** **{cost['compute_savings_pct']}%** (loại bỏ {cost['records_eliminated_total']:,} bản ghi trùng/rác)\n",
        "## 6. Bảng Kiểm định Cổng Pha 1 (`p1_gate_corpus`)",
        "| Tiêu chí kiểm định | Giá trị thực tế | Ngưỡng yêu cầu | Kết quả |",
        "|---|---:|---:|:---:|",
    ]
    for name, chk in p1["checks"].items():
        mark = "PASS" if chk["passed"] else "FAIL"
        L.append(f"| `{name}` | {chk['actual']} | {chk['threshold']} | **{mark}** |")
    return "\n".join(L) + "\n"


def run(
    shards_dir: str,
    out_dir: str,
    parser_sanity_json: Optional[str] = None,
    limit: Optional[int] = None,
    soft_mode: bool = True,
    min_distinct_docs: Optional[int] = None,
    git_head: str = "?",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    scanned = min(len(shards), limit) if limit else len(shards)

    gate_summary = gate_and_dedup_records(iter_raw_shards(shards_dir, limit))
    ocr_queue_items = gate_summary.pop("_ocr_queue_items", [])

    parser_summary = None
    if parser_sanity_json and os.path.exists(parser_sanity_json):
        with open(parser_sanity_json, "r", encoding="utf-8") as pf:
            pdata = json.load(pf)
            parser_summary = pdata.get("summary") or pdata

    thresholds = dict(DEFAULT_P1_THRESHOLDS)
    if min_distinct_docs is not None:
        thresholds["min_distinct_valid_docs"] = min_distinct_docs
    elif limit:
        # Nếu chạy giới hạn shard (smoke test), hạ ngưỡng số văn bản tối thiểu tỷ lệ theo số shard
        thresholds["min_distinct_valid_docs"] = max(1, int(300 * limit))

    p1_gate = evaluate_p1_gate(
        gate_summary,
        parser_summary=parser_summary,
        thresholds=thresholds,
        soft_mode=soft_mode,
    )

    stamp = date.today().strftime("%Y%m%d")
    ocr_queue_path = os.path.join(out_dir, f"fallback_pdf_ocr_queue_{stamp}.jsonl")
    with open(ocr_queue_path, "w", encoding="utf-8") as qf:
        for item in ocr_queue_items:
            qf.write(json.dumps(item, ensure_ascii=False) + "\n")

    env = {
        "date": date.today().isoformat(),
        "shards_dir": shards_dir,
        "shards_scanned": scanned,
        "limit": limit,
        "git_head": git_head,
        "ocr_queue_path": ocr_queue_path,
    }
    payload = {
        "gate_summary": gate_summary,
        "p1_gate": p1_gate,
        "env": env,
    }

    json_path = os.path.join(out_dir, f"corpus_gate_{stamp}.json")
    md_path = os.path.join(out_dir, f"corpus_gate_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(_md(payload))

    return payload


if __name__ == "__main__":
    from configs.paths import RAW_SHARDS_DIR, REPORTS_DIR

    default_soft = os.environ.get("GATE_SOFT", "1").strip() not in ("0", "false", "False")
    ap = argparse.ArgumentParser(
        description="Ba cổng chặn rác MUST + gộp định danh canonical + cổng kiểm định Pha 1 (p1_gate_corpus)."
    )
    ap.add_argument("--shards-dir", default=str(RAW_SHARDS_DIR))
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument(
        "--parser-sanity-json",
        default=str(REPORTS_DIR / f"parser_sanity_{date.today().strftime('%Y%m%d')}.json"),
        help="Đường dẫn JSON kết quả parser_sanity (nếu có) để gộp vào cổng Pha 1",
    )
    ap.add_argument("--limit", type=int, default=None, help="Giới hạn số shard quét (smoke test)")
    ap.add_argument(
        "--hard",
        action="store_true",
        help="Bật chế độ chặn cứng (tương đương GATE_SOFT=0): trả exit code 1 nếu cổng ĐỎ",
    )
    ap.add_argument("--min-distinct-docs", type=int, default=None)
    args = ap.parse_args()

    soft_mode = False if args.hard else default_soft
    head = "?"
    try:
        import subprocess
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        pass

    res = run(
        shards_dir=args.shards_dir,
        out_dir=args.out,
        parser_sanity_json=args.parser_sanity_json,
        limit=args.limit,
        soft_mode=soft_mode,
        min_distinct_docs=args.min_distinct_docs,
        git_head=head,
    )
    dedup = res["gate_summary"]["identity_deduplication"]
    cost = res["gate_summary"]["cloud_cost_estimate"]
    p1 = res["p1_gate"]
    print(
        f"[p1_gate_corpus] Trạng thái: {'PASS' if p1['gate_passed'] else 'WARN/FAIL'} "
        f"(soft={p1['soft_mode']}) | Bản ghi thô: {cost['raw_records_if_ungated']:,} -> "
        f"Văn bản duy nhất ĐẠT cổng: {dedup['distinct_effective_docs_passed']:,} "
        f"(tiết kiệm {cost['compute_savings_pct']}%)"
    )
    sys.exit(p1["exit_code"])
