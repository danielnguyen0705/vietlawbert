# -*- coding: utf-8 -*-
"""
Unit-test cho quality/corpus_audit/corpus_gate.py.
Sử dụng 100% dữ liệu tổng hợp (synthetic), không kết nối DB, không đọc dữ liệu thật/VLQA.
"""
import os
import sys

from quality.corpus_audit.corpus_gate import (
    CorpusIngestGate,
    compute_vi_ratio,
    evaluate_p1_gate,
    evaluate_record_gate,
    gate_and_dedup_records,
)


def test_compute_vi_ratio_chuan_xac():
    assert compute_vi_ratio("") is None
    assert compute_vi_ratio("12345 !@#") == 0.0
    # Tiếng Việt chuẩn có dấu >> 0.05
    vi_text = "Nghị định quy định chi tiết thi hành một số điều của Luật Đất đai."
    r_vi = compute_vi_ratio(vi_text)
    assert r_vi is not None and r_vi >= 0.20
    # Văn bản lỗi mã hóa / ASCII thuần không dấu -> 0.0 < 0.05
    ascii_garbage = "O6i vOi t6 chric ca nhan vi pham hanh chinh trong linh vuc giao thong"
    r_bad = compute_vi_ratio(ascii_garbage)
    assert r_bad is not None and r_bad < 0.05


def test_evaluate_record_gate_ba_cong_must():
    # 1. Bản ghi hợp lệ chuẩn -> ĐẠT
    good = {
        "doc_id": "raw_1",
        "doc_number": "100/2019/NĐ-CP",
        "content_source": "detail_html",
        "html_status": "VALID",
        "full_text": "Điều 1. Phạm vi điều chỉnh của Nghị định này quy định về xử phạt vi phạm hành chính.",
    }
    ev_good = evaluate_record_gate(good)
    assert ev_good["passed"] is True
    assert ev_good["reject_reason"] is None
    assert ev_good["effective_doc_id"].startswith("100-2019-ND-CP-")

    # 2. Cổng 1: Số hiệu CONTAMINATED (rò rỉ tiêu đề kỳ họp) -> CHẶN
    contaminated = {
        "doc_id": "raw_2",
        "doc_number": "Kỳ họp thứ 5 Quốc hội khóa XV về việc ban hành nghị quyết",
        "content_source": "detail_html",
        "html_status": "VALID",
        "full_text": "Điều 1. Nội dung văn bản hợp lệ có đầy đủ tiếng Việt chuẩn xác để vượt qua cổng ngôn ngữ.",
    }
    ev_cont = evaluate_record_gate(contaminated)
    assert ev_cont["passed"] is False
    assert ev_cont["reject_reason"] == "contaminated_doc_number"

    # 3. Cổng 2a: HTML rỗng -> CHẶN
    empty_rec = {
        "doc_id": "raw_3",
        "doc_number": "15/2020/NĐ-CP",
        "content_source": None,
        "html_status": "EMPTY",
        "full_text": "",
    }
    ev_empty = evaluate_record_gate(empty_rec)
    assert ev_empty["passed"] is False
    assert ev_empty["reject_reason"] == "empty_html"

    # 4. Cổng 2b: OCR_PENDING -> CHẶN
    ocr_rec = {
        "doc_id": "raw_4",
        "doc_number": "16/2020/NĐ-CP",
        "content_source": None,
        "html_status": "EMPTY",
        "ocr_status": "OCR_PENDING",
        "full_text": "",
    }
    ev_ocr = evaluate_record_gate(ocr_rec)
    assert ev_ocr["passed"] is False
    assert ev_ocr["reject_reason"] == "ocr_pending"

    # 5. Cổng 3: fallback_pdf dưới ngưỡng tiếng Việt (< 0.05) -> CHẶN & gắn cờ hàng đợi Vision OCR
    pdf_bad = {
        "doc_id": "raw_5",
        "doc_number": "17/2020/TT-BTC",
        "content_source": "fallback_pdf",
        "html_status": "VALID",
        "full_text": "Scan broken text without Vietnamese diacritics at all for over fifty characters long.",
    }
    ev_pdf = evaluate_record_gate(pdf_bad)
    assert ev_pdf["passed"] is False
    assert ev_pdf["reject_reason"] == "fallback_pdf_low_vi"
    assert ev_pdf["is_fallback_pdf_low_vi"] is True


def test_gate_and_dedup_records_gop_dinh_danh_va_cuu_ban_loi():
    records = [
        # Văn bản A: bản 1 bị rỗng, bản 2 (tái-crawl) có detail_html sạch -> phải được CỨU
        {
            "doc_id": "a_1",
            "doc_number": "10/2026/NĐ-CP",
            "content_source": None,
            "html_status": "EMPTY",
            "full_text": "",
        },
        {
            "doc_id": "a_2",
            "doc_number": "Số: 10/2026/NĐ-CP",
            "content_source": "detail_html",
            "html_status": "VALID",
            "full_text": "Điều 1. Quy định chi tiết về việc quản lý và bảo vệ môi trường tự nhiên tại Việt Nam.",
        },
        # Văn bản A: bản 3 trùng lặp hợp lệ -> bị bỏ qua khi đếm distinct
        {
            "doc_id": "a_3",
            "doc_number": "10/2026/NĐ-CP.",
            "content_source": "fallback_html",
            "html_status": "VALID",
            "full_text": "Điều 1. Quy định chi tiết về việc quản lý và bảo vệ môi trường tự nhiên.",
        },
        # Văn bản B (VBHN): 2 phiên bản khác ngày ký -> phải tách thành 2 canonical_doc_id riêng (TN-3)
        {
            "doc_id": "vbhn_1",
            "doc_number": "05/VBHN-BTC",
            "issue_date": "2022-01-10",
            "content_source": "detail_html",
            "html_status": "VALID",
            "full_text": "Điều 1. Văn bản hợp nhất năm 2022 hướng dẫn thi hành Luật Thuế thu nhập doanh nghiệp.",
        },
        {
            "doc_id": "vbhn_2",
            "doc_number": "05/VBHN-BTC",
            "issue_date": "2025-06-20",
            "content_source": "detail_html",
            "html_status": "VALID",
            "full_text": "Điều 1. Văn bản hợp nhất năm 2025 sửa đổi hướng dẫn Luật Thuế thu nhập doanh nghiệp.",
        },
        # Văn bản C: fallback_pdf lỗi tiếng Việt và KHÔNG có bản sạch thay thế -> vào hàng đợi Vision OCR
        {
            "doc_id": "pdf_1",
            "doc_number": "99/2020/QĐ-TTg",
            "content_source": "fallback_pdf",
            "html_status": "VALID",
            "full_text": "Scanned PDF garbage ASCII text only without any accented characters exceeding fifty chars.",
        },
    ]

    summary = gate_and_dedup_records(records)
    raw = summary["raw_records"]
    dedup = summary["identity_deduplication"]
    pdf_q = summary["fallback_pdf_vision_queue"]

    assert raw["total"] == 6
    assert raw["passed"] == 4
    assert raw["rejected"] == 2

    # Tổng văn bản định danh duy nhất: 1 (10/2026/NĐ-CP) + 2 (2 bản VBHN) + 1 (99/2020/QĐ-TTg) = 4
    assert dedup["distinct_effective_docs_total"] == 4
    assert dedup["distinct_effective_docs_passed"] == 3
    assert dedup["distinct_effective_docs_rejected"] == 1
    assert dedup["vbhn_distinct_doc_numbers"] == 1
    assert dedup["vbhn_distinct_version_ids"] == 2
    assert dedup["vbhn_extra_versions_preserved"] == 1
    assert dedup["docs_rescued_by_healthy_duplicate"] == 1
    assert dedup["duplicate_valid_records_skipped"] == 1

    # Hàng đợi Vision OCR chỉ giữ đúng 1 văn bản PDF chưa được cứu
    assert pdf_q["raw_low_vi_or_pending_records"] == 1
    assert pdf_q["distinct_docs_needing_vision_ocr"] == 1


def test_evaluate_p1_gate_soft_vs_hard():
    records = [{
        "doc_id": "r1",
        "doc_number": "01/2026/QH15",
        "content_source": "detail_html",
        "html_status": "VALID",
        "full_text": "Điều 1. Luật này quy định về tổ chức bộ máy nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam.",
    }]
    g_sum = gate_and_dedup_records(records)
    # Ngưỡng yêu cầu 50.000 văn bản nhưng mẫu thử chỉ có 1 văn bản -> gate_passed = False
    p1_soft = evaluate_p1_gate(g_sum, soft_mode=True)
    assert p1_soft["gate_passed"] is False
    assert p1_soft["exit_code"] == 0  # GATE_SOFT=1 không làm dừng tiến trình

    p1_hard = evaluate_p1_gate(g_sum, soft_mode=False)
    assert p1_hard["gate_passed"] is False
    assert p1_hard["exit_code"] == 1  # GATE_SOFT=0 chặn cứng


def test_corpus_ingest_gate_chan_trung_va_rac():
    gate = CorpusIngestGate()
    good_1 = {
        "doc_id": "1",
        "doc_number": "20/2026/NĐ-CP",
        "content_source": "detail_html",
        "html_status": "VALID",
        "full_text": "Điều 1. Nội dung tiếng Việt đầy đủ và hợp lệ cho lần nạp đầu tiên của văn bản.",
    }
    good_dup = {
        "doc_id": "2",
        "doc_number": "Số: 20/2026/NĐ-CP",
        "content_source": "detail_html",
        "html_status": "VALID",
        "full_text": "Điều 1. Nội dung tiếng Việt đầy đủ và hợp lệ nhưng là bản cào trùng số hiệu.",
    }
    bad_empty = {
        "doc_id": "3",
        "doc_number": "21/2026/NĐ-CP",
        "content_source": None,
        "html_status": "EMPTY",
        "full_text": "",
    }
    ok1, id1, r1 = gate.check_and_register(good_1)
    ok2, id2, r2 = gate.check_and_register(good_dup)
    ok3, id3, r3 = gate.check_and_register(bad_empty)

    assert ok1 is True and r1 == "ok"
    assert ok2 is False and r2 == "duplicate_in_run" and id1 == id2
    assert ok3 is False and r3 == "empty_html"
    assert gate.counts["accepted"] == 1
    assert gate.counts["skipped:duplicate_in_run"] == 1
    assert gate.counts["rejected:empty_html"] == 1


def test_corpus_ingest_gate_primed_chon_ban_nguon_tot_nhat():
    """Khi primed: dù bản fallback_pdf (ĐẠT) tới trước, chỉ bản detail_html tốt hơn được nhúng."""
    pdf_copy = {
        "doc_id": "a", "doc_number": "20/2026/NĐ-CP", "content_source": "fallback_pdf",
        "html_status": "VALID",
        "full_text": "Điều 1. Nội dung tiếng Việt đầy đủ hợp lệ trích từ bản PDF dự phòng.",
    }
    html_copy = {
        "doc_id": "b", "doc_number": "Số: 20/2026/NĐ-CP", "content_source": "detail_html",
        "html_status": "VALID",
        "full_text": "Điều 1. Nội dung tiếng Việt đầy đủ hợp lệ trích từ trang chi tiết gốc.",
    }
    gate = CorpusIngestGate()
    gate.prime_best_copies([pdf_copy, html_copy])
    ok_pdf, _, r_pdf = gate.check_and_register(pdf_copy)   # tới trước nhưng nguồn kém hơn
    ok_html, _, r_html = gate.check_and_register(html_copy)

    assert ok_pdf is False and r_pdf == "superseded_by_better_copy"
    assert ok_html is True and r_html == "ok"
    assert gate.counts["accepted"] == 1
    assert gate.counts["skipped:superseded_by_better_copy"] == 1
