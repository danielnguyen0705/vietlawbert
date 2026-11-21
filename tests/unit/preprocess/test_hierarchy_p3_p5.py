# -*- coding: utf-8 -*-
"""Unit-test HybridASTParser P3 (chapter_id/clause_id first-class 4 tầng) + P5 (normalize text).

P3: mỗi chunk mang doc/chương/điều/khoản tách bạch để lọc truy hồi theo Điều/Khoản.
P5: NFKC (NBSP→space) + cắt prefix rác 'Document Content' + gộp space ngang, GIỮ newline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from preprocess.ast_parser import HybridASTParser  # noqa: E402

NBSP = "\xa0"
_TEXT = (
    "Document Content\n"
    "CHƯƠNG I QUY ĐỊNH CHUNG\n"
    "Điều 1. Phạm" + NBSP + "vi  điều chỉnh\n"
    "Nghị định này quy định chi tiết việc áp dụng cho các cơ quan nhà nước.\n"
    "Điều 2. Đối tượng áp dụng\n"
    "1. Khoản một áp dụng cho cơ quan nhà nước và tổ chức xã hội có liên quan.\n"
    "2. Khoản hai áp dụng cho cá nhân có liên quan theo quy định pháp luật hiện hành.\n"
)
_META = {"doc_id": "TESTDOC", "doc_number": "99/2026/TEST", "title": "VB test"}


def _parse():
    return HybridASTParser().parse_document(_TEXT, _META)


def test_p5_normalize_sach_prefix_nbsp_multispace():
    chunks = _parse()
    assert chunks
    for c in chunks:
        assert "Document Content" not in c["text"]
        assert NBSP not in c["text"]
        # gộp space ngang (bỏ newline khi kiểm): không còn chuỗi >=2 space
        assert "  " not in c["content"].replace("\n", "")


def test_p3_chapter_id_tren_moi_dieu():
    art_chunks = [c for c in _parse() if c["article_id"]]
    assert art_chunks
    assert all(c["chapter_id"] == "TESTDOC_ch_I" for c in art_chunks)


def test_p3_clause_id_dung_cho_khoan_rong_cho_non_khoan():
    chunks = _parse()
    clause = sorted(c["clause_id"] for c in chunks if c["clause_id"])
    assert clause == ["TESTDOC_art_2_cl_1", "TESTDOC_art_2_cl_2"]
    d1 = [c for c in chunks if c["article_id"] == "TESTDOC_art_1"]
    assert d1 and all(c["clause_id"] == "" for c in d1)


def test_p5_giu_newline_de_parser_hoat_dong():
    # Nếu normalize nuốt newline, parser dòng-cơ-sở sẽ không tách được Điều → chỉ còn 0-1 chunk.
    assert len({c["article_id"] for c in _parse() if c["article_id"]}) == 2
