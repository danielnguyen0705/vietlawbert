# -*- coding: utf-8 -*-
"""Unit-test cho quality/corpus_audit/chunk_schema.py (Pydantic LegalChunkSchema)."""
import os
import sys
import pytest

from quality.corpus_audit.chunk_schema import LegalChunkSchema


def test_legal_chunk_schema_hop_le_va_tu_dong_gan_co_detector():
    src_text = "Chương I\nĐiều 1. Phạm vi điều chỉnh của Nghị định này áp dụng đối với tổ chức, cá nhân."
    raw_chunk = {
        "chunk_id": "100-2019-ND-CP-12345678_art_1",
        "doc_id": "100-2019-ND-CP-12345678",
        "doc_number": "100/2019/NĐ-CP",
        "article_id": "100-2019-ND-CP-12345678_art_1",
        "article_num": "1",
        "macro_label": "Chương I",
        "hierarchy_path": "Chương I > Điều 1",
        "content": (
            "[META] Văn bản: 100/2019/NĐ-CP | Tiêu đề: Nghị định xử phạt\n"
            "[HIERARCHY] Chương I > Điều 1\n"
            "[CONTENT] Điều 1. Phạm vi điều chỉnh của Nghị định này áp dụng đối với tổ chức, cá nhân."
        ),
    }
    validated = LegalChunkSchema.from_parser_chunk(
        raw_chunk,
        doc_type="Nghị định/Nghị quyết Chính phủ",
        source_format="detail_html",
        source_text=src_text,
    )
    assert validated.chunk_id == "100-2019-ND-CP-12345678_art_1"
    assert validated.structure_method == "ast"
    assert validated.source_format == "detail_html"
    assert validated.char_span is not None
    assert validated.quality.is_clean is True


def test_legal_chunk_schema_chan_node_cut_hoac_thieu_the_content():
    with pytest.raises(ValueError):
        LegalChunkSchema(
            chunk_id="c1",
            doc_id="d1",
            doc_number="01/2026/NĐ-CP",
            hierarchy_path="Điều 1",
            content="Chuỗi không có thẻ CONTENT hợp lệ",
        )
    with pytest.raises(ValueError):
        LegalChunkSchema(
            chunk_id="c2",
            doc_id="d1",
            doc_number="01/2026/NĐ-CP",
            hierarchy_path="Điều 1",
            content="[META] x [HIERARCHY] Điều 1 [CONTENT] ngắn",
        )
