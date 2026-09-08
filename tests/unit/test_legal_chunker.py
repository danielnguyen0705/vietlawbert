"""Unit test động cơ phân đoạn pháp lý nhận thức cấu trúc (Legal Chunker AST)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.legal_chunker import (
    chunk_legal_document,
    chunk_by_dieu,
    chunk_by_khoan,
    extract_cross_references,
    LegalLevel,
)


def test_chunk_id_semantic_preservation():
    raw_md = """
##### Điều 15. Thẩm quyền xử phạt
1. Chủ tịch Ủy ban nhân dân cấp xã có quyền phạt tiền đến 5.000.000 đồng đối với các hành vi vi phạm trật tự an toàn giao thông đường bộ.
2. Trưởng Công an cấp xã có quyền phạt cảnh cáo.
"""
    chunks = chunk_legal_document(raw_md, doc_id="ND_100")
    assert len(chunks) >= 2
    assert "ND_100_Điều15_K1" in chunks[0].chunk_id
    assert chunks[0].hierarchy.dieu == "Điều 15"
    assert chunks[0].hierarchy.khoan == "Khoản 1"


def test_extract_cross_references_external_and_internal():
    text = (
        "Hành vi vi phạm quy định tại Khoản 2 Điều 5 Nghị định 100/2019/NĐ-CP "
        "sẽ bị áp dụng biện pháp khắc phục hậu quả theo Điều này."
    )
    refs = extract_cross_references(text)
    assert len(refs) >= 2

    external_refs = [r for r in refs if r["target_type"] == "Điều"]
    assert len(external_refs) >= 1
    assert external_refs[0]["target_number"] == "5"
    assert "Nghị định 100/2019/NĐ-CP" in external_refs[0]["target_document"]

    internal_refs = [r for r in refs if r["target_type"] == "Nội bộ"]
    assert len(internal_refs) >= 1


def test_subword_boundary_split():
    long_clause = "1. Quy định về xử phạt hành chính: " + ("hành vi vi phạm trật tự an toàn; " * 80)
    text = f"##### Điều 1. Phạm vi\n{long_clause}"
    chunks = chunk_legal_document(text, doc_id="DOC_LONG", max_chunk_size=400)

    assert len(chunks) > 1
    for ck in chunks:
        assert len(ck.text) <= 400
        assert ck.hierarchy.dieu == "Điều 1"
        assert "_p" in ck.chunk_id.lower()