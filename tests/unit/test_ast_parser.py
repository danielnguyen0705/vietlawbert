"""Unit test bộ bóc tách cú pháp phân cấp pháp lý lai (Hybrid AST Parser)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.ast_parser import HybridASTParser, LegalHierarchyTracker


def test_hierarchy_tracker_state():
    tracker = LegalHierarchyTracker()
    tracker.current_part = "Phần I"
    tracker.current_chapter = "Chương II"
    tracker.current_article = "12"
    tracker.current_article_title = "Xử phạt vi phạm"

    path = tracker.get_hierarchy_path(clause="1", point="a")
    assert path == "Phần I > Chương II > Điều 12 (Xử phạt vi phạm) > Khoản 1 > Điểm a"
    assert tracker.get_macro_label() == "Chương II"


def test_ast_parser_structured_document():
    raw_legal_text = """
Phần I: QUY ĐỊNH CHUNG
Chương I: PHẠM VI VÀ ĐỐI TƯỢNG
Điều 1. Phạm vi điều chỉnh
Văn bản này quy định về xử lý vi phạm hành chính trong lĩnh vực bảo vệ môi trường.
Điều 2. Đối tượng áp dụng
1. Cá nhân, tổ chức có hành vi vi phạm.
a) Cá nhân trong nước và nước ngoài;
b) Doanh nghiệp nhà nước.
2. Cơ quan có thẩm quyền xử phạt.
"""
    metadata = {
        "doc_id": "TEST_DOC_01",
        "doc_number": "99/2026/NĐ-CP",
        "title": "Nghị định quy định xử phạt môi trường",
        "effective_date": "2026-05-01",
        "status": "Còn hiệu lực",
        "co_quan": "Chính phủ"
    }

    parser = HybridASTParser()
    chunks = parser.parse_document(raw_legal_text, metadata)

    assert len(chunks) >= 3
    # Kiểm tra metadata injection
    first_chunk = chunks[0]
    assert "[META]" in first_chunk["text"]
    assert "99/2026/NĐ-CP" in first_chunk["text"]
    assert "[HIERARCHY]" in first_chunk["text"]
    assert "[CONTENT]" in first_chunk["text"]

    # Kiểm tra nhãn phân cấp vĩ mô cho hàm mất mát MRL d=64
    assert any(c["macro_label"] == "Chương I" for c in chunks)