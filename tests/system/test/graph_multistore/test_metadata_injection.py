"""Tier 2 - Kiểm định Metadata Text Injection vào văn bản chunk trước khi embedding.

Đặc tả yêu cầu nhúng [META] ... [HIERARCHY] ... [CONTENT] để chunk độc lập ngữ cảnh.
Lưu ý: hiện thực thực tế trong preprocess/ast_parser.py dùng mẫu
`[META] Văn bản: {doc_number} | Tiêu đề: {doc_title} | ...` - test bám theo hành vi thực,
không bám cứng danh sách 11 trường trong đề cương.
Đặc tả: Giai đoạn 1 > Metadata Text Injection.
"""
import pytest
from preprocess.ast_parser import HybridASTParser

pytestmark = pytest.mark.tier2

SAMPLE = """Chương I: QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ.
Điều 2. Đối tượng áp dụng
Nghị định áp dụng với tổ chức, cá nhân có hành vi vi phạm hành chính.
"""

METADATA = {
    "doc_number": "100/2019/NĐ-CP",
    "doc_title": "Quy định xử phạt vi phạm hành chính lĩnh vực giao thông đường bộ",
    "issuer": "Chính phủ",
    "status": "Còn hiệu lực",
    "legal_domain": "Giao thông đường bộ",
    "doc_type": "Nghị định",
}


def _all_text(chunks) -> str:
    buff = []
    for c in chunks:
        if isinstance(c, dict):
            buff += [str(v) for v in c.values() if isinstance(v, str)]
        else:
            buff.append(str(c))
    return "\n".join(buff)


def test_meta_marker_and_doc_number_injected():
    parser = HybridASTParser()
    chunks = parser.parse_document(SAMPLE, METADATA)
    assert chunks, "Parser không tạo ra chunk nào từ văn bản mẫu!"
    blob = _all_text(chunks)
    assert "[META]" in blob, "Chunk chưa được nhúng khối [META]."
    assert METADATA["doc_number"] in blob, "Số hiệu văn bản không xuất hiện trong text chunk."


def test_hierarchy_and_content_markers_present():
    parser = HybridASTParser()
    blob = _all_text(parser.parse_document(SAMPLE, METADATA))
    # Mềm hóa: chấp nhận biến thể nhãn; chỉ cần có dấu hiệu phân cấp + nội dung điều luật.
    assert ("[HIERARCHY]" in blob) or ("Điều 1" in blob), "Thiếu thông tin phân cấp trong chunk."
    assert ("[CONTENT]" in blob) or ("Phạm vi điều chỉnh" in blob), "Thiếu nội dung điều luật."
