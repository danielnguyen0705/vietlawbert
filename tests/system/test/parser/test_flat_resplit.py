"""Tier-1: văn bản 1-dòng (HTML phẳng) phải được re-split → vẫn bóc được >=2 Điều."""
from preprocess.ast_parser import HybridASTParser

# 2 điều dồn vào MỘT dòng, không có ký tự xuống dòng giữa chúng
FLAT = ("CHƯƠNG I QUY ĐỊNH CHUNG Điều 1. Phạm vi điều chỉnh "
        + ("Quy định này điều chỉnh các hoạt động nêu tại văn bản. " * 60)
        + "Điều 2. Đối tượng áp dụng "
        + ("Áp dụng với mọi tổ chức, cá nhân có liên quan theo quy định. " * 60))

META = {"doc_id": "D-TEST-FLAT", "doc_number": "99/2026/TT-TEST", "title": "VB phẳng"}


def test_flat_document_resplit_recovers_articles():
    parser = HybridASTParser()
    chunks = parser.parse_document(FLAT, META)
    arts = {c.get("article_id") for c in chunks if c.get("article_id")}
    assert len(arts) >= 2, f"Phải bóc được >=2 Điều từ văn bản 1-dòng, nhận {len(arts)}: {arts}"
    # không còn chunk nào nuốt cả 2 điều (độ dài hợp lý)
    assert all(len(str(c.get("macro_label", ""))) <= 200 for c in chunks)
