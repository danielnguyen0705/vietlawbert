"""Tier-1: làm sạch/phân loại doc_number bẩn (25 ví dụ thật từ kho). 3 nhóm:
VALID (phục hồi số thật) · NO_NUMBER ('Không số' không gộp) · CONTAMINATED (rò rỉ tiêu đề)."""
from configs.global_id import classify_doc_number, canonical_doc_id, normalize_doc_number


def _status(x): return classify_doc_number(x)[0]


def test_contaminated_prefix_recovered_to_same_canonical():
    # số thật bị dính 'Số:', 'số', 'TT', 'VB tạm nhập', 'bản nháp .sua' → phải khớp bản sạch
    pairs = [
        ("Sô: 28 /2016/QĐ-UBND", "28/2016/QĐ-UBND"),
        ("số 88/2025/QĐ-UBND",   "88/2025/QĐ-UBND"),
        ("TT 14/2023/TT-BNNPTNT","14/2023/TT-BNNPTNT"),
        ("VB tạm nhập 05/2008/TTLT-BKHCN-BNV", "05/2008/TTLT-BKHCN-BNV"),
        ("ban nhap 192/NQ-HĐND.sua", "192/NQ-HĐND"),
    ]
    for dirty, clean in pairs:
        assert _status(dirty) == "VALID", f"{dirty} phải VALID"
        assert canonical_doc_id(dirty) == canonical_doc_id(clean), f"{dirty} != {clean}"


def test_no_number_not_collapsed():
    # 'Không số' là văn bản THẬT nhưng không số → canonical trả "" (caller cấp khóa riêng), KHÔNG gộp
    for x in ["Không số", "KhôngSố/2007/QĐ-UBND", "KhôngSố/QĐ-UBND",
              "KhôngSố/UBND-XDCB", "KhôngSố_không có nội dung"]:
        assert _status(x) == "NO_NUMBER", f"{x} phải NO_NUMBER"
        assert canonical_doc_id(x) == "", f"{x} KHÔNG được sinh canonical gộp"


def test_title_leak_is_contaminated():
    for x in ["HĐND khóa 13 kỳ họp bất thường",
              "HĐND khóa 13 kỳ họp bất thường (1)",
              "QĐ Về việc áp dụng văn bản quy phạm pháp luật do Ủy ban nhân dân tỉnh Quảng Bình ban hành"]:
        assert _status(x) == "CONTAMINATED", f"{x} phải CONTAMINATED"
        assert canonical_doc_id(x) == "", f"{x} KHÔNG được làm khóa"


def test_legit_old_numbers_still_valid():
    # số hiệu cũ/đặc thù hợp lệ vẫn giữ (không bị diệt oan)
    for x in ["001/CA", "001/SL", "002-LB", "01 - TC/TCCB", "88/2025/QĐ-UBND"]:
        assert _status(x) == "VALID", f"{x} phải VALID"
        assert canonical_doc_id(x) != "", f"{x} phải có canonical"


def test_clean_number_unchanged():
    # không phá hành vi cũ với số sạch
    assert canonical_doc_id("72/2026/QĐ-UBND") == canonical_doc_id("72/2026/QD-UBND")
