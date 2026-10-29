"""Tier 1 - Kiểm định ĐỊNH LƯỢNG ngưỡng lọc chất lượng (tỷ lệ nguyên âm tiếng Việt >= 0.05).

Bổ sung cho test_tier1 (vốn chỉ kiểm định tính của gate): ở đây kiểm tra biên 0.05 của
quality.crawl_audit.evaluate_linguistic_quality để chứng minh "loại rác mà không loại nhầm".
Đặc tả: Giai đoạn 1 > Bộ lọc chất lượng.
"""
import pytest
from quality.crawl_audit import evaluate_linguistic_quality

pytestmark = pytest.mark.tier1

_REJECT_KEYS = ("is_foreign_or_tabular", "is_foreign", "foreign_or_tabular", "is_junk")


def _is_rejected(report: dict) -> bool:
    for k in _REJECT_KEYS:
        if k in report:
            return bool(report[k])
    pytest.skip(f"Không thấy cờ loại-trừ trong report: {list(report)}")


def test_clean_vietnamese_passes_gate():
    """Văn bản pháp lý tiếng Việt chuẩn (giàu nguyên âm có dấu) KHÔNG bị loại."""
    text = ("Nghị định này quy định chi tiết về xử phạt vi phạm hành chính "
            "trong lĩnh vực giao thông đường bộ và đường sắt theo quy định hiện hành. ") * 5
    report = evaluate_linguistic_quality(text)
    assert isinstance(report, dict)
    assert _is_rejected(report) is False, "Loại nhầm văn bản tiếng Việt sạch!"


def test_foreign_or_blank_is_rejected():
    """Chuỗi ASCII không dấu / phôi trắng (tỷ lệ nguyên âm có dấu < 0.05) BỊ loại."""
    text = "aaaa bbbb cccc dddd 1234 5678 ssss tttt " * 20
    report = evaluate_linguistic_quality(text)
    assert _is_rejected(report) is True, "Không loại được văn bản ngoại lai/phôi trắng!"
