"""Unit test bộ lọc rác hành chính và chuẩn hóa siêu dữ liệu (Text Cleaner)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date,
)


def test_clean_boilerplate_preserves_enactment_clause():
    raw_text = """
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

Số: 100/2019/NĐ-CP
Hà Nội, ngày 30 tháng 12 năm 2019

##### Điều 82. Hiệu lực thi hành
Nghị định này có hiệu lực thi hành từ ngày 15 tháng 01 năm 2020.

Nơi nhận:
- Ban Bí thư Trung ương Đảng;
- Thủ tướng Chính phủ;

TM. CHÍNH PHỦ
THỦ TƯỚNG
(Đã ký)
Nguyễn Xuân Phúc
"""
    cleaned = clean_boilerplate(raw_text)
    assert "CỘNG HÒA XÃ HỘI" not in cleaned
    assert "Độc lập - Tự do" not in cleaned
    assert "TM. CHÍNH PHỦ" not in cleaned
    assert "Điều 82. Hiệu lực thi hành" in cleaned
    assert "Nghị định này có hiệu lực thi hành" in cleaned


def test_metadata_extraction():
    sample = "NGHỊ ĐỊNH\nSố: 123/2021/NĐ-CP\nCó hiệu lực từ ngày 01 tháng 01 năm 2022"
    assert extract_doc_type(sample) == "Nghị định"
    assert extract_doc_number(sample) == "123/2021/NĐ-CP"
    assert extract_effective_date(sample) == "2022-01-01"