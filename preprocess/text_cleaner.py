"""
text_cleaner.py - Bộ lọc nhiễu và trích xuất siêu dữ liệu hành chính cho văn bản pháp luật VN.
Khử boilerplate (quốc hiệu, tiêu ngữ, chữ ký, nơi nhận) mà không làm mất điều khoản thi hành.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ============================================================
# CÁC MẪU BIỂU THỨC CHÍNH QUY KHỬ NHIỄU (BOILERPLATE NOISE)
# ============================================================

# 1. Phần mở đầu hành chính (Header Patterns)
HEADER_PATTERNS = [
    re.compile(r"^\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*CHÍNH\s*PHỦ\s*[-–—|]*\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Độc\s*lập\s*[-–—]\s*Tự\s*do\s*[-–—]\s*Hạnh\s*phúc\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*{0,2}\s*(?:CHÍNH\s*PHỦ|QUỐC\s*HỘI|ỦY\s*BAN\s*THƯỜNG\s*VỤ\s*QUỐC\s*HỘI)\s*\*{0,2}\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\*{0,2}\s*(?:NGHỊ\s*ĐỊNH|QUYẾT\s*ĐỊNH|THÔNG\s*TƯ)\s*\*{0,2}\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Số:\s*[\d]+[-–/][\w\-/]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Hà\s*Nội,\s*ngày\s+\d+\s+tháng\s+\d+\s+năm\s+\d+", re.IGNORECASE | re.MULTILINE),
]

# 2. Khối chữ ký cuối văn bản (Signature Patterns)
SIGNATURE_PATTERNS = [
    re.compile(r"^\s*TM\.\s*(?:CHÍNH\s*PHỦ|THỦ\s*TƯỚNG|BỘ\s*TRƯỞNG|ỦY\s*BAN)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*KT\.\s*(?:THỦ\s*TƯỚNG|BỘ\s*TRƯỞNG|CHỦ\s*TỊCH)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*_\s*\(Đã\s*ký\)\s*_\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\(Đã\s*ký(?:\s*,\s*đóng\s*dấu)?\)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Ký\s*(?:như|ngày|tên)\s*[:\.]?", re.IGNORECASE | re.MULTILINE),
]

# 3. Phân vùng "Nơi nhận" (Giới hạn dòng quét an toàn, tránh nuốt nội dung điều luật)
NOI_NHAN_START_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?Nơi\s*nhận\s*[:\.]?", re.IGNORECASE)
STRUCTURAL_BOUNDARY_PATTERN = re.compile(r"^\s*(?:#{1,6}\s*|Điều\s+\d+|CHƯƠNG\s+[IVXLCDM\d]+|MỤC\s+\d+)", re.IGNORECASE)

# 4. Ký tự phân trang & rác định dạng Markdown
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:Trang|Page|P)\s*[:\.]?\s*\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE)
NUMERIC_PAGE_PATTERN = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
EMPTY_TABLE_PATTERN = re.compile(r"^\|[\s\-|:]*\|$")
MARKDOWN_RULE_PATTERN = re.compile(r"^\s*\\?[-–—*]{3,}\s*$")

# 5. Phần căn cứ pháp lý mở đầu (Preamble)
PREAMBLE_LINE_PATTERN = re.compile(r"^\s*_?(?:Căn\s*cứ|Theo\s+đề\s+nghị|Xét\s+đề\s+nghị|Để\s+tạo\s+nguồn)\b", re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa toàn bộ văn bản về chuẩn Unicode dựng sẵn (NFKC)."""
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def clean_boilerplate(text: str, keep_preamble: bool = False) -> str:
    """
    Loại bỏ siêu dữ liệu thừa và boilerplate khỏi văn bản pháp luật.
    Đảm bảo tuyệt đối không xóa nhầm các điều khoản quy định hiệu lực thi hành.
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    lines = text.split("\n")
    cleaned_lines = []
    in_noi_nhan = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Kiểm tra bắt đầu khối "Nơi nhận"
        if NOI_NHAN_START_PATTERN.match(stripped):
            in_noi_nhan = True
            continue

        # Thoát khỏi khối "Nơi nhận" khi gặp cấu trúc điều luật mới hoặc dòng kẻ kết thúc
        if in_noi_nhan:
            if STRUCTURAL_BOUNDARY_PATTERN.match(stripped) or MARKDOWN_RULE_PATTERN.match(stripped):
                in_noi_nhan = False
            elif stripped.startswith("-") or stripped.startswith("*") or len(stripped) < 80:
                # Vẫn đang trong danh sách cơ quan nhận
                continue
            else:
                in_noi_nhan = False

        # Kiểm tra mẫu Header
        if any(pat.search(stripped) for pat in HEADER_PATTERNS):
            continue

        # Kiểm tra mẫu Chữ ký
        if any(pat.search(stripped) for pat in SIGNATURE_PATTERNS):
            continue

        # Kiểm tra mẫu Số trang
        if PAGE_NUMBER_PATTERN.match(stripped) or NUMERIC_PAGE_PATTERN.match(stripped):
            continue

        # Kiểm tra rác định dạng bảng rỗng và đường kẻ ngang
        if EMPTY_TABLE_PATTERN.match(stripped) or MARKDOWN_RULE_PATTERN.match(stripped):
            continue

        # Kiểm tra phần căn cứ mở đầu (nếu cấu hình không giữ preamble)
        if not keep_preamble and PREAMBLE_LINE_PATTERN.match(stripped):
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    # Rút gọn các dòng trống liên tiếp
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_preamble(text: str) -> Optional[str]:
    """Trích xuất các dòng căn cứ pháp lý mở đầu phục vụ đối soát cạnh đồ thị Neo4j."""
    if not text:
        return None

    text = normalize_unicode(text)
    preamble_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if PREAMBLE_LINE_PATTERN.match(stripped):
            preamble_lines.append(stripped)

    return "\n".join(preamble_lines) if preamble_lines else None


def extract_doc_type(text: str) -> str:
    """Xác định loại văn bản quy phạm pháp luật theo Luật Ban hành VBQPPL."""
    text_upper = normalize_unicode(text).upper()

    doc_types = [
        ("HIẾN PHÁP", "Hiến pháp"),
        ("BỘ LUẬT", "Bộ luật"),
        ("LUẬT", "Luật"),
        ("NGHỊ QUYẾT LIÊN TỊCH", "Nghị quyết liên tịch"),
        ("NGHỊ QUYẾT", "Nghị quyết"),
        ("PHÁP LỆNH", "Pháp lệnh"),
        ("LỆNH", "Lệnh"),
        ("NGHỊ ĐỊNH", "Nghị định"),
        ("QUYẾT ĐỊNH", "Quyết định"),
        ("THÔNG TƯ LIÊN TỊCH", "Thông tư liên tịch"),
        ("THÔNG TƯ", "Thông tư"),
        ("CHỈ THỊ", "Chỉ thị"),
        ("QUY CHẾ", "Quy chế"),
        ("ĐIỀU LỆ", "Điều lệ"),
    ]

    for keyword, label in doc_types:
        # Kiểm tra từ khóa xuất hiện ở đầu dòng hoặc trong 500 ký tự đầu tiên
        pattern = rf"(?:^|\n|\b){keyword}\b"
        if re.search(pattern, text_upper[:1000]):
            return label

    return "Văn bản khác"


def extract_doc_number(text: str) -> Optional[str]:
    """Trích xuất số hiệu văn bản pháp luật (VD: 100/2019/NĐ-CP, 45/2019/QH14)."""
    if not text:
        return None

    text_clean = normalize_unicode(text)
    # Ưu tiên mẫu chuẩn dạng "Số: .../..."
    match = re.search(r"Số\s*:\s*([\d]+[-–/][\w\-/]+)", text_clean, re.IGNORECASE)
    if match:
        return match.group(1).replace(" ", "").upper()

    # Mẫu tìm kiếm trong tiêu đề văn bản
    match_alt = re.search(r"\b(?:số|số\s+hiệu)\s+([\d]+/[A-Z0-9\-]+)", text_clean, re.IGNORECASE)
    if match_alt:
        return match_alt.group(1).replace(" ", "").upper()

    return None


def extract_effective_date(text: str) -> Optional[str]:
    """
    Trích xuất ngày hiệu lực của văn bản và chuẩn hóa về chuỗi ISO YYYY-MM-DD.
    """
    if not text:
        return None

    text_clean = normalize_unicode(text)

    # 1. Tìm mẫu "ngày ... tháng ... năm ..."
    date_match = re.search(
        r"(?:có\s+hiệu\s+lực|hiệu\s+lực\s+thi\s+hành)\s+(?:từ\s+ngày|kể\s+từ\s+ngày|\b)\s*(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        text_clean,
        re.IGNORECASE,
    )
    if date_match:
        day, month, year = date_match.group(1), date_match.group(2), date_match.group(3)
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # 2. Tìm mẫu rút gọn "kể từ ngày ký / ban hành"
    if re.search(r"kể\s+từ\s+ngày\s+(?:ký|ban\s+hành|thông\s+qua)", text_clean, re.IGNORECASE):
        return "Kể từ ngày ban hành"

    return None