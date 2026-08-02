import re
from typing import Optional


# ============================================================
# PATTERNS LOẠI BỎ BOILERPLATE TRONG VĂN BẢN PHÁP LUẬT VN
# ============================================================

# 1. Header patterns (mở đầu văn bản)
HEADER_PATTERNS = [
    # Quốc hiệu + khẩu hiệu
    r'CỘNG\s*HOÀ\s*Xã\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM',
    r'Độc\s*lập\s*[-–—]\s*Tự\s*do\s*[-–—]\s*Hạnh\s*phúc',
    # Cơ quan ban hành (dòng đầu)
    r'^(?:CHÍNH\s*PHỦ|QUỐC\s*HỘI|ỦY\s*BAN\s*THƯỜNG\s*VỤ\s*QUỐC\s*HỘI)\s*$',
    # Số hiệu + ngày ban hành (dòng riêng)
    r'^Số:\s*[\d]+[-–/][\w\-/]+',
    r'^Hà\s*Nội,\s*ngày\s+\d+\s+tháng\s+\d+\s+năm\s+\d+',
]

# 2. Signature block patterns (chữ ký, ký tên)
SIGNATURE_PATTERNS = [
    r'TM\.\s*(?:CHÍNH\s*PHỦ|THỦ\s*TƯỚNG|BỘ\s*TRƯỞNG)',
    r'KT\.\s*(?:THỦ\s*TƯỚNG\s*CHÍNH\s*PHỦ|THỦ\s*TƯỚNG|BỘ\s*TRƯỞNG)',
    r'Ký\s*(?:như|ngày|tên)',
    r'\(Đã\s*ký\)',
    r'_\(Đã\s*ký\)_',
    r'_\s*\(Đã\s*ký\)\s*_',
    # Tên người ký (thường nằm cuối dòng có "Đã ký")
    r'(?:Phó?\s*(?:Thủ\s*Tướng|Bộ\s*Trưởng|Chủ\s*tịch)[\s\S]{0,30}?\(Đã\s*ký\))',
]

# 3. "Nơi nhận" section
NOI_NHAN_PATTERN = r'Nơi\s*nhận\s*[:\.]?\s*[\s\S]*?(?=\n(?:#####|\*\*|$))'

# 4. Page number patterns
PAGE_NUMBER_PATTERNS = [
    r'(?:Trang|Page|P)\s*[:\.]?\s*\d+',
    r'\b\d+\s*/\s*\d+\b',  #页码格式: 5/10
]

# 5. Markdown artifacts
MARKDOWN_ARTIFACTS = [
    r'\|[\s\-|]*\|',  # Bảng trống |---|---|
    r'^[\-\*]{3,}\s*$',  # Đường kẻ ngang ---
    r'\*{3,}',  # ****
    r'^\s*\|\s*$',  # Dòng |
]

# 6. Legal preamble patterns (Căn cứ..., Theo đề nghị...)
PREAMBLE_PATTERNS = [
    r'_Căn\s*cứ\s+[\s\S]*?;',  # Căn cứ...;
    r'_Theo\s+(?:đề\s+nghị|xét\s+đề\s+nghị)[\s\S]*?;',  # Theo đề nghị...;
    r'_Để\s+tạo\s+nguồn[\s\S]*?;',  # Để tạo nguồn...;
    r'_Xét\s+đề\s+nghị[\s\S]*?;',  # Xét đề nghị...;
]

# 7. "Hiệu lực thi hành" section (giữ lại, không xóa)
# 8. "Điều khoản thi hành" section header


def clean_boilerplate(text: str, keep_preamble: bool = False) -> str:
    """
    Loại bỏ boilerplate noise khỏi văn bản pháp luật.

    Args:
        text: Văn bản đầu vào
        keep_preamble: Nếu True, giữ lại phần "Căn cứ..." (mục đích cho metadata)
    Returns:
        Văn bản đã được làm sạch
    """
    lines = text.split('\n')
    cleaned_lines = []
    in_noi_nhan = False

    for line in lines:
        stripped = line.strip()

        # Bỏ qua dòng trống
        if not stripped:
            cleaned_lines.append('')
            continue

        # Kiểm tra "Nơi nhận" section
        if re.match(r'Nơi\s*nhận', stripped, re.IGNORECASE):
            in_noi_nhan = True
            continue
        if in_noi_nhan:
            # Kết thúc khi gặp header mới hoặc Điều
            if re.match(r'(?:#####|CHƯƠNG|MỤC|TIỂU\s*MỤC)', stripped):
                in_noi_nhan = False
            else:
                continue

        # Kiểm tra các patterns
        skip = False

        # Header patterns
        for pattern in HEADER_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                skip = True
                break

        # Signature patterns
        if not skip:
            for pattern in SIGNATURE_PATTERNS:
                if re.search(pattern, stripped, re.IGNORECASE):
                    skip = True
                    break

        # Page numbers
        if not skip:
            for pattern in PAGE_NUMBER_PATTERNS:
                if re.search(pattern, stripped, re.IGNORECASE):
                    skip = True
                    break

        # Markdown artifacts
        if not skip:
            for pattern in MARKDOWN_ARTIFACTS:
                if re.match(pattern, stripped):
                    skip = True
                    break

        # Preamble (optional)
        if not skip and not keep_preamble:
            for pattern in PREAMBLE_PATTERNS:
                if re.search(pattern, stripped, re.IGNORECASE):
                    skip = True
                    break

        if not skip:
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)

    # Xóa nhiều dòng trống liên tiếp
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def extract_preamble(text: str) -> Optional[str]:
    """
    Trích xuất phần preamble (Căn cứ..., Theo đề nghị...)
    để sử dụng cho metadata.cross_references.
    """
    preamble_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if re.match(r'_?Căn\s*cứ\s+', stripped, re.IGNORECASE):
            preamble_lines.append(stripped)
        elif re.match(r'_?(?:Theo|Xét)\s+(?:đề\s+nghị)', stripped, re.IGNORECASE):
            preamble_lines.append(stripped)
        elif re.match(r'_?Để\s+tạo\s+nguồn', stripped, re.IGNORECASE):
            preamble_lines.append(stripped)

    return '\n'.join(preamble_lines) if preamble_lines else None


def extract_doc_type(text: str) -> str:
    """
    Trích xuất loại văn bản pháp luật từ nội dung.
    """
    text_upper = text.upper()

    # Kiểm tra theo thứ tự ưu tiên (loại văn bản cụ thể trước)
    doc_types = [
        ('THÔNG TƯ LIÊN TỊCH', 'Thông tư liên tịch'),
        ('NGHỊ ĐỊNH', 'Nghị định'),
        ('QUYẾT ĐỊNH', 'Quyết định'),
        ('Nghị quyết', 'Nghị quyết'),
        ('THÔNG TƯ', 'Thông tư'),
        ('CHỈ THỊ', 'Chỉ thị'),
        ('QUY CHẾ', 'Quy chế'),
        ('ĐIỀU LỆ', 'Điều lệ'),
        ('PHÁP LỆNH', 'Pháp lệnh'),
        ('LUẬT', 'Luật'),
    ]

    for keyword, doc_type in doc_types:
        if keyword in text_upper:
            return doc_type

    return 'Văn bản khác'


def extract_doc_number(text: str) -> Optional[str]:
    """
    Trích xuất số hiệu văn bản.
    """
    match = re.search(r'Số:\s*([\d]+[-–/][\w\-/]+)', text)
    if match:
        return match.group(1)

    # Tìm số hiệu trong tiêu đề
    match = re.search(r'(?:số|Số)\s+([\d]+[-–/][\w\-/]+)', text)
    if match:
        return match.group(1)

    return None


def extract_effective_date(text: str) -> Optional[str]:
    """
    Trích xuất ngày hiệu lực.
    """
    # Tìm "có hiệu lực thi hành từ ngày..."
    match = re.search(
        r'(?:có\s+hiệu\s+lực|hiệu\s+lực\s+thi\s+hành)\s+(?:từ|sau)\s+(?:ngày\s+)?(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})',
        text, re.IGNORECASE
    )
    if match:
        return match.group(1)

    # Tìm "kể từ ngày ban hành"
    match = re.search(
        r'kể\s+từ\s+ngày\s+(?:ban\s+hành|ký|đăng\s+Công\s+báo)',
        text, re.IGNORECASE
    )
    if match:
        return 'kể từ ngày ban hành'

    return None


# ============================================================
# TEST FUNCTION
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_cleaner.py <file.md>")
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        text = f.read()

    print("=" * 60)
    print("TRƯỚC KHI LÀM SẠCH:")
    print("=" * 60)
    print(text[:500])
    print("...")

    cleaned = clean_boilerplate(text)

    print("\n" + "=" * 60)
    print("SAU KHI LÀM SẠCH:")
    print("=" * 60)
    print(cleaned[:500])
    print("...")

    print("\n" + "=" * 60)
    print("METADATA EXTRACTED:")
    print("=" * 60)
    print(f"Loại VBPL: {extract_doc_type(text)}")
    print(f"Số hiệu: {extract_doc_number(text)}")
    print(f"Ngày hiệu lực: {extract_effective_date(text)}")
    print(f"Preamble: {extract_preamble(text)[:200] if extract_preamble(text) else 'N/A'}")
