from __future__ import annotations

import re


STRUCTURE_START_PATTERNS = [
    r"^Phần\s+[IVXLC\d]+\.?$",
    r"^Chương\s+[IVXLC\d]+\.?$",
    r"^Mục\s+\d+\.?$",
    r"^Điều\s+\d+\.?$",
    r"^Khoản\s+\d+\.?$",
    r"^\d+\.$",
    r"^[a-zđ]\)$",
    r"^[A-ZĐ]\.$",
    r"^QUYẾT ĐỊNH:$",
    r"^NGHỊ ĐỊNH:$",
    r"^THÔNG TƯ:$",
    r"^CHỈ THỊ:$",
    r"^PHỤ LỤC.*$",
    r"^CHƯƠNG TRÌNH.*$",
    r"^QUY CHẾ.*$",
    r"^QUY ĐỊNH.*$",
]


def _normalize_line_spaces(line: str) -> str:
    line = line.replace("\xa0", " ").replace("\u200b", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _is_decorative_line(line: str) -> bool:
    """
    Dòng kiểu:
    _____________
    ---------------
    =================
    ____________________________
    """
    s = line.strip()
    return bool(re.fullmatch(r"[\s_\-—=]{5,}", s))


def _is_signature_marker(line: str) -> bool:
    s = line.strip().upper()
    markers = [
        "BỘ TRƯỞNG",
        "THỨ TRƯỞNG",
        "KT. BỘ TRƯỞNG",
        "TL. BỘ TRƯỞNG",
        "Q. TỔNG CỤC TRƯỞNG",
        "TỔNG CỤC TRƯỞNG",
        "CỤC TRƯỞNG",
        "(ĐÃ KÝ)",
    ]
    return any(m in s for m in markers)


def _is_structure_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    for pat in STRUCTURE_START_PATTERNS:
        if re.match(pat, s, flags=re.IGNORECASE):
            return True

    # các dòng bắt đầu bằng mốc cấu trúc + có nội dung ngay phía sau
    if re.match(
        r"^(Phần\s+[IVXLC\d]+|Chương\s+[IVXLC\d]+|Mục\s+\d+|Điều\s+\d+\.|Khoản\s+\d+|\d+\.|[a-zđ]\))\s+",
        s,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def clean_legal_text(text: str, remove_signature: bool = False) -> str:
    if not text:
        return ""

    # 1) Chuẩn hóa newline và khoảng trắng cơ bản
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line_spaces(line) for line in text.split("\n")]

    # 2) Bỏ dòng trống và dòng trang trí
    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        if _is_decorative_line(line):
            continue
        cleaned_lines.append(line)

    lines = cleaned_lines

    # 3) Optional: bỏ block chữ ký một cách thận trọng
    if remove_signature:
        tmp: list[str] = []
        skip_mode = False

        for line in lines:
            upper = line.upper()

            if _is_signature_marker(line):
                skip_mode = True
                continue

            if skip_mode:
                if (
                    upper.startswith("CHƯƠNG ")
                    or upper.startswith("MỤC ")
                    or upper.startswith("PHẦN ")
                    or upper.startswith("PHỤ LỤC")
                    or upper.startswith("QUY ĐỊNH")
                    or upper.startswith("CHƯƠNG TRÌNH")
                    or upper.startswith("ĐIỀU ")
                    or upper.startswith("KHOẢN ")
                    or upper.startswith("NƠI NHẬN")
                ):
                    skip_mode = False
                    tmp.append(line)
                else:
                    continue
            else:
                tmp.append(line)

        lines = tmp

    text = "\n".join(lines)

    # 4) Ghép mốc cấu trúc bị vỡ dòng
    text = re.sub(r"\bĐiều\s*\n\s*(\d+)\s*\.", r"Điều \1.", text, flags=re.IGNORECASE)
    text = re.sub(r"\bChương\s*\n\s*([IVXLC]+|\d+)\.?", r"Chương \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMục\s*\n\s*(\d+)\.?", r"Mục \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPhần\s*\n\s*([IVXLC]+|\d+)\.?", r"Phần \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bKhoản\s*\n\s*(\d+)\.?", r"Khoản \1", text, flags=re.IGNORECASE)

    # 5) Ghép Điều với tên điều nếu bị tách
    # Điều 1.\nPhạm vi điều chỉnh -> Điều 1. Phạm vi điều chỉnh
    text = re.sub(r"(?m)^(Điều\s+\d+\.)\s*\n([^\n]+)", r"\1 \2", text, flags=re.IGNORECASE)

    # 6) Ghép khoản số bị vỡ, nhưng chỉ khi dòng sau không phải cấu trúc lớn
    # 1.\nNội dung -> 1. Nội dung
    text = re.sub(
        r"(?m)^(\d+\.)\s*\n(?!Chương\s+|Mục\s+|Phần\s+|Điều\s+|Khoản\s+)([^\n]+)",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    # 7) Ghép điểm chữ bị vỡ
    # a)\nNội dung -> a) Nội dung
    text = re.sub(
        r"(?mi)^([a-zđ]\))\s*\n(?!Chương\s+|Mục\s+|Phần\s+|Điều\s+|Khoản\s+)([^\n]+)",
        r"\1 \2",
        text,
    )

    # 8) Ghép các số hiệu văn bản bị vỡ sau chữ "số"
    text = re.sub(
        r"\bsố\s*\n\s*([0-9]+/[0-9]+/[A-ZĐa-z0-9\-.]+)",
        r"số \1",
        text,
        flags=re.IGNORECASE,
    )

    # 9) Sửa bullet bị dính
    text = re.sub(r"(?m)^-\s*", "- ", text)

    # 10) Tách lại các mốc lớn nếu vô tình dính chung dòng
    text = re.sub(r"\s+(Chương\s+[IVXLC]+|\bChương\s+\d+\b)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Mục\s+\d+)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Phần\s+[IVXLC]+|\bPhần\s+\d+\b)", r"\n\1", text, flags=re.IGNORECASE)

    # chỉ tách Điều nếu trước đó có nhiều khoảng trắng bất thường
    text = re.sub(r"[ \t]{2,}(Điều\s+\d+\.)", r"\n\1", text, flags=re.IGNORECASE)

    # 11) Dọn dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text