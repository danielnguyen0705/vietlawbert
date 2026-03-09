from __future__ import annotations

import re


def clean_legal_text(text: str) -> str:
    if not text:
        return ""

    # 1) Chuẩn hóa ký tự trắng cơ bản
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Chuẩn hóa khoảng trắng từng dòng
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 3) Bỏ nhiều dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4) Sửa các bullet bị dính: -Căn cứ -> - Căn cứ
    text = re.sub(r"(?m)^-\s*", "- ", text)

    # 5) Ghép lại "Điều" bị xuống dòng sai: Điều \n 1. -> Điều 1.
    text = re.sub(r"\bĐiều\s*\n\s*(\d+)\s*\.", r"Điều \1.", text)

    # 6) Ghép lại "Khoản", "Chương", "Mục" nếu bị vỡ dòng
    text = re.sub(r"\bKhoản\s*\n\s*(\d+)", r"Khoản \1", text)
    text = re.sub(r"\bChương\s*\n\s+([IVXLC]+)\b", r"Chương \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMục\s*\n\s*(\d+)\b", r"Mục \1", text)

    # 7) Ghép số hiệu văn bản bị tách dòng sau "số"
    # ví dụ: "Nghị định số\n34/2003/NĐ-CP" -> "Nghị định số 34/2003/NĐ-CP"
    text = re.sub(r"\bsố\s*\n\s*([0-9]+/[0-9]+/[A-ZĐ\-]+)", r"số \1", text, flags=re.IGNORECASE)

    # 8) Ghép các điểm a) b) c) nếu bị tách sai
    text = re.sub(r"\n([a-zđ])\)\s*", r"\n\1) ", text)

    # 9) Ghép dòng ngắn bị vỡ không hợp lý
    # Ý tưởng:
    # - nếu dòng trước không kết thúc bằng dấu câu mạnh
    # - và dòng sau không phải là tiêu đề cấu trúc mới
    # thì nối lại bằng 1 khoảng trắng
    structure_pat = re.compile(
        r"^(Điều\s+\d+\.?|Khoản\s+\d+|Chương\s+[IVXLC]+|Mục\s+\d+|[a-zđ]\)|- )",
        flags=re.IGNORECASE
    )

    merged_lines = []
    raw_lines = text.split("\n")

    for line in raw_lines:
        line = line.strip()
        if not line:
            if merged_lines and merged_lines[-1] != "":
                merged_lines.append("")
            continue

        if not merged_lines:
            merged_lines.append(line)
            continue

        prev = merged_lines[-1]

        should_merge = (
            prev != ""
            and not re.search(r"[.:;?!]$", prev)
            and not structure_pat.match(line)
        )

        if should_merge:
            merged_lines[-1] = prev + " " + line
        else:
            merged_lines.append(line)

    text = "\n".join(merged_lines)

    # 10) Làm sạch lần cuối
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text