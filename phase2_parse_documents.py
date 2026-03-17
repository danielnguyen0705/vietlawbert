from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from law_dataset.utils.extractor import (
    fetch_html,
    extract_document_from_html,
    canonicalize_agency_name,
    canonicalize_agency_code,
)
from law_dataset.utils.cleaner import clean_legal_text
from law_dataset.utils.io_utils import slugify_filename, write_text_file, append_jsonl

BASE_DIR = Path(__file__).resolve().parent
LAW_DATASET_DIR = BASE_DIR / "law_dataset"
DATA_DIR = LAW_DATASET_DIR / "data"

RAW_LINKS_PATH = DATA_DIR / "raw_links.json"
DOCUMENTS_JSONL = DATA_DIR / "documents.jsonl"
PARSE_ERRORS_JSONL = DATA_DIR / "logs" / "parse_errors.jsonl"
RAW_TEXT_DIR = DATA_DIR / "docs" / "raw"
CLEANED_TEXT_DIR = DATA_DIR / "docs" / "cleaned"

TITLE_INVALID_PATTERNS = [
    r"^CỦA\s+(BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|BỘ|ỦY BAN|UỶ BAN|TỔNG CỤC|CỤC)\b",
    r"^(BỘ TRƯỞNG|THỨ TRƯỞNG|CHỦ TỊCH|TỔNG CỤC TRƯỞNG|CỤC TRƯỞNG)\b",
    r"^BAN\s+HÀNH\s+KÈM\s+THEO\b",
    r"^KÈM\s+THEO\b",
    r"^CHƯƠNG\s+[IVXLC0-9]+\b",
    r"^MỤC\s+[IVXLC0-9]+\b",
    r"^PHẦN\s+[IVXLC0-9]+\b",
    r"^ĐIỀU\s+\d+\.?\b",
]

TITLE_LEADS = [
    "VỀ VIỆC", "BAN HÀNH", "QUY ĐỊNH", "QUY CHẾ", "HƯỚNG DẪN", "PHÊ DUYỆT", "CÔNG BỐ", "SỬA ĐỔI", "BỔ SUNG"
]


def _safe_str(value: object) -> str:
    return str(value or "").strip()


def _clean_spaces(text: str) -> str:
    text = _safe_str(text)
    text = text.replace("\\", " ").replace("/", " ")
    text = text.replace("\u00ad", " ").replace("\xa0", " ").replace("\u200b", " ")
    text = text.replace('"', '“')
    text = re.sub(r"[_\-=—–]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_bad_title(title: str) -> bool:
    s = _clean_spaces(title)
    if not s or len(s) < 5:
        return True
    upper = s.upper()
    if any(re.match(p, upper, flags=re.IGNORECASE) for p in TITLE_INVALID_PATTERNS):
        return True
    if re.fullmatch(r"[_\-=—–\s.]+", s):
        return True
    if re.search(r"\bCHƯƠNG\s+[IVXLC0-9]+\b", upper):
        return True
    if re.search(r"\b(BỘ TRƯỞNG|BỘ GIAO THÔNG VẬN TẢI|BỘ CÔNG AN|CỦA BỘ TRƯỞNG)\b", upper):
        if not any(word in upper for word in TITLE_LEADS):
            return True
    return False


def _derive_title_fallback(raw_text: str) -> str:
    lines = [_clean_spaces(line) for line in _safe_str(raw_text).splitlines()]
    lines = [line for line in lines if line]
    candidates: list[str] = []
    for line in lines[:40]:
        upper = line.upper()
        if _looks_like_bad_title(line):
            continue
        if upper in {"QUYẾT ĐỊNH", "NGHỊ ĐỊNH", "THÔNG TƯ", "CHỈ THỊ", "NGHỊ QUYẾT", "LUẬT", "BỘ LUẬT"}:
            continue
        if len(line) > 300:
            continue
        if re.search(r"\b(CĂN CỨ|THEO ĐỀ NGHỊ|NƠI NHẬN|KT\.|TM\.)\b", upper):
            continue
        if any(lead in upper for lead in TITLE_LEADS):
            candidates.append(line)
            if len(candidates) >= 2:
                break
    title = _clean_spaces(" ".join(candidates[:2]))
    return title


def normalize_title(title: str, raw_text: str) -> str:
    title = _clean_spaces(title)
    if not _looks_like_bad_title(title):
        return title
    fallback = _derive_title_fallback(raw_text)
    return fallback if fallback and not _looks_like_bad_title(fallback) else title


def normalize_agencies(parsed: dict, item: dict) -> tuple[list[str], list[str], str]:
    codes = []
    for code in parsed.get("agency_codes") or []:
        c = canonicalize_agency_code(code)
        if c:
            codes.append(c)
    item_ministry = _safe_str(item.get("ministry"))
    if item_ministry:
        c = canonicalize_agency_code(item_ministry)
        if c:
            codes.append(c)
    codes = list(dict.fromkeys(codes))
    names = [canonicalize_agency_name(code) for code in codes]
    ministry = " - ".join(dict.fromkeys(names))
    return codes, names, ministry


def build_file_stem(item_id: str, doc_type: str, doc_number: str, title: str, fallback_doc_id: str) -> str:
    if doc_type and doc_number:
        base = f"{doc_type} {doc_number}"
    elif title:
        base = title
    elif fallback_doc_id:
        base = fallback_doc_id
    elif item_id:
        base = item_id
    else:
        base = "unknown_document"
    return slugify_filename(base)


def build_document_record(*, item: dict, parsed: dict, raw_text_path: Path, cleaned_text_path: Path, retrieved_at: str) -> dict:
    agency_codes, agencies, ministry = normalize_agencies(parsed, item)
    return {
        "id": _safe_str(parsed.get("id")),
        "doc_type": _safe_str(parsed.get("doc_type")),
        "doc_number": _safe_str(parsed.get("doc_number")),
        "title": _safe_str(parsed.get("title")),
        "url": _safe_str(parsed.get("url")) or _safe_str(item.get("url")),
        "domain": _safe_str(item.get("type")),
        "ministry": ministry,
        "agency_codes": agency_codes,
        "agencies": agencies,
        "raw_text_path": str(raw_text_path.relative_to(BASE_DIR)),
        "cleaned_text_path": str(cleaned_text_path.relative_to(BASE_DIR)),
        "retrieved_at": retrieved_at,
        "parse_status": "success",
    }


def main():
    if not RAW_LINKS_PATH.exists():
        raise FileNotFoundError(f"raw_links.json not found: {RAW_LINKS_PATH}")

    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    PARSE_ERRORS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with RAW_LINKS_PATH.open("r", encoding="utf-8") as f:
        links = json.load(f)

    print(f"Total links: {len(links)}")

    for idx, item in enumerate(links, start=1):
        url = _safe_str(item.get("url"))
        doc_id_from_links = _safe_str(item.get("doc_id"))
        retrieved_at = datetime.now().astimezone().isoformat()

        try:
            print(f"[{idx}] Fetching {url}")
            html = fetch_html(url)
            parsed = extract_document_from_html(html, url=url, fallback_doc_id=doc_id_from_links)

            item_id = _safe_str(parsed.get("id"))
            doc_type = _safe_str(parsed.get("doc_type"))
            doc_number = _safe_str(parsed.get("doc_number")) or doc_id_from_links
            raw_text = _safe_str(parsed.get("raw_text"))
            if not raw_text:
                raise ValueError("raw_text empty")

            parsed["doc_number"] = doc_number
            parsed["title"] = normalize_title(_safe_str(parsed.get("title")), raw_text)
            title = _safe_str(parsed.get("title"))

            file_stem = build_file_stem(item_id, doc_type, doc_number, title, doc_id_from_links)
            file_name = f"{file_stem}.txt"
            raw_text_path = RAW_TEXT_DIR / file_name
            cleaned_text_path = CLEANED_TEXT_DIR / file_name

            cleaned_text = clean_legal_text(raw_text)
            write_text_file(raw_text_path, raw_text)
            write_text_file(cleaned_text_path, cleaned_text)

            record = build_document_record(
                item=item,
                parsed=parsed,
                raw_text_path=raw_text_path,
                cleaned_text_path=cleaned_text_path,
                retrieved_at=retrieved_at,
            )
            append_jsonl(DOCUMENTS_JSONL, record)

        except Exception as e:
            err = {
                "url": url,
                "doc_id": doc_id_from_links,
                "retrieved_at": retrieved_at,
                "error": repr(e),
            }
            append_jsonl(PARSE_ERRORS_JSONL, err)
            print(f"[ERROR] {url} -> {e}")

    print("Done.")
    print(f"Documents JSONL: {DOCUMENTS_JSONL}")
    print(f"Parse errors JSONL: {PARSE_ERRORS_JSONL}")


if __name__ == "__main__":
    main()
