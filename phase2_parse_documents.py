# law_dataset/phase2_parse_documents.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from law_dataset.utils.extractor import (
    fetch_html,
    extract_document_from_html,
    extract_item_id,
)

from law_dataset.utils.cleaner import clean_legal_text
from law_dataset.utils.io_utils import (
    slugify_filename,
    write_text_file,
    append_jsonl,
)


BASE_DIR = Path(__file__).resolve().parent
LAW_DATASET_DIR = BASE_DIR / "law_dataset" 
DATA_DIR = LAW_DATASET_DIR / "data"

RAW_LINKS_PATH = DATA_DIR / "raw_links.json"

RAW_DOCS_JSONL = DATA_DIR / "raw_documents.jsonl"
CLEANED_DOCS_JSONL = DATA_DIR / "cleaned_documents.jsonl"

PARSE_ERRORS_JSONL = DATA_DIR / "logs" / "parse_errors.jsonl"

RAW_TEXT_DIR = DATA_DIR / "docs" / "raw"
CLEANED_TEXT_DIR = DATA_DIR / "docs" / "cleaned"


def main():

    if not RAW_LINKS_PATH.exists():
        raise FileNotFoundError("raw_links.json not found")

    with RAW_LINKS_PATH.open("r", encoding="utf-8") as f:
        links = json.load(f)

    print(f"Total links: {len(links)}")

    for idx, item in enumerate(links, start=1):

        url = item["url"]
        doc_id_from_links = (item.get("doc_id") or "").strip()

        retrieved_at = datetime.now().astimezone().isoformat()

        try:

            print(f"[{idx}] Fetching {url}")

            html = fetch_html(url)

            parsed = extract_document_from_html(
                html,
                url=url,
                fallback_doc_id=doc_id_from_links,
            )

            doc_id = parsed.get("doc_id") or doc_id_from_links or "unknown_document"
            title = parsed.get("title", "")
            effective_date = parsed.get("effective_date", "")
            raw_text = (parsed.get("raw_text") or "").strip()

            if not raw_text:
                raise ValueError("raw_text empty")

            safe_name = slugify_filename(doc_id)

            raw_file_name = f"{safe_name}.txt"

            raw_text_path = RAW_TEXT_DIR / raw_file_name
            cleaned_text_path = CLEANED_TEXT_DIR / raw_file_name

            write_text_file(raw_text_path, raw_text)

            item_id = extract_item_id(url)
            source_id = f"vbpl_{item_id}" if item_id else ""

            raw_record = {
                "source_id": source_id,
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "type": item.get("type", ""),
                "ministry": item.get("ministry", ""),
                "effective_date": effective_date,
                "raw_text_path": str(raw_text_path.relative_to(BASE_DIR)),
                "retrieved_at": retrieved_at,
                "parse_status": "success",
            }

            append_jsonl(RAW_DOCS_JSONL, raw_record)

            cleaned_text = clean_legal_text(raw_text)

            write_text_file(cleaned_text_path, cleaned_text)

            cleaned_record = {
                "source_id": source_id,
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "type": item.get("type", ""),
                "ministry": item.get("ministry", ""),
                "effective_date": effective_date,
                "cleaned_text_path": str(cleaned_text_path.relative_to(BASE_DIR)),
                "retrieved_at": retrieved_at,
                "parse_status": "success",
            }

            append_jsonl(CLEANED_DOCS_JSONL, cleaned_record)

            print(f"Parsed: {doc_id}")

        except Exception as e:

            print(f"ERROR: {doc_id_from_links} -> {e}")

            error_record = {
                "doc_id": doc_id_from_links,
                "url": url,
                "type": item.get("type", ""),
                "ministry": item.get("ministry", ""),
                "error": str(e),
                "retrieved_at": retrieved_at,
            }

            append_jsonl(PARSE_ERRORS_JSONL, error_record)


if __name__ == "__main__":
    main()