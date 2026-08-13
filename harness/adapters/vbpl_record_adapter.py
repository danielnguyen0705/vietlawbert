from __future__ import annotations

from typing import Any

KNOWN_NATIVE_STATUSES = {
    "HTML_VALID",
    "CONTENT_TOO_SHORT",
    "PDF_ONLY",
    "METADATA_ONLY",
    "SECURITY_CHALLENGE",
    "HTTP_ERROR",
    "CONTRACT_ERROR",
    "PARSE_ERROR",
}

KNOWN_WARNING_CODES = {
    "MISSING_OFFICIAL_PARENT_GROUP",
    "MISSING_DOCUMENT_NUMBER_FOR_GROUP_LOOKUP",
    "OFFICIAL_GROUP_NOT_FOUND",
    "DUPLICATE_ID_IN_GROUP_LOOKUP",
    "DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED",
    "OFFICIAL_GROUP_ADMINISTRATIVE_FLAG_CONFLICT",
    "CONTENT_RECOVERED_FROM_PDF",
    "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
    "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
    "PDF_EXTRACTION_INSUFFICIENT",
    "PDF_FETCH_FAILED",
}


def _normalized_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _normalized_warning_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = _normalized_string(item)
        if text:
            normalized.append(text)
    return normalized


def _normalized_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _content_text(record: dict[str, Any]) -> str:
    value = record.get("content_html")
    return value if isinstance(value, str) else ""


def _content_sha256(record: dict[str, Any]) -> str | None:
    value = record.get("content_sha256")
    return _normalized_string(value)


def _content_chars(record: dict[str, Any]) -> int | None:
    value = record.get("content_chars")
    if isinstance(value, int):
        return value
    content = _content_text(record)
    return len(content) if content else None


def _source_url(source: dict[str, Any]) -> str | None:
    value = source.get("url")
    return value.strip() if isinstance(value, str) and value.strip() else None


def is_vbpl_native_record(record: dict[str, Any]) -> bool:
    return isinstance(record.get("source"), dict) and (
        "content_html" in record or "official_group_code" in record or "files" in record
    )


def _pick_title(metadata: dict[str, Any]) -> str | None:
    for key in ("title", "name", "docName", "documentName"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def adapt_vbpl_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map native VBPL pilot records to harness canonical record shape.

    Missing values stay missing. Native status is preserved verbatim.
    """

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    native_status = _normalized_string(record.get("status"))
    normalized_status = native_status.upper() if native_status else None
    content = _content_text(record)

    return {
        "document_id": _normalized_string(record.get("document_id")),
        "source_url": _source_url(source),
        "title": _pick_title(metadata),
        "status": normalized_status,
        "native_status": normalized_status,
        "content": content,
        "content_chars": _content_chars(record),
        "content_sha256": _content_sha256(record),
        "metadata": metadata,
        "official_group_code": _normalized_string(record.get("official_group_code")),
        "official_group_name": _normalized_string(record.get("official_group_name")),
        "official_form_code": _normalized_string(record.get("official_form_code")),
        "official_form_name": _normalized_string(record.get("official_form_name")),
        "files": _normalized_files(record.get("files")),
        "warnings": _normalized_warning_codes(record.get("warnings")),
        "warning_codes": _normalized_warning_codes(record.get("warnings")),
        "scope": _normalized_string(source.get("scope")),
        "id_type": _normalized_string(source.get("id_type")),
        "observed_group": _normalized_string(source.get("observed_group")),
        "crawl_timestamp": _normalized_string(record.get("fetched_at")),
        "backend": _normalized_string(record.get("backend")),
        "error": _normalized_string(record.get("error")),
        "extraction_method": _normalized_string(record.get("extraction_method")),
        "extraction_source_file": _normalized_string(record.get("extraction_source_file")),
        "extraction_confidence": record.get("extraction_confidence") if isinstance(record.get("extraction_confidence"), (int, float)) else None,
        "extraction_note": _normalized_string(record.get("extraction_note")),
        "ocr_attempted": bool(record.get("ocr_attempted")) if record.get("ocr_attempted") is not None else False,
        "pdf_url": _normalized_string(record.get("pdf_url")),
        "pdf_text_chars": record.get("pdf_text_chars") if isinstance(record.get("pdf_text_chars"), int) else None,
        "pdf_ocr_chars": record.get("pdf_ocr_chars") if isinstance(record.get("pdf_ocr_chars"), int) else None,
        "pdf_fetch_error": _normalized_string(record.get("pdf_fetch_error")),
        "recovered_text_normalized": record.get("recovered_text_normalized").strip() if isinstance(record.get("recovered_text_normalized"), str) and record.get("recovered_text_normalized").strip() else None,
        "recovered_text_normalization": record.get("recovered_text_normalization") if isinstance(record.get("recovered_text_normalization"), dict) else None,
        "recovered_text_metrics": record.get("recovered_text_metrics") if isinstance(record.get("recovered_text_metrics"), dict) else None,
        "recovered_text_selection": record.get("recovered_text_selection") if isinstance(record.get("recovered_text_selection"), dict) else None,
        "manual_corpus_review": record.get("manual_corpus_review") if isinstance(record.get("manual_corpus_review"), dict) else None,
    }
