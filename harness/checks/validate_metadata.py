from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from harness.adapters.vbpl_record_adapter import KNOWN_WARNING_CODES
from harness.config import HarnessConfig

DATE_FIELDS = ("issued_date", "effective_date", "expiration_date", "crawl_timestamp", "legislationDate")
WARNING_FIELDS = ("warning_codes", "warnings")
RECOMMENDED_WARNING_CODES = {
    "MISSING_OFFICIAL_PARENT_GROUP",
    "MISSING_DOCUMENT_NUMBER_FOR_GROUP_LOOKUP",
    "OFFICIAL_GROUP_NOT_FOUND",
    "DUPLICATE_ID_IN_GROUP_LOOKUP",
    "DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED",
    "OFFICIAL_GROUP_ADMINISTRATIVE_FLAG_CONFLICT",
}
KNOWN_METADATA_FIELDS = {
    "document_id",
    "source_url",
    "title",
    "status",
    "official_group_code",
    "official_form_code",
    "official_form_name",
    "document_type",
    "warning_codes",
}
KNOWN_SOURCE_SCOPES = {"central", "local"}
KNOWN_ID_TYPES = {"numeric", "uuid", "legacy_prefixed"}
KNOWN_OBSERVED_GROUPS = {
    "LEGAL_FORM_CANDIDATE",
    "CONSOLIDATED",
    "SYSTEMATIZED",
    "TRANSLATION",
}
RECOMMENDED_OFFICIAL_GROUP_CODES = {"VBQPPL", "VBHN", "VBHTH", "BD"}
RECOMMENDED_OFFICIAL_FORM_FALLBACK_KEYS = ("official_form_name", "official_form_code", "docType")
RECOMMENDED_FIELD_FALLBACK_KEYS = ("field", "linh_vuc", "sector", "topic")
RECOMMENDED_ISSUER_FALLBACK_KEYS = ("issuer", "issuing_authority", "legislationPassedBy")
RECOMMENDED_DOCUMENT_NUMBER_FALLBACK_KEYS = ("document_number", "docNum", "legislationIdentifier")
RECOMMENDED_SCOPE_FALLBACK_KEYS = ("scope", "central_local")
RECOMMENDED_SOURCE_METADATA_KEYS = (
    "official_group_code",
    "official_form_code",
    "official_form_name",
    "document_number",
    "issuer",
    "issued_date",
    "effective_date",
    "expiration_date",
    "field",
    "scope",
    "warning_codes",
)
RECOMMENDED_SOURCE_METADATA_KEYS_SET = set(RECOMMENDED_SOURCE_METADATA_KEYS)
RECOMMENDED_WARNING_CODES_SET = set(KNOWN_WARNING_CODES) | RECOMMENDED_WARNING_CODES
CORPUS_REVIEW_WARNING_CODES = {
    "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
    "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
}
RECOMMENDED_RECORD_WARNING_REASONS = {
    "missing_official_group_code",
    "missing_official_form",
    "missing_document_number",
    "missing_issuer",
    "missing_field",
    "missing_scope",
    "unknown_scope",
    "unknown_id_type",
    "unknown_observed_group",
    "unknown_official_group_code",
    "unknown_warning_code",
    "warning_codes_not_list",
}
RECOMMENDED_COMPLETENESS_FIELDS = tuple(sorted(KNOWN_METADATA_FIELDS | RECOMMENDED_RECORD_WARNING_REASONS))
VALID_STATUSES = {
    "HTML_VALID",
    "CONTENT_TOO_SHORT",
    "PDF_ONLY",
    "METADATA_ONLY",
    "SECURITY_CHALLENGE",
    "HTTP_ERROR",
    "CONTRACT_ERROR",
    "PARSE_ERROR",
}
ACCEPTED_CORPUS_STATUSES = {"HTML_VALID"}
REJECTED_STATUSES = {"HTTP_ERROR", "CONTRACT_ERROR", "PARSE_ERROR", "SECURITY_CHALLENGE"}
REVIEW_ONLY_STATUSES = {"CONTENT_TOO_SHORT", "PDF_ONLY", "METADATA_ONLY"}


@dataclass(frozen=True)
class MetadataValidationResult:
    passed: bool
    reasons: tuple[str, ...]
    completeness: dict[str, bool]


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value
    return None


def _valid_date(value: Any) -> bool:
    if value in (None, ""):
        return True
    if not isinstance(value, str):
        return False
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            datetime.strptime(value.replace("Z", "+0000"), fmt)
            return True
        except ValueError:
            continue
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _normalized_warning_codes(record: dict[str, Any]) -> tuple[list[str], bool]:
    for key in WARNING_FIELDS:
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            return [], False
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())
        return normalized, True
    return [], True


def _metadata_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _manual_corpus_review(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("manual_corpus_review")
    return value if isinstance(value, dict) else None


def _manual_review_override_codes(review: dict[str, Any] | None) -> set[str]:
    if not isinstance(review, dict):
        return set()
    if review.get("status") != "approved":
        return set()
    override_codes = review.get("override_warning_codes")
    if not isinstance(override_codes, list):
        return set()
    normalized: set[str] = set()
    for code in override_codes:
        if isinstance(code, str) and code.strip() in CORPUS_REVIEW_WARNING_CODES:
            normalized.add(code.strip())
    return normalized


def _manual_review_valid(review: dict[str, Any] | None) -> bool:
    if not isinstance(review, dict):
        return False
    if review.get("status") != "approved":
        return False
    reviewer = review.get("reviewer")
    reviewed_at = review.get("reviewed_at")
    notes = review.get("notes")
    if not isinstance(reviewer, str) or not reviewer.strip():
        return False
    if not isinstance(notes, str) or not notes.strip():
        return False
    return _valid_date(reviewed_at)


def validate_metadata(record: dict[str, Any], config: HarnessConfig | None = None) -> MetadataValidationResult:
    config = config or HarnessConfig()
    reasons: list[str] = []

    document_id = _first_present(record, ("document_id", "item_id", "id"))
    source_url = _first_present(record, ("source_url", "url", "canonical_url"))
    title = _first_present(record, ("title", "name"))
    status = _first_present(record, ("status", "html_status"))
    official_group_code = _first_present(record, ("official_group_code", "document_group"))
    official_form = _first_present(record, RECOMMENDED_OFFICIAL_FORM_FALLBACK_KEYS)
    document_number = _first_present(record, RECOMMENDED_DOCUMENT_NUMBER_FALLBACK_KEYS) or _metadata_value(
        record, RECOMMENDED_DOCUMENT_NUMBER_FALLBACK_KEYS
    )
    issuer = _first_present(record, RECOMMENDED_ISSUER_FALLBACK_KEYS) or _metadata_value(
        record, RECOMMENDED_ISSUER_FALLBACK_KEYS
    )
    field_value = _first_present(record, RECOMMENDED_FIELD_FALLBACK_KEYS) or _metadata_value(
        record, RECOMMENDED_FIELD_FALLBACK_KEYS
    )
    scope = _first_present(record, RECOMMENDED_SCOPE_FALLBACK_KEYS) or _metadata_value(
        record, RECOMMENDED_SCOPE_FALLBACK_KEYS
    )
    id_type = _first_present(record, ("id_type",))
    observed_group = _first_present(record, ("observed_group", "document_group"))
    warning_codes, warning_list_ok = _normalized_warning_codes(record)
    manual_review = _manual_corpus_review(record)
    manual_override_codes = _manual_review_override_codes(manual_review)
    manual_review_ok = _manual_review_valid(manual_review)

    completeness = {
        "document_id": bool(document_id),
        "source_url": bool(source_url),
        "title": bool(title),
        "status": bool(status),
        "official_group_code": bool(official_group_code),
        "official_form_code": bool(_first_present(record, ("official_form_code",))),
        "official_form_name": bool(_first_present(record, ("official_form_name",))),
        "document_type": bool(official_form),
        "warning_codes": bool(warning_codes),
        "missing_document_number": not bool(document_number),
        "missing_issuer": not bool(issuer),
        "missing_field": not bool(field_value),
        "missing_scope": not bool(scope),
        "missing_official_group_code": not bool(official_group_code),
        "missing_official_form": not bool(official_form),
        "warning_codes_not_list": not warning_list_ok,
        "unknown_scope": bool(scope) and str(scope).lower() not in KNOWN_SOURCE_SCOPES,
        "unknown_id_type": bool(id_type) and str(id_type).lower() not in KNOWN_ID_TYPES,
        "unknown_observed_group": bool(observed_group) and str(observed_group) not in KNOWN_OBSERVED_GROUPS,
        "unknown_official_group_code": bool(official_group_code) and str(official_group_code) not in RECOMMENDED_OFFICIAL_GROUP_CODES,
        "unknown_warning_code": any(code not in RECOMMENDED_WARNING_CODES_SET for code in warning_codes),
    }

    if config.require_document_id and not document_id:
        reasons.append("missing_document_id")
    if config.require_source_url and not source_url:
        reasons.append("missing_source_url")
    if config.require_title and not title:
        reasons.append("missing_title")

    normalized_status = str(status).upper() if status else ""
    if not normalized_status:
        reasons.append("missing_status")
    elif normalized_status not in VALID_STATUSES:
        reasons.append(f"unknown_status:{normalized_status}")
    elif normalized_status in REJECTED_STATUSES:
        reasons.append(f"bad_status:{normalized_status}")
    elif normalized_status in REVIEW_ONLY_STATUSES:
        reasons.append(f"non_accepted_status:{normalized_status}")

    for key in DATE_FIELDS:
        value = _first_present(record, (key,))
        if not _valid_date(value):
            reasons.append(f"malformed_date:{key}")

    if not warning_list_ok:
        reasons.append("warning_codes_not_list")
    for code in warning_codes:
        if code not in RECOMMENDED_WARNING_CODES_SET:
            reasons.append(f"unknown_warning_code:{code}")
    if manual_review and not manual_review_ok:
        reasons.append("invalid_manual_corpus_review")

    if normalized_status in ACCEPTED_CORPUS_STATUSES:
        for code in warning_codes:
            if code in CORPUS_REVIEW_WARNING_CODES:
                if code in manual_override_codes and manual_review_ok:
                    continue
                reasons.append(f"corpus_review_required:{code}")

    return MetadataValidationResult(passed=not reasons, reasons=tuple(reasons), completeness=completeness)
