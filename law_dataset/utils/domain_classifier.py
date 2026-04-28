from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List


def load_domain_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Domain config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(text: str) -> str:
    return strip_accents(normalize_text(text))


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result = []

    for value in values:
        key = normalize_for_match(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)

    return result


def keyword_matches(text: str, keywords: List[str]) -> List[str]:
    normalized_text = normalize_for_match(text)
    matches = []

    for keyword in keywords:
        normalized_keyword = normalize_for_match(keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            matches.append(keyword)

    return unique_keep_order(matches)


def pattern_matches(text: str, patterns: List[str]) -> List[str]:
    normalized_text = normalize_for_match(text)
    matches = []

    for pattern in patterns:
        normalized_pattern = normalize_for_match(pattern)
        try:
            if re.search(normalized_pattern, normalized_text, flags=re.IGNORECASE):
                matches.append(pattern)
        except re.error:
            if normalized_pattern in normalized_text:
                matches.append(pattern)

    return unique_keep_order(matches)


def get_document_search_text(document: Dict[str, Any]) -> str:
    values = [
        document.get("title", ""),
        document.get("doc_type", ""),
        document.get("doc_number", ""),
        document.get("ministry", ""),
        document.get("domain", ""),
        document.get("raw_text", ""),
        document.get("cleaned_text", ""),
        document.get("text", "")
    ]

    for key in ["agencies", "agency_codes", "fields", "field"]:
        value = document.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))

    return "\n".join(str(value or "") for value in values)


def classify_document_domain(
    document: Dict[str, Any],
    domain_config: Dict[str, Any],
) -> Dict[str, Any]:
    title = str(document.get("title") or "")
    doc_type = str(document.get("doc_type") or "")
    body = get_document_search_text(document)

    score = 0
    matched_keywords: List[str] = []
    matched_patterns: List[str] = []

    priority_title_matches = keyword_matches(
        title,
        domain_config.get("priority_keywords", []),
    )
    priority_body_matches = keyword_matches(
        body[:10000],
        domain_config.get("priority_keywords", []),
    )
    secondary_body_matches = keyword_matches(
        body[:10000],
        domain_config.get("secondary_keywords", []),
    )
    include_title_matches = pattern_matches(
        title,
        domain_config.get("include_title_patterns", []),
    )
    exclude_title_matches = pattern_matches(
        title,
        domain_config.get("exclude_title_patterns", []),
    )
    field_matches = keyword_matches(
        body[:3000],
        domain_config.get("field_boost_keywords", []),
    )

    score += len(priority_title_matches) * 5
    score += len(priority_body_matches) * 2
    score += len(secondary_body_matches) * 1
    score += len(include_title_matches) * 5
    score += len(field_matches) * 3
    score -= len(exclude_title_matches) * 5

    allowed_doc_types = [
        normalize_for_match(item)
        for item in domain_config.get("document_types", [])
    ]
    if allowed_doc_types and normalize_for_match(doc_type) in allowed_doc_types:
        score += 1

    matched_keywords.extend(priority_title_matches)
    matched_keywords.extend(priority_body_matches)
    matched_keywords.extend(secondary_body_matches)
    matched_keywords.extend(field_matches)
    matched_patterns.extend(include_title_matches)
    matched_patterns.extend(exclude_title_matches)

    min_score = int(domain_config.get("min_relevance_score", 8))
    review_min = int(domain_config.get("review_score_range", {}).get("min", 4))

    if score >= min_score:
        status = "relevant"
    elif score >= review_min:
        status = "need_review"
    else:
        status = "ignore"

    return {
        "domain_id": domain_config.get("domain_id", ""),
        "domain_name": domain_config.get("domain_name", ""),
        "domain_score": score,
        "domain_status": status,
        "matched_keywords": unique_keep_order(matched_keywords),
        "matched_patterns": unique_keep_order(matched_patterns),
    }