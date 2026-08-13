from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from harness.config import HarnessConfig

SECURITY_PATTERNS = (
    "cloudflare",
    "performing security verification",
    "ray id",
    "access denied",
    "attention required",
    "checking your browser",
    "security verification",
    "captcha",
)

STRONG_NON_DOCUMENT_PATTERNS = (
    "đăng nhập",
    "login",
    "sign in",
    "javascript is disabled",
    "page not found",
    "404 not found",
    "404 page not found",
)

WEAK_NON_DOCUMENT_PATTERNS = (
    "trang chủ",
    "not found",
    "404",
)

VIETNAMESE_LEGAL_TERMS = (
    "điều",
    "khoản",
    "văn bản",
    "quy định",
    "pháp luật",
    "nghị định",
    "thông tư",
    "quyết định",
    "luật",
    "căn cứ",
    "thi hành",
    "ủy ban nhân dân",
    "chính phủ",
)

VIETNAMESE_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


@dataclass(frozen=True)
class ContentValidationResult:
    passed: bool
    reasons: tuple[str, ...]
    vietnamese_score: float
    content_chars: int


def extract_text(record: dict[str, Any]) -> str:
    for key in ("content", "text", "html", "html_raw", "legal_text", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("content") or metadata.get("documentContent")
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            return value["content"].strip()
    return ""


def vietnamese_legal_score(text: str) -> float:
    """CI-safe heuristic, not a formal language detector."""
    if not text:
        return 0.0
    lowered = text.lower()
    tokens = TOKEN_RE.findall(lowered)
    if not tokens:
        return 0.0
    diacritic_hits = len(VIETNAMESE_DIACRITIC_RE.findall(lowered))
    term_hits = sum(1 for term in VIETNAMESE_LEGAL_TERMS if term in lowered)
    alpha_tokens = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
    token_quality = alpha_tokens / max(len(tokens), 1)
    diacritic_score = min(diacritic_hits / 20.0, 1.0)
    term_score = min(term_hits / 5.0, 1.0)
    return round((0.45 * diacritic_score) + (0.45 * term_score) + (0.10 * token_quality), 4)


def has_security_challenge(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in SECURITY_PATTERNS)


def looks_like_non_document_page(text: str) -> bool:
    lowered = text.lower()
    head = lowered[:4000]
    if any(pattern in head for pattern in STRONG_NON_DOCUMENT_PATTERNS):
        return True
    if any(pattern in head for pattern in WEAK_NON_DOCUMENT_PATTERNS):
        has_legal_signal = any(term in lowered for term in VIETNAMESE_LEGAL_TERMS)
        return len(text.strip()) < 1500 and not has_legal_signal
    return False


def validate_document_content(record: dict[str, Any], config: HarnessConfig | None = None) -> ContentValidationResult:
    config = config or HarnessConfig()
    text = extract_text(record)
    reasons: list[str] = []

    if not text:
        reasons.append("empty_content")
    elif len(text) < config.minimum_content_chars:
        reasons.append("content_too_short")

    if text and has_security_challenge(text):
        reasons.append("security_challenge")
    if text and looks_like_non_document_page(text):
        reasons.append("non_document_page")

    score = vietnamese_legal_score(text)
    status = str(record.get("status") or record.get("html_status") or "").upper()
    recoverable_without_text = status in {"PDF_ONLY", "METADATA_ONLY"}
    if text and score < config.minimum_vietnamese_score and not recoverable_without_text:
        reasons.append("language_quality")

    return ContentValidationResult(
        passed=not reasons,
        reasons=tuple(reasons),
        vietnamese_score=score,
        content_chars=len(text),
    )
