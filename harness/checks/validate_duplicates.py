from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from harness.checks.validate_crawl import extract_text

SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DuplicateFinding:
    kind: str
    value: str
    first_index: int
    duplicate_index: int


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = re.sub(r"/+", "/", parts.path.rstrip("/"))
    return urlunsplit((scheme, netloc, path, "", ""))


def normalize_content(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip().lower())


def content_sha256(text: str) -> str:
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def find_duplicates(records: list[dict[str, Any]]) -> list[DuplicateFinding]:
    findings: list[DuplicateFinding] = []
    seen_urls: dict[str, int] = {}
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}

    for index, record in enumerate(records):
        source_url = record.get("source_url") or record.get("url") or record.get("canonical_url")
        if isinstance(source_url, str) and source_url.strip():
            key = normalize_url(source_url)
            if key in seen_urls:
                findings.append(DuplicateFinding("duplicate_source_url", key, seen_urls[key], index))
            else:
                seen_urls[key] = index

        document_id = record.get("document_id") or record.get("item_id") or record.get("id")
        if document_id not in (None, ""):
            key = str(document_id).strip()
            if key in seen_ids:
                findings.append(DuplicateFinding("duplicate_document_id", key, seen_ids[key], index))
            else:
                seen_ids[key] = index

        text = extract_text(record)
        status = str(record.get("status") or record.get("html_status") or "").upper()
        should_hash_content = (not status) or status == "HTML_VALID"
        if should_hash_content and text and len(normalize_content(text)) >= 80:
            key = content_sha256(text)
            if key in seen_hashes:
                findings.append(DuplicateFinding("duplicate_content", key, seen_hashes[key], index))
            else:
                seen_hashes[key] = index

    return findings
