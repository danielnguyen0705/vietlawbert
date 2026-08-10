"""Minimal parser for the React Server Components action responses used by VBPL."""

from __future__ import annotations

import json
import re
from typing import Any


TEXT_REFERENCE_RE = re.compile(r"^\$([0-9a-fA-F]+)$")


class RSCParseError(ValueError):
    pass


def parse_rsc_records(payload: bytes) -> dict[str, Any]:
    """Parse JSON-line and length-prefixed text records from a Flight response.

    VBPL detail responses commonly contain a text record such as
    ``2:T1da33,<html>...</html>`` followed by a JSON record that references it as
    ``"$2"``. Lengths are byte lengths encoded as hexadecimal.
    """

    records: dict[str, Any] = {}
    cursor = 0
    length = len(payload)
    while cursor < length:
        while cursor < length and payload[cursor : cursor + 1] in (b"\n", b"\r"):
            cursor += 1
        if cursor >= length:
            break
        colon = payload.find(b":", cursor)
        if colon < 0:
            raise RSCParseError(f"Missing record separator at byte {cursor}")
        record_id = payload[cursor:colon].decode("ascii", "strict")
        body_start = colon + 1
        if body_start < length and payload[body_start : body_start + 1] == b"T":
            comma = payload.find(b",", body_start + 1)
            if comma < 0:
                raise RSCParseError(f"Missing text-record length terminator for {record_id}")
            try:
                text_length = int(payload[body_start + 1 : comma].decode("ascii"), 16)
            except ValueError as exc:
                raise RSCParseError(f"Invalid text-record length for {record_id}") from exc
            content_start = comma + 1
            content_end = content_start + text_length
            if content_end > length:
                raise RSCParseError(f"Truncated text record {record_id}")
            records[record_id] = payload[content_start:content_end].decode("utf-8", "replace")
            cursor = content_end
            continue
        newline = payload.find(b"\n", body_start)
        if newline < 0:
            newline = length
        raw_value = payload[body_start:newline].decode("utf-8", "replace")
        try:
            records[record_id] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise RSCParseError(f"Invalid JSON record {record_id}") from exc
        cursor = newline + 1
    return records


def resolve_text_references(value: Any, records: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = TEXT_REFERENCE_RE.fullmatch(value)
        if match and match.group(1) in records:
            return records[match.group(1)]
        return value
    if isinstance(value, list):
        return [resolve_text_references(item, records) for item in value]
    if isinstance(value, dict):
        return {key: resolve_text_references(item, records) for key, item in value.items()}
    return value


def parse_action_result(payload: bytes, result_record_id: str = "1") -> Any:
    records = parse_rsc_records(payload)
    if result_record_id not in records:
        raise RSCParseError(f"Missing action result record {result_record_id}")
    return resolve_text_references(records[result_record_id], records)
