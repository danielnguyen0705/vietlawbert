#!/usr/bin/env python3
"""Audit a stratified sample of VBPL public document pages.

The audit classifies metadata quality and document groups. It deliberately does not
claim that dynamically loaded legal text is valid; that requires the separate
rendered-content audit described in the generated report.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from crawler.probe_vbpl import (
        DEFAULT_USER_AGENT,
        DocumentRef,
        PoliteFetcher,
        inspect_page,
    )
except ModuleNotFoundError:  # direct execution: python src/crawler/audit_vbpl.py
    from probe_vbpl import (  # type: ignore[no-redef]
        DEFAULT_USER_AGENT,
        DocumentRef,
        PoliteFetcher,
        inspect_page,
    )


NORMATIVE_TYPES = {
    "hiến pháp",
    "bộ luật",
    "luật",
    "pháp lệnh",
    "nghị định",
    "thông tư",
    "thông tư liên tịch",
    "thông tư liên bộ",
    "quyết định",
    "lệnh",
    "nghị quyết",
    "nghị quyết liên tịch",
    "chỉ thị",
    "sắc lệnh",
    "sắc luật",
    "quy định",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def agency_name(legislation: dict[str, Any]) -> str | None:
    value = legislation.get("legislationPassedBy")
    if isinstance(value, dict):
        name = value.get("name")
        return str(name).strip() if name else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def classify_document_group(legislation: dict[str, Any] | None, page_title: str, document_id: str) -> str:
    legislation = legislation or {}
    doc_type = normalize_text(legislation.get("legislationType"))
    combined = " ".join(
        [
            doc_type,
            normalize_text(legislation.get("name")),
            normalize_text(page_title),
            normalize_text(document_id),
        ]
    )
    if "đính chính" in combined or document_id.lower().startswith("vbpqdinhchinh_"):
        return "CORRECTION"
    if "văn bản hợp nhất" in doc_type or "văn bản hợp nhất" in combined:
        return "CONSOLIDATED"
    if "hệ thống hóa" in doc_type or "hệ thống hóa" in combined or document_id.lower().startswith("hht"):
        return "SYSTEMATIZED"
    if "bản dịch" in doc_type or "văn bản tiếng anh" in combined:
        return "TRANSLATION"
    if "văn bản hành chính liên quan" in doc_type or "văn bản hành chính liên quan" in combined:
        return "RELATED_ADMIN"
    if doc_type in NORMATIVE_TYPES:
        # A legal form is not enough to prove that a record is normative. For
        # example, VBPL also exposes individual administrative Decisions.
        return "LEGAL_FORM_CANDIDATE"
    return "UNKNOWN"


def quality_warnings(page: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not page.get("ok"):
        return ["PAGE_UNREACHABLE"]
    legislation = page.get("legislation")
    if not isinstance(legislation, dict):
        return ["MISSING_LEGISLATION_JSON_LD"]
    if not legislation.get("name"):
        warnings.append("MISSING_NAME")
    if not legislation.get("legislationIdentifier"):
        warnings.append("MISSING_IDENTIFIER")
    if not legislation.get("legislationType"):
        warnings.append("MISSING_TYPE")
    if not legislation.get("legislationDate"):
        warnings.append("MISSING_DATE")
    if not agency_name(legislation):
        warnings.append("MISSING_AGENCY")
    canonical = page.get("canonical")
    if not canonical:
        warnings.append("MISSING_CANONICAL")
    elif canonical.rstrip("/") != str(page.get("url", "")).rstrip("/"):
        warnings.append("CANONICAL_MISMATCH")
    if page.get("json_ld_errors"):
        warnings.append("INVALID_JSON_LD_BLOCK")
    return warnings


def metadata_disposition(group: str, warnings: list[str]) -> str:
    if "PAGE_UNREACHABLE" in warnings or "MISSING_LEGISLATION_JSON_LD" in warnings:
        return "REJECT_METADATA"
    if group == "UNKNOWN":
        return "MANUAL_REVIEW"
    if any(item in warnings for item in ("MISSING_IDENTIFIER", "MISSING_TYPE", "MISSING_DATE", "MISSING_AGENCY")):
        return "MANUAL_REVIEW"
    if group in {"LEGAL_FORM_CANDIDATE", "CONSOLIDATED"}:
        return "GROUP_AND_CONTENT_CHECK_REQUIRED"
    return "REFERENCE_ONLY"


def logical_key(page: dict[str, Any]) -> str | None:
    legislation = page.get("legislation")
    if not isinstance(legislation, dict):
        return None
    identifier = normalize_text(legislation.get("legislationIdentifier"))
    agency = normalize_text(agency_name(legislation))
    date = normalize_text(legislation.get("legislationDate"))
    if not identifier or not agency or not date:
        return None
    return "|".join((identifier, agency, date))


def load_inventory(path: Path) -> list[DocumentRef]:
    opener = gzip.open if path.suffix == ".gz" else open
    documents: list[DocumentRef] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            documents.append(DocumentRef(**value))
    return documents


def evenly_spaced(items: list[DocumentRef], count: int) -> list[DocumentRef]:
    if not items or count <= 0:
        return []
    if count >= len(items):
        return items[:]
    if count == 1:
        return [items[len(items) // 2]]
    indexes = {round(index * (len(items) - 1) / (count - 1)) for index in range(count)}
    return [items[index] for index in sorted(indexes)]


def stratified_sample(documents: list[DocumentRef], sample_size: int) -> list[DocumentRef]:
    groups: dict[tuple[str, str], list[DocumentRef]] = defaultdict(list)
    for document in documents:
        groups[(document.scope, document.id_type)].append(document)
    allocation: dict[tuple[str, str], int] = {}
    remaining = sample_size
    # Guarantee coverage of every observed scope/identifier stratum first.
    for key in sorted(groups):
        allocation[key] = 1
        remaining -= 1
    if remaining > 0:
        total = sum(len(group) for group in groups.values())
        raw = {key: remaining * len(group) / total for key, group in groups.items()}
        for key in sorted(groups):
            add = math.floor(raw[key])
            allocation[key] += add
            remaining -= add
        for key in sorted(groups, key=lambda item: raw[item] - math.floor(raw[item]), reverse=True):
            if remaining <= 0:
                break
            allocation[key] += 1
            remaining -= 1
    sample: list[DocumentRef] = []
    for key in sorted(groups):
        sample.extend(evenly_spaced(groups[key], allocation[key]))
    return sample[:sample_size]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0}
    estimate = successes / total
    denominator = 1 + z * z / total
    centre = estimate + z * z / (2 * total)
    margin = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total))
    return {
        "estimate": round(estimate, 4),
        "lower": round((centre - margin) / denominator, 4),
        "upper": round((centre + margin) / denominator, 4),
    }


def audit_one(fetcher: PoliteFetcher, document: DocumentRef) -> dict[str, Any]:
    page = inspect_page(fetcher, document)
    legislation = page.get("legislation") if isinstance(page.get("legislation"), dict) else None
    group = classify_document_group(legislation, str(page.get("title") or ""), document.document_id)
    warnings = quality_warnings(page)
    page["document_group"] = group
    page["metadata_disposition"] = metadata_disposition(group, warnings)
    page["quality_warnings"] = warnings
    page["agency_name"] = agency_name(legislation or {})
    page["logical_key"] = logical_key(page)
    return page


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# VBPL content-readiness audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## What this audit proves",
        "",
        f"- Public pages sampled: **{report['sample']['attempted']}**",
        f"- HTTP-successful pages: **{report['sample']['successful']}**",
        f"- Pages with Legislation JSON-LD: **{report['sample']['with_legislation_json_ld']}**",
        "- The classification below concerns public metadata. Dynamic full-text eligibility is not assumed.",
        "",
        "## Document groups",
        "",
        "| Group | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in report["classification"]["by_group"].items())
    lines.extend(["", "## Metadata disposition", "", "| Disposition | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in report["classification"]["by_disposition"].items())
    lines.extend(["", "## Quality warnings", "", "| Warning | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in report["quality"]["warnings"].items())
    lines.extend(
        [
            "",
            "## Duplicate screening",
            "",
            f"- Duplicate URLs in sample: **{report['duplicates']['duplicate_urls']}**",
            f"- Repeated legal keys (number + agency + date): **{report['duplicates']['repeated_logical_keys']}**",
            "- Full-text hash duplication was not tested because the public HTML does not server-render the legal body.",
            "",
            "## Interpretation",
            "",
            "- `GROUP_AND_CONTENT_CHECK_REQUIRED` is a candidate, not an accepted RAG document.",
            "- `REFERENCE_ONLY` should be stored outside the core normative corpus.",
            "- `MANUAL_REVIEW` must not be silently discarded.",
            "- Final inclusion requires rendered HTML or a verified attachment plus successful legal-structure parsing.",
            "- See `RENDERED_CONTENT_AUDIT.md` when a rendered-browser audit was run for this output directory.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-dir", default="artifacts/vbpl_content_audit")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay-per-worker", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inventory = load_inventory(Path(args.inventory))
    sample = stratified_sample(inventory, args.sample_size)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thread_local = threading.local()

    def worker(document: DocumentRef) -> dict[str, Any]:
        if not hasattr(thread_local, "fetcher"):
            thread_local.fetcher = PoliteFetcher(
                args.user_agent,
                args.delay_per_worker,
                args.timeout,
                args.retries,
            )
        return audit_one(thread_local.fetcher, document)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, document): document for document in sample}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(sample)}] {result['scope']}/{result['id_type']} "
                f"{result['document_id']} -> {result['metadata_disposition']}",
                flush=True,
            )

    results.sort(key=lambda row: row["url"])
    with (output_dir / "audit_records.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for document in sample:
            stream.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

    group_counts = Counter(result["document_group"] for result in results)
    disposition_counts = Counter(result["metadata_disposition"] for result in results)
    warning_counts = Counter(warning for result in results for warning in result["quality_warnings"])
    url_counts = Counter(result["url"] for result in results)
    logical_counts = Counter(result["logical_key"] for result in results if result.get("logical_key"))
    successful = sum(bool(result.get("ok")) for result in results)
    json_ld = sum(bool(result.get("has_legislation_json_ld")) for result in results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "inventory_size": len(inventory),
            "sampling": "deterministic stratified by central/local and identifier family, then evenly across sitemap order",
            "sample_size": len(sample),
            "workers": max(1, args.workers),
            "delay_per_worker_seconds": args.delay_per_worker,
        },
        "sample": {
            "attempted": len(results),
            "successful": successful,
            "with_legislation_json_ld": json_ld,
            "http_success_wilson_95": wilson_interval(successful, len(results)),
            "json_ld_wilson_95": wilson_interval(json_ld, len(results)),
            "by_scope": dict(sorted(Counter(result["scope"] for result in results).items())),
            "by_id_type": dict(sorted(Counter(result["id_type"] for result in results).items())),
        },
        "classification": {
            "by_group": dict(sorted(group_counts.items())),
            "by_disposition": dict(sorted(disposition_counts.items())),
        },
        "quality": {
            "warnings": dict(sorted(warning_counts.items())),
            "pages_without_warning": sum(not result["quality_warnings"] for result in results),
            "has_ssr_provision_content": sum(bool(result.get("has_ssr_provision_content")) for result in results),
            "has_document_content_marker": sum(bool(result.get("has_document_content_marker")) for result in results),
        },
        "duplicates": {
            "duplicate_urls": sum(count - 1 for count in url_counts.values() if count > 1),
            "repeated_logical_keys": sum(count - 1 for count in logical_counts.values() if count > 1),
            "logical_key_groups": {
                key: count for key, count in logical_counts.items() if count > 1
            },
            "content_hash_checked": False,
        },
        "limits": [
            "The sample is stratified and deterministic, not a simple random sample.",
            "Public HTML metadata was audited; dynamically loaded full legal text was not accepted as valid.",
            "Wilson intervals are descriptive only because the sample is not simple random.",
            "Classification is rule-based and must be reviewed against rendered content fixtures.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "REPORT.md", report)
    print(f"Report written to {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
