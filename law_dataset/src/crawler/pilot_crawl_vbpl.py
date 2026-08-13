#!/usr/bin/env python3
"""Controlled VBPL full-text pilot with stratification, checkpointing and gates."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from crawler.vbpl_content_adapter import (
        ActionContract,
        ContentStatus,
        NextActionClient,
        PDFExtractor,
        VBPLContentAdapter,
        classify_official_doc_type,
    )
except ModuleNotFoundError:  # direct execution support
    from vbpl_content_adapter import (  # type: ignore[no-redef]
        ActionContract,
        ContentStatus,
        NextActionClient,
        PDFExtractor,
        VBPLContentAdapter,
        classify_official_doc_type,
    )


HARD_FAILURES = {
    ContentStatus.SECURITY_CHALLENGE.value,
    ContentStatus.HTTP_ERROR.value,
    ContentStatus.CONTRACT_ERROR.value,
    ContentStatus.PARSE_ERROR.value,
}
STOP_STATUSES = {
    ContentStatus.SECURITY_CHALLENGE.value,
    ContentStatus.CONTRACT_ERROR.value,
    ContentStatus.PARSE_ERROR.value,
}
CORPUS_REVIEW_WARNING_CODES = {
    "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
    "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
}
NON_CLASSIFICATION_WARNING_CODES = {
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
ACCEPTED_CORPUS_STATUSES = {ContentStatus.HTML_VALID.value}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize audit and manifest rows without changing their document IDs."""

    value = dict(row)
    if "document_group" not in value and "observed_group" in value:
        value["document_group"] = value["observed_group"]
    return value


def evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not items or count <= 0:
        return []
    if count >= len(items):
        return items[:]
    if count == 1:
        return [items[len(items) // 2]]
    indexes = {round(index * (len(items) - 1) / (count - 1)) for index in range(count)}
    return [items[index] for index in sorted(indexes)]


def stratified_sample(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    """Allocate proportionally after giving every observed three-axis stratum one slot."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("scope") or "unknown"),
            str(row.get("id_type") or "unknown"),
            str(row.get("document_group") or "unknown"),
        )
        groups[key].append(row)
    target = min(max(0, sample_size), len(rows))
    if not groups or target == 0:
        return []

    allocation = {key: 0 for key in groups}
    remaining = target
    for key in sorted(groups, key=lambda item: (len(groups[item]), item)):
        if remaining <= 0:
            break
        allocation[key] = 1
        remaining -= 1

    capacity = {key: len(group) - allocation[key] for key, group in groups.items()}
    while remaining > 0:
        available = {key: value for key, value in capacity.items() if value > 0}
        if not available:
            break
        total_capacity = sum(available.values())
        raw = {key: remaining * value / total_capacity for key, value in available.items()}
        added = 0
        for key in sorted(available):
            amount = min(capacity[key], math.floor(raw[key]))
            allocation[key] += amount
            capacity[key] -= amount
            remaining -= amount
            added += amount
        if remaining <= 0:
            break
        ranked = sorted(available, key=lambda key: (raw[key] - math.floor(raw[key]), key), reverse=True)
        for key in ranked:
            if remaining <= 0:
                break
            if capacity[key] > 0:
                allocation[key] += 1
                capacity[key] -= 1
                remaining -= 1
                added += 1
        if added == 0:
            break

    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        selected.extend(evenly_spaced(groups[key], allocation[key]))
    return sorted(selected, key=lambda row: (str(row.get("scope")), str(row.get("document_id"))))


def compact_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Keep metadata but remove fields whose full text is stored separately."""

    value = dict(metadata or {})
    value.pop("documentContent", None)
    value.pop("documentContentEn", None)
    return value


def make_record(source: dict[str, Any], result: Any) -> dict[str, Any]:
    html = result.content_html or ""
    return {
        "source": {
            "url": source.get("url"),
            "lastmod": source.get("lastmod"),
            "scope": source.get("scope"),
            "id_type": source.get("id_type"),
            "observed_group": source.get("document_group"),
        },
        "document_id": result.document_id,
        "status": result.status.value,
        "metadata": compact_metadata(result.metadata),
        "content_html": html,
        "content_chars": len(html),
        "content_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest() if html else None,
        "files": result.files,
        "official_group_code": result.official_group_code,
        "official_group_name": result.official_group_name,
        "official_form_code": result.official_form_code,
        "official_form_name": result.official_form_name,
        "warnings": result.warnings,
        "backend": result.backend,
        "error": result.error,
        "extraction_method": result.extraction_method,
        "extraction_source_file": result.extraction_source_file,
        "extraction_confidence": result.extraction_confidence,
        "extraction_note": result.extraction_note,
        "ocr_attempted": result.ocr_attempted,
        "pdf_url": result.pdf_url,
        "pdf_text_chars": result.pdf_text_chars,
        "pdf_ocr_chars": result.pdf_ocr_chars,
        "pdf_fetch_error": result.pdf_fetch_error,
        "recovered_text_normalized": result.recovered_text_normalized,
        "recovered_text_normalization": result.recovered_text_normalization,
        "recovered_text_metrics": result.recovered_text_metrics,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_record_classification(record: dict[str, Any]) -> dict[str, Any]:
    """Refresh checkpoint records after classification logic improves, without refetching."""

    classification = classify_official_doc_type(record.get("metadata") or {})
    if classification is None:
        return record
    group_code, group_name, form_code, form_name, classification_warnings = classification
    retained_warnings = [
        warning
        for warning in record.get("warnings", [])
        if warning
        not in {
            "OFFICIAL_GROUP_NOT_FOUND",
            "MISSING_OFFICIAL_PARENT_GROUP",
            "VBQPPL_DECISION_REVIEW_RECOMMENDED",
            "DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED",
            "OFFICIAL_GROUP_ADMINISTRATIVE_FLAG_CONFLICT",
        }
    ]
    record.update(
        {
            "official_group_code": group_code,
            "official_group_name": group_name,
            "official_form_code": form_code,
            "official_form_name": form_name,
            "warnings": retained_warnings + classification_warnings,
        }
    )
    return record


def evaluate(records: list[dict[str, Any]], requested: int, stopped_early: bool) -> dict[str, Any]:
    total = len(records)
    statuses = Counter(row["status"] for row in records)
    hard_failures = sum(statuses.get(status, 0) for status in HARD_FAILURES)
    content_available = sum(
        row["status"] == ContentStatus.HTML_VALID.value
        or row["status"] == ContentStatus.PDF_ONLY.value
        or any(str(item.get("fileName") or "").lower().endswith(".pdf") for item in row.get("files", []))
        for row in records
    )
    missing_group = sum(not row.get("official_group_code") for row in records)
    hashes = Counter(
        row["content_sha256"]
        for row in records
        if row.get("content_sha256") and row["status"] == ContentStatus.HTML_VALID.value
    )
    duplicate_content = sum(count - 1 for count in hashes.values() if count > 1)
    classification_review = sum(
        any(code not in NON_CLASSIFICATION_WARNING_CODES for code in row.get("warnings", []))
        for row in records
    )
    corpus_review_required = sum(
        row.get("status") in ACCEPTED_CORPUS_STATUSES
        and any(code in CORPUS_REVIEW_WARNING_CODES for code in row.get("warnings", []))
        for row in records
    )
    corpus_ready = sum(
        row.get("status") in ACCEPTED_CORPUS_STATUSES
        and not any(code in CORPUS_REVIEW_WARNING_CODES for code in row.get("warnings", []))
        for row in records
    )
    recovered_quality_review = sum(
        "CONTENT_RECOVERED_FROM_PDF" in (row.get("warnings") or [])
        and any(code in CORPUS_REVIEW_WARNING_CODES for code in row.get("warnings", []))
        for row in records
    )

    hard_failure_ratio = hard_failures / total if total else 1.0
    content_available_ratio = content_available / total if total else 0.0
    missing_group_ratio = missing_group / total if total else 1.0
    duplicate_ratio = duplicate_content / total if total else 1.0
    classification_review_ratio = classification_review / total if total else 1.0
    corpus_review_required_ratio = corpus_review_required / total if total else 1.0
    corpus_ready_ratio = corpus_ready / total if total else 0.0
    recovered_quality_review_ratio = recovered_quality_review / total if total else 0.0

    technical_gates = {
        "completed_requested_sample": total == requested and not stopped_early,
        "hard_failure_ratio_lte_2pct": hard_failure_ratio <= 0.02,
        "content_or_pdf_ratio_gte_95pct": content_available_ratio >= 0.95,
        "missing_official_group_ratio_lte_5pct": missing_group_ratio <= 0.05,
        "duplicate_content_ratio_lte_1pct": duplicate_ratio <= 0.01,
        "classification_review_ratio_lte_20pct": classification_review_ratio <= 0.20,
    }
    technical_decision = (
        "READY_FOR_LARGER_PILOT" if all(technical_gates.values()) else "REVIEW_AND_REPEAT"
    )
    corpus_gates = {
        "completed_requested_sample": total == requested and not stopped_early,
        "content_or_pdf_ratio_gte_95pct": content_available_ratio >= 0.95,
        "corpus_review_required_records_eq_0": corpus_review_required == 0,
        "recovered_quality_review_records_eq_0": recovered_quality_review == 0,
    }
    corpus_decision = "READY_FOR_PHASE_3" if all(corpus_gates.values()) else "REVIEW_REQUIRED"
    metrics = {
        "records": total,
        "hard_failures": hard_failures,
        "hard_failure_ratio": round(hard_failure_ratio, 4),
        "content_or_pdf": content_available,
        "content_or_pdf_ratio": round(content_available_ratio, 4),
        "missing_official_group": missing_group,
        "missing_official_group_ratio": round(missing_group_ratio, 4),
        "duplicate_content_records": duplicate_content,
        "duplicate_content_ratio": round(duplicate_ratio, 4),
        "classification_review_records": classification_review,
        "classification_review_ratio": round(classification_review_ratio, 4),
        "corpus_ready_records": corpus_ready,
        "corpus_ready_ratio": round(corpus_ready_ratio, 4),
        "corpus_review_required_records": corpus_review_required,
        "corpus_review_required_ratio": round(corpus_review_required_ratio, 4),
        "recovered_quality_review_records": recovered_quality_review,
        "recovered_quality_review_ratio": round(recovered_quality_review_ratio, 4),
    }
    return {
        "decision": technical_decision,
        "gates": technical_gates,
        "metrics": metrics,
        "technical_readiness": {
            "decision": technical_decision,
            "gates": technical_gates,
            "metrics": metrics,
        },
        "corpus_readiness": {
            "decision": corpus_decision,
            "gates": corpus_gates,
            "metrics": {
                "records": total,
                "content_or_pdf": content_available,
                "content_or_pdf_ratio": round(content_available_ratio, 4),
                "corpus_ready_records": corpus_ready,
                "corpus_ready_ratio": round(corpus_ready_ratio, 4),
                "corpus_review_required_records": corpus_review_required,
                "corpus_review_required_ratio": round(corpus_review_required_ratio, 4),
                "recovered_quality_review_records": recovered_quality_review,
                "recovered_quality_review_ratio": round(recovered_quality_review_ratio, 4),
            },
        },
    }


def _fallback_recovery_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [row for row in records if "CONTENT_RECOVERED_FROM_PDF" in (row.get("warnings") or [])]
    recovered_ocr = [row for row in records if "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED" in (row.get("warnings") or [])]
    fetch_failed = [row for row in records if "PDF_FETCH_FAILED" in (row.get("warnings") or [])]
    insufficient = [row for row in records if "PDF_EXTRACTION_INSUFFICIENT" in (row.get("warnings") or [])]
    return {
        "recovered_from_pdf": len(recovered),
        "recovered_by_ocr": len(recovered_ocr),
        "pdf_fetch_failed": len(fetch_failed),
        "pdf_extraction_insufficient": len(insufficient),
        "recovered_document_ids": [str(row.get("document_id")) for row in recovered],
        "ocr_recovered_document_ids": [str(row.get("document_id")) for row in recovered_ocr],
    }


def build_report(
    records: list[dict[str, Any]], requested: int, input_rows: int, stopped_early: bool
) -> dict[str, Any]:
    lengths = [row["content_chars"] for row in records if row["content_chars"] > 0]
    evaluation = evaluate(records, requested, stopped_early)
    fallback_metrics = _fallback_recovery_metrics(records)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pilot": {
            "input_metadata_rows": input_rows,
            "requested": requested,
            "completed": len(records),
            "stopped_early": stopped_early,
            "sampling": "deterministic stratification by scope, id_type and observed_group",
        },
        "counts": {
            "by_status": dict(sorted(Counter(row["status"] for row in records).items())),
            "by_scope": dict(sorted(Counter(row["source"]["scope"] for row in records).items())),
            "by_id_type": dict(sorted(Counter(row["source"]["id_type"] for row in records).items())),
            "by_observed_group": dict(
                sorted(Counter(row["source"]["observed_group"] for row in records).items())
            ),
            "by_official_group": dict(
                sorted(Counter(row.get("official_group_code") or "UNKNOWN" for row in records).items())
            ),
            "warnings": dict(
                sorted(Counter(warning for row in records for warning in row.get("warnings", [])).items())
            ),
        },
        "content_chars": {
            "total": sum(lengths),
            "min": min(lengths) if lengths else 0,
            "median": round(statistics.median(lengths), 1) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "evaluation": evaluation,
        "fallback": fallback_metrics,
        "limits": [
            "This pilot is stratified and deterministic, not a simple random sample.",
            "Observed groups come from JSON-LD; official groups/forms come from VBPL detail metadata.",
            "PDF availability is not proof that PDF text extraction or OCR will succeed.",
            "Decision-number warnings are triage heuristics, not legal conclusions.",
            "A larger crawl still needs rate limiting, checkpointing and periodic contract verification.",
        ],
    }


def report_markdown(report: dict[str, Any]) -> str:
    pilot = report["pilot"]
    evaluation = report["evaluation"]
    technical = evaluation.get("technical_readiness", evaluation)
    corpus = evaluation.get("corpus_readiness", {})
    lines = [
        "# VBPL controlled full-text pilot",
        "",
        f"- Technical retrieval decision: **{technical['decision']}**",
        f"- Corpus readiness decision: **{corpus.get('decision', 'UNKNOWN')}**",
        f"- Completed: **{pilot['completed']}/{pilot['requested']}**",
        f"- Stopped early: **{pilot['stopped_early']}**",
        "- Sampling: central/local + identifier family + observed document group",
        "",
        "## Technical quality gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value} |" for key, value in technical["gates"].items())
    lines.extend(["", "## Technical metrics", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| `{key}` | {value} |" for key, value in technical["metrics"].items())
    if corpus:
        lines.extend(["", "## Corpus readiness gates", "", "| Gate | Passed |", "|---|---:|"])
        lines.extend(f"| `{key}` | {value} |" for key, value in corpus["gates"].items())
        lines.extend(["", "## Corpus readiness metrics", "", "| Metric | Value |", "|---|---:|"])
        lines.extend(f"| `{key}` | {value} |" for key, value in corpus["metrics"].items())
    fallback = report.get("fallback") or {}
    if fallback:
        lines.extend(["", "## PDF/OCR fallback", "", "| Metric | Value |", "|---|---:|"])
        for key in ("recovered_from_pdf", "recovered_by_ocr", "pdf_fetch_failed", "pdf_extraction_insufficient"):
            lines.append(f"| `{key}` | {fallback.get(key, 0)} |")
    for title, key in (
        ("Status", "by_status"),
        ("Scope", "by_scope"),
        ("Identifier type", "by_id_type"),
        ("Observed group", "by_observed_group"),
        ("Official group", "by_official_group"),
        ("Warnings", "warnings"),
    ):
        lines.extend(["", f"## {title}", "", "| Value | Count |", "|---|---:|"])
        values = report["counts"][key]
        lines.extend(f"| {name} | {count} |" for name, count in values.items())
        if not values:
            lines.append("| (none) | 0 |")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in report["limits"])
    lines.extend(["", "## Recommended next step", ""])
    if technical["decision"] != "READY_FOR_LARGER_PILOT":
        lines.append("- Fix failed technical retrieval gates and repeat the 120-document pilot before increasing load.")
    elif corpus.get("decision") != "READY_FOR_PHASE_3":
        lines.extend(
            [
                "- Keep Phase 2 open: review and repair recovered records flagged for OCR or encoding quality.",
                "- Do not treat recovered HTML_VALID records as corpus-ready until corpus review warnings return to zero.",
                "- Rebuild the pilot report after fixes, then re-evaluate Phase 3 entry criteria.",
                "- Do not start a 500–1,000 document pilot or full-sitemap crawl from this report alone.",
            ]
        )
    else:
        lines.extend(
            [
                "- Technical retrieval and corpus readiness gates passed on this pilot.",
                "- Run a larger 500–1,000 document pilot with the same gates and evidence capture.",
                "- Keep contract verification, checkpointing and rate limits enabled.",
                "- Do not start a full-sitemap crawl from this result alone.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_records(records: list[dict[str, Any]], checkpoint: Path, target: Path) -> None:
    temporary = checkpoint.with_suffix(".tmp")
    with temporary.open("wt", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "artifacts" / "vbpl_content_audit_120" / "audit_records.jsonl",
    )
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Authoritative existing manifest. Replays its exact rows without resampling.",
    )
    parser.add_argument(
        "--contract", type=Path, default=root / "config" / "vbpl_action_contract.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "vbpl_pilot_120"
    )
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--enable-pdf-fallback", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.manifest:
        source_rows = [normalize_manifest_row(row) for row in read_jsonl(args.manifest)]
        selected = source_rows
    else:
        source_rows = [normalize_manifest_row(row) for row in read_jsonl(args.input)]
        selected = stratified_sample(source_rows, args.sample_size)

    if not selected:
        raise SystemExit("No pilot rows selected; check --manifest or --input.")
    if len({str(row.get("document_id")) for row in selected}) != len(selected):
        raise SystemExit("Manifest contains duplicate document_id values; refusing ambiguous resume state.")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "sample_manifest.jsonl"
    checkpoint_path = args.output / "checkpoint.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps({
                "url": row.get("url"),
                "document_id": row.get("document_id"),
                "scope": row.get("scope"),
                "id_type": row.get("id_type"),
                "observed_group": row.get("document_group"),
            }, ensure_ascii=False) + "\n")

    existing = read_jsonl(checkpoint_path)
    completed_ids = {str(row["document_id"]) for row in existing}
    contract = ActionContract.load(args.contract)
    client = NextActionClient(
        contract, delay=args.delay, timeout=args.timeout, retries=args.retries
    )
    adapter = VBPLContentAdapter(
        client,
        pdf_extractor=PDFExtractor(enable_ocr=not args.disable_ocr),
        enable_pdf_fallback=args.enable_pdf_fallback,
    )
    stopped_early = False
    with checkpoint_path.open("a", encoding="utf-8") as stream:
        for index, source in enumerate(selected, start=1):
            document_id = str(source["document_id"])
            if document_id in completed_ids:
                print(f"[{index}/{len(selected)}] {document_id} -> RESUMED", flush=True)
                continue
            result = adapter.fetch(document_id)
            record = make_record(source, result)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            completed_ids.add(document_id)
            print(
                f"[{index}/{len(selected)}] {source.get('scope')}/{source.get('id_type')}/"
                f"{source.get('document_group')} {document_id} -> {result.status.value}",
                flush=True,
            )
            if result.status.value in STOP_STATUSES:
                stopped_early = True
                print(f"Stopping on unsafe status: {result.status.value}", flush=True)
                break

    records_by_id = {str(row["document_id"]): row for row in read_jsonl(checkpoint_path)}
    records = [
        refresh_record_classification(records_by_id[str(row["document_id"])])
        for row in selected
        if str(row["document_id"]) in records_by_id
    ]
    write_records(records, checkpoint_path, args.output / "documents.jsonl.gz")
    report = build_report(records, len(selected), len(source_rows), stopped_early)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    print(f"Report written to {args.output / 'REPORT.md'}")
    return 0 if report["evaluation"]["decision"] == "READY_FOR_LARGER_PILOT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
