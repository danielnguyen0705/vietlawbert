from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config import HarnessConfig
from harness.adapters.vbpl_record_adapter import adapt_vbpl_record, is_vbpl_native_record
from harness.checks.validate_crawl import validate_document_content
from harness.checks.validate_duplicates import find_duplicates
from harness.checks.validate_jsonl import JSONLineError, validate_jsonl_file
from harness.checks.validate_metadata import validate_metadata

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "fixtures" / "sample_documents.jsonl"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "latest_validation.json"


def _load_records(path: Path) -> tuple[list[dict[str, Any]], list[JSONLineError]]:
    suffixes = path.suffixes
    if suffixes[-2:] == [".jsonl", ".gz"] or path.suffix == ".jsonl":
        return validate_jsonl_file(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        records: list[dict[str, Any]] = []
        errors: list[JSONLineError] = []
        for index, value in enumerate(data, start=1):
            if not isinstance(value, dict):
                errors.append(JSONLineError(str(path), index, "record_not_object"))
                continue
            records.append(adapt_vbpl_record(value) if is_vbpl_native_record(value) else value)
        return records, errors
    if isinstance(data, dict):
        for key in ("records", "documents", "items"):
            value = data.get(key)
            if isinstance(value, list):
                records = []
                errors = []
                for index, item in enumerate(value, start=1):
                    if not isinstance(item, dict):
                        errors.append(JSONLineError(str(path), index, f"record_not_object:{key}"))
                        continue
                    records.append(adapt_vbpl_record(item) if is_vbpl_native_record(item) else item)
                return records, errors
        if is_vbpl_native_record(data):
            return [adapt_vbpl_record(data)], []
        return [data], []
    return [], [JSONLineError(str(path), 0, "unsupported_json_shape")]


def run_validation(input_path: Path, config: HarnessConfig) -> dict[str, Any]:
    records, json_errors = _load_records(input_path)
    records = [adapt_vbpl_record(record) if is_vbpl_native_record(record) else record for record in records]
    rejection_reasons: Counter[str] = Counter()
    metadata_counts: defaultdict[str, int] = defaultdict(int)
    record_results: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        content_result = validate_document_content(record, config)
        metadata_result = validate_metadata(record, config)
        reasons = list(content_result.reasons) + list(metadata_result.reasons)
        for reason in reasons:
            rejection_reasons[reason] += 1
        for field, present in metadata_result.completeness.items():
            if present:
                metadata_counts[field] += 1
        record_results.append(
            {
                "index": index,
                "document_id": record.get("document_id") or record.get("id") or record.get("item_id"),
                "source_url": record.get("source_url") or record.get("url"),
                "native_status": str(record.get("status") or record.get("html_status") or "").upper() or None,
                "valid": not reasons,
                "reasons": reasons,
                "content_chars": content_result.content_chars,
                "vietnamese_score": content_result.vietnamese_score,
            }
        )

    duplicate_findings = find_duplicates(records)
    duplicate_record_indexes: set[int] = set()
    duplicate_groups: set[tuple[str, str]] = set()
    for finding in duplicate_findings:
        rejection_reasons[finding.kind] += 1
        duplicate_record_indexes.add(finding.first_index)
        duplicate_record_indexes.add(finding.duplicate_index)
        duplicate_groups.add((finding.kind, finding.value))
        for record_index in (finding.first_index, finding.duplicate_index):
            if record_index < len(record_results):
                record_results[record_index]["valid"] = False
                if finding.kind not in record_results[record_index]["reasons"]:
                    record_results[record_index]["reasons"].append(finding.kind)

    for error in json_errors:
        rejection_reasons[error.reason] += 1

    documents_scanned = len(records)
    rejected_documents = sum(1 for item in record_results if not item["valid"])
    duplicate_ratio = len(duplicate_record_indexes) / max(documents_scanned, 1)
    status = "PASS"
    if json_errors or rejected_documents or duplicate_ratio > config.maximum_duplicate_ratio:
        status = "FAIL"
    elif records and all(item.get("native_status") == "HTML_VALID" for item in record_results) and rejected_documents == 0:
        status = "PASS"

    metadata_completeness = {
        field: round(count / max(documents_scanned, 1), 4) for field, count in sorted(metadata_counts.items())
    }

    status_distribution = dict(
        sorted(Counter((item.get("native_status") or "MISSING") for item in record_results).items())
    )

    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "documents_scanned": documents_scanned,
        "valid_documents": documents_scanned - rejected_documents,
        "rejected_documents": rejected_documents,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "duplicate_count": len(duplicate_groups),
        "duplicate_record_count": len(duplicate_record_indexes),
        "duplicate_findings": [finding.__dict__ for finding in duplicate_findings],
        "metadata_completeness": metadata_completeness,
        "status_distribution": status_distribution,
        "json_errors": [error.__dict__ for error in json_errors],
        "record_results": record_results,
        "thresholds": config.__dict__,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("=== VietLaw Harness Validation ===")
    print(f"Input: {report['input']}")
    print(f"Documents scanned: {report['documents_scanned']}")
    print(f"Valid: {report['valid_documents']}")
    print(f"Rejected: {report['rejected_documents']}")
    print("Rejection reasons:")
    if report["rejection_reasons"]:
        for reason, count in report["rejection_reasons"].items():
            print(f"- {reason}: {count}")
    else:
        print("- none: 0")
    print("Metadata completeness:")
    for field, ratio in report["metadata_completeness"].items():
        print(f"{field}: {ratio:.1%}")
    print("Status distribution:")
    for name, count in report["status_distribution"].items():
        print(f"- {name}: {count}")
    print(f"Duplicate documents: {report['duplicate_count']}")
    print(f"Result: {report['status']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Vietnamese legal crawl records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSON/JSONL file to validate.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Small JSON report path.")
    parser.add_argument("--min-content-chars", type=int, default=None)
    parser.add_argument("--min-vietnamese-score", type=float, default=None)
    parser.add_argument("--max-duplicate-ratio", type=float, default=None)
    parser.add_argument("--allow-missing-source-url", action="store_true")
    parser.add_argument("--allow-missing-document-id", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = HarnessConfig.from_env()
    config = HarnessConfig(
        minimum_content_chars=args.min_content_chars if args.min_content_chars is not None else base.minimum_content_chars,
        minimum_vietnamese_score=args.min_vietnamese_score if args.min_vietnamese_score is not None else base.minimum_vietnamese_score,
        maximum_duplicate_ratio=args.max_duplicate_ratio if args.max_duplicate_ratio is not None else base.maximum_duplicate_ratio,
        require_source_url=not args.allow_missing_source_url and base.require_source_url,
        require_document_id=not args.allow_missing_document_id and base.require_document_id,
        require_title=base.require_title,
    )
    report = run_validation(args.input, config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
