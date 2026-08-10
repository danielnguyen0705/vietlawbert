"""Run a small, non-destructive verification of the VBPL content contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from crawler.vbpl_content_adapter import (
        ActionContract,
        ContentStatus,
        NextActionClient,
        VBPLContentAdapter,
    )
except ModuleNotFoundError:  # direct execution support
    from vbpl_content_adapter import (  # type: ignore[no-redef]
        ActionContract,
        ContentStatus,
        NextActionClient,
        VBPLContentAdapter,
    )


DEFAULT_DOCUMENTS = [
    ("175440", "numeric_law"),
    ("3dedecb0-9239-11f1-889f-513c5fa29d01", "uuid_decree"),
    ("vbpqta_1964", "legacy_translation"),
    ("hhtquyetdinh_228", "legacy_systematized"),
    ("121624", "numeric_consolidated"),
    ("90831", "decision_review_case"),
]


def compact_record(result: Any, fixture_name: str) -> dict[str, Any]:
    html = result.content_html or ""
    metadata = result.metadata or {}
    return {
        "fixture": fixture_name,
        "document_id": result.document_id,
        "status": result.status.value,
        "content_chars": len(html),
        "content_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest() if html else None,
        "file_count": len(result.files),
        "document_number": metadata.get("docNum"),
        "default_language": metadata.get("defaultLanguage"),
        "official_group_code": result.official_group_code,
        "official_group_name": result.official_group_name,
        "official_form_code": result.official_form_code,
        "official_form_name": result.official_form_name,
        "diagram_checked": result.diagram is not None,
        "warnings": result.warnings,
        "error": result.error,
    }


def make_report(records: list[dict[str, Any]], contract: ActionContract) -> dict[str, Any]:
    counts = Counter(record["status"] for record in records)
    hard_failures = {
        ContentStatus.SECURITY_CHALLENGE.value,
        ContentStatus.HTTP_ERROR.value,
        ContentStatus.CONTRACT_ERROR.value,
        ContentStatus.PARSE_ERROR.value,
    }
    passed = bool(records) and not any(record["status"] in hard_failures for record in records)
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "contract_verified_at": contract.verified_at,
        "source_chunk": contract.source_chunk,
        "requires_cookie": contract.requires_cookie,
        "requires_router_state": contract.requires_router_state,
        "passed": passed,
        "status_counts": dict(sorted(counts.items())),
        "records": records,
        "notes": [
            "This is a representative contract check, not proof that every sitemap URL is valid.",
            "Content hashes are build/data drift signals; raw legal text is intentionally not stored.",
            "Official group/form values come from VBPL search results and may still require data-quality review.",
        ],
    }


def report_markdown(report: dict[str, Any]) -> str:
    verdict = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# VBPL content-contract verification",
        "",
        f"- Result: **{verdict}**",
        f"- Checked at (UTC): `{report['verified_at']}`",
        f"- Contract verified at: `{report['contract_verified_at']}`",
        f"- Cookie required: `{report['requires_cookie']}`",
        f"- Router state required: `{report['requires_router_state']}`",
        "",
        "| Fixture | ID | Status | Chars | Files | Group | Form | Warnings |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for record in report["records"]:
        warnings = ", ".join(record["warnings"]) or "-"
        row = {key: (value if value is not None else "-") for key, value in record.items()}
        row["warnings"] = warnings
        lines.append(
            "| {fixture} | `{document_id}` | {status} | {content_chars} | {file_count} | "
            "{official_group_code} | {official_form_code} | {warnings} |".format(
                **row
            )
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    return "\n".join(lines)


def parse_documents(values: Iterable[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_DOCUMENTS.copy()
    parsed = []
    for value in values:
        document_id, separator, label = value.partition(":")
        parsed.append((document_id, label if separator else "custom"))
    return parsed


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=root / "config" / "vbpl_action_contract.json",
    )
    parser.add_argument(
        "--document",
        action="append",
        help="Document fixture as ID or ID:label; repeatable. Defaults to six representative fixtures.",
    )
    parser.add_argument(
        "--diagram-id",
        action="append",
        default=[],
        help="Also verify the diagram action for this ID; repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "vbpl_contract_verification",
    )
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    contract = ActionContract.load(args.contract)
    client = NextActionClient(contract, delay=args.delay, timeout=args.timeout)
    adapter = VBPLContentAdapter(client)
    diagram_ids = {str(value) for value in args.diagram_id}
    records = []
    for document_id, fixture_name in parse_documents(args.document):
        result = adapter.fetch(document_id, include_diagram=document_id in diagram_ids)
        records.append(compact_record(result, fixture_name))

    report = make_report(records, contract)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
