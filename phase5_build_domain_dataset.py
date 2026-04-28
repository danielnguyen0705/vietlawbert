from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from law_dataset.utils.domain_classifier import (
    classify_document_domain,
    load_domain_config,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "law_dataset" / "data"

DOCUMENTS_JSONL = DATA_DIR / "documents.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"

DOMAIN_DATASET_DIR = DATA_DIR / "domain_datasets"


def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip invalid JSON line {line_no} in {path}: {e}")


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def load_documents_with_domain(config: Dict) -> list[Dict]:
    documents = []

    for doc in iter_jsonl(DOCUMENTS_JSONL):
        result = classify_document_domain(doc, config)
        enriched = {
            **doc,
            **result,
        }
        documents.append(enriched)

    return documents


def build_chunk_dataset(
    *,
    documents: list[Dict],
    config: Dict,
    include_review: bool,
) -> list[Dict]:
    allowed_statuses = {"relevant"}
    if include_review:
        allowed_statuses.add("need_review")

    doc_by_id = {
        str(doc.get("id") or doc.get("document_id") or ""): doc
        for doc in documents
        if doc.get("domain_status") in allowed_statuses
    }

    chunks = []

    for chunk in iter_jsonl(CHUNKS_JSONL):
        doc_id = str(chunk.get("document_id") or "")
        doc = doc_by_id.get(doc_id)

        if not doc:
            continue

        chunks.append(
            {
                **chunk,
                "domain_id": config.get("domain_id", ""),
                "domain_name": config.get("domain_name", ""),
                "domain_score": doc.get("domain_score", 0),
                "domain_status": doc.get("domain_status", ""),
                "matched_keywords": doc.get("matched_keywords", []),
                "matched_patterns": doc.get("matched_patterns", []),
                "metadata": {
                    "document_id": doc.get("id", ""),
                    "doc_number": doc.get("doc_number", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "title": doc.get("title", ""),
                    "ministry": doc.get("ministry", ""),
                    "agencies": doc.get("agencies", []),
                    "url": doc.get("url", ""),
                    "domain_id": config.get("domain_id", ""),
                    "domain_name": config.get("domain_name", ""),
                    "domain_score": doc.get("domain_score", 0),
                    "domain_status": doc.get("domain_status", ""),
                    "section_header": chunk.get("section_header", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                },
            }
        )

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a domain-specific legal dataset from documents.jsonl and chunks.jsonl."
    )
    parser.add_argument(
        "--domain-config",
        default="configs/domains/traffic_law.json",
        help="Path to domain config JSON.",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Include documents with domain_status=need_review.",
    )
    args = parser.parse_args()

    config = load_domain_config(BASE_DIR / args.domain_config)

    domain_id = config["domain_id"]
    output_dir = DOMAIN_DATASET_DIR / domain_id

    print(f"Building domain dataset: {domain_id} - {config.get('domain_name', '')}")
    print(f"Domain config: {args.domain_config}")

    documents = load_documents_with_domain(config)

    relevant_documents = [
        doc for doc in documents
        if doc.get("domain_status") == "relevant"
    ]
    review_documents = [
        doc for doc in documents
        if doc.get("domain_status") == "need_review"
    ]
    ignored_documents = [
        doc for doc in documents
        if doc.get("domain_status") == "ignore"
    ]

    allowed_statuses = {"relevant"}
    if args.include_review:
        allowed_statuses.add("need_review")

    export_documents = [
        doc for doc in documents
        if doc.get("domain_status") in allowed_statuses
    ]

    domain_chunks = build_chunk_dataset(
        documents=documents,
        config=config,
        include_review=args.include_review,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    review_path = output_dir / "need_review_documents.jsonl"
    summary_path = output_dir / "summary.json"

    documents_count = write_jsonl(documents_path, export_documents)
    chunks_count = write_jsonl(chunks_path, domain_chunks)
    review_count = write_jsonl(review_path, review_documents)

    summary = {
        "domain_id": domain_id,
        "domain_name": config.get("domain_name", ""),
        "total_documents": len(documents),
        "relevant_documents": len(relevant_documents),
        "need_review_documents": len(review_documents),
        "ignored_documents": len(ignored_documents),
        "exported_documents": documents_count,
        "exported_chunks": chunks_count,
        "include_review": args.include_review,
        "outputs": {
            "documents": str(documents_path.relative_to(BASE_DIR)),
            "chunks": str(chunks_path.relative_to(BASE_DIR)),
            "need_review_documents": str(review_path.relative_to(BASE_DIR)),
            "summary": str(summary_path.relative_to(BASE_DIR)),
        },
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== DOMAIN DATASET SUMMARY ===")
    print(f"Domain              : {domain_id}")
    print(f"Total documents     : {len(documents)}")
    print(f"Relevant documents  : {len(relevant_documents)}")
    print(f"Need review docs    : {len(review_documents)}")
    print(f"Ignored documents   : {len(ignored_documents)}")
    print(f"Exported documents  : {documents_count}")
    print(f"Exported chunks     : {chunks_count}")
    print(f"Output dir          : {output_dir}")


if __name__ == "__main__":
    main()