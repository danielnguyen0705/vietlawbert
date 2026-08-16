"""Quality gate cho crawl artifact và dữ liệu ingestion Milvus/Neo4j."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ARTICLE_HEADING = re.compile(r"^[\s#*_“”\"'‘’(\[]*Điều\s+\d+", re.IGNORECASE)
BOILERPLATE = re.compile(
    r"CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM|Độc\s+lập\s*[-–—]\s*Tự\s+do\s*[-–—]\s*Hạnh\s+phúc",
    re.IGNORECASE,
)


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL lỗi ở dòng {line_number}: {exc}") from exc


def audit_crawl(
    path: Path,
    expected_documents: int | None,
    allow_upstream_missing: bool = False,
    allow_ocr_pending: bool = False,
):
    # Không giữ payload HTML trong RAM: full corpus được ước tính hơn 10 GiB.
    unique_ids: set[str] = set()
    unresolved_keys: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    for record in read_jsonl(path):
        counters["records"] += 1
        item_id = str(record.get("item_id", "")).strip()
        if item_id:
            unique_ids.add(item_id)
        else:
            counters["missing_document_id"] += 1

        status_valid = record.get("html_status") == "VALID"
        content_valid = status_valid and len(str(record.get("html_raw") or "").strip()) >= 100
        counters["html_status_valid"] += status_valid
        counters["html_valid"] += content_valid
        metadata_detail = record.get("metadata_detail") or {}
        upstream_missing = (
            not content_valid
            and str((record.get("rescue_file") or {}).get("fileName") or "").lower()
            == "template.pdf"
            and int((record.get("rescue_file") or {}).get("size") or 0) == 32052
            and record.get("upstream_content_unavailable") is True
            and str(record.get("rescue_status") or "")
            in {"OCR_EMPTY", "FILE_NOT_FOUND", "UPSTREAM_TEMPLATE"}
        )
        counters["upstream_content_unavailable"] += upstream_missing
        ocr_pending = not content_valid and str(record.get("ocr_status") or "") == "OCR_PENDING"
        counters["ocr_pending"] += ocr_pending
        counters[f"diagram_{str(record.get('diagram_status') or '').lower()}"] += 1

        metadata_api = record.get("metadata_api") or {}
        doc_type = metadata_api.get("docType") or {}
        counters["translated_documents"] += str(doc_type.get("code") or "").upper() == "BD"
        counters["source_is_lw_false"] += metadata_api.get("isLw") is False
        counters["vbqppl_parent_documents"] += (
            str(doc_type.get("parentCode") or "").upper() == "VBQPPL"
        )
        counters["administrative_documents"] += metadata_detail.get("isAdministrativeDocument") is True
        unresolved_keys.update(
            f"{entry.get('group')}:{entry.get('key')}"
            for entry in (record.get("diagram_unresolved_keys") or [])
        )

    record_count = counters["records"]
    html_content_valid = counters["html_valid"]
    result = {
        "records": record_count,
        "unique_documents": len(unique_ids),
        "duplicate_documents": record_count - counters["missing_document_id"] - len(unique_ids),
        "missing_document_id": counters["missing_document_id"],
        "html_valid": html_content_valid,
        "html_invalid": record_count - html_content_valid,
        "upstream_content_unavailable": counters["upstream_content_unavailable"],
        "ocr_pending": counters["ocr_pending"],
        "html_invalid_unexplained": (
            record_count
            - html_content_valid
            - counters["upstream_content_unavailable"]
            - counters["ocr_pending"]
        ),
        "html_status_valid_but_empty": counters["html_status_valid"] - html_content_valid,
        "translated_documents": counters["translated_documents"],
        "source_is_lw_false": counters["source_is_lw_false"],
        "vbqppl_parent_documents": counters["vbqppl_parent_documents"],
        "administrative_documents": counters["administrative_documents"],
        "diagram_valid": counters["diagram_valid"],
        "diagram_empty": counters["diagram_empty"],
        "diagram_review_required": counters["diagram_inconsistent"],
        "unresolved_relation_keys": dict(sorted(unresolved_keys.items())),
    }
    failures = []
    if expected_documents is not None and len(unique_ids) != expected_documents:
        failures.append(f"expected {expected_documents} documents, got {len(unique_ids)}")
    if result["duplicate_documents"]:
        failures.append("crawl artifact contains duplicate document IDs")
    if result["missing_document_id"]:
        failures.append("crawl artifact contains records without item_id")
    if result["html_invalid_unexplained"]:
        failures.append(f"{result['html_invalid_unexplained']} documents do not have valid HTML")
    if result["upstream_content_unavailable"] and not allow_upstream_missing:
        failures.append(
            f"{result['upstream_content_unavailable']} documents are unavailable from upstream"
        )
    if result["ocr_pending"] and not allow_ocr_pending:
        failures.append(f"{result['ocr_pending']} documents are pending OCR")
    if result["translated_documents"]:
        failures.append(f"{result['translated_documents']} translated documents are not primary corpus records")
    if result["administrative_documents"]:
        failures.append(f"{result['administrative_documents']} administrative documents entered the legal corpus")
    if result["diagram_review_required"]:
        failures.append(f"{result['diagram_review_required']} diagrams are inconsistent")
    result["failures"] = failures
    result["passed"] = not failures
    return result


def fetch_all_milvus_rows(client, collection_name):
    output_fields = ["chunk_id", "doc_id", "hierarchy", "original_text"]
    if hasattr(client, "query_iterator"):
        iterator = client.query_iterator(
            collection_name=collection_name,
            batch_size=1000,
            filter="chunk_id != ''",
            output_fields=output_fields,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                yield from batch
        finally:
            iterator.close()
        return

    yield from client.query(
        collection_name=collection_name,
        filter="chunk_id != ''",
        output_fields=output_fields,
        limit=16384,
    )


def audit_databases(expected_documents: int | None, document_ids: set[str] | None = None):
    from neo4j import GraphDatabase
    from pymilvus import MilvusClient

    collection = os.getenv("MILVUS_COLLECTION", "vietlaw_chunks")
    milvus_uri = os.getenv(
        "MILVUS_URI",
        f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}",
    )
    milvus = MilvusClient(uri=milvus_uri)
    milvus.flush(collection_name=collection)
    rows = list(fetch_all_milvus_rows(milvus, collection))
    if document_ids is not None:
        rows = [row for row in rows if str(row.get("doc_id")) in document_ids]
    milvus_ids = {str(row.get("chunk_id")) for row in rows}
    source_doc_ids = {str(row.get("doc_id")) for row in rows if row.get("doc_id")}

    hierarchy_mismatches = 0
    boilerplate_chunks = 0
    empty_chunks = 0
    max_chunk_chars = int(os.getenv("AUDIT_MAX_CHUNK_CHARS", "1600"))
    oversized_chunks = 0
    for row in rows:
        text = str(row.get("original_text") or "").strip()
        try:
            hierarchy = json.loads(row.get("hierarchy") or "{}")
        except (TypeError, json.JSONDecodeError):
            hierarchy = {}
        hierarchy_mismatches += bool(ARTICLE_HEADING.match(text) and not hierarchy.get("điều"))
        boilerplate_chunks += bool(BOILERPLATE.search(text))
        empty_chunks += not bool(text)
        oversized_chunks += len(text) > max_chunk_chars

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "vietlawbert")),
    )
    with driver.session() as session:
        neo_ids = {
            record["id"]
            for record in session.run(
                """
                MATCH (c:Chunk)
                WHERE $doc_ids IS NULL OR c.doc_id IN $doc_ids
                RETURN c.chunk_id AS id
                """,
                doc_ids=sorted(document_ids) if document_ids is not None else None,
            )
        }
        duplicate_relations = session.run(
            """
            MATCH (a:LawDocument)-[r]->(b:LawDocument)
            WHERE $doc_ids IS NULL OR a.doc_id IN $doc_ids
            WITH a.doc_id AS source, type(r) AS kind, b.doc_id AS target, count(r) AS copies
            WHERE copies > 1
            RETURN count(*) AS groups, coalesce(sum(copies - 1), 0) AS extras
            """,
            doc_ids=sorted(document_ids) if document_ids is not None else None,
        ).single()
    driver.close()

    failures = []
    if milvus_ids != neo_ids:
        failures.append("Milvus and Neo4j chunk ID sets differ")
    if expected_documents is not None and len(source_doc_ids) != expected_documents:
        failures.append(f"expected {expected_documents} source documents, got {len(source_doc_ids)}")
    if duplicate_relations["groups"]:
        failures.append("Neo4j contains duplicate semantic relations")
    if hierarchy_mismatches:
        failures.append(f"{hierarchy_mismatches} article headings have missing article hierarchy")
    if boilerplate_chunks:
        failures.append(f"{boilerplate_chunks} chunks still contain boilerplate")
    if empty_chunks:
        failures.append(f"{empty_chunks} chunks are empty")
    if oversized_chunks:
        failures.append(f"{oversized_chunks} chunks exceed {max_chunk_chars} characters")

    return {
        "milvus_chunks": len(milvus_ids),
        "neo4j_chunks": len(neo_ids),
        "milvus_only_chunks": len(milvus_ids - neo_ids),
        "neo4j_only_chunks": len(neo_ids - milvus_ids),
        "source_documents": len(source_doc_ids),
        "duplicate_relation_groups": duplicate_relations["groups"],
        "duplicate_relation_extras": duplicate_relations["extras"],
        "article_hierarchy_mismatches": hierarchy_mismatches,
        "boilerplate_chunks": boilerplate_chunks,
        "empty_chunks": empty_chunks,
        "oversized_chunks": oversized_chunks,
        "max_chunk_chars": max_chunk_chars,
        "failures": failures,
        "passed": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit pilot crawl/ingestion")
    parser.add_argument("--crawl-file", type=Path)
    parser.add_argument("--databases", action="store_true")
    parser.add_argument("--expect-documents", type=int)
    parser.add_argument(
        "--allow-upstream-missing",
        action="store_true",
        help="cho phép record hasContent=false + Template.pdf vào quarantine thay vì fail shard",
    )
    parser.add_argument(
        "--allow-ocr-pending",
        action="store_true",
        help="cho phép PDF scan vào quarantine để OCR ở pha worker riêng",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.crawl_file and not args.databases:
        parser.error("chọn --crawl-file, --databases hoặc cả hai")

    report = {}
    document_ids = None
    if args.crawl_file:
        document_ids = {
            str(record.get("item_id", "")).strip()
            for record in read_jsonl(args.crawl_file)
            if str(record.get("item_id", "")).strip()
        }
        report["crawl"] = audit_crawl(
            args.crawl_file,
            args.expect_documents,
            allow_upstream_missing=args.allow_upstream_missing,
            allow_ocr_pending=args.allow_ocr_pending,
        )
    if args.databases:
        report["databases"] = audit_databases(args.expect_documents, document_ids)
    report["passed"] = all(section["passed"] for key, section in report.items() if key != "passed")

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
