"""
crawl_audit.py - Công cụ kiểm toán chất lượng văn bản và tính nhất quán giữa các CSDL.
Triển khai các tiêu chuẩn kiểm định chặt chẽ phục vụ công bố khoa học (ACL/EMNLP Data Sanity).
"""

from __future__ import annotations

import os
import sys
import json
import gzip
import re
import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Iterator

from configs.paths import ROOT_DIR, ARTIFACTS_DIR, get_log_path
from configs.config import config
from artifacts.canonical import read_jsonl

logger = logging.getLogger("VietLawBERT_CrawlAudit")

# Tập hợp các ký tự tiếng Việt có dấu chuẩn Unicode dựng sẵn và tổ hợp
VIETNAMESE_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

ARTICLE_HEADING = re.compile(r"^[\s#*_“”\"'‘’(\[]*Điều\s+\d+", re.IGNORECASE)
BOILERPLATE = re.compile(
    r"CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM|Độc\s+lập\s*[-–—]\s*Tự\s+do\s*[-–—]\s*Hạnh\s+phúc",
    re.IGNORECASE,
)


def evaluate_linguistic_quality(text: str) -> Dict[str, Any]:
    """Kiểm tra tỷ lệ nguyên âm tiếng Việt và phát hiện lỗi vỡ bảng mã Unicode."""
    if not text:
        return {"vietnamese_ratio": 0.0, "has_encoding_error": False, "is_valid": False}

    alpha_chars = [c for c in text if c.isalpha()]
    total_alpha = len(alpha_chars)
    if total_alpha == 0:
        return {"vietnamese_ratio": 0.0, "has_encoding_error": False, "is_valid": False}

    vn_count = sum(1 for c in alpha_chars if c in VIETNAMESE_CHARS)
    ratio = vn_count / total_alpha
    has_encoding_error = ("\ufffd" in text) or ("\x00" in text)

    return {
        "vietnamese_ratio": round(ratio, 4),
        "has_encoding_error": has_encoding_error,
        "is_valid": ratio >= 0.05 and not has_encoding_error,
    }


def audit_crawl(
    path: Path | str,
    expected_documents: Optional[int] = None,
    allow_upstream_missing: bool = False,
    allow_ocr_pending: bool = False,
) -> Dict[str, Any]:
    """
    Kiểm toán toàn diện một tệp Shard (.jsonl hoặc .jsonl.gz).
    Bảo đảm không giữ payload HTML trong RAM để tối ưu hóa bộ nhớ O(1).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp artifact: {p}")

    unique_ids: Set[str] = set()
    unresolved_keys: Counter = Counter()
    counters: Counter = Counter()

    for record in read_jsonl(p):
        counters["records"] += 1
        item_id = str(record.get("item_id") or record.get("id") or "").strip()
        if item_id:
            unique_ids.add(item_id)
        else:
            counters["missing_document_id"] += 1

        html_raw = str(record.get("html_raw") or "").strip()
        status_valid = record.get("html_status") == "VALID"
        content_valid = status_valid and len(html_raw) >= 100

        counters["html_status_valid"] += status_valid
        counters["html_valid"] += content_valid

        # Kiểm định chất lượng ngôn ngữ học
        if content_valid:
            ling_eval = evaluate_linguistic_quality(html_raw)
            if not ling_eval["is_valid"]:
                counters["linguistic_quality_rejected"] += 1

        # Phân tích nguyên nhân thiếu văn bản
        rescue_file = record.get("rescue_file") or {}
        upstream_missing = (
            not content_valid
            and str(rescue_file.get("fileName") or "").lower() == "template.pdf"
            and int(rescue_file.get("size") or 0) == 32052
            and record.get("upstream_content_unavailable") is True
            and str(record.get("rescue_status") or "") in {"OCR_EMPTY", "FILE_NOT_FOUND", "UPSTREAM_TEMPLATE"}
        )
        counters["upstream_content_unavailable"] += upstream_missing
        ocr_pending = not content_valid and str(record.get("ocr_status") or "") == "OCR_PENDING"
        counters["ocr_pending"] += ocr_pending

        diagram_status = str(record.get("diagram_status") or "EMPTY").lower()
        counters[f"diagram_{diagram_status}"] += 1

        metadata_api = record.get("metadata_api") or {}
        metadata_detail = record.get("metadata_detail") or {}
        doc_type = metadata_api.get("docType") or {}

        counters["translated_documents"] += str(doc_type.get("code") or "").upper() == "BD"
        counters["administrative_documents"] += metadata_detail.get("isAdministrativeDocument") is True

        for entry in record.get("diagram_unresolved_keys") or []:
            unresolved_keys.update([f"{entry.get('group')}:{entry.get('key')}"])

    record_count = counters["records"]
    html_content_valid = counters["html_valid"]

    result = {
        "file": p.name,
        "records": record_count,
        "unique_documents": len(unique_ids),
        "duplicate_documents": record_count - counters["missing_document_id"] - len(unique_ids),
        "missing_document_id": counters["missing_document_id"],
        "html_valid": html_content_valid,
        "html_invalid": record_count - html_content_valid,
        "linguistic_quality_rejected": counters["linguistic_quality_rejected"],
        "upstream_content_unavailable": counters["upstream_content_unavailable"],
        "ocr_pending": counters["ocr_pending"],
        "html_invalid_unexplained": (
            record_count
            - html_content_valid
            - counters["upstream_content_unavailable"]
            - counters["ocr_pending"]
        ),
        "translated_documents": counters["translated_documents"],
        "administrative_documents": counters["administrative_documents"],
        "diagram_valid": counters["diagram_valid"],
        "diagram_empty": counters["diagram_empty"],
        "diagram_review_required": counters["diagram_inconsistent"],
        "unresolved_relation_keys": dict(sorted(unresolved_keys.items())),
    }

    failures = []
    if expected_documents is not None and len(unique_ids) != expected_documents:
        failures.append(f"Kỳ vọng {expected_documents} văn bản, thực tế phát hiện {len(unique_ids)}")
    if result["duplicate_documents"] > 0:
        failures.append(f"Tồn tại {result['duplicate_documents']} văn bản trùng lặp ID")
    if result["missing_document_id"] > 0:
        failures.append(f"Tồn tại {result['missing_document_id']} bản ghi thiếu trường định danh item_id")
    if result["html_invalid_unexplained"] > 0:
        failures.append(f"{result['html_invalid_unexplained']} văn bản hỏng không rõ nguyên nhân")
    if result["upstream_content_unavailable"] > 0 and not allow_upstream_missing:
        failures.append(f"{result['upstream_content_unavailable']} văn bản chưa được công bố nội dung gốc")
    if result["ocr_pending"] > 0 and not allow_ocr_pending:
        failures.append(f"{result['ocr_pending']} văn bản PDF scan đang chờ xử lý OCR")
    if result["translated_documents"] > 0:
        failures.append(f"{result['translated_documents']} văn bản dịch thuật (BD) không thuộc kho chính quy")
    if result["administrative_documents"] > 0:
        failures.append(f"{result['administrative_documents']} văn bản hành chính thông thường bị lẫn vào kho luật")

    result["failures"] = failures
    result["passed"] = len(failures) == 0
    return result


def fetch_all_milvus_rows(client, collection_name: str) -> Iterator[Dict[str, Any]]:
    """Trích xuất dữ liệu Milvus theo lô lớn có kiểm soát bộ nhớ."""
    output_fields = ["chunk_id", "doc_id", "hierarchy", "original_text"]
    try:
        # Sử dụng Query Iterator của PyMilvus nếu khả dụng
        iterator = client.query_iterator(
            collection_name=collection_name,
            batch_size=1000,
            filter="chunk_id != ''",
            output_fields=output_fields,
        )
        while True:
            batch = iterator.next()
            if not batch:
                iterator.close()
                break
            for item in batch:
                yield item
    except Exception:
        # Fallback phân trang bằng LIMIT/OFFSET an toàn
        offset = 0
        limit = 10000
        while True:
            batch = client.query(
                collection_name=collection_name,
                filter="chunk_id != ''",
                output_fields=output_fields,
                limit=limit,
                offset=offset,
            )
            if not batch:
                break
            for item in batch:
                yield item
            if len(batch) < limit:
                break
            offset += limit


def audit_databases(
    expected_documents: Optional[int] = None,
    document_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Kiểm toán tính nhất quán 1:1 giữa Milvus (Vector) và Neo4j (Graph).
    Phát hiện Chunks mồ côi, văn bản dính boilerplate và lệch cấu trúc phân cấp.
    """
    from pymilvus import MilvusClient
    from neo4j import GraphDatabase

    collection = getattr(config, "MILVUS_COLLECTION_NAME", "vietlawbert_chunks")
    milvus_uri = getattr(config, "MILVUS_URI", "http://localhost:19530")

    logger.info(f"Đang kết nối Milvus [{milvus_uri}] và đối soát collection [{collection}]...")
    milvus = MilvusClient(uri=milvus_uri)
    if not milvus.has_collection(collection_name=collection):
        return {
            "passed": False,
            "error": f"Collection '{collection}' không tồn tại trong Milvus!",
            "failures": [f"Collection '{collection}' missing"],
        }

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
        except Exception:
            hierarchy = {}

        hierarchy_mismatches += bool(ARTICLE_HEADING.match(text) and not hierarchy.get("điều"))
        boilerplate_chunks += bool(BOILERPLATE.search(text))
        empty_chunks += len(text) == 0
        oversized_chunks += len(text) > max_chunk_chars

    # Kết nối Neo4j đối soát cấu trúc
    driver = GraphDatabase.driver(
        getattr(config, "NEO4J_URI", "bolt://localhost:7687"),
        auth=(getattr(config, "NEO4J_USER", "neo4j"), getattr(config, "NEO4J_PASSWORD", "vietlawbert")),
    )
    with driver.session() as session:
        neo_ids = {
            record["id"]
            for record in session.run(
                "MATCH (c:Chunk) "
                "WHERE $doc_ids IS NULL OR c.doc_id IN $doc_ids "
                "RETURN c.chunk_id AS id",
                doc_ids=sorted(list(document_ids)) if document_ids is not None else None,
            )
        }

        # Kiểm tra Chunk mồ côi (không gắn với Article nào)
        orphan_chunks = session.run(
            "MATCH (c:Chunk) WHERE NOT ()-[:HAS_CHUNK]->(c) RETURN count(c) AS total"
        ).single()["total"]

        # Kiểm tra trùng lặp cạnh quan hệ ngữ nghĩa
        duplicate_relations = session.run(
            "MATCH (a:LawDocument)-[r]->(b:LawDocument) "
            "WHERE $doc_ids IS NULL OR a.doc_id IN $doc_ids "
            "WITH a.doc_id AS source, type(r) AS kind, b.doc_id AS target, count(r) AS copies "
            "WHERE copies > 1 "
            "RETURN count(*) AS groups, coalesce(sum(copies - 1), 0) AS extras",
            doc_ids=sorted(list(document_ids)) if document_ids is not None else None,
        ).single()

    driver.close()

    failures = []
    if milvus_ids != neo_ids:
        failures.append(
            f"Tập Chunk ID giữa Milvus ({len(milvus_ids)}) và Neo4j ({len(neo_ids)}) không khớp!"
        )
    if expected_documents is not None and len(source_doc_ids) != expected_documents:
        failures.append(f"Kỳ vọng {expected_documents} văn bản nguồn, thực tế chỉ có {len(source_doc_ids)}")
    if orphan_chunks > 0:
        failures.append(f"Phát hiện {orphan_chunks} Chunk mồ côi (không kết nối vào Article)")
    if duplicate_relations["groups"] > 0:
        failures.append(f"Tồn tại {duplicate_relations['groups']} nhóm quan hệ ngữ nghĩa bị trùng lặp trên Neo4j")
    if hierarchy_mismatches > 0:
        failures.append(f"{hierarchy_mismatches} tiêu đề Điều bị khuyết metadata phân cấp")
    if boilerplate_chunks > 0:
        failures.append(f"{boilerplate_chunks} chunks vẫn còn sót boilerplate hành chính")
    if empty_chunks > 0:
        failures.append(f"{empty_chunks} chunks rỗng không có nội dung")
    if oversized_chunks > 0:
        failures.append(f"{oversized_chunks} chunks vượt quá độ dài quy định {max_chunk_chars} ký tự")

    return {
        "milvus_chunks": len(milvus_ids),
        "neo4j_chunks": len(neo_ids),
        "milvus_only_chunks": len(milvus_ids - neo_ids),
        "neo4j_only_chunks": len(neo_ids - milvus_ids),
        "source_documents": len(source_doc_ids),
        "orphan_chunks": orphan_chunks,
        "duplicate_relation_groups": duplicate_relations["groups"],
        "duplicate_relation_extras": duplicate_relations["extras"],
        "article_hierarchy_mismatches": hierarchy_mismatches,
        "boilerplate_chunks": boilerplate_chunks,
        "empty_chunks": empty_chunks,
        "oversized_chunks": oversized_chunks,
        "failures": failures,
        "passed": len(failures) == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Bộ kiểm toán chất lượng kho ngữ liệu VietLawBERT")
    parser.add_argument("--crawl-file", type=Path, help="Đường dẫn file shard cần kiểm toán")
    parser.add_argument("--databases", action="store_true", help="Kiểm tra đối soát Milvus và Neo4j")
    parser.add_argument("--expect-documents", type=int, help="Số lượng văn bản kỳ vọng")
    parser.add_argument("--allow-upstream-missing", action="store_true", help="Chấp nhận template rỗng cách ly")
    parser.add_argument("--allow-ocr-pending", action="store_true", help="Chấp nhận PDF chờ OCR")
    parser.add_argument("--output", type=Path, help="Đường dẫn tệp xuất báo cáo JSON")
    args = parser.parse_args()

    if not args.crawl_file and not args.databases:
        parser.error("Vui lòng chọn ít nhất một tác vụ: --crawl-file hoặc --databases")

    report = {}
    doc_ids = None

    if args.crawl_file:
        report["crawl"] = audit_crawl(
            args.crawl_file,
            expected_documents=args.expect_documents,
            allow_upstream_missing=args.allow_upstream_missing,
            allow_ocr_pending=args.allow_ocr_pending,
        )
        doc_ids = {
            str(r.get("item_id") or r.get("id")).strip()
            for r in read_jsonl(args.crawl_file)
            if str(r.get("item_id") or r.get("id")).strip()
        }

    if args.databases:
        report["databases"] = audit_databases(
            expected_documents=args.expect_documents,
            document_ids=doc_ids,
        )

    all_passed = all(sec.get("passed", False) for sec in report.values())
    report["overall_passed"] = all_passed

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())