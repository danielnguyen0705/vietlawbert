"""
build_hin_graph.py - Động cơ nạp dữ liệu Heterogeneous Information Network (HIN) vào Neo4j.
Ánh xạ đồng bộ cấu trúc 3 tầng: LawDocument -> Article -> Chunk và 22 quan hệ liên văn bản.
Tự động khởi tạo Constraints & Indices và nạp quan hệ từ pipeline ontology chuẩn hóa.
"""

from __future__ import annotations

import json
import gzip
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from configs.config import config
from database.neo4j_client import Neo4jClient
from preprocess.ast_parser import HybridASTParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_HINBuilder")

HIN_CHECKPOINT_FILE = config.STORAGE_ROOT / ".hin_graph_checkpoint.json"


def load_hin_checkpoint() -> Set[str]:
    if HIN_CHECKPOINT_FILE.exists():
        try:
            with open(HIN_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_hin_checkpoint(completed: Set[str]) -> None:
    try:
        tmp = HIN_CHECKPOINT_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(completed), f, ensure_ascii=False, indent=2)
        tmp.replace(HIN_CHECKPOINT_FILE)
    except Exception as exc:
        logger.warning("Không thể lưu HIN checkpoint: %s", exc)


class HINGraphBuilder:
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self.parser = HybridASTParser()
        # Neo4jClient đã tự động khởi tạo constraints trong hàm khởi tạo của nó

    def ingest_documents_batch(self, docs: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """Nạp các nút Văn bản pháp luật."""
        cypher = """
        UNWIND $batch AS doc
        MERGE (d:LawDocument {doc_id: doc.doc_id})
        SET d.doc_number = doc.doc_number,
            d.title = doc.title,
            d.effective_date = doc.effective_date,
            d.status = doc.status,
            d.org = doc.org,
            d.issue_date = doc.issue_date
        """
        return self.client.execute_batch(cypher, docs, batch_size=batch_size)

    def ingest_ast_hierarchy_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Nạp cấu trúc phân cấp: (LawDocument) -> [:HAS_ARTICLE] -> (Article) -> [:HAS_CHUNK] -> (Chunk)."""
        cypher = """
        UNWIND $batch AS ch
        MERGE (c:Chunk {chunk_id: ch.chunk_id})
        SET c.hierarchy_path = ch.hierarchy_path,
            c.macro_label = ch.macro_label,
            c.content = ch.text,
            c.doc_id = ch.doc_id

        WITH c, ch
        MERGE (d:LawDocument {doc_id: ch.doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)

        WITH c, ch, d
        WHERE ch.article_id IS NOT NULL AND ch.article_id <> ''
        MERGE (a:Article {article_id: ch.article_id})
        SET a.doc_id = ch.doc_id,
            a.article_num = ch.article_num,
            a.macro_label = ch.macro_label
        MERGE (d)-[:HAS_ARTICLE]->(a)
        MERGE (a)-[:HAS_CHUNK]->(c)
        """
        return self.client.execute_batch(cypher, chunks, batch_size=batch_size)

    def ingest_legal_relations_batch(self, relations: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Nạp 22 loại quan hệ pháp lý liên văn bản có định danh r.type."""
        cypher = """
        UNWIND $batch AS rel
        MERGE (src:LawDocument {doc_id: rel.source_id})
        MERGE (tgt:LawDocument {doc_id: rel.target_id})
        MERGE (src)-[r:LEGAL_RELATION {type: rel.relation_type}]->(tgt)
        SET r.updated_at = timestamp()
        """
        return self.client.execute_batch(cypher, relations, batch_size=batch_size)

    def build_from_stream(self, file_path: Path, parse_chunks: bool = True) -> None:
        logger.info("Đang nạp dữ liệu đồ thị từ tệp: %s...", file_path.name)
        opener = gzip.open(file_path, "rt", encoding="utf-8") if file_path.suffix == ".gz" else open(file_path, "r", encoding="utf-8")

        docs_buffer: List[Dict[str, Any]] = []
        relations_buffer: List[Dict[str, Any]] = []
        chunks_buffer: List[Dict[str, Any]] = []

        with opener as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    item = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue

                meta_detail = item.get("metadata_detail") or {}
                meta_api = item.get("metadata_api") or {}

                doc_id = str(item.get("doc_id") or item.get("item_id") or "UNKNOWN")
                doc_number = str(
                    item.get("doc_number")
                    or meta_detail.get("docNum")
                    or meta_api.get("docNum")
                    or "N/A"
                )
                title = str(item.get("title") or meta_detail.get("title") or meta_api.get("title") or "")
                effective_date = str(
                    item.get("effective_date") or meta_detail.get("effFrom") or meta_api.get("effFrom") or ""
                )
                status_raw = str(
                    item.get("status")
                    or (meta_detail.get("effStatus") or {}).get("name")
                    or (meta_api.get("effStatus") or {}).get("name")
                    or "Còn hiệu lực"
                )
                org = str(
                    item.get("co_quan")
                    or meta_detail.get("agencyName")
                    or meta_api.get("agencyName")
                    or "N/A"
                )

                metadata = {
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "title": title,
                    "effective_date": effective_date,
                    "status": status_raw,
                    "org": org,
                    "issue_date": str(meta_detail.get("issueDate") or meta_api.get("issueDate") or ""),
                }
                docs_buffer.append(metadata)

                # Trích xuất quan hệ ontology đã chuẩn hóa
                if item.get("relationships") and isinstance(item["relationships"], list):
                    for rel in item["relationships"]:
                        tgt_id = str(rel.get("target_id") or "").strip()
                        rel_type = str(rel.get("edge_type") or "CAN_CU_BAN_HANH").upper().replace(" ", "_")
                        direction = rel.get("direction", "OUTGOING")
                        if tgt_id and tgt_id != doc_id:
                            src_id = tgt_id if direction == "INCOMING" else doc_id
                            dst_id = doc_id if direction == "INCOMING" else tgt_id
                            relations_buffer.append({
                                "source_id": src_id,
                                "target_id": dst_id,
                                "relation_type": rel_type,
                            })
                else:
                    diagram = item.get("diagram_json") or item.get("diagram_data") or {}
                    if isinstance(diagram, dict):
                        for group_name in ["documentNamesByType", "documentNamesBySource"]:
                            group_data = diagram.get(group_name) or {}
                            is_incoming = (group_name == "documentNamesBySource")
                            if isinstance(group_data, dict):
                                for rel_k, doc_list in group_data.items():
                                    rel_type_clean = f"REL_{rel_k}"
                                    if isinstance(doc_list, list):
                                        for d_entry in doc_list:
                                            tgt = str(d_entry.get("id") or "").strip()
                                            if tgt and tgt != doc_id:
                                                relations_buffer.append({
                                                    "source_id": tgt if is_incoming else doc_id,
                                                    "target_id": doc_id if is_incoming else tgt,
                                                    "relation_type": rel_type_clean,
                                                })

                # Bóc tách AST phân rã cây phân cấp
                if parse_chunks:
                    raw_text = item.get("full_text") or item.get("text") or item.get("html_raw") or ""
                    if len(raw_text) > 100:
                        try:
                            parsed = self.parser.parse_document(raw_text, metadata)
                            for c in parsed:
                                meta_c = c.get("metadata", {})
                                chunks_buffer.append({
                                    "chunk_id": str(c.get("chunk_id") or meta_c.get("chunk_id") or ""),
                                    "doc_id": doc_id,
                                    "article_id": str(c.get("article_id") or ""),
                                    "article_num": str(c.get("article_num") or ""),
                                    "hierarchy_path": str(c.get("hierarchy_path") or meta_c.get("hierarchy_path") or ""),
                                    "macro_label": str(c.get("macro_label") or meta_c.get("macro_label") or "CHUNG"),
                                    "text": str(c.get("text") or c.get("content") or ""),
                                })
                        except Exception as parse_err:
                            logger.debug("Lỗi parse AST doc %s: %s", doc_id, parse_err)

                if len(docs_buffer) >= 500:
                    self.ingest_documents_batch(docs_buffer)
                    docs_buffer.clear()

                if len(relations_buffer) >= 1000:
                    self.ingest_legal_relations_batch(relations_buffer)
                    relations_buffer.clear()

                if len(chunks_buffer) >= 1000:
                    self.ingest_ast_hierarchy_batch(chunks_buffer)
                    chunks_buffer.clear()

        if docs_buffer:
            self.ingest_documents_batch(docs_buffer)
        if relations_buffer:
            self.ingest_legal_relations_batch(relations_buffer)
        if chunks_buffer:
            self.ingest_ast_hierarchy_batch(chunks_buffer)

        logger.info("✓ Hoàn tất nạp đồ thị Shard %s.", file_path.name)


def run_build_hin(
    metadata_path: str,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    parse_chunks: bool = True,
) -> None:
    client = Neo4jClient(uri=uri, user=user, password=password)
    builder = HINGraphBuilder(neo4j_client=client)

    completed = load_hin_checkpoint()
    path = Path(metadata_path)

    if path.is_dir():
        files = sorted([
            f for f in path.glob("*.jsonl*")
            if not f.name.endswith(".quarantine.jsonl") and not f.name.endswith(".corrupted")
        ])
        logger.info("Tìm thấy %d shards (Đã nạp HIN trước đó: %d).", len(files), len(completed))
        for f in files:
            if f.name in completed:
                logger.info("[HIN SKIP] Bỏ qua Shard đã nạp: %s", f.name)
                continue
            builder.build_from_stream(f, parse_chunks=parse_chunks)
            completed.add(f.name)
            save_hin_checkpoint(completed)
    else:
        builder.build_from_stream(path, parse_chunks=parse_chunks)

    client.close()
    logger.info("Quá trình xây dựng mạng HIN trên Neo4j đã hoàn tất mỹ mãn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng mạng thông tin dị thể HIN trên Neo4j")
    parser.add_argument("--metadata", default=str(config.STORAGE_ROOT / "raw_shards"))
    parser.add_argument("--uri", default=config.NEO4J_URI)
    parser.add_argument("--user", default=config.NEO4J_USER)
    parser.add_argument("--password", default=config.NEO4J_PASSWORD)
    parser.add_argument("--no-chunks", action="store_true", help="Chỉ nạp văn bản và quan hệ, bỏ qua nạp Chunk")
    args = parser.parse_args()

    run_build_hin(
        metadata_path=args.metadata,
        uri=args.uri,
        user=args.user,
        password=args.password,
        parse_chunks=not args.no_chunks,
    )