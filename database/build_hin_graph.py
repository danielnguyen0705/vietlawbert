"""
build_hin_graph.py - Động cơ nạp dữ liệu Heterogeneous Information Network (HIN) vào Neo4j.
Ánh xạ đồng bộ văn bản pháp luật, cây phân cấp AST (Phần -> Chương -> Điều -> Khoản) và 22 quan hệ.
"""

from __future__ import annotations

import json
import gzip
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from configs.config import config
from database.neo4j_client import Neo4jClient
from preprocess.ast_parser import HybridASTParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_HINBuilder")


class HINGraphBuilder:
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self.parser = HybridASTParser()

    def ingest_documents_batch(self, docs: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """Nạp các nút Văn bản pháp luật cùng siêu dữ liệu định danh."""
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

    def ingest_ast_chunks_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Nạp các nút Chunk và cạnh phân cấp HAS_CHUNK nối từ LawDocument."""
        cypher = """
        UNWIND $batch AS ch
        MERGE (c:Chunk {chunk_id: ch.chunk_id})
        SET c.hierarchy_path = ch.hierarchy_path,
            c.macro_label = ch.macro_label,
            c.content = ch.text
        WITH c, ch
        MATCH (d:LawDocument {doc_id: ch.doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        return self.client.execute_batch(cypher, chunks, batch_size=batch_size)

    def ingest_legal_relations_batch(self, relations: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Nạp 22 loại quan hệ pháp lý liên văn bản (SỬA_ĐỔI, BỔ_SUNG, DẪN_CHIẾU...)."""
        cypher = """
        UNWIND $batch AS rel
        MATCH (src:LawDocument {doc_id: rel.source_id})
        MATCH (tgt:LawDocument {doc_id: rel.target_id})
        MERGE (src)-[r:LEGAL_RELATION {type: rel.relation_type}]->(tgt)
        SET r.updated_at = timestamp()
        """
        return self.client.execute_batch(cypher, relations, batch_size=batch_size)

    def build_from_stream(self, file_path: Path, parse_chunks: bool = True) -> None:
        """Đọc tệp Shard thô (.jsonl hoặc .jsonl.gz), bóc tách AST và nạp HIN đồ thị."""
        logger.info("Đang nạp dữ liệu đồ thị từ tệp: %s...", file_path.name)
        opener = gzip.open(file_path, "rt", encoding="utf-8") if file_path.suffix == ".gz" else open(file_path, "r", encoding="utf-8")

        docs = []
        relations = []
        all_chunks = []

        with opener as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    item = json.loads(clean_line)
                except json.JSONDecodeError:
                    continue

                doc_id = str(item.get("doc_id") or item.get("item_id") or "UNKNOWN")
                doc_number = str(item.get("doc_number") or item.get("docNum") or "N/A")
                metadata = {
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "title": str(item.get("title", "")),
                    "effective_date": str(item.get("effective_date", "")),
                    "status": str(item.get("status", "Còn hiệu lực")),
                    "co_quan": str(item.get("co_quan") or item.get("org", "N/A")),
                    "issue_date": str(item.get("issue_date", "")),
                }
                docs.append(metadata)

                # Trích xuất 22 quan hệ pháp lý từ diagram_data
                diagram = item.get("diagram_data") or item.get("diagram_json") or {}
                if isinstance(diagram, dict):
                    for rel_name, targets in diagram.items():
                        if isinstance(targets, list):
                            for tgt_id in targets:
                                relations.append({
                                    "source_id": doc_id,
                                    "target_id": str(tgt_id),
                                    "relation_type": str(rel_name).upper().replace(" ", "_"),
                                })

                # Phân rã cây AST Chunk nếu có văn bản toàn văn
                if parse_chunks:
                    raw_text = item.get("full_text") or item.get("text") or item.get("html_raw") or ""
                    if len(raw_text) > 100:
                        parsed_chunks = self.parser.parse_document(raw_text, metadata)
                        all_chunks.extend(parsed_chunks)

        self.ingest_documents_batch(docs)
        self.ingest_legal_relations_batch(relations)
        if all_chunks:
            self.ingest_ast_chunks_batch(all_chunks)

        logger.info("✓ Hoàn tất nạp đồ thị cho %s: %d docs, %d cạnh, %d chunks.", file_path.name, len(docs), len(relations), len(all_chunks))


def run_build_hin(metadata_path: str, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
    client = Neo4jClient(uri=uri, user=user, password=password)
    builder = HINGraphBuilder(neo4j_client=client)

    path = Path(metadata_path)
    if path.is_dir():
        for f in sorted(path.glob("*.jsonl*")):
            builder.build_from_stream(f)
    else:
        builder.build_from_stream(path)

    client.close()
    logger.info("Quá trình xây dựng mạng HIN trên Neo4j đã hoàn tất mỹ mãn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng mạng thông tin dị thể HIN trên Neo4j")
    parser.add_argument("--metadata", default="./data/raw/enriched_metadata.jsonl")
    parser.add_argument("--uri", default=config.NEO4J_URI)
    parser.add_argument("--user", default=config.NEO4J_USER)
    parser.add_argument("--password", default=config.NEO4J_PASSWORD)
    args = parser.parse_args()

    run_build_hin(args.metadata, args.uri, args.user, args.password)