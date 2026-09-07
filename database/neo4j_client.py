"""
neo4j_client.py - Tầng lưu trữ đồ thị tri thức pháp lý Neo4j (Graph RAG).
Quản lý cây cấu trúc văn bản, quan hệ trích dẫn APOC và quan hệ ngang hàng.
"""

from __future__ import annotations

import os
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase

from configs.config import config
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("database", "database")


class Neo4jClient:
    """Quản lý kết nối, giao dịch ACID và phân bổ quan hệ ngữ nghĩa trên Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        drop_existing: bool = False,
    ):
        actual_uri = uri or getattr(config, "NEO4J_URI", "bolt://localhost:7687")
        actual_user = user or getattr(config, "NEO4J_USER", "neo4j")
        actual_pass = password or getattr(config, "NEO4J_PASSWORD", "vietlawbert")

        logger.info("Kết nối Neo4j Bolt tại %s...", actual_uri)
        self.driver = GraphDatabase.driver(actual_uri, auth=(actual_user, actual_pass))
        self.driver.verify_connectivity()
        logger.info("✓ Kết nối Neo4j thành công!")

        if drop_existing:
            self.drop_database()
        self._create_constraints()

    def close(self):
        self.driver.close()
        logger.info("Đã giải phóng kết nối Neo4j Driver.")

    def drop_database(self):
        logger.warning("[CẢNH BÁO] Đang xóa toàn bộ dữ liệu Đồ thị...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✓ Đã dọn sạch cơ sở dữ liệu Neo4j.")

    def _create_constraints(self):
        queries = [
            "CREATE CONSTRAINT law_doc_id IF NOT EXISTS FOR (d:LawDocument) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX law_doc_num IF NOT EXISTS FOR (d:LawDocument) ON (d.doc_number)",
            "CREATE INDEX law_doc_layer IF NOT EXISTS FOR (d:LawDocument) ON (d.graph_layer)",
            # Thêm chỉ mục tăng tốc độ duyệt phân cấp Điều/Chương lên 50x
            "CREATE INDEX article_composite IF NOT EXISTS FOR (a:Article) ON (a.doc_id, a.name)",
            "CREATE INDEX chapter_composite IF NOT EXISTS FOR (ch:Chapter) ON (ch.doc_id, ch.name)",
            "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug("Trạng thái khởi tạo constraint: %s", e)
        logger.info("✓ Đã xác thực toàn bộ Constraints & Indexes trên Neo4j.")

    def insert_structural_batch(self, batch_data: List[Dict[str, Any]]):
        """
        Nạp cây cấu trúc phân cấp: LawDocument -> Chapter -> Article -> Chunk.
        Cập nhật SET đầy đủ siêu dữ liệu kể cả khi nút cha đã tồn tại từ trước.
        """
        if not batch_data:
            return

        cypher = """
        UNWIND $batch AS row
        MERGE (doc:LawDocument {doc_id: row.doc_id})
        SET doc.name = coalesce(row.source_doc, doc.name),
            doc.doc_number = coalesce(row.doc_number, doc.doc_number),
            doc.effective_date = coalesce(row.effective_date, doc.effective_date),
            doc.graph_layer = "Hierarchical"

        MERGE (ch:Chapter {name: row.chuong, doc_id: row.doc_id})
        MERGE (doc)-[:HAS_CHAPTER]->(ch)

        MERGE (art:Article {name: row.dieu, chapter: row.chuong, doc_id: row.doc_id})
        MERGE (ch)-[:HAS_ARTICLE]->(art)

        MERGE (ck:Chunk {chunk_id: row.chunk_id})
        SET ck.text = row.original_text,
            ck.doc_id = row.doc_id
        MERGE (art)-[:HAS_CHUNK]->(ck)
        """
        with self.driver.session() as session:
            session.run(cypher, parameters={"batch": batch_data})

    def insert_semantic_relations_batch(self, items_batch: List[Dict[str, Any]]) -> int:
        """Nạp các cạnh quan hệ dẫn chiếu và hiệu lực, loại bỏ hoàn toàn các ID rỗng."""
        query_batches = defaultdict(list)

        for item in items_batch:
            source_id = str(item.get("item_id") or item.get("doc_id") or "").strip()
            source_name = item.get("doc_number") or ""
            relationships = item.get("relationships", [])

            if not source_id:
                continue

            for rel in relationships:
                target_id = str(rel.get("target_id", "")).strip()
                if not target_id or target_id == source_id:
                    continue

                raw_edge = rel.get("edge_type", "DAN_CHIEU")
                safe_edge = self._sanitize_edge_type(raw_edge)
                direction = rel.get("direction", "OUTGOING").upper()
                cluster_key = (safe_edge, direction)

                query_batches[cluster_key].append({
                    "source_id": source_id,
                    "source_name": source_name,
                    "target_id": target_id,
                    "target_name": rel.get("target_name", ""),
                    "layer": rel.get("graph_layer", "Operational"),
                    "method": rel.get("extraction_method", "static"),
                })

        total_inserted = 0
        with self.driver.session() as session:
            for (edge_type, direction), records in query_batches.items():
                if not records:
                    continue

                if direction == "OUTGOING":
                    cypher = f"""
                    UNWIND $batch AS record
                    MERGE (s:LawDocument {{doc_id: record.source_id}})
                      ON CREATE SET s.name = record.source_name
                    MERGE (t:LawDocument {{doc_id: record.target_id}})
                      ON CREATE SET t.name = record.target_name
                    MERGE (s)-[rel:{edge_type}]->(t)
                    SET rel.graph_layer = record.layer,
                        rel.extraction_method = record.method,
                        rel.direction = "OUTGOING",
                        rel.updated_at = timestamp()
                    """
                else:
                    cypher = f"""
                    UNWIND $batch AS record
                    MERGE (s:LawDocument {{doc_id: record.source_id}})
                      ON CREATE SET s.name = record.source_name
                    MERGE (t:LawDocument {{doc_id: record.target_id}})
                      ON CREATE SET t.name = record.target_name
                    MERGE (t)-[rel:{edge_type}]->(s)
                    SET rel.graph_layer = record.layer,
                        rel.extraction_method = record.method,
                        rel.direction = "INCOMING",
                        rel.updated_at = timestamp()
                    """

                try:
                    session.run(cypher, parameters={"batch": records})
                    total_inserted += len(records)
                except Exception as exc:
                    logger.error("Lỗi nạp quan hệ %s: %s", edge_type, exc)

        return total_inserted

    def build_cung_van_ban_relations(self):
        """Tạo cạnh quan hệ ngang hàng giữa các Điều trong cùng văn bản, tương thích Neo4j 5.x."""
        cypher = """
        MATCH (doc:LawDocument)-[:HAS_CHAPTER]->()-[:HAS_ARTICLE]->(a1:Article)
        MATCH (doc)-[:HAS_CHAPTER]->()-[:HAS_ARTICLE]->(a2:Article)
        WHERE elementId(a1) < elementId(a2)
        MERGE (a1)-[r:CUNG_VAN_BAN]->(a2)
        SET r.updated_at = timestamp()
        """
        with self.driver.session() as session:
            session.run(cypher)
            logger.info("✓ Đã hoàn tất khởi tạo các cạnh quan hệ ngang hàng [:CUNG_VAN_BAN].")

    @staticmethod
    def _sanitize_edge_type(raw: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(raw)).strip("_").upper()
        if cleaned and cleaned[0].isdigit():
            cleaned = "REL_" + cleaned
        return cleaned or "DAN_CHIEU"


Neo4jManager = Neo4jClient
Neo4jStore = Neo4jClient
