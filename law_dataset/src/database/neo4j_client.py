"""
neo4j_client.py - Nạp cấu trúc phân cấp và mối quan hệ liên kết đồ thị (Graph RAG) vào Neo4j.

Định dạng các cạnh quan hệ ngữ nghĩa (CAN_CO_BAN_HANH, VAN_BAN_BI_BAI_BO,...) dựa trên API diagram.
Đã refactor: dùng node label LawDocument, đọc relationships đã chuẩn hoá từ metadata.jsonl.
"""

import os
import sys
import json
import logging
from collections import defaultdict
from neo4j import GraphDatabase

from paths import METADATA_FILE, get_log_path

INPUT_FILE = METADATA_FILE
LOG_FILE_PATH = get_log_path("neo4j_client")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Neo4jClient")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "vietlawbert")

# MA TRẬN ÁNH XẠ THỰC NGHIỆM (Semantic Mapping Matrix)
# Dựa trên quan sát thực tế từ API vbpl-bientap-gateway.moj.gov.vn
# Key: (group_name, raw_key) để tránh collision giữa 2 namespace
MAP_RELATION = {
    "documentNamesByType": {
        "1": "NGHI_QUYET_HUONG_DAN_AP_DUNG",
        "3": "CAN_CO_BAN_HANH",
        "4": "PHAP_LENH",
        "9": "LUAT_HOP_NHAT",
        "10": "VAN_BAN_CHI_TIET_THI_HANH",
        "12": "VAN_BAN_DUOC_THAY_THE",
    },
    "documentNamesBySource": {
        "1": "VAN_BAN_BI_BAI_BO",
        "2": "VAN_BAN_DUOC_SU_DUNG",
        "3": "VAN_BAN_AP_DUNG",
        "4": "VAN_BAN_HUONG_DAN_AP_DUNG",
        "7": "VAN_BAN_DAN_CHIEU_LIEN_QUAN",
        "8": "VAN_BAN_LIEN_QUAN_KHAC",
        "9": "VAN_BAN_HUONG_DAN_DIA_PHUONG",
        "10": "VAN_BAN_DAN_CHIEU",
        "12": "VAN_BAN_TAM_NGUNG",
    }
}


class Neo4jManager:
    def __init__(self, drop_existing=False):
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}...")
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            self.driver.verify_connectivity()
            logger.info("✅ Connection successful!")
            if drop_existing:
                self.drop_database()
            self._create_constraints()
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise RuntimeError(f"Neo4j connection failed: {e}") from e

    def close(self):
        self.driver.close()
        logger.info("Closed Neo4j connection.")

    def drop_database(self):
        """Xóa toàn bộ node/edge hiện có. Dùng trước khi nạp batch mới trong môi trường research."""
        logger.warning("[DROP] Xóa toàn bộ database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("[DROP] Done.")

    def _create_constraints(self):
        logger.info("Creating constraints & indexes...")
        queries = [
            "CREATE CONSTRAINT law_doc_id IF NOT EXISTS FOR (d:LawDocument) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX law_doc_layer IF NOT EXISTS FOR (d:LawDocument) ON (d.graph_layer)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception:
                    pass

    def build_structural_graph(self):
        """Dựng cấu trúc phân cấp LawDocument -> Chapter -> Article -> Chunk."""
        from paths import CONTEXTUAL_CHUNKS_FILE
        logger.info(f"Building structural tree from: {CONTEXTUAL_CHUNKS_FILE}")
        if not os.path.exists(CONTEXTUAL_CHUNKS_FILE):
            logger.error(f"Contextual chunks file not found: {CONTEXTUAL_CHUNKS_FILE}")
            return

        batch_size = 500
        batch_data = []
        total_inserted = 0

        with open(CONTEXTUAL_CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = record.get("metadata", {})
                hierarchy = meta.get("hierarchy_path", {})

                node_data = {
                    "chunk_id": record.get("chunk_id", ""),
                    "original_text": record.get("original_text", ""),
                    "doc_id": meta.get("doc_id", "Unknown"),
                    "doc_number": meta.get("doc_number", "N/A"),
                    "effective_date": meta.get("effective_date", "Chưa xác định"),
                    "source_doc": meta.get("doc_type", "") + " " + meta.get("doc_number", ""),
                    "chuong": hierarchy.get("chương") or "Chương N/A",
                    "dieu": hierarchy.get("điều") or "Điều N/A"
                }
                batch_data.append(node_data)

                if len(batch_data) == batch_size:
                    self._insert_structural_batch(batch_data)
                    total_inserted += len(batch_data)
                    batch_data.clear()

            if batch_data:
                self._insert_structural_batch(batch_data)
                total_inserted += len(batch_data)

        logger.info(f"✅ Structural tree finished: {total_inserted} chunks.")
        self.build_chunk_references()

    def build_chunk_references(self):
        """Đọc cross_references từ final_contextual_chunks.jsonl để dựng quan hệ liên kết mức độ Chunk."""
        from paths import CONTEXTUAL_CHUNKS_FILE
        logger.info(f"Dựng quan hệ dẫn chiếu mức độ Chunk từ: {CONTEXTUAL_CHUNKS_FILE}")
        if not os.path.exists(CONTEXTUAL_CHUNKS_FILE):
            return

        batch_size = 500
        batch_data = []

        with open(CONTEXTUAL_CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_id = record.get("chunk_id", "")
                meta = record.get("metadata", {})
                cross_refs = meta.get("cross_references", [])

                for ref in cross_refs:
                    target_doc = ref.get("target_document", "")
                    target_num = ref.get("target_number", "")

                    if target_doc and target_num:
                        # Chuẩn hóa tên/số hiệu văn bản để link
                        batch_data.append({
                            "chunk_id": chunk_id,
                            "target_doc": target_doc,
                            "target_article": f"Điều {target_num}"
                        })

                if len(batch_data) >= batch_size:
                    self._insert_chunk_references_batch(batch_data)
                    batch_data.clear()

            if batch_data:
                self._insert_chunk_references_batch(batch_data)
        logger.info("✅ Hoàn thành quan hệ dẫn chiếu mức độ Chunk.")

    def _insert_chunk_references_batch(self, batch_data):
        # Cypher nối Chunk của văn bản A tới Article của văn bản B thông qua text matching
        # Chỉ MERGE quan hệ nếu target_doc và target_art ĐÃ TỒN TẠI trong hệ thống (MATCH)
        cypher = """
        UNWIND $batch AS row
        MATCH (ck:Chunk {chunk_id: row.chunk_id})
        MATCH (target_doc:LawDocument)
          WHERE target_doc.name = row.target_doc OR target_doc.doc_number = row.target_doc
        MATCH (target_doc)-[:HAS_CHAPTER]->()-[:HAS_ARTICLE]->(target_art:Article {name: row.target_article})
        MERGE (ck)-[r:REFERENCES]->(target_art)
        SET r.type = "CROSS_REFERENCE", r.updated_at = timestamp()
        """
        with self.driver.session() as session:
            try:
                session.run(cypher, batch=batch_data)
            except Exception as e:
                logger.error(f"Lỗi nạp chunk references batch: {e}")

    def _insert_structural_batch(self, batch_data):
        # Thiết lập mô hình Canonical Text Version (CTV) & Temporal Versioning trên đồ thị
        # Chunks được kết nối với Article thông qua CTV node có thuộc tính temporal
        cypher = """
        UNWIND $batch AS row
        MERGE (doc:LawDocument {doc_id: row.doc_id})
        SET doc.name = row.source_doc,
            doc.doc_number = row.doc_number,
            doc.effective_date = row.effective_date

        MERGE (ch:Chapter {name: row.chuong, doc_id: row.doc_id})
        MERGE (doc)-[:HAS_CHAPTER]->(ch)

        MERGE (art:Article {name: row.dieu, chapter: row.chuong, doc_id: row.doc_id})
        MERGE (ch)-[:HAS_ARTICLE]->(art)

        // Tạo node phiên bản văn bản (Canonical Text Version - CTV) có gắn mốc thời gian hiệu lực
        MERGE (ctv:CTV {ctv_id: row.doc_id + "_" + row.dieu + "_" + row.effective_date})
        SET ctv.effective_date = row.effective_date,
            ctv.doc_id = row.doc_id,
            ctv.article = row.dieu

        MERGE (art)-[:HAS_VERSION]->(ctv)

        MERGE (ck:Chunk {chunk_id: row.chunk_id})
        SET ck.text = row.original_text
        MERGE (ctv)-[:HAS_CHUNK]->(ck)
        """
        with self.driver.session() as session:
            session.run(cypher, batch=batch_data)

    @staticmethod
    def _sanitize_edge_type(raw: str) -> str:
        """Chuẩn hóa edge_type cho Cypher: bỏ ký tự đặc biệt, không bắt đầu bằng số."""
        import re as _re
        cleaned = _re.sub(r'[^A-Za-z0-9_]', '_', raw).strip('_')
        if cleaned and cleaned[0].isdigit():
            cleaned = "REL_" + cleaned
        return cleaned or "UNKNOWN_REL"

    def insert_semantic_relations_batch(self, items_batch):
        """
        Nạp một lô relationships đã chuẩn hoá từ LegalOntologyMappingPipeline.
        Gom nhóm theo (edge_type, direction) → giảm số query xuống O(k) thay vì O(n).
        """
        query_batches = defaultdict(list)

        for item in items_batch:
            source_id = item.get("item_id")
            relationships = item.get("relationships", [])

            for rel in relationships:
                safe_edge = self._sanitize_edge_type(rel["edge_type"])
                cluster_key = (safe_edge, rel["direction"])
                payload = {
                    "source_id": str(source_id),
                    "source_name": item.get("doc_number") or item.get("metadata_api", {}).get("name") or "",
                    "target_id": str(rel["target_id"]),
                    "target_name": rel.get("target_name", ""),
                    "layer": rel["graph_layer"],
                    "method": rel.get("extraction_method", "unknown"),
                }
                query_batches[cluster_key].append(payload)

        total_inserted = 0
        with self.driver.session() as session:
            for (edge_type, direction), batch_data in query_batches.items():
                if not batch_data:
                    continue

                if direction == "OUTGOING":
                    cypher = f"""
                    UNWIND $batch AS record
                    MERGE (s:LawDocument {{doc_id: record.source_id}})
                      ON CREATE SET s.name = record.source_name
                    MERGE (t:LawDocument {{doc_id: record.target_id}})
                      ON CREATE SET t.name = record.target_name
                    WITH s, t, record
                    CALL apoc.create.relationship(s, $edge_type, {{}}, t) YIELD rel
                    SET rel.graph_layer = record.layer,
                        rel.extraction_method = record.method,
                        rel.direction = "OUTGOING",
                        rel.updated_at = timestamp()
                    """
                else:  # INCOMING
                    cypher = f"""
                    UNWIND $batch AS record
                    MERGE (s:LawDocument {{doc_id: record.source_id}})
                      ON CREATE SET s.name = record.source_name
                    MERGE (t:LawDocument {{doc_id: record.target_id}})
                      ON CREATE SET t.name = record.target_name
                    WITH s, t, record
                    CALL apoc.create.relationship(t, $edge_type, {{}}, s) YIELD rel
                    SET rel.graph_layer = record.layer,
                        rel.extraction_method = record.method,
                        rel.direction = "INCOMING",
                        rel.updated_at = timestamp()
                    """

                try:
                    session.run(cypher, batch=batch_data, edge_type=edge_type)
                    total_inserted += len(batch_data)
                    logger.info(f"[BATCH] {edge_type} ({direction}): {len(batch_data)} relations")
                except Exception as e:
                    logger.error(f"[ERROR] {edge_type}: {e}")

        logger.info(f"✅ Total semantic relations inserted: {total_inserted}")
        return total_inserted

    def load_items_from_metadata(self):
        """Đọc items từ metadata.jsonl (đã qua pipeline chuẩn hoá)."""
        if not os.path.exists(INPUT_FILE):
            logger.error(f"Input file not found: {INPUT_FILE}")
            return []

        items = []
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(items)} items from {INPUT_FILE}")
        return items


# DEPRECATED: Code cũ đọc từ RAW_DIAGRAM_DIR - không còn dùng trong pipeline mới
# Giữ lại dưới dạng comment để tham khảo.
#
# def build_relations_graph_legacy(self):
#     """DEPRECATED: Đọc file diagram JSON thô - không còn dùng. Thay bằng insert_semantic_relations_batch."""
#     pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="Xóa DB trước khi nạp")
    args = parser.parse_args()

    neo = Neo4jManager(drop_existing=args.drop)
    items = neo.load_items_from_metadata()
    if items:
        neo.insert_semantic_relations_batch(items)
    neo.close()
