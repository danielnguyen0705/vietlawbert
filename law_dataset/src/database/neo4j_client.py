import os
import sys
import json
import logging
from neo4j import GraphDatabase

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & LOGGING
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "json", "final_contextual_chunks.jsonl")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

current_filename = os.path.basename(__file__).split('.')[0]
log_filepath = os.path.join(LOG_DIR, f"log_{current_filename}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "vietlawbert")

class Neo4jManager:
    def __init__(self):
        logger.info(f"Đang kết nối tới Neo4j tại {NEO4J_URI}...")
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            self.driver.verify_connectivity()
            logger.info("✅ Kết nối Neo4j thành công!")
            self._create_constraints()
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối Neo4j: {str(e)}")
            sys.exit(1)

    def close(self):
        self.driver.close()
        logger.info("Đã đóng kết nối Neo4j.")

    def _create_constraints(self):
        logger.info("Đang kiểm tra và tạo Graph Constraints & Indexes...")
        queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX chapter_name IF NOT EXISTS FOR (ch:Chapter) ON (ch.name)",
            "CREATE INDEX article_name IF NOT EXISTS FOR (a:Article) ON (a.name)"
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    pass

    # ==========================================
    # 3. PIPELINE DỰNG ĐỒ THỊ BẰNG BATCH (UNWIND)
    # ==========================================
    def build_structural_graph(self):
        logger.info(f"Bắt đầu đọc dữ liệu từ: {INPUT_FILE}")
        
        batch_size = 1000
        batch_data = []
        total_inserted = 0

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                meta = record.get("metadata", {})
                hierarchy = meta.get("hierarchy", {})
                
                # Trích xuất dữ liệu BỔ SUNG ĐỂ ĐƯA VÀO GRAPH
                node_data = {
                    "chunk_id": record.get("chunk_id", ""),
                    "original_text": record.get("original_text", ""),
                    "doc_id": meta.get("doc_id", "Unknown"),
                    "doc_number": meta.get("doc_number", "N/A"),
                    "effective_date": meta.get("effective_date", "Chưa xác định"),
                    "source_doc": meta.get("source_doc", "Văn bản chưa xác định"),
                    "chuong": hierarchy.get("chuong", "Chương N/A"),
                    "dieu": hierarchy.get("dieu", "Điều N/A")
                }
                batch_data.append(node_data)
                
                if len(batch_data) == batch_size:
                    try:
                        self._insert_batch(batch_data)
                        total_inserted += batch_size
                        logger.info(f"   Đã dựng đồ thị thành công cho {total_inserted} chunks...")
                    except Exception as e:
                        logger.error(f"❌ Lỗi khi dựng đồ thị batch này: {e}")
                    batch_data = [] 

            if len(batch_data) > 0:
                try:
                    self._insert_batch(batch_data)
                    total_inserted += len(batch_data)
                    logger.info(f"   Đã dựng đồ thị thành công nốt {len(batch_data)} chunks cuối...")
                except Exception as e:
                    logger.error(f"❌ Lỗi khi dựng đồ thị đợt cuối: {e}")

        logger.info(f"🎉 HOÀN TẤT! Đã dựng xong Structural Graph cho tổng cộng {total_inserted} chunks.")

    def _insert_batch(self, batch_data):
        # LƯU Ý: Đã sửa lại cypher để lưu doc_number và effective_date vào Node Document
        cypher_query = """
        UNWIND $batch AS row
        
        // 1. Văn bản (Dùng doc_id để định danh duy nhất)
        MERGE (doc:Document {doc_id: row.doc_id})
        SET doc.name = row.source_doc,
            doc.doc_number = row.doc_number,
            doc.effective_date = row.effective_date
        
        // 2. Chương (thuộc Văn bản)
        MERGE (ch:Chapter {name: row.chuong, doc_id: row.doc_id})
        MERGE (doc)-[:HAS_CHAPTER]->(ch)
        
        // 3. Điều (thuộc Chương)
        MERGE (art:Article {name: row.dieu, chapter: row.chuong, doc_id: row.doc_id})
        MERGE (ch)-[:HAS_ARTICLE]->(art)
        
        // 4. Chunk luật (thuộc Điều)
        MERGE (ck:Chunk {chunk_id: row.chunk_id})
        SET ck.text = row.original_text
        MERGE (art)-[:HAS_CHUNK]->(ck)
        """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(cypher_query, batch=batch_data))

if __name__ == "__main__":
    neo_db = Neo4jManager()
    neo_db.build_structural_graph()
    neo_db.close()