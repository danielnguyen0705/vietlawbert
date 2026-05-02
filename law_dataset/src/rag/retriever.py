import os
import sys
import logging
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & LOGGING TỰ ĐỘNG
# ==========================================
current_file_name = os.path.splitext(os.path.basename(__file__))[0]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filepath = os.path.join(LOG_DIR, f"log_{current_file_name}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(current_file_name.capitalize())

class LegalRetriever:
    def __init__(self):
        logger.info("Đang khởi tạo Retriever...")
        self.encoder = SentenceTransformer('BAAI/bge-m3')
        
        connections.connect("default", host="localhost", port="19530")
        self.milvus_collection = Collection("vietlaw_chunks")
        self.milvus_collection.load()
        
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "vietlawbert"))
        )
        logger.info("Retriever đã sẵn sàng!")

    def search_context(self, query: str, top_k: int = 3):
        logger.info(f"Tìm kiếm context cho: '{query}'")
        query_vector = self.encoder.encode([query])[0].tolist()
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        milvus_results = self.milvus_collection.search(
            data=[query_vector], anns_field="embedding", param=search_params,
            limit=top_k, output_fields=["chunk_id", "original_text"]
        )
        
        retrieved_chunks = [{"chunk_id": h.entity.get("chunk_id"), "text": h.entity.get("original_text")} 
                            for hits in milvus_results for h in hits]
        
        final_context = []
        with self.neo4j_driver.session() as session:
            for chunk in retrieved_chunks:
                cypher = """
                MATCH (ck:Chunk {chunk_id: $chunk_id})<-[:HAS_CHUNK]-(art:Article)<-[:HAS_ARTICLE]-(ch:Chapter)<-[:HAS_CHAPTER]-(doc:Document) 
                RETURN doc.name AS DocName, doc.doc_number AS DocNum, doc.effective_date AS EffDate, art.name AS Article
                """
                graph_result = session.run(cypher, chunk_id=chunk["chunk_id"]).data()
                meta = graph_result[0] if graph_result else {'DocName': 'N/A', 'DocNum': 'N/A', 'EffDate': 'Chưa xác định', 'Article': 'N/A'}
                
                # --- XỬ LÝ LÀM ĐẸP NGÀY HIỆU LỰC ---
                raw_date = meta.get('EffDate', 'Chưa xác định')
                clean_date = "Chưa xác định"
                if raw_date and raw_date not in ['N/A', 'None']:
                    try:
                        # Cắt "2025-01-01T00:00:00" lấy "2025-01-01" rồi đảo ngược thành "01/01/2025"
                        date_part = str(raw_date).split('T')[0]
                        y, m, d = date_part.split('-')
                        clean_date = f"{d}/{m}/{y}"
                    except Exception:
                        clean_date = raw_date # Lỗi thì giữ nguyên bản gốc
                
                final_context.append({
                    "doc_info": f"{meta['DocName']} (Số hiệu: {meta['DocNum']})",
                    "effective_date": clean_date,
                    "article": meta['Article'],
                    "content": chunk["text"]
                })
        return final_context