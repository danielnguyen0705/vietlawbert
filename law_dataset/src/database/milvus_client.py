import os
import sys
import json
import logging
from datetime import datetime
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & LOGGING ĐỘNG
# ==========================================

# Xác định thư mục gốc law_dataset/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

# Đường dẫn file input (Dữ liệu từ bước Contextualizer)
INPUT_FILE = os.path.join(BASE_DIR, "json", "final_contextual_chunks.jsonl")

# Cấu hình Thư mục Log theo ngày (Y-m-d)
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", TODAY_STR)
os.makedirs(LOG_DIR, exist_ok=True)

# Tự động lấy tên file hiện tại làm tên file log
CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = os.path.join(LOG_DIR, f"log_{CURRENT_FILENAME}.log")

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # mode="a" để ghi nối tiếp nếu chạy nhiều lần trong ngày
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. KHỞI TẠO EMBEDDING MODEL (BGE-M3)
# ==========================================
logger.info("Đang tải mô hình BAAI/bge-m3 (Lần đầu có thể mất vài phút)...")
model = SentenceTransformer('BAAI/bge-m3')
VECTOR_DIM = 1024 

# ==========================================
# 3. KẾT NỐI MILVUS & TẠO COLLECTION
# ==========================================
COLLECTION_NAME = "vietlaw_chunks"

def setup_milvus():
    logger.info("Đang kết nối tới Milvus Docker (localhost:19530)...")
    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection(COLLECTION_NAME):
        logger.warning(f"Collection '{COLLECTION_NAME}' đã tồn tại. Xóa để tạo mới với Schema nâng cấp...")
        utility.drop_collection(COLLECTION_NAME)

    # ĐÃ THÊM CÁC TRƯỜNG METADATA MỚI VÀO ĐÂY
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="doc_number", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="hierarchy", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="original_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    ]
    
    schema = CollectionSchema(fields, "Kho chứa Vector Luật Giao Thông")
    collection = Collection(COLLECTION_NAME, schema)
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    logger.info(f"✅ Đã tạo thành công Collection '{COLLECTION_NAME}' và Index.")
    return collection

# ==========================================
# 4. CHẠY PIPELINE NHỒI DỮ LIỆU
# ==========================================
def ingest_data():
    collection = setup_milvus()
    logger.info(f"Bắt đầu đọc dữ liệu từ: {INPUT_FILE}")
    
    batch_size = 5 
    data_batch = {
        "chunk_id": [], "doc_id": [], "doc_number": [], "effective_date": [],
        "source_doc": [], "hierarchy": [], "original_text": [], "texts_to_embed": []
    }
    
    total_inserted = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            meta = record.get("metadata", {})
            
            data_batch["chunk_id"].append(record.get("chunk_id", ""))
            data_batch["doc_id"].append(meta.get("doc_id", ""))
            data_batch["doc_number"].append(meta.get("doc_number", "N/A"))
            data_batch["effective_date"].append(meta.get("effective_date", "Chưa xác định"))
            data_batch["source_doc"].append(meta.get("source_doc", ""))
            data_batch["hierarchy"].append(json.dumps(meta.get("hierarchy", {}), ensure_ascii=False))
            data_batch["original_text"].append(record.get("original_text", ""))
            data_batch["texts_to_embed"].append(record.get("contextualized_text", ""))
            
            if len(data_batch["chunk_id"]) == batch_size:
                try:
                    embeddings = model.encode(data_batch["texts_to_embed"], batch_size=2, show_progress_bar=False)
                    collection.insert([
                        data_batch["chunk_id"], data_batch["doc_id"], data_batch["doc_number"], 
                        data_batch["effective_date"], data_batch["source_doc"], 
                        data_batch["hierarchy"], data_batch["original_text"], embeddings.tolist()
                    ])
                    total_inserted += batch_size
                    logger.info(f"   Đã nhồi thành công {total_inserted} chunks vào Milvus...")
                except Exception as e:
                    logger.error(f"❌ Lỗi khi nhúng vector: {e}")
                data_batch = {k: [] for k in data_batch}

    if len(data_batch["chunk_id"]) > 0:
        try:
            embeddings = model.encode(data_batch["texts_to_embed"], batch_size=2, show_progress_bar=False)
            collection.insert([
                data_batch["chunk_id"], data_batch["doc_id"], data_batch["doc_number"], 
                data_batch["effective_date"], data_batch["source_doc"], 
                data_batch["hierarchy"], data_batch["original_text"], embeddings.tolist()
            ])
            total_inserted += len(data_batch["chunk_id"])
            logger.info(f"   Đã nhồi thành công nốt {len(data_batch['chunk_id'])} chunks cuối...")
        except Exception as e:
            logger.error(f"❌ Lỗi khi nhúng vector ở đợt cuối: {e}")
        
    collection.flush()
    logger.info(f"🎉 HOÀN TẤT! Tổng cộng {total_inserted} chunks đã nằm an toàn trong Milvus.")

if __name__ == "__main__":
    ingest_data()