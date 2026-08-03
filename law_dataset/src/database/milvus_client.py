"""
milvus_client.py - Nap chunks vao Milvus (Vector RAG)

Embedding model: BAAI/bge-m3 (1024 dim)
Dung transformers + torch truc tiep de tranh DLL conflict cua scipy/sklearn.

Tuong thich: Ubuntu 22.04+, Docker Compose v2, Python 3.12
"""

import os
import sys
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient, DataType

from paths import get_log_path, BASE_DIR

CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = get_log_path(CURRENT_FILENAME)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(CURRENT_FILENAME.capitalize())

MODEL_NAME = "BAAI/bge-m3"
VECTOR_DIM = 1024
COLLECTION_NAME = "vietlaw_chunks"
EMBED_BATCH_SIZE = 8
INSERT_BATCH_SIZE = 32

# Doc tu env de tuong thich Docker
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")


def load_encoder():
    logger.info(f"Dang tai mo hinh {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def encode_texts(tokenizer, model, texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def setup_milvus() -> MilvusClient:
    logger.info(f"Ket noi Milvus tai {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(collection_name=COLLECTION_NAME):
        logger.warning(f"Collection '{COLLECTION_NAME}' ton tai. Xoa de tao moi...")
        client.drop_collection(collection_name=COLLECTION_NAME)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id",      DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("doc_id",        DataType.VARCHAR, max_length=128)
    schema.add_field("doc_number",    DataType.VARCHAR, max_length=128)
    schema.add_field("effective_date",DataType.VARCHAR, max_length=64)
    schema.add_field("source_doc",    DataType.VARCHAR, max_length=256)
    schema.add_field("hierarchy",     DataType.VARCHAR, max_length=512)
    schema.add_field("original_text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding",     DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index("embedding", metric_type="COSINE", index_type="HNSW",
                           params={"M": 16, "efConstruction": 256})

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"Da tao Collection '{COLLECTION_NAME}'.")
    return client


def ingest_from_kafka_consumer(consumer_batch_texts: list[str], consumer_batch_rows: list[dict]) -> int:
    """
    Nạp batch dữ liệu từ Kafka Consumer (không qua JSONL trung gian).
    Returns: số chunks đã insert thành công.
    """
    if not consumer_batch_texts:
        return 0

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(collection_name=COLLECTION_NAME):
        setup_milvus()

    try:
        vectors = encode_texts(load_encoder()[0], load_encoder()[1], consumer_batch_texts)
        data = []
        for row, vec in zip(consumer_batch_rows, vectors):
            row["embedding"] = vec
            data.append(row)
        client.insert(collection_name=COLLECTION_NAME, data=data)
        logger.info(f"[MILVUS] Da insert {len(data)} chunks.")
        return len(data)
    except Exception as e:
        logger.error(f"[MILVUS ERROR] {e}")
        return 0


def ingest_data():
    """Backward-compat: doc tu CONTEXTUAL_CHUNKS_FILE neu can."""
    from paths import CONTEXTUAL_CHUNKS_FILE
    tokenizer, model = load_encoder()
    client = setup_milvus()

    if not os.path.exists(CONTEXTUAL_CHUNKS_FILE):
        logger.error(f"Khong tim thay file input: {CONTEXTUAL_CHUNKS_FILE}")
        return

    logger.info(f"Doc du lieu tu: {CONTEXTUAL_CHUNKS_FILE}")

    pending_texts: list[str] = []
    pending_rows: list[dict] = []
    total_inserted = 0

    def flush_batch():
        nonlocal total_inserted
        if not pending_rows:
            return
        try:
            vectors = encode_texts(tokenizer, model, pending_texts)
            data = []
            for row, vec in zip(pending_rows, vectors):
                row["embedding"] = vec
                data.append(row)
            client.insert(collection_name=COLLECTION_NAME, data=data)
            total_inserted += len(data)
            logger.info(f"Da nhoi {total_inserted} chunks...")
        except Exception as e:
            logger.error(f"Loi nhoi vector: {e}")
        finally:
            pending_texts.clear()
            pending_rows.clear()

    with open(CONTEXTUAL_CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("metadata", {})
            hierarchy = meta.get("hierarchy_path", {})

            pending_rows.append({
                "chunk_id": record.get("chunk_id", ""),
                "doc_id": meta.get("doc_id", ""),
                "doc_number": meta.get("doc_number", "N/A"),
                "effective_date": meta.get("effective_date", "Chua xac dinh"),
                "source_doc": meta.get("doc_type", "") + " " + meta.get("doc_number", ""),
                "hierarchy": json.dumps(hierarchy, ensure_ascii=False)[:500],
                "original_text": record.get("original_text", "")[:65000],
            })
            pending_texts.append(record.get("contextualized_text") or record.get("original_text", ""))

            if len(pending_rows) >= INSERT_BATCH_SIZE:
                flush_batch()

    flush_batch()
    logger.info(f"HOAN TAT! {total_inserted} chunks trong Milvus.")


if __name__ == "__main__":
    ingest_data()
