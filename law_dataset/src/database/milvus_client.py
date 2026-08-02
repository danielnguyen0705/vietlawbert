"""
milvus_client.py - Nap chunks vao Milvus (Vector RAG)

Embedding model: BAAI/bge-m3 (1024 dim)
Dung transformers + torch truc tiep de tranh DLL conflict cua scipy/sklearn.
"""

import os
import sys
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import MilvusClient, DataType

from paths import get_log_path, CONTEXTUAL_CHUNKS_FILE, BASE_DIR

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
        # CLS token embedding
        embeddings = output.last_hidden_state[:, 0, :]
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def setup_milvus() -> MilvusClient:
    logger.info("Ket noi Milvus Docker (localhost:19530)...")
    client = MilvusClient(uri="http://localhost:19530")

    if client.has_collection(collection_name=COLLECTION_NAME):
        logger.warning(f"Collection '{COLLECTION_NAME}' ton tai. Xoa de tao moi...")
        client.drop_collection(collection_name=COLLECTION_NAME)

    # Dung schema explicit de tranh conflict voi auto-generated 'id' field cua MilvusClient v3
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


def ingest_data():
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
