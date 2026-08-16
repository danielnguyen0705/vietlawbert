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
from dataclasses import dataclass
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

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
VECTOR_DIM = 1024
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "vietlaw_chunks")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto").lower()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()
INSERT_BATCH_SIZE = 32

# Doc tu env de tuong thich Docker
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")


@dataclass
class RemoteEmbeddingModel:
    client: object
    model_name: str


def load_encoder():
    if EMBED_PROVIDER == "openai_compatible":
        from openai import OpenAI

        base_url = os.environ["EMBED_API_BASE"]
        model_name = os.environ["EMBED_API_MODEL"]
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("EMBED_API_KEY", "not-required"),
            timeout=float(os.getenv("EMBED_API_TIMEOUT", "120")),
            max_retries=int(os.getenv("EMBED_API_MAX_RETRIES", "3")),
        )
        logger.info("Embedding remote ready: base_url=%s, model=%s", base_url, model_name)
        return None, RemoteEmbeddingModel(client=client, model_name=model_name)
    if EMBED_PROVIDER != "local":
        raise ValueError(f"EMBED_PROVIDER không được hỗ trợ: {EMBED_PROVIDER}")

    if EMBED_DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = EMBED_DEVICE

    logger.info(f"Dang tai mo hinh {MODEL_NAME} tren {device}...")
    local_only = os.getenv("EMBED_LOCAL_FILES_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=local_only)
    model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=local_only)
    model.to(device)
    if device == "cuda" and os.getenv("EMBED_FP16", "1") == "1":
        model.half()
    if device == "cpu" and os.getenv("EMBED_CPU_INT8", "0") == "1":
        logger.info("Áp dụng dynamic INT8 quantization cho CPU...")
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    model.eval()
    model._vietlawbert_device = device
    logger.info(
        "Embedding ready: device=%s, batch_size=%d, max_length=%d",
        device, EMBED_BATCH_SIZE, EMBED_MAX_LENGTH,
    )
    return tokenizer, model


def encode_texts(tokenizer, model, texts: list[str]) -> list[list[float]]:
    if isinstance(model, RemoteEmbeddingModel):
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            response = model.client.embeddings.create(
                model=model.model_name,
                input=texts[i:i + EMBED_BATCH_SIZE],
                encoding_format="float",
            )
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            if any(len(vector) != VECTOR_DIM for vector in batch_embeddings):
                raise ValueError(f"Embedding API phải trả vector {VECTOR_DIM} chiều")
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    all_embeddings = []
    device = getattr(model, "_vietlawbert_device", None)
    if device is None:
        device = next(model.parameters()).device

    # Gom các văn bản có độ dài gần nhau để giảm padding CPU/GPU, sau đó trả
    # embedding về đúng thứ tự đầu vào trước khi ghép với metadata.
    ordered_indices = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    ordered_embeddings = [None] * len(texts)
    for i in range(0, len(ordered_indices), EMBED_BATCH_SIZE):
        batch_indices = ordered_indices[i:i + EMBED_BATCH_SIZE]
        batch = [texts[index] for index in batch_indices]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=EMBED_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        for original_index, embedding in zip(batch_indices, embeddings.float().cpu().tolist()):
            ordered_embeddings[original_index] = embedding
    return ordered_embeddings


def setup_milvus() -> MilvusClient:
    logger.info(f"Ket noi Milvus tai {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(collection_name=COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' da ton tai. Dung lai schema cu.")
        return client

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
        client.upsert(collection_name=COLLECTION_NAME, data=data)
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
            client.upsert(collection_name=COLLECTION_NAME, data=data)
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
