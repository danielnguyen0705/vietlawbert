"""
contextualizer.py - Sinh Ngữ Cảnh cho Chunk và đẩy thẳng vào Kafka.

Sử dụng vLLM serving engine với prefix caching thay cho Ollama.
Tăng throughput lên ~20x so với Ollama batching đơn lẻ.

Khởi động vLLM server:
    vllm serve qwen2.5:7b-instruct --enable-prefix-caching --port 8000
hoặc:
    vllm serve qwen2.5:14b-instruct --enable-prefix-caching --port 8000

Khởi động Kafka:
    docker compose up -d kafka redpanda
"""

import os
import sys
import json
import time
import logging
from dotenv import load_dotenv
from confluent_kafka import Producer
import socket

from preprocess.text_cleaner import (
    clean_boilerplate,
    extract_doc_type,
    extract_doc_number,
    extract_effective_date
)
from preprocess.legal_chunker import (
    chunk_legal_document,
    extract_cross_references,
    LegalChunk
)

from paths import BASE_DIR, MD_DIR, get_log_path, ensure_dirs

ensure_dirs()
LOG_FILE_PATH = get_log_path("contextualizer")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Contextualizer")

from config import config
from openai import OpenAI

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# ============================================================
# CẤU HÌNH vLLM SERVER (Thay thế Ollama để tăng throughput ~20x)
# ============================================================
client = OpenAI(base_url=config.LLM_API_BASE, api_key=config.LLM_API_KEY)
CONTEXTUALIZER_MODEL = config.CONTEXTUALIZER_MODEL
REQUEST_DELAY = float(os.getenv("CONTEXTUALIZER_DELAY", "0.05"))

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC_CHUNKS = os.getenv("KAFKA_TOPIC_CHUNKS", "law-documents")


SYSTEM_PROMPT = """Ban la chuyen gia NLP phap ly cao cap, chuyen phan tich van ban phap luat Viet Nam.

NHIEM VU: Viet dung 1-2 cau ngon ngu giai thich ngan gon cho doan trich luat.

YEU CAU BAT BUOC:
1. Xac dinh ro doan trich nay ap dung cho doi tuong nao (VD: nguoi thanh nien hay chua thanh nien, do tuoi cu the, to chuc, ca nhan...)
2. Xac dinh hanh vi hoac toi danh duoc dieu chinh
3. Xac dinh moi lien ket logic voi cac dieu khoan khac neu co (dan chieu)
4. Neu ten va muc dich cot loi va so hieu cua van ban

CAU TRUC CAU:
"Day la quy dinh ve [MUC DICH] doi voi [DOI TUONG], thuoc [LOAI VAN BAN] so [SO HIEU]. [NOI DUNG CHINH]."

TUYET DOI KHONG:
- Su dung cau truc rap khuon vo nghia
- Viet qua dai (gioi han 2-3 cau)
- Bo sot thong tin quan trong ve doi tuong ap dung
"""


def call_ollama(prompt: str, model: str = None, max_tokens: int = 512) -> str:
    """
    Gọi vLLM server qua OpenAI-compatible API.
    Prefix caching giảm latency cho các prompt có chung system prompt.
    """
    model = model or CONTEXTUALIZER_MODEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[vLLM ERROR] {e}")
        return ""


def build_prompt(chunk: LegalChunk, doc_number: str, doc_type: str, effective_date: str) -> str:
    """Xay dung prompt cho chunk van ban phap luat."""
    hierarchy_str = []
    if chunk.hierarchy.chuong:
        hierarchy_str.append(chunk.hierarchy.chuong)
    if chunk.hierarchy.dieu:
        hierarchy_str.append(chunk.hierarchy.dieu)
    if chunk.hierarchy.khoan:
        hierarchy_str.append(chunk.hierarchy.khoan)
    if chunk.hierarchy.diem:
        hierarchy_str.append(chunk.hierarchy.diem)

    hierarchy_text = " > ".join(hierarchy_str) if hierarchy_str else "Toan van"

    refs = extract_cross_references(chunk.text)
    refs_text = ""
    if refs:
        refs_text = "\nDan chieu trong doan trich: " + "; ".join([
            f"{r['target_type']} {r['target_number']} {r['target_document']}"
            for r in refs[:3]
        ])

    return f"""{SYSTEM_PROMPT}

THONG TIN VAN BAN:
- Loai: {doc_type}
- So hieu: {doc_number}
- Hieu luc: {effective_date}
- Vi tri: {hierarchy_text}
{refs_text}

DOAN TRICH:
{chunk.text[:2000]}
"""


def build_kafka_producer() -> Producer:
    """Khởi tạo Kafka producer với cấu hình tối ưu throughput."""
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'client.id': socket.gethostname(),
        'acks': 'all',
        'linger.ms': 50,
        'batch.size': 128 * 1024,
        'compression.type': 'snappy',
        'queue.buffering.max.messages': 100000,
    }
    return Producer(conf)


def process_and_push():
    """
    Đọc từ MD_DIR, sinh ngữ cảnh, đẩy thẳng vào Kafka.
    KHÔNG ghi ra file JSONL trung gian.
    """
    logger.info(f"[BAT DAU] Contextualizer - Model: {CONTEXTUALIZER_MODEL}")

    producer = build_kafka_producer()
    total_chunks = 0
    total_files = 0

    md_files = sorted([f for f in os.listdir(MD_DIR) if f.endswith('.md')])
    logger.info(f"[INFO] Tìm thấy {len(md_files)} file markdown.")

    for filename in md_files:
        file_path = os.path.join(MD_DIR, filename)
        doc_id = filename.replace(".md", "")

        logger.info(f"[DANG XU LY] {doc_id}")

        with open(file_path, "r", encoding="utf-8") as f_in:
            raw_md = f_in.read()

        doc_type = extract_doc_type(raw_md)
        doc_number = extract_doc_number(raw_md) or doc_id
        effective_date = extract_effective_date(raw_md) or "Chua xac dinh"

        cleaned_md = clean_boilerplate(raw_md)
        chunks = chunk_legal_document(cleaned_md, doc_id)

        for chunk in chunks:
            if chunk.metadata.get('type') == 'preamble':
                continue
            if len(chunk.text.strip()) < 50:
                continue

            prompt = build_prompt(chunk, doc_number, doc_type, effective_date)
            context = call_ollama(prompt, max_tokens=150)

            cross_refs = extract_cross_references(chunk.text)
            contextualized_text = (
                f"Context: {context}\n\nContent:\n{chunk.text}"
                if context
                else chunk.text
            )

            record = {
                "chunk_id": chunk.chunk_id,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "doc_type": doc_type,
                    "effective_date": effective_date,
                    "hierarchy_path": chunk.hierarchy.to_dict(),
                    "cross_references": cross_refs[:5],
                },
                "original_text": chunk.text,
                "contextualized_text": contextualized_text,
            }

            producer.produce(
                KAFKA_TOPIC_CHUNKS,
                value=json.dumps(record, ensure_ascii=False).encode('utf-8')
            )
            producer.poll(0)
            total_chunks += 1

        total_files += 1
        logger.info(f"  -> {len(chunks)} chunks đã đẩy vào Kafka.")

    producer.flush(30)
    logger.info(f"[HOAN TAT] {total_files} files, {total_chunks} chunks đã được đẩy vào Kafka.")
    logger.info(f"[OUTPUT] Kafka topic: {KAFKA_TOPIC_CHUNKS}")


if __name__ == "__main__":
    process_and_push()
