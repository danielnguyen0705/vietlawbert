"""
contextualizer.py - Utility functions for contextualization.
Chỉ chứa các hàm utility: build_prompt, call_ollama.
Logic batch processing đã được chuyển hoàn toàn vào streaming consumer.
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

from paths import BASE_DIR, get_log_path

from config import config
from openai import OpenAI

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# ============================================================
# CẤU HÌNH LLM SERVER (Ollama/vLLM/OpenAI compatible)
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
    Gọi LLM server qua OpenAI-compatible API.
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
        logger.error(f"[LLM ERROR] {e}")
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