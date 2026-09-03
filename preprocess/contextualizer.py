"""
contextualizer.py - Module làm giàu ngữ cảnh (Context Enrichment) cho từng đoạn trích pháp luật.
Sử dụng LLM server (vLLM/Ollama) tóm lược mục đích, đối tượng áp dụng và quan hệ dẫn chiếu.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from typing import Optional

from openai import OpenAI
from confluent_kafka import Producer

from configs.paths import ROOT_DIR, get_log_path
from configs.config import config
from .legal_chunker import LegalChunk, extract_cross_references

logger = logging.getLogger("VietLawBERT_Contextualizer")

# Khởi tạo OpenAI Client kết nối tới dịch vụ suy luận LLM (vLLM / Ollama)
_llm_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=getattr(config, "LLM_API_KEY", "ollama"),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0")),
            max_retries=1,
        )
    return _llm_client


SYSTEM_PROMPT = """Bạn là chuyên gia NLP pháp lý cao cấp, chuyên phân tích văn bản quy phạm pháp luật Việt Nam.

NHIỆM VỤ: Viết đúng 1-2 câu giải thích ngắn gọn làm rõ ngữ cảnh cho đoạn trích luật.

YÊU CẦU BẮT BUỘC:
1. Xác định rõ quy định này áp dụng cho đối tượng nào (VD: người chưa thành niên, cơ quan nhà nước, doanh nghiệp...).
2. Xác định hành vi, thẩm quyền hoặc chế tài được điều chỉnh.
3. Nêu tên và số hiệu của văn bản chứa đoạn trích.

CẤU TRÚC PHẢN HỒI:
"Quy định này điều chỉnh về [HÀNH VI/CHẾ TÀI] áp dụng đối với [ĐỐI TƯỢNG], thuộc [LOẠI VĂN BẢN] số [SỐ HIỆU]. [NỘI DUNG CHÍNH]."
"""

_llm_warning_logged = False


def call_ollama(prompt: str, model: Optional[str] = None, max_tokens: int = 256) -> str:
    """Gọi LLM Server để tạo đoạn ngữ cảnh ngắn cho văn bản pháp lý."""
    global _llm_warning_logged
    client = get_llm_client()
    target_model = model or getattr(config, "CONTEXTUALIZER_MODEL", "qwen2.5:1.5b")

    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        if not _llm_warning_logged:
            logger.warning(f"Dịch vụ LLM không phản hồi ({exc}). Tự động chuyển sang sử dụng văn bản gốc.")
            _llm_warning_logged = True
        return ""


def build_prompt(chunk: LegalChunk, doc_number: str, doc_type: str, effective_date: str) -> str:
    """Xây dựng Prompt hoàn chỉnh cung cấp cây phả hệ pháp luật cho LLM."""
    hierarchy = chunk.hierarchy
    hierarchy_parts = []
    if hierarchy.phan:
        hierarchy_parts.append(hierarchy.phan)
    if hierarchy.chuong:
        hierarchy_parts.append(hierarchy.chuong)
    if hierarchy.muc:
        hierarchy_parts.append(hierarchy.muc)
    if hierarchy.tieu_muc:
        hierarchy_parts.append(hierarchy.tieu_muc)
    if hierarchy.dieu:
        hierarchy_parts.append(hierarchy.dieu)
    if hierarchy.khoan:
        hierarchy_parts.append(hierarchy.khoan)
    if hierarchy.diem:
        hierarchy_parts.append(hierarchy.diem)

    hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else "Toàn văn"

    refs = extract_cross_references(chunk.text)
    refs_text = ""
    if refs:
        refs_formatted = [f"{r['target_type']} {r['target_number']} ({r['target_document']})" for r in refs[:3]]
        refs_text = "\nDẫn chiếu trong điều khoản: " + "; ".join(refs_formatted)

    return f"""THÔNG TIN VĂN BẢN:
- Loại văn bản: {doc_type}
- Số hiệu: {doc_number}
- Ngày hiệu lực: {effective_date}
- Vị trí phân cấp: {hierarchy_str}{refs_text}

NỘI DUNG ĐOẠN TRÍCH:
{chunk.text[:2000]}
"""


def build_kafka_producer() -> Producer:
    """Khởi tạo Kafka Producer phát đoạn trích lên Message Bus."""
    broker = getattr(config, "KAFKA_BROKER", "localhost:9092")
    conf = {
        "bootstrap.servers": broker,
        "acks": "all",
        "linger.ms": 20,
        "compression.type": "zstd",
        "batch.size": 65536,
    }
    return Producer(conf)