"""
config.py - Trung tâm điều phối tham số cấu hình hệ thống VietLawBERT (Kiến trúc v3 - 2026).
Nạp biến môi trường từ .env và đồng bộ với Docker Compose (Qdrant, ES, Neo4j, Redis, Mongo).
Tương thích toàn diện GPU Compute Capability >= 7.0 (Volta, Turing, Ampere, Ada, Hopper).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from dotenv import load_dotenv

from .paths import ROOT_DIR, DATA_STORAGE_ROOT

logger = logging.getLogger("VietLawBERT_Config")

ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

import torch


def _check_valid_cuda() -> bool:
    """Kiểm tra CUDA và hỗ trợ các GPU hiện đại từ Compute Capability >= 7.0 (Volta, Ampere, Hopper)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 7
    except Exception:
        return False


def _resolve_embed_device() -> str:
    target_device = os.getenv("EMBED_DEVICE", "").strip().lower()
    if not target_device:
        return "cuda" if _check_valid_cuda() else "cpu"
    if "cuda" in target_device and not _check_valid_cuda():
        logger.warning("GPU hiện tại không hỗ trợ kernel sm_70+ (GTX 1050 Ti CC 6.1). Fallback an toàn về 'cpu'.")
        return "cpu"
    return target_device


class Config:
    STORAGE_ROOT: Path = DATA_STORAGE_ROOT

    # ==============================================================================
    # 1. CƠ SỞ DỮ LIỆU LAI (HYBRID STORAGE STACK)
    # ==============================================================================
    # MongoDB Document Store
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "vietlawbert_db")

    # Neo4j Knowledge Graph (HIN 22 Quan hệ pháp lý)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "vietlawbert2026")

    # Qdrant Dense Vector Store (MRL d=256)
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "vietlawbert_chunks")
    QDRANT_VECTOR_DIM: int = int(os.getenv("QDRANT_VECTOR_DIM", 256))

    # Elasticsearch 8.x Sparse Retriever
    ES_HOST: str = os.getenv("ES_HOST", "http://localhost:9200")
    ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME", "vietlaw_sparse_idx")

    # Redis In-Memory Cache (Pre-computed Graph Embeddings 128d)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # ==============================================================================
    # 2. MÔ HÌNH BIỂU DIỄN & HUẤN LUYỆN (VIETLAWBERT-MRL CORE)
    # ==============================================================================
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 1024))
    MATRYOSHKA_DIMS: Tuple[int, ...] = (64, 128, 256, 512, 768, 1024)
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.05))
    HIERARCHY_WEIGHT: float = float(os.getenv("HIERARCHY_WEIGHT", 0.15))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", 512))

    EMBED_DEVICE: str = _resolve_embed_device()
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 32))

    # ==============================================================================
    # 3. CHUỖI ĐIỀU PHỐI LLM ĐA TẦNG (RAG & CASCADING FALLBACK)
    # ==============================================================================
    # Tầng 1: Grounded Generator chính thức
    PRIMARY_LLM_MODEL: str = os.getenv("PRIMARY_LLM_MODEL", "anthropic/claude-3.5-sonnet")
    PRIMARY_LLM_API_KEY: str = os.getenv("PRIMARY_LLM_API_KEY", "")
    PRIMARY_LLM_API_BASE: str = os.getenv("PRIMARY_LLM_API_BASE", "https://openrouter.ai/api/v1")

    # Tầng 2: Fallback Generator
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "gemini-3.8-flash")
    FALLBACK_LLM_API_KEY: str = os.getenv("FALLBACK_LLM_API_KEY", "")
    FALLBACK_LLM_API_BASE: str = os.getenv("FALLBACK_LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")

    # Tầng 3: Local Engine (Ollama / vLLM)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "ollama")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # Tác vụ Tiền xử lý & Làm giàu ngữ cảnh (Contextualizer qua HHTechAPI / Anthropic)
    CONTEXTUALIZER_API_BASE: str = os.getenv("CONTEXTUALIZER_API_BASE", "https://hhtechapi.net/v1")
    CONTEXTUALIZER_API_KEY: str = os.getenv("CONTEXTUALIZER_API_KEY", "")
    CONTEXTUALIZER_MODEL: str = os.getenv("CONTEXTUALIZER_MODEL", "claude-3-5-haiku")

    # Tác vụ Query Rewriting & Router
    ROUTER_LLM_MODEL: str = os.getenv("ROUTER_LLM_MODEL", "anthropic/claude-3.5-haiku")
    ROUTER_LLM_API_KEY: str = os.getenv("ROUTER_LLM_API_KEY", "")
    ROUTER_LLM_API_BASE: str = os.getenv("ROUTER_LLM_API_BASE", "https://openrouter.ai/api/v1")

    # ==============================================================================
    # 4. ĐÁNH GIÁ THỰC NGHIỆM KHOA HỌC (BENCHMARK & LLM-AS-A-JUDGE RQ4)
    # ==============================================================================
    EVAL_MODEL: str = os.getenv("EVAL_MODEL", "anthropic/claude-3.5-sonnet")
    EVAL_API_KEY: str = os.getenv("EVAL_API_KEY", "")
    EVAL_API_BASE: str = os.getenv("EVAL_API_BASE", "https://openrouter.ai/api/v1")

    # Siêu tham số hợp nhất RRF & Bơm điểm đồ thị
    RRF_K: int = int(os.getenv("RRF_K", 60))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    GRAPH_ALPHA: float = float(os.getenv("GRAPH_ALPHA", 0.2))

    # ==============================================================================
    # 5. THU THẬP DỮ LIỆU & OCR PIPELINE
    # ==============================================================================
    CRAWLER_CONCURRENCY: int = int(os.getenv("CRAWLER_CONCURRENCY", 8))
    CRAWLER_DOWNLOAD_DELAY: float = float(os.getenv("CRAWLER_DOWNLOAD_DELAY", 0.1))
    CRAWL_PAGE_SIZE: int = int(os.getenv("CRAWL_PAGE_SIZE", 100))
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "1") in ("1", "true", "True")
    OCR_INLINE_ENABLED: bool = os.getenv("OCR_INLINE_ENABLED", "1") in ("1", "true", "True")
    OCR_LANG: str = os.getenv("OCR_LANG", "vie+eng")
    OCR_DPI: int = int(os.getenv("OCR_DPI", 200))
    OCR_MAX_PAGES: int = int(os.getenv("OCR_MAX_PAGES", 200))
    OCR_CONCURRENCY: int = int(os.getenv("OCR_CONCURRENCY", 2))

    # ==============================================================================
    # 6. ĐIỀU PHỐI CLOUD GPU & HUGGING FACE
    # ==============================================================================
    ENABLE_CLOUD_GPU: bool = os.getenv("ENABLE_CLOUD_GPU", "false").lower() in ("true", "1", "yes")
    CLOUD_GPU_PROVIDER: str = os.getenv("CLOUD_GPU_PROVIDER", "runpod")
    CLOUD_GPU_INSTANCE_ID: str = os.getenv("CLOUD_GPU_INSTANCE_ID", "")
    CLOUD_GPU_SSH_HOST: str = os.getenv("CLOUD_GPU_SSH_HOST", "")
    CLOUD_GPU_SSH_PORT: int = int(os.getenv("CLOUD_GPU_SSH_PORT", 22))
    CLOUD_GPU_SSH_USER: str = os.getenv("CLOUD_GPU_SSH_USER", "root")
    CLOUD_GPU_SSH_KEY_PATH: str = os.getenv("CLOUD_GPU_SSH_KEY_PATH", "~/.ssh/id_rsa")
    CLOUD_GPU_API_KEY: str = os.getenv("CLOUD_GPU_API_KEY", "")
    CLOUD_AUTO_SHUTDOWN: bool = os.getenv("CLOUD_AUTO_SHUTDOWN", "true").lower() in ("true", "1", "yes")
    CLOUD_COST_ALERT_THRESHOLD_USD: float = float(os.getenv("CLOUD_COST_ALERT_THRESHOLD_USD", 5.0))


config = Config()