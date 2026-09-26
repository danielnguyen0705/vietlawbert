"""
config.py - Trung tâm điều phối tham số cấu hình hệ thống VietLawBERT (Kiến trúc v3).
Nạp biến môi trường từ .env và đồng bộ với Docker Compose (Qdrant, ES, Neo4j, Redis, Mongo).
Tích hợp hàm kiểm tra Compute Capability an toàn và cấu hình điều phối Cascading LLM.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from dotenv import load_dotenv

from .paths import ROOT_DIR, DATA_STORAGE_ROOT

# Thiết lập Logger hệ thống
logger = logging.getLogger("VietLawBERT_Config")

# 1. NẠP BIẾN MÔI TRƯỜNG .ENV TRƯỚC KHI IMPORT TORCH
# Bắt buộc nạp sớm để các cờ CUDA_VISIBLE_DEVICES có hiệu lực ở tầng C-extension
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

import torch  # Nạp sau khi môi trường đã được đồng bộ


def _check_valid_cuda() -> bool:
    """Kiểm tra CUDA khả dụng và phần cứng đạt chuẩn tối thiểu sm_75 (Turing/Ampere/Ada Lovelace)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
        # Ngăn chặn các GPU cũ như Pascal sm_61 (GTX 1050 Ti) gây sập PyTorch cu130
        return (major > 7) or (major == 7 and minor >= 5)
    except Exception:
        return False


def _resolve_embed_device() -> str:
    """Cơ chế phòng thủ đa tầng: Ép về CPU nếu GPU không hỗ trợ Compute Capability >= 7.5."""
    target_device = os.getenv("EMBED_DEVICE", "").strip().lower()
    
    if not target_device:
        return "cuda" if _check_valid_cuda() else "cpu"
    
    if "cuda" in target_device and not _check_valid_cuda():
        logger.warning(
            "Phát hiện cấu hình EMBED_DEVICE='cuda' nhưng GPU hiện tại không hỗ trợ kernel sm_75+. "
            "Tự động fallback an toàn về 'cpu' để tránh ngắt tiến trình."
        )
        return "cpu"
        
    return target_device


class Config:
    """Singleton Configuration Loader đảm bảo tính toàn vẹn kiểu dữ liệu và an toàn phần cứng."""

    # ==========================================
    # 1. HẠ TẦNG LƯU TRỮ TẬP TRUNG (STORAGE)
    # ==========================================
    STORAGE_ROOT: Path = DATA_STORAGE_ROOT

    # ==========================================
    # 2. CƠ SỞ DỮ LIỆU TÀI LIỆU (MONGODB)
    # ==========================================
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "vietlawbert_db")

    # ==========================================
    # 3. CƠ SỞ DỮ LIỆU ĐỒ THỊ DỊ THỂ (NEO4J - HIN 22 QUAN HỆ)
    # ==========================================
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "vietlawbert_secure_pass")

    # ==========================================
    # 4. DENSE VECTOR ENGINE (QDRANT - MRL d=256)
    # ==========================================
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "vietlawbert_chunks")
    QDRANT_VECTOR_DIM: int = int(os.getenv("QDRANT_VECTOR_DIM", 256))

    # ==========================================
    # 5. SPARSE LEXICAL ENGINE (ELASTICSEARCH - CUSTOM ANALYZER)
    # ==========================================
    ES_HOST: str = os.getenv("ES_HOST", "http://localhost:9200")
    ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME", "vietlaw_sparse_idx")

    # ==========================================
    # 6. IN-MEMORY GRAPH CACHE (REDIS - COMPILE-TIME EMBEDDINGS)
    # ==========================================
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # ==========================================
    # 7. MÔ HÌNH NHÚNG VIETLAWBERT-MRL & THIẾT BỊ TÍNH TOÁN
    # ==========================================
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "/mnt/data/vietlawbert_data/models/vietlawbert_mrl_final")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "/mnt/data/vietlawbert_data/models/vietlawbert_mrl_final")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 1024))
    MATRYOSHKA_DIMS: Tuple[int, ...] = (64, 128, 256, 512, 768, 1024)
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.05))
    HIERARCHY_WEIGHT: float = float(os.getenv("HIERARCHY_WEIGHT", 0.15))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", 512))

    # Tự động phòng thủ phần cứng qua _resolve_embed_device()
    EMBED_DEVICE: str = _resolve_embed_device()
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 16))

    # ==========================================
    # 8. ĐỒNG BỘ HUGGING FACE HUB (KHI THUÊ CLOUD GPU)
    # ==========================================
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    HF_MODEL_REPO_ID: Optional[str] = os.getenv("HF_MODEL_REPO_ID", None)

    # ==========================================
    # 9. ĐIỀU PHỐI GENERATOR ĐA TẦNG (CASCADING DISPATCHER)
    # ==========================================
    # Tầng 1: Cloud SOTA / Fast (Google AI Studio - Gemini 3.8 Flash)
    PRIMARY_LLM_MODEL: str = os.getenv("PRIMARY_LLM_MODEL", "gemini-3.8-flash")
    PRIMARY_LLM_API_KEY: str = os.getenv("PRIMARY_LLM_API_KEY", "")
    PRIMARY_LLM_API_BASE: str = os.getenv("PRIMARY_LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")

    # Tầng 2: Cloud Open-weights (OpenRouter Free - Qwen 2.5 72B)
    FALLBACK_LLM_MODEL: str = os.getenv("FALLBACK_LLM_MODEL", "qwen/qwen-2.5-72b-instruct:free")
    FALLBACK_LLM_API_KEY: str = os.getenv("FALLBACK_LLM_API_KEY", "")
    FALLBACK_LLM_API_BASE: str = os.getenv("FALLBACK_LLM_API_BASE", "https://openrouter.ai/api/v1")

    # Tầng 3: Local Safety Net (Ollama cục bộ khi ngoại tuyến)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "ollama")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    CONTEXTUALIZER_MODEL: str = os.getenv("CONTEXTUALIZER_MODEL", "Qwen/Qwen2.5-14B-Instruct")

    # Cấu hình RRF & Rerank
    RRF_K: int = int(os.getenv("RRF_K", 60))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    GRAPH_ALPHA: float = float(os.getenv("GRAPH_ALPHA", 0.2))

    # ==========================================
    # 10. ĐIỀU PHỐI CLOUD GPU (KHI ĐÀO TẠO PHÂN TÁN)
    # ==========================================
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
