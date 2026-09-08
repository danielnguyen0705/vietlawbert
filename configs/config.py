"""
config.py - Trung tâm điều phối tham số cấu hình hệ thống VietLawBERT (Kiến trúc v3).
Nạp biến môi trường từ .env và đồng bộ với Docker Compose (Qdrant, ES, Neo4j, Redis, Mongo).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

# Nhập khẩu an toàn nội bộ package
from .paths import ROOT_DIR, DATA_STORAGE_ROOT

# Nạp tệp cấu hình bảo mật .env từ gốc dự án
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Singleton Configuration Loader đảm bảo tính toàn vẹn kiểu dữ liệu."""

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
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "vietlawbert2026")

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
    # 7. MÔ HÌNH NHÚNG VIETLAWBERT-MRL & HUẤN LUYỆN
    # ==========================================
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 1024))
    MATRYOSHKA_DIMS: Tuple[int, ...] = (64, 128, 256, 512, 768, 1024)
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.05))
    HIERARCHY_WEIGHT: float = float(os.getenv("HIERARCHY_WEIGHT", 0.15))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", 512))

    # Cấu hình tính toán Embedding
    EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cuda")
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 32))

    # ==========================================
    # 8. CỤM TRÍ TUỆ NHÂN TẠO & TRUY XUẤT LAI (RAG / LLM)
    # ==========================================
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "ollama")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # Siêu tham số hợp nhất RRF & Bơm điểm đồ thị
    RRF_K: int = int(os.getenv("RRF_K", 60))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    GRAPH_ALPHA: float = float(os.getenv("GRAPH_ALPHA", 0.2))

    # ==========================================
    # 9. ĐIỀU PHỐI CLOUD GPU (NẾU SỬ DỤNG RUNPOD / SSH)
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


# Xuất đối tượng cấu hình duy nhất dùng cho toàn hệ sinh thái
config = Config()