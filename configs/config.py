"""
config.py - Trung tâm điều phối tham số cấu hình hệ thống VietLawBERT.
Nạp biến môi trường từ .env và đồng bộ với Docker Compose, Kafka, Milvus, Neo4j.
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
    # 2. CƠ SỞ DỮ LIỆU ĐỒ THỊ (NEO4J)
    # ==========================================
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "vietlawbert")

    # ==========================================
    # 3. CƠ SỞ DỮ LIỆU VECTOR PHÂN TÁN (MILVUS)
    # ==========================================
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", 19530))
    MILVUS_URI: str = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "vietlawbert_chunks")

    # ==========================================
    # 4. CƠ SỞ DỮ LIỆU TÀI LIỆU (MONGODB)
    # ==========================================
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "vietlawbert_db")

    # ==========================================
    # 5. ĐIỀU PHỐI LUỒNG SỰ KIỆN (KAFKA / REDPANDA)
    # ==========================================
    KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")
    KAFKA_TOPIC: str = os.getenv("KAFKA_TOPIC", "law-documents-v5")
    KAFKA_GROUP_ID: str = os.getenv("KAFKA_GROUP_ID", "vietlawbert-consumers-v5-bounded")
    CONSUMER_DOC_BATCH_SIZE: int = int(os.getenv("CONSUMER_DOC_BATCH_SIZE", 10))
    CONSUMER_CHUNK_BATCH_SIZE: int = int(os.getenv("CONSUMER_CHUNK_BATCH_SIZE", 64))
    CONSUMER_FLUSH_INTERVAL_SECONDS: int = int(os.getenv("CONSUMER_FLUSH_INTERVAL_SECONDS", 5))

    # ==========================================
    # 6. MÔ HÌNH NHÚNG VIETLAWBERT-MRL & HUẤN LUYỆN
    # ==========================================
    # Kiến trúc Bi-Encoder nền tảng BAAI/bge-m3 kết hợp Matryoshka Representation Learning
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 1024))
    MATRYOSHKA_DIMS: Tuple[int, ...] = (64, 128, 256, 512, 768, 1024)
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.05))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", 512))

    # Cấu hình tính toán Embedding
    EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cpu")
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 16))
    EMBED_CPU_INT8: bool = os.getenv("EMBED_CPU_INT8", "1").lower() in ("1", "true", "yes")

    # ==========================================
    # 7. CỤM TRÍ TUỆ NHÂN TẠO & PHỤC VỤ (LLM / RAG)
    # ==========================================
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "ollama")
    CONTEXTUALIZER_MODEL: str = os.getenv("CONTEXTUALIZER_MODEL", "qwen2.5:1.5b")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "qwen2.5:1.5b")

    # Hệ số dung hợp thông tin RRF (Reciprocal Rank Fusion)
    RRF_K: int = int(os.getenv("RRF_K", 60))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", 5))

    # ==========================================
    # 8. CẤU HÌNH OCR & BÓC TÁCH VĂN BẢN QUÉT
    # ==========================================
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "1") in ("1", "true", "yes")
    OCR_INLINE_ENABLED: bool = os.getenv("OCR_INLINE_ENABLED", "1") in ("1", "true", "yes")
    OCR_LANG: str = os.getenv("OCR_LANG", "vie+eng")
    OCR_DPI: int = int(os.getenv("OCR_DPI", 200))
    OCR_MAX_PAGES: int = int(os.getenv("OCR_MAX_PAGES", 200))
    OCR_CONCURRENCY: int = int(os.getenv("OCR_CONCURRENCY", 2))

    # ==========================================
    # 9. KHÓA BẢO MẬT & DỊCH VỤ ĐÁM MÂY MỞ RỘNG
    # ==========================================
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Điều phối Cloud GPU (RunPod / SSH Instance)
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