"""
config.py - Cấu hình và tiện ích import từ .env

Dùng chung cho toàn bộ project.
"""

import os
from dotenv import load_dotenv
from paths import BASE_DIR


load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config:
    """Singleton config loader."""

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "vietlawbert")

    # Cấu hình API LLM dùng chung chuẩn OpenAI (Tương thích Ollama, Llama.cpp, SGLang, vLLM)
    LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

    CONTEXTUALIZER_MODEL = os.getenv("CONTEXTUALIZER_MODEL", "qwen2.5:1.5b")
    GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen2.5:1.5b")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


config = Config()