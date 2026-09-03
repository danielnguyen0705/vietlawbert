"""
model_router.py - Bộ điều phối định tuyến mô hình trí tuệ nhân tạo (Dynamic Model Router).
Tự động ánh xạ từng tác vụ chuyên biệt tới mô hình và siêu tham số tối ưu.
"""

from __future__ import annotations

import os
from typing import Dict, Any
from configs.config import config


class ModelRouter:
    """Điều hướng mô hình linh hoạt giữa Local Ollama, vLLM và Cloud Endpoints."""

    @staticmethod
    def get_model_for_task(task_type: str) -> str:
        """
        Lấy định danh mô hình dựa trên tác vụ cụ thể:
        - 'text' / 'contextualize': Mô hình làm giàu ngữ cảnh (Qwen2.5-1.5B/7B).
        - 'generator' / 'qa': Mô hình sinh câu trả lời (Qwen2.5-1.5B hoặc DeepSeek-R1).
        - 'embedding': Mô hình nhúng vector (BAAI/bge-m3).
        - 'reranker': Mô hình tái xếp hạng (BAAI/bge-reranker-v2-m3).
        - 'ocr' / 'vision': Mô hình phân tích ảnh/PDF scan (Qwen2-VL-2B).
        - 'evaluation': Mô hình đánh giá RAGAS/DeepEval.
        """
        task = str(task_type).strip().lower()

        if task in ("text", "contextualize", "context"):
            return getattr(config, "CONTEXTUALIZER_MODEL", "qwen2.5:1.5b")
        elif task in ("generator", "qa", "generation"):
            return getattr(config, "GENERATOR_MODEL", "qwen2.5:1.5b")
        elif task in ("embedding", "dense"):
            return getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        elif task in ("reranker", "cross_encoder"):
            return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        elif task in ("ocr", "vision"):
            return os.getenv("OCR_MODEL", "qwen2-vl:2b")
        elif task in ("evaluation", "eval"):
            return os.getenv("EVAL_MODEL", "qwen2.5:1.5b")

        return getattr(config, "GENERATOR_MODEL", "qwen2.5:1.5b")

    @staticmethod
    def get_task_parameters(task_type: str) -> Dict[str, Any]:
        """Trả về cấu hình siêu tham số mặc định cho từng loại tác vụ."""
        task = str(task_type).strip().lower()

        if task in ("contextualize", "context"):
            return {
                "model": ModelRouter.get_model_for_task("text"),
                "temperature": 0.1,
                "max_tokens": 256,
                "api_base": getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
            }
        elif task in ("generator", "qa"):
            return {
                "model": ModelRouter.get_model_for_task("generator"),
                "temperature": 0.1,
                "max_tokens": 1024,
                "api_base": getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
            }

        return {
            "model": ModelRouter.get_model_for_task(task),
            "temperature": 0.0,
            "max_tokens": 512,
            "api_base": getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
        }