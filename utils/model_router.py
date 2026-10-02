"""
model_router.py - Bộ điều phối định tuyến mô hình trí tuệ nhân tạo (Dynamic Model Router).
Tự động ánh xạ từng tác vụ chuyên biệt tới mô hình và siêu tham số tối ưu (Kiến trúc v3).
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
        - 'generator' / 'qa': Mô hình sinh câu trả lời (Claude 3.5 Sonnet / Qwen2.5-7B).
        - 'embedding' / 'dense': Mô hình nhúng vector (BAAI/bge-m3).
        - 'contextualizer': Mô hình tạo ngữ cảnh phân mảnh (Claude 3.5 Haiku).
        - 'reranker': Mô hình tái xếp hạng (BAAI/bge-reranker-v2-m3).
        - 'ocr' / 'vision': Mô hình xử lý ảnh/PDF scan (Qwen2-VL-2B).
        - 'evaluation' / 'judge': Mô hình trọng tài RAGAS (Claude 3.5 Sonnet).
        """
        task = str(task_type).strip().lower()

        if task in ("generator", "qa", "generation"):
            return getattr(config, "PRIMARY_LLM_MODEL", "claude-3.5-sonnet")
        elif task in ("contextualizer", "context"):
            return getattr(config, "CONTEXTUALIZER_MODEL", "claude-3-5-haiku")
        elif task in ("embedding", "dense"):
            return getattr(config, "BASE_MODEL_NAME", "BAAI/bge-m3")
        elif task in ("reranker", "cross_encoder"):
            return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        elif task in ("ocr", "vision"):
            return os.getenv("OCR_MODEL", "qwen2-vl:2b")
        elif task in ("evaluation", "eval", "judge"):
            return getattr(config, "EVAL_MODEL", "anthropic/claude-3.5-sonnet")

        return getattr(config, "GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    @staticmethod
    def get_task_parameters(task_type: str) -> Dict[str, Any]:
        """Trả về cấu hình siêu tham số mặc định cho từng loại tác vụ."""
        task = str(task_type).strip().lower()

        if task in ("generator", "qa", "generation"):
            return {
                "model": ModelRouter.get_model_for_task("generator"),
                "temperature": 0.1,
                "max_tokens": 1024,
                "api_base": getattr(config, "PRIMARY_LLM_API_BASE", "https://openrouter.ai/api/v1"),
            }
        elif task in ("contextualizer", "context"):
            return {
                "model": ModelRouter.get_model_for_task("contextualizer"),
                "temperature": 0.1,
                "max_tokens": 150,
                "api_base": getattr(config, "CONTEXTUALIZER_API_BASE", "https://hhtechapi.net/v1"),
            }
        elif task in ("evaluation", "eval", "judge"):
            return {
                "model": ModelRouter.get_model_for_task("evaluation"),
                "temperature": 0.0,
                "max_tokens": 512,
                "api_base": getattr(config, "EVAL_API_BASE", "https://openrouter.ai/api/v1"),
            }

        return {
            "model": ModelRouter.get_model_for_task(task),
            "temperature": 0.0,
            "max_tokens": 512,
            "api_base": getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
        }