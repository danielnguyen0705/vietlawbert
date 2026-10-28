"""
demo/ui/handlers.py — Gradio Event Handlers
=============================================
Tầng trung gian giữa Gradio UI và microservices (server/).
Mỗi handler nhận input từ Gradio components → gọi services → trả output.

Handlers:
  handle_submit()   — gửi câu hỏi, chạy pipeline RAG
  handle_clear()    — xóa toàn bộ state
  handle_retry()    — gửi lại câu hỏi cuối
  handle_feedback() — ghi like/dislike RLHF
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

import gradio as gr

from demo.config import USE_MOCK_BACKEND, DEFAULT_TOP_K
from demo.server import retrieval_service, generation_service, feedback_service
from demo.ui.panels import render_evidence, render_metrics

logger = logging.getLogger("demo.handlers")

ChatHistory = List[List[str | None]]
SessionMeta = List[Dict[str, Any]]


def _run_pipeline(query: str, mode: str, top_k: int):
    """Chạy full pipeline: Retrieval → Generation."""
    use_mock = USE_MOCK_BACKEND

    retr_result = retrieval_service.retrieve(
        query=query, top_k=top_k, mode=mode, use_mock=use_mock
    )
    docs              = retr_result["docs"]
    retrieval_latency = retr_result["latency_ms"]

    gen_result = generation_service.generate(
        query=query, docs=docs, mode=mode, use_mock=use_mock
    )
    answer             = gen_result["answer"]
    generation_latency = gen_result["latency_ms"]
    model_used         = gen_result["model_used"]
    attribution_score  = gen_result["attribution_score"]
    is_mock            = use_mock

    return answer, docs, retrieval_latency, generation_latency, model_used, attribution_score, is_mock


def handle_submit(
    message: str,
    history: ChatHistory,
    meta_state: SessionMeta,
    mode: str,
    top_k: int,
):
    """Xử lý sự kiện Gửi câu hỏi."""
    if not message.strip():
        return history, history, meta_state, "", render_evidence([]), render_metrics()

    query = message.strip()
    logger.info("Nhận câu hỏi: '%.60s...'", query)

    answer, docs, retr_lat, gen_lat, model_used, attribution_score, is_mock = _run_pipeline(
        query, mode, int(top_k)
    )

    new_history = history + [[query, answer]]
    new_meta    = meta_state + [{
        "query": query, "answer": answer,
        "mode": mode, "top_k": int(top_k), "model_used": model_used,
    }]

    evidence = render_evidence(docs)
    metrics  = render_metrics(
        retrieval_latency_ms  = retr_lat,
        generation_latency_ms = gen_lat,
        n_docs                = len(docs),
        mode                  = mode,
        top_k                 = int(top_k),
        attribution_score     = attribution_score,
        model_used            = model_used,
        is_mock               = is_mock,
    )

    return new_history, new_history, new_meta, "", evidence, metrics


def handle_clear():
    """Reset toàn bộ state."""
    return [], [], [], render_evidence([]), render_metrics()


def handle_retry(history: ChatHistory, meta_state: SessionMeta, mode: str, top_k: int):
    """Gửi lại câu hỏi cuối cùng."""
    if not history:
        return history, history, meta_state, render_evidence([]), render_metrics()
    last_query      = history[-1][0]
    trimmed_history = history[:-1]
    trimmed_meta    = meta_state[:-1] if meta_state else []
    return handle_submit(last_query, trimmed_history, trimmed_meta, mode, top_k)


def handle_feedback(event: gr.LikeData, history: ChatHistory, meta_state: SessionMeta) -> None:
    """Xử lý sự kiện like/dislike từ Chatbot.like()."""
    idx = event.index
    if not history or idx >= len(history):
        return
    query  = history[idx][0] or ""
    answer = history[idx][1] or ""
    meta   = meta_state[idx] if meta_state and idx < len(meta_state) else {}
    feedback_service.log_feedback(
        query=query, answer=answer, liked=event.liked, index=idx,
        mode=meta.get("mode", ""), top_k=meta.get("top_k", DEFAULT_TOP_K),
        model_used=meta.get("model_used", ""),
    )
