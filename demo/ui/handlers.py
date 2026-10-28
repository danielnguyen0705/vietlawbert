"""
demo/ui/handlers.py — Gradio Event Handlers
=============================================
Tầng trung gian giữa Gradio UI và microservices (server/).
Mỗi handler nhận input từ Gradio components → gọi services → trả output.

Gradio 6.x: ChatHistory format thay đổi từ [[user, bot], ...] sang
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

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

from demo.config import USE_MOCK_BACKEND, DEFAULT_TOP_K, DEFAULT_MODE
from demo.server import retrieval_service, generation_service, feedback_service
from demo.ui.panels import render_evidence, render_metrics

logger = logging.getLogger("demo.handlers")

# Type aliases — Gradio 6.x format
# ChatHistory: list of {"role": "user"|"assistant", "content": str}
ChatHistory = List[Dict[str, str]]
# SessionMeta: metadata mỗi lượt hỏi-đáp (để feedback_service dùng)
# Index i trong meta tương ứng với lượt thứ i (mỗi lượt = 1 user + 1 assistant msg)
SessionMeta = List[Dict[str, Any]]


def _msg(role: str, content: str) -> Dict[str, str]:
    """Tạo message dict theo Gradio 6.x format."""
    return {"role": role, "content": content}


def _run_pipeline(
    query: str,
    mode: str,
    top_k: int,
) -> Tuple[str, List[Dict], float, float, str, float, bool]:
    """
    Chạy full pipeline: Retrieval → Generation.

    Returns
    -------
    (answer, docs, retrieval_latency_ms, generation_latency_ms, model_used,
     attribution_score, is_mock)
    """
    use_mock = USE_MOCK_BACKEND

    # ① Retrieval
    retr_result = retrieval_service.retrieve(
        query=query, top_k=top_k, mode=mode, use_mock=use_mock
    )
    docs              = retr_result["docs"]
    retrieval_latency = retr_result["latency_ms"]

    # ② Generation (truyền docs đã retrieved → không fetch lại)
    gen_result = generation_service.generate(
        query=query, docs=docs, mode=mode, use_mock=use_mock
    )
    answer             = gen_result["answer"]
    generation_latency = gen_result["latency_ms"]
    model_used         = gen_result["model_used"]
    attribution_score  = gen_result["attribution_score"]

    return answer, docs, retrieval_latency, generation_latency, model_used, attribution_score, use_mock


def handle_submit(
    message: str,
    history: ChatHistory,
    meta_state: SessionMeta,
    mode: str,
    top_k: int,
) -> Tuple[ChatHistory, ChatHistory, SessionMeta, str, str, str]:
    """
    Xử lý sự kiện Gửi câu hỏi.

    Returns (chatbot, history_state, meta_state, txt_clear, evidence_html, metrics_html)
    """
    empty_evidence = render_evidence([])
    empty_metrics  = render_metrics()

    if not message.strip():
        return history, history, meta_state, "", empty_evidence, empty_metrics

    query = message.strip()
    logger.info("Nhận câu hỏi: '%.80s'", query)

    answer, docs, retr_lat, gen_lat, model_used, attribution_score, is_mock = _run_pipeline(
        query, mode, int(top_k)
    )

    # Gradio 6.x: append từng message riêng với role
    new_history = history + [
        _msg("user", query),
        _msg("assistant", answer),
    ]

    # Lưu metadata lượt này để feedback_service dùng
    # index = số lượt đã có (trước khi append lượt mới này)
    turn_index = len(meta_state)
    new_meta = meta_state + [{
        "query"      : query,
        "answer"     : answer,
        "mode"       : mode,
        "top_k"      : int(top_k),
        "model_used" : model_used,
        "turn_index" : turn_index,
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


def handle_clear() -> Tuple[ChatHistory, ChatHistory, SessionMeta, str, str]:
    """Reset toàn bộ state."""
    return [], [], [], render_evidence([]), render_metrics()


def handle_retry(
    history: ChatHistory,
    meta_state: SessionMeta,
    mode: str,
    top_k: int,
) -> Tuple[ChatHistory, ChatHistory, SessionMeta, str, str, str]:
    """
    Gửi lại câu hỏi cuối cùng trong lịch sử.
    Gradio 6.x: mỗi lượt = 2 entries (user + assistant) trong history list.
    """
    if not history:
        return history, history, meta_state, "", render_evidence([]), render_metrics()

    # Tìm user message cuối cùng
    last_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        return history, history, meta_state, "", render_evidence([]), render_metrics()

    # Cắt bỏ lượt cuối (2 entries: user + assistant)
    trimmed_history = history[:-2] if len(history) >= 2 else []
    trimmed_meta    = meta_state[:-1] if meta_state else []

    return handle_submit(last_user_msg, trimmed_history, trimmed_meta, mode, top_k)


def handle_feedback(
    event: gr.LikeData,
    history: ChatHistory,
    meta_state: SessionMeta,
) -> None:
    """
    Xử lý sự kiện like/dislike từ Chatbot.like().
    Gradio 6.x: event.index là index vào flat message list.
    Chuyển đổi sang turn_index (số lượt hỏi-đáp) để match meta_state.
    """
    msg_index = event.index if hasattr(event, 'index') else None
    liked     = event.liked if hasattr(event, 'liked') else None

    if msg_index is None or not history:
        logger.warning("Feedback event thiếu thông tin, bỏ qua.")
        return

    # Gradio 6.x: flat list [user0, asst0, user1, asst1, ...]
    # turn_index = msg_index // 2
    turn_index = msg_index // 2
    if turn_index >= len(history) // 2:
        logger.warning("Feedback turn_index %d vượt quá số lượt chat", turn_index)
        return

    # Lấy nội dung từ history
    user_msg_idx = turn_index * 2
    asst_msg_idx = user_msg_idx + 1
    query  = history[user_msg_idx]["content"] if user_msg_idx < len(history) else ""
    answer = history[asst_msg_idx]["content"] if asst_msg_idx < len(history) else ""

    meta = meta_state[turn_index] if meta_state and turn_index < len(meta_state) else {}

    feedback_service.log_feedback(
        query      = query,
        answer     = answer,
        liked      = liked,
        index      = turn_index,
        mode       = meta.get("mode", ""),
        top_k      = meta.get("top_k", DEFAULT_TOP_K),
        model_used = meta.get("model_used", ""),
    )
    logger.info("Feedback logged: turn=%d liked=%s", turn_index, liked)
