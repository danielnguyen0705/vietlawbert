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
from configs.logging_config import get_subsystem_logger
from typing import List, Dict, Any, Tuple

import gradio as gr

from demo.config import USE_MOCK_BACKEND, DEFAULT_TOP_K, DEFAULT_MODE
from demo.server import retrieval_service, generation_service, feedback_service, interaction_service
from demo.ui.panels import render_evidence, render_metrics

logger = get_subsystem_logger("demo", "handlers")

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
):
    """
    Xử lý Gửi câu hỏi — GENERATOR streaming (chatbox chảy chữ).
    Mỗi lần yield: (chatbot, history_state, meta_state, txt_clear, evidence_html, metrics_html).
    history_state chỉ cập nhật ở yield CUỐI → không hỏng state giữa chừng.
    """
    empty_evidence = render_evidence([])
    empty_metrics  = render_metrics()

    if not message.strip():
        yield history, history, meta_state, "", empty_evidence, empty_metrics
        return

    query = message.strip()
    is_mock = USE_MOCK_BACKEND
    logger.info("Nhận câu hỏi: '%.80s'", query)

    # ① Retrieval (blocking) → hiện Evidence ngay trước khi câu trả lời chảy ra
    retr = retrieval_service.retrieve(query=query, top_k=int(top_k), mode=mode, use_mock=is_mock)
    docs     = retr["docs"]
    retr_lat = retr["latency_ms"]
    evidence = render_evidence(docs)
    streaming_metrics = render_metrics(
        retrieval_latency_ms=retr_lat, n_docs=len(docs), mode=mode,
        top_k=int(top_k), is_mock=is_mock,
    )

    base = history + [_msg("user", query)]
    # Hiện bong bóng "đang soạn…" + Evidence, xoá ô nhập; GIỮ history_state cũ
    yield base + [_msg("assistant", "⏳ Đang soạn…")], history, meta_state, "", evidence, streaming_metrics

    # ② Generation — stream token
    answer, model_used, attribution_score, gen_lat = "", "Unknown", 0.0, 0.0
    for ev in generation_service.generate_stream(query, docs, mode, is_mock):
        answer            = ev.get("answer", answer)
        model_used        = ev.get("model_used", model_used)
        attribution_score = ev.get("attribution_score", attribution_score)
        gen_lat           = ev.get("latency_ms", gen_lat)
        yield base + [_msg("assistant", answer or "…")], history, meta_state, "", evidence, streaming_metrics

    # ③ Chốt lượt: cập nhật history_state + meta + metrics đầy đủ + log
    new_history = base + [_msg("assistant", answer)]
    turn_index  = len(meta_state)
    new_meta = meta_state + [{
        "query": query, "answer": answer, "mode": mode,
        "top_k": int(top_k), "model_used": model_used, "turn_index": turn_index,
    }]
    metrics = render_metrics(
        retrieval_latency_ms=retr_lat, generation_latency_ms=gen_lat,
        n_docs=len(docs), mode=mode, top_k=int(top_k),
        attribution_score=attribution_score, model_used=model_used, is_mock=is_mock,
    )
    interaction_service.log_interaction(
        query=query, answer=answer, docs=docs, mode=mode, top_k=int(top_k),
        model_used=model_used, attribution_score=attribution_score,
        retrieval_latency_ms=retr_lat, generation_latency_ms=gen_lat,
        is_mock=is_mock, turn_index=turn_index,
    )
    yield new_history, new_history, new_meta, "", evidence, metrics


def handle_clear() -> Tuple[ChatHistory, ChatHistory, SessionMeta, str, str]:
    """Reset toàn bộ state."""
    return [], [], [], render_evidence([]), render_metrics()


def handle_retry(
    history: ChatHistory,
    meta_state: SessionMeta,
    mode: str,
    top_k: int,
):
    """Gửi lại câu hỏi cuối — GENERATOR (stream như handle_submit). 6 outputs khớp submit."""
    if not history:
        yield history, history, meta_state, "", render_evidence([]), render_metrics()
        return

    last_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        yield history, history, meta_state, "", render_evidence([]), render_metrics()
        return

    # Cắt lượt cuối (2 entries: user + assistant) rồi chạy lại streaming
    trimmed_history = history[:-2] if len(history) >= 2 else []
    trimmed_meta    = meta_state[:-1] if meta_state else []
    yield from handle_submit(last_user_msg, trimmed_history, trimmed_meta, mode, top_k)


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
