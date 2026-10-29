"""
gradio_chat_provider.py — Provider promptfoo gọi CHATBOX Gradio qua endpoint /chat (gradio_client).

Khác với generation_provider.py (gọi thẳng pipeline Python), provider này đi đúng đường người dùng đi:
    promptfoo -> HTTP -> Gradio server -> handle_submit -> retrieval + generation -> UI components
nên kiểm được cả lớp UI/handlers: định dạng trả về, bảng Evidence, bảng Metrics, chế độ (mode) và top_k.

Chạy chatbox trước (một cửa sổ riêng):
    python -m demo.app                        # USE_MOCK_BACKEND=false để dùng backend thật
Địa chỉ mặc định http://127.0.0.1:7860 (đổi bằng biến môi trường CHATBOX_URL hoặc config.url).

Cấu hình provider (promptfoo `config:`):
    url         : địa chỉ chatbox                        (mặc định $CHATBOX_URL hoặc http://127.0.0.1:7860)
    mode        : một trong demo.config.RETRIEVAL_MODES  (mặc định "Hybrid-RRF")
    top_k       : 1..20                                  (mặc định 5)
    timeout_s   : giới hạn chờ một lượt                  (mặc định 600)
    reuse_client: True = dùng chung 1 phiên Gradio (lịch sử/turn_index tích luỹ); mặc định False = phiên mới mỗi ca

Output (chuỗi JSON, cùng khoá với generation_provider để dùng lại assertion):
    query, mode, top_k, answer, retrieved_docs[], contexts[], latency_ms (đo ở phía client, gồm cả mạng/UI),
    metrics{retrieval_ms, generation_ms, n_docs, attribution, is_mock}, n_evidence
LƯU Ý: mỗi ca test cũng được chatbox ghi vào flagging/interactions.jsonl như một lượt hỏi thật.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_URL = "http://127.0.0.1:7860"
_CLIENT: Optional[Tuple[str, Any]] = None   # (url, Client) khi reuse_client=True


def _get_client(url: str, reuse: bool, timeout_s: float):
    """Tạo gradio_client.Client; phiên mới mỗi ca để các ca test không ảnh hưởng nhau (State của Gradio theo phiên)."""
    global _CLIENT
    from gradio_client import Client
    if reuse and _CLIENT and _CLIENT[0] == url:
        return _CLIENT[1]
    client = Client(url, verbose=False, httpx_kwargs={"timeout": timeout_s})
    if reuse:
        _CLIENT = (url, client)
    return client


def _text(fragment: str) -> str:
    """HTML -> văn bản thuần (bỏ thẻ, giải mã &nbsp; ...)."""
    s = re.sub(r"<[^>]+>", " ", fragment or "")
    return re.sub(r"\s+", " ", _html.unescape(s).replace("\xa0", " ")).strip()


def _split_outputs(result: Any) -> Tuple[List[Dict[str, str]], str, str]:
    """
    Tách (chatbot, evidence_html, metrics_html) khỏi giá trị trả về của /chat.
    gradio_client bỏ các gr.State khỏi output nên vị trí có thể khác giữa các bản Gradio →
    nhận diện theo NỘI DUNG thay vì theo chỉ số.
    """
    items = list(result) if isinstance(result, (list, tuple)) else [result]
    chatbot: List[Dict[str, str]] = []
    strings: List[str] = []
    for it in items:
        if isinstance(it, list) and (not it or isinstance(it[0], dict)):
            chatbot = it
        elif isinstance(it, str):
            strings.append(it)
    metrics_html = next((s for s in strings if "Docs retrieved" in s or "Retrieval" in s and "Generation" in s), "")
    evidence_html = next((s for s in strings if s is not metrics_html and ("height:480px" in s or "#1" in s)), "")
    return chatbot, evidence_html, metrics_html


def _parse_evidence(evidence_html: str) -> Tuple[List[str], List[str]]:
    """Trả về (doc_numbers, nội dung từng thẻ) từ bảng Evidence."""
    doc_numbers = [
        _text(m.group(2))
        for m in re.finditer(r"#(\d+)(?:&nbsp;|\s)+([^<]+?)\s*(?=<span|</div>)", evidence_html)
    ]
    cards = [c for c in re.split(r'<div style="border:1px solid #dde[^>]*>', evidence_html)[1:]]
    return doc_numbers, [_text(c) for c in cards]


def _parse_metrics(metrics_html: str) -> Dict[str, Any]:
    t = _text(metrics_html)

    def num(pattern: str) -> Optional[float]:
        m = re.search(pattern, t)
        return float(m.group(1)) if m else None

    return {
        "retrieval_ms": num(r"([\d.]+)\s*ms\s*Retrieval"),
        "generation_ms": num(r"([\d.]+)\s*ms\s*Generation"),
        "n_docs": (lambda v: int(v) if v is not None else None)(num(r"(\d+)\s*Docs retrieved")),
        "attribution": num(r"([\d.]+)\s*Attribution"),
        "is_mock": "MOCK backend" in t,
    }


def call_api(prompt: str, options: dict = None, context: dict = None) -> dict:
    """Entrypoint promptfoo (Python provider)."""
    cfg = (options or {}).get("config", {}) or {}
    url = cfg.get("url") or os.environ.get("CHATBOX_URL") or DEFAULT_URL
    mode = cfg.get("mode", "Hybrid-RRF")
    top_k = int(cfg.get("top_k", 5))
    timeout_s = float(cfg.get("timeout_s", 600))
    query = (prompt or "").strip()

    t0 = time.perf_counter()
    try:
        client = _get_client(url, bool(cfg.get("reuse_client", False)), timeout_s)
    except Exception as e:  # server chưa chạy / sai địa chỉ
        return {"error": f"Không kết nối được chatbox tại {url} — hãy chạy `python -m demo.app` trước. Chi tiết: {e}"}

    try:
        job = client.submit(query, mode, top_k, api_name="/chat")
        result = job.result(timeout=timeout_s)
    except Exception as e:
        return {"error": f"Gọi /chat thất bại (mode={mode}, top_k={top_k}): {e}"}
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    chatbot, evidence_html, metrics_html = _split_outputs(result)
    answer = next((m.get("content", "") for m in reversed(chatbot) if m.get("role") == "assistant"), "")
    if isinstance(answer, list):                      # Gradio có thể trả content dạng list các khối
        answer = " ".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in answer)
    doc_numbers, cards = _parse_evidence(evidence_html)

    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "answer": answer,
        "retrieved_docs": doc_numbers,
        "contexts": cards,
        "n_evidence": len(doc_numbers),
        "metrics": _parse_metrics(metrics_html),
        "latency_ms": latency_ms,
    }
    return {"output": json.dumps(payload, ensure_ascii=False)}
