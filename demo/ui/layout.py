"""
demo/ui/layout.py — Gradio Blocks Layout Builder
=================================================
Xây dựng toàn bộ giao diện gr.Blocks.
Tách biệt khỏi handlers để dễ thay đổi UI mà không ảnh hưởng logic.

Layout 2 cột:
  ┌─────────────────────────┬────────────────────┐
  │  Chatbot + Controls     │  Evidence Inspector│
  │  (scale=3)              │  (scale=2)         │
  │                         ├────────────────────┤
  │                         │  Scientific Metrics│
  │                         │  (scale=2)         │
  └─────────────────────────┴────────────────────┘

Gradio 6.x notes:
  - css= đã chuyển ra khỏi gr.Blocks() → truyền vào .launch() bên app.py
  - show_copy_button, bubble_full_width đã bị xóa khỏi gr.Chatbot()
  - avatar_images vẫn được hỗ trợ
"""

from __future__ import annotations

import gradio as gr

from demo.config import RETRIEVAL_MODES, DEFAULT_MODE, DEFAULT_TOP_K
from demo.ui.handlers import (
    handle_submit,
    handle_clear,
    handle_retry,
    handle_feedback,
)
from demo.ui.panels import render_evidence, render_metrics

# ─── CSS ────────────────────────────────────────────────────────────────────
# NOTE: CSS này được truyền vào demo.launch(css=CSS) trong app.py
#       KHÔNG truyền vào gr.Blocks() — Gradio 6.x đã chuyển param này sang launch()
CSS = """
#vietlawbert-chatbot   { flex-grow:1; min-height:440px; }
#evidence-panel        { min-height:320px; max-height:520px; overflow-y:auto; }
#metrics-panel         { min-height:130px; }
.gradio-container      { font-family:'Segoe UI', 'Helvetica Neue', sans-serif; }
footer                 { display:none !important; }
#send-btn              { min-width:90px; }
"""


# ─── Build ──────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """
    Khởi tạo và trả về gr.Blocks chứa toàn bộ UI.
    Hàm này được gọi từ demo/app.py.
    CSS được export riêng để app.py truyền vào launch().
    """
    with gr.Blocks(title="VietLawBERT Demo") as demo_block:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            "## ⚖️ VietLawBERT — Hệ thống Hỏi-Đáp Pháp luật Việt Nam\n"
            "_Truy xuất lai (Dense + BM25 + RRF) · Đồ thị HIN · MRL@256d · "
            "Cascading LLM Fallback_"
        )

        # ── States ──────────────────────────────────────────────────────
        history_state = gr.State([])     # ChatHistory: [[user, bot], ...]
        meta_state    = gr.State([])     # SessionMeta: [{query, mode, ...}, ...]

        # ── Layout ──────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Cột trái: Chat ──────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="vietlawbert-chatbot",
                    label="Hội thoại pháp lý",
                    height=440,
                    avatar_images=(None, "https://img.icons8.com/color/48/scales--v1.png"),
                    # Gradio 6.x: show_copy_button và bubble_full_width đã bị xóa
                )

                with gr.Row():
                    txt_input = gr.Textbox(
                        placeholder=(
                            "Nhập câu hỏi pháp luật… "
                            "(VD: Thủ tục đăng ký doanh nghiệp tư nhân?)"
                        ),
                        show_label=False,
                        scale=5,
                        container=False,
                        lines=2,
                        max_lines=4,
                    )
                    btn_send = gr.Button(
                        "Gửi ▶",
                        variant="primary",
                        scale=1,
                        elem_id="send-btn",
                    )

                with gr.Row():
                    btn_clear = gr.Button("🗑 Xoá", scale=1)
                    btn_retry = gr.Button("🔄 Thử lại", scale=1)

                with gr.Accordion("⚙️ Tham số truy xuất (Ablation)", open=False):
                    dd_mode = gr.Dropdown(
                        choices=RETRIEVAL_MODES,
                        value=DEFAULT_MODE,
                        label="Chế độ truy xuất",
                        info=(
                            "Dense: chỉ Qdrant | Sparse: chỉ BM25 | "
                            "Hybrid-RRF: fusion | Full: + Graph Rerank"
                        ),
                        interactive=True,
                    )
                    sl_topk = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=DEFAULT_TOP_K,
                        step=1,
                        label="Top-K tài liệu truy xuất",
                        interactive=True,
                    )

            # ── Cột phải: Evidence + Metrics ────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Evidence Inspector")
                evidence_html = gr.HTML(
                    value=render_evidence([]),
                    elem_id="evidence-panel",
                )
                gr.Markdown("### 📊 Scientific Metrics")
                metrics_html = gr.HTML(
                    value=render_metrics(),
                    elem_id="metrics-panel",
                )

        # ── Event wiring ────────────────────────────────────────────────

        submit_inputs  = [txt_input, history_state, meta_state, dd_mode, sl_topk]
        submit_outputs = [chatbot, history_state, meta_state, txt_input, evidence_html, metrics_html]

        btn_send.click(
            fn=handle_submit,
            inputs=submit_inputs,
            outputs=submit_outputs,
            api_name="chat",   # WP12 — gradio_client endpoint
        )
        txt_input.submit(
            fn=handle_submit,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )

        btn_clear.click(
            fn=handle_clear,
            inputs=[],
            outputs=[chatbot, history_state, meta_state, evidence_html, metrics_html],
        )

        btn_retry.click(
            fn=handle_retry,
            inputs=[history_state, meta_state, dd_mode, sl_topk],
            outputs=[chatbot, history_state, meta_state, evidence_html, metrics_html],
        )

        chatbot.like(
            fn=handle_feedback,
            inputs=[history_state, meta_state],
            outputs=[],
        )

    return demo_block
