"""
demo/config.py — Cấu hình tập trung cho VietLawBERT Demo
=========================================================
Tất cả constants, env vars, và paths đều được định nghĩa ở đây.
Các module khác import từ đây, không định nghĩa lại.

Gradio 6.x notes:
  - show_api không còn là launch() param — đã bị xóa trong Gradio 6.0
  - API endpoint (api_name="chat") vẫn hoạt động qua btn_send.click(..., api_name="chat")
"""

from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DEMO_DIR  = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent

# ── Gradio Server ─────────────────────────────────────────────────────────────
GRADIO_HOST  = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT  = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
# show_api đã bị xóa trong Gradio 6.x — API endpoint vẫn khả dụng qua api_name= trong .click()

# ── RAG defaults ──────────────────────────────────────────────────────────────
DEFAULT_TOP_K   = 5
MAX_TOP_K       = 20
RETRIEVAL_MODES = [
    "Dense",
    "Sparse (BM25)",
    "Hybrid-RRF",
    "Full (+ Graph Rerank)",
]
DEFAULT_MODE = "Full (+ Graph Rerank)"

# ── Feedback / RLHF ──────────────────────────────────────────────────────────
FLAGGING_DIR  = REPO_ROOT / "flagging"
FEEDBACK_FILE = FLAGGING_DIR / "feedback.jsonl"

# ── Interaction logging (reproducible evidence) ──────────────────────────────
# Ghi TOÀN BỘ mỗi lượt hỏi-đáp (query, answer, top-k evidence, metrics) vào JSONL
# phục vụ phân tích định lượng và viết test hồi quy. File là dữ liệu runtime → gitignored.
INTERACTIONS_FILE = FLAGGING_DIR / "interactions.jsonl"

# ── Backend switch ────────────────────────────────────────────────────────────
# USE_MOCK_BACKEND=true (mặc định): chạy UI không cần Qdrant/ES
# USE_MOCK_BACKEND=false: kết nối backend RAG thật
USE_MOCK_BACKEND = os.getenv("USE_MOCK_BACKEND", "true").lower() == "true"
