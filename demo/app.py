"""
demo/app.py — VietLawBERT Gradio Demo Entry Point
==================================================
Điểm khởi động duy nhất của toàn bộ demo.
File này chỉ làm 3 việc:
  1. Đọc cấu hình từ demo/config.py
  2. Gọi build_demo() từ demo/ui/layout.py để lấy gr.Blocks
  3. Gọi demo.launch()

Để chạy:
    cd D:\\UIT-HK7\\VietLawBERT\\vietlawbert
    python -m demo.app

Hoặc chạy trực tiếp (cần cwd là thư mục vietlawbert/):
    python demo/app.py

Môi trường:
    USE_MOCK_BACKEND=true   (mặc định) — UI test không cần Qdrant/ES
    USE_MOCK_BACKEND=false  — kết nối backend thật khi services đã sẵn sàng
    GRADIO_PORT=7860        (mặc định)
    GRADIO_SHARE=false      (mặc định)

Gradio 6.x notes:
    - css= truyền vào launch(), KHÔNG phải gr.Blocks()
    - show_api= đã bị xóa khỏi launch() — api endpoint vẫn hoạt động qua api_name=
    - show_copy_button, bubble_full_width đã bị xóa khỏi gr.Chatbot()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Đảm bảo Python tìm thấy package vietlawbert/ (khi chạy demo/app.py trực tiếp)
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent          # .../vietlawbert/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── Thiết lập logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo.app")

# ── Import sau khi sys.path đã ổn định
from demo.config import (
    GRADIO_HOST,
    GRADIO_PORT,
    GRADIO_SHARE,
    FEEDBACK_FILE,
    USE_MOCK_BACKEND,
)
from demo.ui.layout import build_demo, CSS   # CSS export từ layout để truyền vào launch()


def main() -> None:
    logger.info("=" * 60)
    logger.info("VietLawBERT Gradio Demo — khởi động")
    logger.info("Gradio version: %s", __import__('gradio').__version__)
    logger.info("Backend mode : %s", "MOCK" if USE_MOCK_BACKEND else "REAL RAG")
    logger.info("Feedback log : %s", FEEDBACK_FILE)
    logger.info("URL          : http://localhost:%d", GRADIO_PORT)
    logger.info("=" * 60)

    if USE_MOCK_BACKEND:
        logger.warning(
            "⚠  USE_MOCK_BACKEND=true — giao diện đang chạy với dữ liệu giả.\n"
            "   Đặt USE_MOCK_BACKEND=false trong .env khi Qdrant + ES đã sẵn sàng."
        )

    # Tạo thư mục flagging/ nếu chưa có
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Xây dựng giao diện
    demo = build_demo()

    # Khởi động server — Gradio 6.x compatible
    # - css= ở đây (đã chuyển ra khỏi gr.Blocks())
    # - show_api= đã bị xóa trong Gradio 6.x
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        inbrowser=False,   # không tự mở browser trên server
        css=CSS,           # Gradio 6.x: css truyền vào launch()
    )


if __name__ == "__main__":
    main()
