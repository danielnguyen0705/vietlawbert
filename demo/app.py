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

    # ── Eager warm-up: khởi tạo TOÀN BỘ stack TRƯỚC demo.launch()
    # Mục tiêu: khi nhận query đầu tiên, tất cả đã sẵn sàng — không có cold-start.
    if not USE_MOCK_BACKEND:
        # Bước 1: Khởi tạo retriever (Qdrant + ES + bge-m3)
        logger.info("🔥 [1/3] Warm-up: Khởi tạo LegalHybridRetriever (BAAI/bge-m3 + Qdrant + ES)...")
        try:
            from demo.server.retrieval_service import _get_retriever
            retriever = _get_retriever()
            logger.info("✅ [1/3] LegalHybridRetriever sẵn sàng.")
        except Exception as e:
            logger.error("❌ [1/3] Warm-up retriever thất bại: %s", e)
            retriever = None

        # Bước 2: Nạp graph vector cache (1.29M vectors) bằng dummy query
        # Graph cache được load lazy bên trong retrieve() — phải trigger ở đây
        # để tránh 12s delay khi nhận query thật đầu tiên.
        if retriever is not None:
            logger.info("🔥 [2/3] Warm-up: Nạp graph vector cache (1.29M vectors từ graph_embeddings_128d.parquet)...")
            try:
                retriever.retrieve("khởi động hệ thống pháp lý", top_k=1)
                logger.info("✅ [2/3] Graph vector cache đã nạp vào RAM — sẵn sàng phục vụ query.")
            except Exception as e:
                logger.error("❌ [2/3] Warm-up graph cache thất bại: %s", e)

        # Bước 3: Khởi tạo generator (LLM dispatcher) — truyền retriever singleton vào
        # để tránh LegalGenerator tự tạo thêm LegalHybridRetriever thứ 2.
        logger.info("🔥 [3/3] Warm-up: Khởi tạo LegalGenerator (ResilientLLMDispatcher)...")
        try:
            from demo.server.generation_service import _get_generator
            _get_generator(retriever=retriever)
            logger.info("✅ [3/3] LegalGenerator sẵn sàng.")
        except Exception as e:
            logger.error("❌ [3/3] Warm-up generator thất bại: %s", e)

        logger.info("=" * 60)
        logger.info("✅ Warm-up hoàn tất — toàn bộ pipeline sẵn sàng nhận query ngay.")
        logger.info("=" * 60)

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
