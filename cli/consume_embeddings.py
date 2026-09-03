"""
consume_embeddings.py - CLI chunk văn bản, tính toán vector embeddings và nạp song song Milvus/Neo4j.
Ủy quyền ngắt tín hiệu êm thuận (Graceful Termination) hoàn toàn cho LawEventConsumer.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

logger = logging.getLogger("VietLawBERT_EmbeddingConsumerCLI")


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT Kafka Embedding Ingestion Consumer")
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=float(os.getenv("CONSUMER_IDLE_EXIT_SECONDS", "0")),
        help="Thoát sau N giây nếu hàng đợi Kafka trống; 0 nghĩa là chạy liên tục dưới dạng Daemon.",
    )
    args = parser.parse_args()

    # Lazy import các gói AI/Database nặng sau khi hoàn tất parse CLI args
    try:
        from ingestion.embedding_consumer import LawEventConsumer
    except ImportError as err:
        logger.error(f"Lỗi nạp thư viện Ingestion: {err}. Hãy đảm bảo môi trường đã cài đặt đủ dependencies.")
        return 1

    try:
        consumer = LawEventConsumer()
        # consumer.run() tự quản lý signal handler để xả nốt buffer an toàn trước khi dừng
        consumer.run(idle_exit_seconds=args.idle_exit_seconds)
    except Exception as exc:
        logger.error(f"Lỗi thực thi Consumer: {exc}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)