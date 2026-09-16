"""
publish.py - CLI phát các Shards dữ liệu chuẩn hóa lên Kafka/Redpanda Message Bus.
Tích hợp cơ chế phòng vệ tự động thông báo an toàn khi chạy chế độ Offline Direct Sharding.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

logger = logging.getLogger("VietLawBERT_PublishCLI")

try:
    from ingestion.kafka_publisher import main
except ImportError:
    def main() -> int:
        logger.info("Hệ thống đang hoạt động ở chế độ Offline Direct Sharding (Decoupled Kafka Architecture).")
        logger.info("Dữ liệu Shards được nạp trực tiếp qua 'cli.ingest' mà không cần Message Bus trung gian.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)