"""
publish.py - CLI phát các Shards dữ liệu chuẩn hóa lên Message Bus.
LƯU Ý KIẾN TRÚC: Ở Kiến trúc V3 (2026), hệ thống đã chuyển dịch toàn diện sang cơ chế
Offline Direct Sharding, loại bỏ tầng trung chuyển Kafka/Redpanda để tối ưu I/O đĩa cứng.
Tệp này đóng vai trò cảnh báo tương thích ngược.
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


def main() -> int:
    logger.info("================================================================================")
    logger.info("THÔNG BÁO KIẾN TRÚC VIETLAWBERT (V3 ARCHITECTURE - 2026):")
    logger.info("Hệ thống đã khai tử Message Bus Kafka/Redpanda và chuyển sang Direct Sharding.")
    logger.info("Dữ liệu Shards (.jsonl.gz) được nạp trực tiếp qua lệnh:")
    logger.info("  python -m cli.ingest --shard-path /mnt/data/vietlawbert_data/raw_shards")
    logger.info("================================================================================")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)