"""
consume_raw.py - CLI nạp bản kê khai tài liệu pháp lý và quan hệ HIN vào Neo4j.
Hỗ trợ cả chế độ Offline Ingestion trực tiếp từ file lẫn Kafka Streaming cũ.
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

from configs.config import config
from database.build_hin_graph import run_build_hin

logger = logging.getLogger("VietLawBERT_RawConsumerCLI")


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT Raw Metadata & Graph Ingestion")
    parser.add_argument(
        "--file",
        default="./data/raw/enriched_metadata.jsonl",
        help="Đường dẫn file metadata thô để nạp trực tiếp vào Neo4j",
    )
    args = parser.parse_args()

    meta_path = Path(args.file)
    if not meta_path.exists():
        logger.warning(f"Không tìm thấy file metadata tại {meta_path}. Tìm kiếm file shard trong data/raw_shards/...")
        raw_shards = list(Path("./data/raw_shards").glob("*.jsonl*"))
        if raw_shards:
            meta_path = raw_shards[0]
            logger.info(f"Sử dụng file shard thay thế: {meta_path}")
        else:
            logger.error("Không tìm thấy nguồn dữ liệu thô để nạp vào Neo4j.")
            return 1

    try:
        logger.info(f"Bắt đầu nạp HIN đồ thị vào Neo4j từ: {meta_path}")
        run_build_hin(
            metadata_path=str(meta_path),
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
        )
        logger.info("✓ Xây dựng mạng Heterogeneous Information Network (HIN) trên Neo4j hoàn tất.")
        return 0
    except Exception as exc:
        logger.error(f"Lỗi nạp đồ thị thô vào Neo4j: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)