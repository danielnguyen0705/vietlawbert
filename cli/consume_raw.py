"""
consume_raw.py - CLI nạp bản kê khai tài liệu pháp lý và quan hệ HIN vào Neo4j (Pha 3).
Đọc trực tiếp từ RAW_SHARDS_DIR (.jsonl.gz), bóc tách cây phân cấp và tạo dựng 22 quan hệ pháp lý.
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

from configs.paths import RAW_SHARDS_DIR
from configs.config import config
from database.build_hin_graph import run_build_hin

logger = logging.getLogger("VietLawBERT_RawConsumerCLI")


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT HIN Graph Ingestion (Phase 3)")
    parser.add_argument(
        "--path",
        "--file",
        dest="path",
        default=str(RAW_SHARDS_DIR),
        help="Đường dẫn file shard đơn lẻ hoặc thư mục chứa các shard thô để nạp vào Neo4j",
    )
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Chỉ nạp văn bản và quan hệ pháp lý, bỏ qua phân rã Chunks cây phân cấp",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Kích thước lô giao dịch Cypher nạp Neo4j (mặc định 500 records)",
    )
    args = parser.parse_args()

    meta_path = Path(args.path).resolve()
    if not meta_path.exists():
        logger.error("Không tìm thấy đường dẫn dữ liệu thô: %s", meta_path)
        return 1

    try:
        logger.info("Bắt đầu nạp HIN đồ thị vào Neo4j từ: %s (Batch size: %d)", meta_path, args.batch_size)
        run_build_hin(
            metadata_path=str(meta_path),
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            parse_chunks=not args.no_chunks,
        )
        logger.info("✓ Xây dựng mạng Heterogeneous Information Network (HIN) trên Neo4j hoàn tất.")
        return 0
    except Exception as exc:
        err_msg = str(exc)
        if "Unauthorized" in err_msg or "authentication" in err_msg.lower():
            logger.error("✗ Lỗi xác thực Neo4j! Hãy kiểm tra biến NEO4J_PASSWORD trong .env so với docker-compose.yml.")
        else:
            logger.error("✗ Lỗi nghiêm trọng khi nạp đồ thị Neo4j: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)