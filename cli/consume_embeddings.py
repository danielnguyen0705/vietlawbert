"""
consume_embeddings.py - CLI điều phối vector hóa ngữ nghĩa và nạp vào Qdrant & Elasticsearch.
Tương thích hoàn toàn với kiến trúc lưu trữ lai v3 (Matryoshka d=256).
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

from configs.paths import RAW_SHARDS_DIR
from configs.config import config
from pipeline.ingest_pipeline import IngestPipelineWorker

logger = logging.getLogger("VietLawBERT_EmbeddingConsumerCLI")


def main() -> int:
    parser = argparse.ArgumentParser(description="VietLawBERT Embedding Ingestion Orchestrator")
    parser.add_argument(
        "--shard-path",
        default=str(RAW_SHARDS_DIR),
        help="Đường dẫn đến thư mục chứa shard thô (.jsonl.gz) hoặc file đơn lẻ",
    )
    parser.add_argument(
        "--model-name",
        default=config.BASE_MODEL_NAME,
        help="Tên mô hình Backbone Bi-Encoder",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=config.QDRANT_VECTOR_DIM,
        help="Số chiều cắt lát Matryoshka nạp Qdrant (mặc định d=256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBED_BATCH_SIZE,
        help="Kích thước lô xử lý embedding trên GPU/CPU",
    )
    # Giữ lại tham số cũ để tương thích các script tự động
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=0.0,
        help="Tham số tương thích ngược kiến trúc streaming cũ",
    )
    args = parser.parse_args()

    target_path = Path(args.shard_path)
    if not target_path.exists():
        logger.error(f"Đường dẫn shard không tồn tại: {target_path}")
        return 1

    try:
        logger.info(f"Khởi động IngestPipelineWorker: Dim={args.dim}, Batch={args.batch_size}, Model={args.model_name}")
        worker = IngestPipelineWorker(
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
            es_host=config.ES_HOST,
            model_name_or_path=args.model_name,
            vector_dim=args.dim,
            batch_size=args.batch_size,
        )

        if target_path.is_dir():
            shard_files = sorted(list(target_path.glob("*.jsonl*")))
            if not shard_files:
                logger.warning(f"Không tìm thấy tệp shard nào trong thư mục: {target_path}")
                return 0
            logger.info(f"Tìm thấy {len(shard_files)} tệp shard cần xử lý.")
            for shard_file in shard_files:
                worker.process_raw_shard(str(shard_file))
        else:
            worker.process_raw_shard(str(target_path))

        logger.info("Hoàn tất tiến trình nạp vector embeddings vào Qdrant và Elasticsearch.")
        return 0

    except Exception as exc:
        logger.error(f"Lỗi nghiêm trọng trong quá trình nạp embeddings: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Nhận tín hiệu dừng từ người dùng. Thoát CLI an toàn.")
        sys.exit(130)