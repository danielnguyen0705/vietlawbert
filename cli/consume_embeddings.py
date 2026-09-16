"""
consume_embeddings.py - CLI điều phối vector hóa ngữ nghĩa và nạp vào Qdrant & Elasticsearch.
Tích hợp Checkpoint Shard tự phục hồi, bộ lọc whitelist file và hỗ trợ điều phối GPU/CPU.
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
from pipeline.ingest_pipeline import IngestPipelineWorker, load_checkpoint, save_checkpoint

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
    parser.add_argument(
        "--device",
        default=config.EMBED_DEVICE,
        choices=["cpu", "cuda"],
        help="Thiết bị tính toán (cpu/cuda)",
    )
    parser.add_argument(
        "--idle-exit-seconds",
        type=float,
        default=0.0,
        help="Tham số tương thích ngược kiến trúc streaming cũ",
    )
    args = parser.parse_args()

    target_path = Path(args.shard_path).resolve()
    if not target_path.exists():
        logger.error("Đường dẫn shard không tồn tại: %s", target_path)
        return 1

    try:
        logger.info(
            "Khởi động IngestPipelineWorker: Dim=%d | Batch=%d | Device=%s | Model=%s",
            args.dim,
            args.batch_size,
            args.device,
            args.model_name,
        )
        worker = IngestPipelineWorker(
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
            es_host=config.ES_HOST,
            model_name_or_path=args.model_name,
            vector_dim=args.dim,
            batch_size=args.batch_size,
            device=args.device,
        )

        completed_shards = load_checkpoint()

        if target_path.is_dir():
            all_candidates = sorted(list(target_path.glob("*.jsonl*")))
            # Chỉ nhận tệp shard chuẩn, loại bỏ tệp quarantine và tệp tạm
            shard_files = [
                f for f in all_candidates
                if (f.name.endswith(".jsonl.gz") or f.name.endswith(".jsonl"))
                and not f.name.endswith(".corrupted")
                and not f.name.endswith(".quarantine.jsonl")
                and not f.name.endswith(".tmp")
            ]

            if not shard_files:
                logger.warning("Không tìm thấy tệp shard hợp lệ nào trong thư mục: %s", target_path)
                return 0

            logger.info("Tìm thấy %d tệp shard hợp lệ (Đã nạp trước đó: %d).", len(shard_files), len(completed_shards))
            for shard_file in shard_files:
                if shard_file.name in completed_shards:
                    logger.info("[CHECKPOINT SKIP] Bỏ qua Shard đã nạp: %s", shard_file.name)
                    continue
                worker.process_raw_shard(shard_file)
                completed_shards.add(shard_file.name)
                save_checkpoint(completed_shards)
        else:
            worker.process_raw_shard(target_path)

        logger.info("✓ Hoàn tất nạp vector embeddings vào Qdrant và Elasticsearch.")
        return 0

    except Exception as exc:
        logger.error("Lỗi nghiêm trọng trong quá trình nạp embeddings: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Nhận tín hiệu dừng từ người dùng. Thoát CLI an toàn.")
        sys.exit(130)