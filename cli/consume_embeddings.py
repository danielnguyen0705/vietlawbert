"""
consume_embeddings.py - CLI điều phối vector hóa ngữ nghĩa và nạp vào Qdrant & Elasticsearch.
Tích hợp Checkpoint Shard tự phục hồi, kiểm soát tài nguyên CPU/GPU và dọn rác bộ nhớ tự động.
"""

from __future__ import annotations

import os
import gc
import sys
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

import torch
from configs.paths import RAW_SHARDS_DIR
from configs.config import config
from pipeline.ingest_pipeline import IngestPipelineWorker, load_checkpoint, save_checkpoint

logger = logging.getLogger("VietLawBERT_EmbeddingConsumerCLI")


def resolve_compute_device(requested_device: str) -> str:
    """Kiểm tra tính tương thích phần cứng phần cứng; tự động lùi về CPU nếu GPU không hỗ trợ."""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA được yêu cầu nhưng torch.cuda.is_available() = False. Tự động lùi về [cpu].")
            return "cpu"
        try:
            # Kiểm thử phân bổ tensor trên GPU để phát hiện lỗi compute capability (như sm_61)
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            return "cuda"
        except Exception as e:
            logger.warning("GPU phát hiện lỗi không tương thích kernel (%s). Tự động lùi về [cpu].", e)
            return "cpu"
    return "cpu"


def configure_cpu_runtime():
    """Khống chế luồng tính toán CPU tránh nghẽn luồng hệ thống trên laptop."""
    max_threads = min(4, os.cpu_count() or 2)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    torch.set_num_threads(max_threads)
    logger.info("Đã thiết lập PyTorch CPU execution threads: %d", max_threads)


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
        default=getattr(config, "EMBED_DEVICE", "cpu"),
        choices=["cpu", "cuda"],
        help="Thiết bị tính toán (cpu/cuda)",
    )
    args = parser.parse_args()

    target_path = Path(args.shard_path).resolve()
    if not target_path.exists():
        logger.error("Đường dẫn shard không tồn tại: %s", target_path)
        return 1

    effective_device = resolve_compute_device(args.device)
    if effective_device == "cpu":
        configure_cpu_runtime()

    try:
        logger.info(
            "Khởi động IngestPipelineWorker: Dim=%d | Batch=%d | Device=%s | Model=%s",
            args.dim,
            args.batch_size,
            effective_device,
            args.model_name,
        )
        worker = IngestPipelineWorker(
            qdrant_host=config.QDRANT_HOST,
            qdrant_port=config.QDRANT_PORT,
            es_host=config.ES_HOST,
            model_name_or_path=args.model_name,
            vector_dim=args.dim,
            batch_size=args.batch_size,
            device=effective_device,
        )

        completed_shards = load_checkpoint()

        if target_path.is_dir():
            all_candidates = sorted(list(target_path.glob("*.jsonl*")))
            shard_files = [
                f for f in all_candidates
                if (f.name.endswith(".jsonl.gz") or f.name.endswith(".jsonl"))
                and not f.name.endswith(".corrupted")
                and not f.name.endswith(".quarantine.jsonl")
                and not f.name.endswith(".tmp")
                and not f.name.endswith(".audit.json")
            ]

            if not shard_files:
                logger.warning("Không tìm thấy tệp shard hợp lệ nào trong thư mục: %s", target_path)
                return 0

            logger.info("Tìm thấy %d tệp shard hợp lệ (Đã nạp trước đó: %d).", len(shard_files), len(completed_shards))
            for idx, shard_file in enumerate(shard_files, 1):
                if shard_file.name in completed_shards:
                    logger.info("[%d/%d] [CHECKPOINT SKIP] Bỏ qua Shard đã nạp: %s", idx, len(shard_files), shard_file.name)
                    continue

                logger.info("[%d/%d] Bắt đầu nạp Shard: %s", idx, len(shard_files), shard_file.name)
                worker.process_raw_shard(shard_file)
                completed_shards.add(shard_file.name)
                save_checkpoint(completed_shards)

                # Dọn dẹp rác bộ nhớ giữa các shard lớn
                gc.collect()
                if effective_device == "cuda":
                    torch.cuda.empty_cache()
        else:
            if target_path.name in completed_shards:
                logger.info("[CHECKPOINT SKIP] Tệp đơn lẻ đã nạp trước đó: %s", target_path.name)
                return 0

            worker.process_raw_shard(target_path)
            completed_shards.add(target_path.name)
            save_checkpoint(completed_shards)

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