"""
cloud_entrypoint.py - Trục điều phối tự động hóa khép kín trên Cloud GPU (A100/RTX 4090).
Tự động nạp cấu hình từ .env, chạy Train MRL 3 Epochs, Bulk Encode Chunks và đồng bộ Hugging Face Hub.
"""

from __future__ import annotations

import os
import sys
import glob
import time
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

import torch
import pyarrow as pa
import pyarrow.parquet as pq

# Nạp cấu hình môi trường
load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_CloudMaster")

from configs.config import config
from configs.paths import RAW_SHARDS_DIR, ARTIFACTS_DIR, MODELS_DIR, DATA_STORAGE_ROOT

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")
TRIPLET_FILE = ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"
OUTPUT_MODEL_DIR = Path(os.getenv("OUTPUT_MODEL_DIR", str(MODELS_DIR / "vietlawbert_mrl_final")))
OUTPUT_VECTOR_DIR = DATA_STORAGE_ROOT / "bulk_vectors"


def run_phase_train():
    logger.info("=" * 65)
    logger.info("GIAI ĐOẠN 1: HUẤN LUYỆN TOÀN DIỆN VIETLAWBERT-MRL (3 EPOCHS - FULL BACKBONE)")
    logger.info("=" * 65)

    from training.train_mrl import train

    args = argparse.Namespace(
        train_parquet=str(TRIPLET_FILE),
        model_name=config.BASE_MODEL_NAME,
        output_dir=str(OUTPUT_MODEL_DIR),
        epochs=int(os.getenv("EPOCHS", 3)),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", 32)),
        gradient_accumulation_steps=1,
        freeze_layers=0,  # Mở khóa toàn bộ 24 tầng khi chạy trên GPU A100
        lr=float(os.getenv("LEARNING_RATE", 2e-5)),
        max_seq_length=int(getattr(config, "MAX_SEQ_LENGTH", 512)),
        matryoshka_dims=[64, 128, 256, 512, 768, 1024],
        tau=float(os.getenv("TEMPERATURE", 0.05)),
        hierarchy_weight=float(os.getenv("HIERARCHY_WEIGHT", 0.15)),
        precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=None,
        max_samples=None,
        logging_steps=50,
        num_workers=4,
        push_to_hub=bool(HF_TOKEN and HF_REPO_ID),
        hub_model_id=HF_REPO_ID,
        hf_token=HF_TOKEN,
    )
    train(args)
    logger.info("✓ Hoàn thành Giai đoạn 1: Trọng số mô hình đã được lưu tại %s", OUTPUT_MODEL_DIR)


def run_phase_bulk_embedding():
    logger.info("=" * 65)
    logger.info("GIAI ĐOẠN 2: MÃ HÓA HÀNG LOẠT VECTOR BẰNG MODEL VỪA HUẤN LUYỆN")
    logger.info("=" * 65)

    from sentence_transformers import SentenceTransformer

    OUTPUT_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(glob.glob(str(DATA_STORAGE_ROOT / "processed" / "parquet" / "*.parquet")))

    if not parquet_files:
        logger.info("Không phát hiện Parquet thô, kiểm tra Shards nén tại %s...", RAW_SHARDS_DIR)
        return

    logger.info("Tìm thấy %d tệp Parquet để mã hóa vector.", len(parquet_files))
    embedder = SentenceTransformer(str(OUTPUT_MODEL_DIR), device="cuda" if torch.cuda.is_available() else "cpu")
    embedder.max_seq_length = 512

    for idx, f_path in enumerate(parquet_files, 1):
        out_f = OUTPUT_VECTOR_DIR / f"vec_{Path(f_path).name}"
        if out_f.exists():
            continue

        table = pq.read_table(f_path, columns=["chunk_id", "content"])
        chunk_ids = table["chunk_id"].to_pylist()
        texts = [str(t) for t in table["content"].to_pylist()]

        vectors = embedder.encode(
            texts,
            batch_size=128 if torch.cuda.is_available() else 32,
            show_progress_bar=False,
            normalize_embeddings=True,
            precision="float32",
        )

        out_table = pa.Table.from_arrays(
            [pa.array(chunk_ids), pa.array(vectors.tolist())],
            names=["chunk_id", "vector_1024d"],
        )
        pq.write_table(out_table, out_f, compression="snappy")

        if idx % 10 == 0 or idx == len(parquet_files):
            logger.info("[%03d/%d] Đã mã hóa và xuất vector: %s", idx, len(parquet_files), out_f.name)

    logger.info("✓ Hoàn thành Giai đoạn 2: Toàn bộ vector đã được xuất sang %s", OUTPUT_VECTOR_DIR)


def run_phase_sync():
    logger.info("=" * 65)
    logger.info("GIAI ĐOẠN 3: ĐỒNG BỘ ARTIFACTS LÊN HUGGING FACE HUB")
    logger.info("=" * 65)

    if HF_TOKEN and HF_REPO_ID:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            vec_repo = f"{HF_REPO_ID}-vectors"
            logger.info("Đang tải tập vector lên Hugging Face Dataset: %s...", vec_repo)
            api.create_repo(repo_id=vec_repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(OUTPUT_VECTOR_DIR),
                repo_id=vec_repo,
                repo_type="dataset",
            )
            logger.info("✓ Đồng bộ vector lên Hugging Face Hub thành công 100%!")
        except Exception as e:
            logger.error("Lỗi đồng bộ Hugging Face: %s", e)
    else:
        logger.warning("Bỏ qua đồng bộ HF do chưa cấu hình HF_TOKEN trong .env.")


def main():
    start_time = time.time()
    try:
        run_phase_train()
        run_phase_bulk_embedding()
        run_phase_sync()
    finally:
        total_min = (time.time() - start_time) / 60
        logger.info("=" * 65)
        logger.info("TOÀN BỘ TIẾN TRÌNH CLOUD HOÀN TẤT TRONG: %.1f PHÚT", total_min)
        logger.info("BẠN CÓ THỂ NGẮT / TẮT CLOUD GPU AN TOÀN!")
        logger.info("=" * 65)


if __name__ == "__main__":
    main()