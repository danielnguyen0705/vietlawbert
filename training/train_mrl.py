"""
train_mrl.py - Động cơ huấn luyện VietLawBERT-MRL sử dụng Matryoshka Representation Learning.
Tối ưu hóa đa tầng hàm mất mát InfoNCE trên không gian đa độ phân giải D = {64, 128, 256, 512, 768, 1024}.
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

from configs.paths import ROOT_DIR, ARTIFACTS_DIR, MODELS_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from artifacts.canonical import read_jsonl

logger = get_subsystem_logger("training", "model_training")


def compute_matryoshka_weights(dims: List[int]) -> List[float]:
    raw_weights = [1.0 / math.log2(float(d) + 2.0) for d in dims]
    total_w = sum(raw_weights)
    normalized = [round(w / total_w, 4) for w in raw_weights]
    logger.info(f"Phân bổ trọng số Matryoshka ({dims}): {normalized}")
    return normalized


def load_triplets(file_path: Path | str, max_samples: Optional[int] = None) -> List[InputExample]:
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Không tìm thấy tệp huấn luyện: {path}")
        return []

    examples: List[InputExample] = []

    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        if max_samples:
            df = df.iloc[:max_samples]
        for _, row in df.iterrows():
            examples.append(InputExample(texts=[str(row["query"]), str(row["positive"]), str(row["hard_negative"])]))
    else:
        for idx, record in enumerate(read_jsonl(path)):
            if max_samples and idx >= max_samples:
                break
            q = record.get("query")
            p = record.get("positive")
            n = record.get("hard_negative")
            if q and p and n:
                examples.append(InputExample(texts=[str(q), str(p), str(n)]))

    logger.info(f"✓ Đã nạp thành công {len(examples):,} bộ ba đối lập phục vụ huấn luyện.")
    return examples


def train(
    train_file: Path | str,
    base_model_name: str = "BAAI/bge-m3",
    output_dir: Path | str = MODELS_DIR / "vietlawbert_mrl_base",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_samples: Optional[int] = None,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    auto_terminate: bool = False,
):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Kích hoạt huấn luyện trên NVIDIA GPU: {gpu_name} (CUDA v{torch.version.cuda})")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"

    train_samples = load_triplets(train_file, max_samples=max_samples)
    if not train_samples:
        logger.error("Dữ liệu đầu vào rỗng. Hủy tiến trình huấn luyện.")
        return

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size, drop_last=True)

    logger.info(f"Đang tải Backbone Encoder nền tảng: [{base_model_name}]...")
    model = SentenceTransformer(base_model_name, device=device)

    matryoshka_dims = [64, 128, 256, 512, 768, 1024]
    matryoshka_weights = compute_matryoshka_weights(matryoshka_dims)

    base_loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)

    train_loss = losses.MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=matryoshka_dims,
        matryoshka_weights=matryoshka_weights,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    logger.info("=== BẮT ĐẦU QUÁ TRÌNH FINE-TUNING VIETLAWBERT-MRL ===")
    logger.info(f"Tham số: Epochs={epochs} | BatchSize={batch_size} | LR={learning_rate} | WarmupSteps={warmup_steps}")

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=str(out_path),
            show_progress_bar=True,
            use_amp=True if device == "cuda" else False,
        )
        logger.info(f"✓ Huấn luyện thành công! Trọng số mô hình đã lưu tại: {out_path.resolve()}")

        if push_to_hub and hub_model_id:
            logger.info(f"Đang tải trọng số lên Hugging Face Hub: {hub_model_id}...")
            model.save_to_hub(repo_id=hub_model_id, private=True)
            logger.info(f"✓ Đã đưa mô hình lên Hugging Face Hub thành công: https://huggingface.co/{hub_model_id}")

    finally:
        if auto_terminate and getattr(config, "ENABLE_CLOUD_GPU", False):
            logger.warning("[SAFETY] Kích hoạt tự hủy RunPod Pod sau khi hoàn tất...")
            pod_id = os.environ.get("RUNPOD_POD_ID") or getattr(config, "CLOUD_GPU_INSTANCE_ID", "")
            if pod_id:
                os.system(f"runpodctl stop pod {pod_id} 2>/dev/null")


def main():
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện VietLawBERT-MRL")
    parser.add_argument("--train-file", default=ARTIFACTS_DIR / "triplet_training_data.jsonl", help="Đường dẫn tệp Triplet Dataset")
    parser.add_argument("--model-name", default=getattr(config, "BASE_MODEL_NAME", "BAAI/bge-m3"), help="Mô hình Backbone")
    parser.add_argument("--output-dir", default=MODELS_DIR / "vietlawbert_mrl_base", help="Thư mục xuất trọng số")
    parser.add_argument("--epochs", type=int, default=3, help="Số lượt huấn luyện")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Tốc độ học (Learning Rate)")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn mẫu để test nhanh PoC")
    parser.add_argument("--push-to-hub", action="store_true", help="Đẩy mô hình lên Hugging Face Hub")
    parser.add_argument("--hub-id", default="vietlawbert/vietlawbert-mrl-base", help="Tên repo trên HF Hub")
    parser.add_argument("--auto-terminate", action="store_true", help="Tự động hủy Cloud Pod khi hoàn thành")
    args = parser.parse_args()

    train(
        train_file=args.train_file,
        base_model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_id,
        auto_terminate=args.auto_terminate,
    )


if __name__ == "__main__":
    main()