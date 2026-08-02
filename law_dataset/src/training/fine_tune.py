"""
fine_tune.py - Domain-Specific Fine-tuning pipeline cho VietLawBERT

Sử dụng Matryoshka Representation Learning (MRL) loss để tối ưu hóa
kích thước embedding từ 1024-dim xuống [64, 128, 256, 512] mà giữ vững chất lượng.
Huấn luyện trên tập dữ liệu sinh ra bởi GG-SLM (triplet_training_data.jsonl).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from paths import BASE_DIR, get_log_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.FileHandler(get_log_path("fine_tune"), encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FineTune")


def load_triplet_dataset(jsonl_path: str) -> list[InputExample]:
    """Đọc triplet_training_data.jsonl và chuyển sang cấu trúc InputExample."""
    examples = []
    if not os.path.exists(jsonl_path):
        logger.error(f"Triplet training file không tồn tại: {jsonl_path}")
        return []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # InputExample cho Triplet: [Query, Positive, Negative]
                examples.append(InputExample(
                    texts=[
                        data["query"],
                        data["positive"],
                        data["hard_negative"]
                    ]
                ))
            except Exception as e:
                logger.warning(f"Bỏ qua dòng lỗi: {e}")

    logger.info(f"Loaded {len(examples):,} ví dụ huấn luyện.")
    return examples


def main():
    parser = argparse.ArgumentParser(description="VietLawBERT Embedding Fine-Tuner")
    parser.add_argument("--input", default="data/json/triplet_training_data.jsonl")
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Base model")
    parser.add_argument("--output-dir", default="models/vietlawbert_mrl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()

    # Môi trường chạy
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    logger.info(f"Huấn luyện trên thiết bị: {device}")

    # Đọc dữ liệu (BASE_DIR = law_dataset/, args.input = data/json/...)
    input_path = os.path.join(BASE_DIR, args.input)
    train_examples = load_triplet_dataset(input_path)
    if not train_examples:
        logger.error("Không có dữ liệu huấn luyện. Thoát.")
        sys.exit(1)

    # Dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Khởi tạo model
    logger.info(f"Đang tải base model: {args.model_name}...")
    model = SentenceTransformer(args.model_name, device=device)

    # Thiết lập Matryoshka Representation Learning Loss
    # Bọc MultipleNegativesRankingLoss bên trong MatryoshkaLoss
    base_loss = losses.MultipleNegativesRankingLoss(model=model)

    # ponytail: Tải trọng số Matryoshka cho các dimensions
    # [64, 128, 256, 512, 1024] giúp scale index size cực kỳ linh hoạt
    train_loss = losses.MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=[64, 128, 256, 512, 1024]
    )

    output_path = os.path.join(BASE_DIR, "law_dataset", args.output_dir)
    os.makedirs(output_path, exist_ok=True)

    logger.info("Bắt đầu huấn luyện...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=output_path,
        show_progress_bar=True
    )
    logger.info(f"✅ Hoàn tất! Model đã lưu tại: {output_path}")


if __name__ == "__main__":
    main()
