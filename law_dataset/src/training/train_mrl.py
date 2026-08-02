"""
train_mrl.py - Script huấn luyện VietLawBERT Embedding sử dụng Matryoshka Representation Learning (MRL).
Thiết kế tối ưu để chạy trên Kaggle GPU (T4/L4) hoặc RunPod.
"""

import os
import argparse
import logging
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT-MRL")

def load_triplets(file_path: str, max_samples: int = None) -> list[InputExample]:
    import json
    examples = []
    if not os.path.exists(file_path):
        logger.error(f"Không tìm thấy file: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                examples.append(InputExample(
                    texts=[data["query"], data["positive"], data["hard_negative"]]
                ))
            except Exception as e:
                logger.debug(f"Lỗi parse dòng {idx}: {e}")

    logger.info(f"Đã nạp {len(examples):,} mẫu huấn luyện.")
    return examples

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="Đường dẫn tới triplet_training_data.jsonl")
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Base model từ HF")
    parser.add_argument("--output-dir", default="./vietlawbert-mrl-model", help="Thư mục lưu model sau train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16, help="Tối ưu batch size cho L4 GPU (16-32)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn mẫu để test nhanh PoC")
    args = parser.parse_args()

    # Thiết bị huấn luyện
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Sử dụng NVIDIA GPU: {torch.cuda.get_device_name(0)}")

    # Đọc dữ liệu triplet
    train_examples = load_triplets(args.train_file, args.max_samples)
    if not train_examples:
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Tải base model
    logger.info(f"Đang tải model nền tảng: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)

    # Cấu hình MRL Loss với MultipleNegativesRankingLoss làm nhân (core loss)
    base_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Định nghĩa các tầng phân giải của Matryoshka (64 đến 1024 chiều)
    mrl_loss = losses.MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=[64, 128, 256, 512, 1024]
    )

    logger.info("Khởi động quá trình Fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, mrl_loss)],
        epochs=args.epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True
    )
    logger.info(f"✅ Hoàn tất! Model đã được xuất ra thư mục: {args.output_dir}")

if __name__ == "__main__":
    train()
