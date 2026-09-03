"""
fine_tune.py - Điểm vào đồng bộ cho quy trình tinh chỉnh mô hình VietLawBERT.
Bảo toàn tính tương thích ngược với các runbook và chuyển tiếp trực tiếp vào train_mrl.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from configs.paths import ARTIFACTS_DIR, MODELS_DIR
from configs.config import config
from .train_mrl import train


def main():
    parser = argparse.ArgumentParser(description="VietLawBERT Embedding Fine-Tuner Entrypoint")
    parser.add_argument("--input", default=ARTIFACTS_DIR / "triplet_training_data.jsonl", help="Đường dẫn tệp Triplet Dataset")
    parser.add_argument("--model-name", default=getattr(config, "BASE_MODEL_NAME", "BAAI/bge-m3"), help="Mô hình Backbone")
    parser.add_argument("--output-dir", default=MODELS_DIR / "vietlawbert_mrl_base", help="Thư mục lưu Checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Tốc độ học")
    parser.add_argument("--max-samples", type=int, default=None, help="Số mẫu chạy PoC")
    parser.add_argument("--auto-terminate", action="store_true", help="Tự hủy instance GPU khi hoàn thành")
    args = parser.parse_args()

    train(
        train_file=args.input,
        base_model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        auto_terminate=args.auto_terminate,
    )


if __name__ == "__main__":
    main()