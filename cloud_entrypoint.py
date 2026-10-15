"""
cloud_entrypoint.py - Trục điều phối tự động hóa khép kín trên Cloud GPU (A100/RTX 3090/4090).
Khắc phục triệt để lỗi đường dẫn Parquet cũ, tích hợp toàn diện:
Pha 1: Khai phá HIN-Guided Triplets
Pha 2: Huấn luyện VietLawBERT-MRL (3 Epochs, Validation Loop, lưu Best Checkpoint)
Pha 3: Khởi tạo bộ kiểm chuẩn chuẩn mực VietLawBench
Pha 4: Thực thi Master Benchmark & Đo lường RQ1 - RQ4
Pha 5: Đóng gói toàn bộ Artifacts thành tệp .tar.gz để tải về máy local
"""

from __future__ import annotations

import os
import sys
import time
import tarfile
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

import torch

# Nạp cấu hình môi trường
load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_CloudMaster")

from configs.config import config
from configs.paths import ARTIFACTS_DIR, MODELS_DIR, ROOT_DIR, BENCHMARK_DIR

TRIPLET_FILE = ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"
OUTPUT_MODEL_DIR = Path(os.getenv("OUTPUT_MODEL_DIR", str(MODELS_DIR / "vietlawbert_mrl_final")))
RESULTS_DIR = ROOT_DIR / "benchmark" / "results"
BENCH_DIR = ARTIFACTS_DIR / "benchmark"


def run_phase_mine_triplets(limit: int = 50000, beta: float = 0.5):
    """Pha 1: Khai phá mẫu khó HIN-Guided Triplets bằng RWR nếu chưa có."""
    logger.info("=" * 70)
    logger.info("PHA 1: KHAI PHÁ MẪU KHÓ ĐỐI LẬP HIN TRIPLETS (RWR GRAPH GUIDED)")
    logger.info("=" * 70)
    
    if TRIPLET_FILE.exists():
        logger.info("✓ Đã tìm thấy tệp Triplet có sẵn tại %s. Bỏ qua khai phá lại.", TRIPLET_FILE)
        return

    from training.generate_hin_triplets import mine_and_export_hin_triplets
    TRIPLET_FILE.parent.mkdir(parents=True, exist_ok=True)
    mine_and_export_hin_triplets(output_path=str(TRIPLET_FILE), limit=limit, beta=beta)
    logger.info("✓ Hoàn tất Pha 1: Đã lưu tập Triplet tại %s", TRIPLET_FILE)


def run_phase_train():
    """Pha 2: Huấn luyện MRL có Validation Loop và lưu Best Checkpoint."""
    logger.info("=" * 70)
    logger.info("PHA 2: HUẤN LUYỆN TOÀN DIỆN VIETLAWBERT-MRL (3 EPOCHS - FULL BACKBONE)")
    logger.info("=" * 70)

    from training.train_mrl import train

    args = argparse.Namespace(
        train_parquet=str(TRIPLET_FILE),
        model_name=os.getenv("BASE_MODEL_NAME", config.BASE_MODEL_NAME),
        output_dir=str(OUTPUT_MODEL_DIR),
        epochs=int(os.getenv("EPOCHS", 3)),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", 16)),
        gradient_accumulation_steps=2,
        freeze_layers=0,  # Mở khóa toàn bộ các tầng khi chạy trên Cloud GPU
        lr=float(os.getenv("LEARNING_RATE", 2e-5)),
        max_seq_length=int(getattr(config, "MAX_SEQ_LENGTH", 256)),
        matryoshka_dims=[64, 128, 256, 512, 768, 1024],
        tau=float(os.getenv("TEMPERATURE", 0.05)),
        hierarchy_weight=float(os.getenv("HIERARCHY_WEIGHT", 0.15)),
        precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_steps=None,
        max_samples=None,
        logging_steps=25,
        num_workers=2,
    )
    train(args)

    # Đảm bảo trọng số best_model nằm ngay tại OUTPUT_MODEL_DIR
    best_sub = OUTPUT_MODEL_DIR / "best_model"
    if best_sub.exists() and (best_sub / "vietlawbert_mrl.pt").exists():
        import shutil
        for item in best_sub.iterdir():
            target = OUTPUT_MODEL_DIR / item.name
            if not target.exists():
                shutil.copy2(item, target)
        logger.info("✓ Đã đồng bộ trọng số Best Model ra gốc: %s", OUTPUT_MODEL_DIR)

    logger.info("✓ Hoàn thành Pha 2: Mô hình tối ưu đã lưu tại %s", OUTPUT_MODEL_DIR)


def run_phase_benchmark_generation():
    """Pha 3: Khởi tạo tập kiểm chuẩn phân tầng VietLawBench."""
    logger.info("=" * 70)
    logger.info("PHA 3: TẠO LẬP BỘ KIỂM CHUẨN VIETLAWBENCH (ZERO LEXICAL LEAKAGE)")
    logger.info("=" * 70)

    bench_file = BENCH_DIR / "vietlawbench_1000.jsonl"
    if bench_file.exists():
        logger.info("✓ Đã có sẵn tập kiểm chuẩn tại %s. Bỏ qua tạo lại.", BENCH_DIR)
        return

    from benchmark.build_vietlawbench import VietLawBenchBuilder
    builder = VietLawBenchBuilder()
    try:
        builder.build_master_benchmark(total_target=1000)
    finally:
        builder.close()
    logger.info("✓ Hoàn tất Pha 3: Đã xuất tập kiểm chuẩn tại %s", BENCH_DIR)


def run_phase_evaluation():
    """Pha 4: Thực thi Master Baseline Comparator và trọn bộ RQ1 - RQ4."""
    logger.info("=" * 70)
    logger.info("PHA 4: THỰC NGHIỆM ĐỐI CHUẨN MASTER & ĐO ĐẠC TOÀN DIỆN RQ1 - RQ4")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Master Baseline Comparison
    from benchmark.evaluate_rqs import MultiModelBaselineComparator
    comparator = MultiModelBaselineComparator(
        benchmark_dir=BENCH_DIR,
        checkpoints_dir=MODELS_DIR,
        device=device,
    )
    comparator.compare_all(output_dir=RESULTS_DIR)

    # 2. Đo 4 câu hỏi nghiên cứu RQ1 - RQ4
    from benchmark.evaluate_rqs import ScientificBenchmarkRunner
    runner = ScientificBenchmarkRunner(
        benchmark_dir=str(BENCH_DIR),
        model_path=str(OUTPUT_MODEL_DIR),
        device=device,
    )
    runner.run_all(output_dir=RESULTS_DIR)
    logger.info("✓ Hoàn thành Pha 4: Toàn bộ bảng CSV đã lưu tại %s", RESULTS_DIR)


def run_phase_package_artifacts():
    """Pha 5: Đóng gói toàn bộ thành phẩm thành file .tar.gz để kéo về máy local."""
    logger.info("=" * 70)
    logger.info("PHA 5: ĐÓNG GÓI THÀNH PHẨM THỰC NGHIỆM ĐỂ TẢI VỀ MÁY LOCAL")
    logger.info("=" * 70)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_name = f"vietlawbert_artifacts_{timestamp}.tar.gz"
    archive_path = ROOT_DIR / archive_name

    items_to_pack = [
        (RESULTS_DIR, "benchmark/results"),
        (OUTPUT_MODEL_DIR, "models/vietlawbert_mrl_final"),
        (BENCH_DIR, "artifacts/benchmark"),
    ]

    with tarfile.open(archive_path, "w:gz") as tar:
        for src_path, arc_name in items_to_pack:
            if Path(src_path).exists():
                tar.add(src_path, arcname=arc_name)
                logger.info("  + Đã đóng gói: %s -> %s", src_path, arc_name)

    logger.info("✓ HOÀN TẤT ĐÓNG GÓI: %s", archive_path.resolve())
    logger.info("Bạn có thể tải gói này về máy Dell G7 bằng lệnh:")
    logger.info("  scp -P <PORT> <USER>@<IP>:%s ~/Desktop/", archive_path.resolve())


def main():
    parser = argparse.ArgumentParser(description="VietLawBERT Cloud GPU Master Orchestrator")
    parser.add_argument("--skip-train", action="store_true", help="Bỏ qua bước huấn luyện nếu đã có model")
    parser.add_argument("--skip-mine", action="store_true", help="Bỏ qua bước khai phá Triplet")
    parser.add_argument("--triplet-limit", type=int, default=50000)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    start_time = time.time()
    try:
        if not args.skip_mine:
            run_phase_mine_triplets(limit=args.triplet_limit, beta=args.beta)
        if not args.skip_train:
            run_phase_train()
        run_phase_benchmark_generation()
        run_phase_evaluation()
        run_phase_package_artifacts()
    finally:
        total_min = (time.time() - start_time) / 60
        logger.info("=" * 70)
        logger.info("TOÀN BỘ TIẾN TRÌNH CLOUD HOÀN TẤT TRONG: %.1f PHÚT", total_min)
        logger.info("BẠN CÓ THỂ NGẮT / TẮT CLOUD GPU AN TOÀN!")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()