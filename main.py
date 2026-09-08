"""
main.py - Cổng điều phối quy trình hợp nhất VietLawBERT (Unified CLI Gateway).
Chuẩn hóa các tác vụ quản trị hạ tầng, điều phối phân đoạn, huấn luyện và kiểm thử RAG.
"""

from __future__ import annotations

import os
import sys
import time
import socket
import argparse
import subprocess
from pathlib import Path

from configs.paths import ROOT_DIR, RAW_SHARDS_DIR
from configs.config import config
from configs.logging_config import setup_hierarchical_logging, get_subsystem_logger

setup_hierarchical_logging()
logger = get_subsystem_logger("master", "main_gateway")


def wait_for_service(host: str, port: int, name: str, timeout: int = 45) -> bool:
    """Kiểm tra tính sẵn sàng của dịch vụ cơ sở dữ liệu qua kết nối TCP socket."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                logger.info("✓ Dịch vụ %s (%s:%d) đã sẵn sàng.", name, host, port)
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(1.5)
    logger.error("✗ Dịch vụ %s (%s:%d) không phản hồi sau %ds.", name, host, port, timeout)
    return False


def run_subcommand(command: list[str], description: str) -> bool:
    """Thực thi tiến trình con với PYTHONPATH tự động điều hướng về gốc dự án."""
    logger.info("==================================================")
    logger.info("BẮT ĐẦU: %s", description)
    logger.info("Lệnh thực thi: %s", " ".join(command))
    logger.info("==================================================")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)

    result = subprocess.run(command, cwd=ROOT_DIR, env=env)
    if result.returncode == 0:
        logger.info("✓ HOÀN TẤT: %s", description)
        return True
    logger.error("✗ THẤT BẠI: %s (Mã lỗi: %d)", description, result.returncode)
    return False


def cmd_check_infrastructure() -> bool:
    """Khởi động Docker Compose và kiểm tra toàn vẹn cụm 5 dịch vụ lưu trữ."""
    logger.info("Khởi động cụm CSDL phân tán qua Docker Compose...")
    if not run_subcommand(["docker", "compose", "up", "-d"], "Khởi động Docker Compose"):
        return False

    services = [
        (config.QDRANT_HOST, config.QDRANT_PORT, "Qdrant Vector Engine"),
        ("localhost", 9200, "Elasticsearch Sparse Search"),
        ("localhost", 7687, "Neo4j HIN Graph"),
        ("localhost", 27017, "MongoDB Document Store"),
        (config.REDIS_HOST, config.REDIS_PORT, "Redis Graph Cache"),
    ]

    for host, port, name in services:
        if not wait_for_service(host, port, name):
            return False

    return run_subcommand([sys.executable, "-m", "quality.database_inspection", "--verbose"], "Kiểm toán CSDL")


def cmd_crawl(total_docs: int = 160660, concurrency: int = 4) -> bool:
    """Kích hoạt bộ cào phân đoạn chống tràn RAM (Shard Runner)."""
    return run_subcommand(
        [
            sys.executable,
            "-m",
            "crawler.shard_runner",
            "--total-documents",
            str(total_docs),
            "--concurrency",
            str(concurrency),
            "--allow-upstream-missing",
            "--defer-ocr",
        ],
        f"Thu thập dữ liệu phân đoạn ({total_docs:,} văn bản)",
    )


def cmd_ingest(batch_size: int = 8, device: str = "cpu") -> bool:
    """Kích hoạt luồng bóc tách Hybrid AST và nạp vào Qdrant & Elasticsearch."""
    return run_subcommand(
        [
            sys.executable,
            "-m",
            "pipeline.ingest_pipeline",
            "--shard-path",
            str(RAW_SHARDS_DIR),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
        ],
        "Bóc tách AST và nạp đồng thời Qdrant (Dense) & Elasticsearch (Sparse)",
    )


def cmd_graph() -> bool:
    """Xây dựng đồ thị 22 quan hệ HIN và tiền tính toán Vector 128d."""
    hin_ok = run_subcommand(
        [sys.executable, "-m", "database.build_hin_graph", "--metadata", str(RAW_SHARDS_DIR)],
        "Xây dựng mạng thông tin dị thể (HIN) trên Neo4j",
    )
    if not hin_ok:
        return False

    return run_subcommand(
        [sys.executable, "-m", "training.compute_graph_embeddings", "--dim", "128"],
        "Tiền tính toán Vector Đồ thị Compile-time (Node2Vec 128d)",
    )


def cmd_train(epochs: int = 3, batch_size: int = 16) -> bool:
    """Khai phá mẫu khó đối lập RWR và huấn luyện VietLawBERT-MRL."""
    triplet_ok = run_subcommand(
        [sys.executable, "-m", "training.generate_hin_triplets", "--limit", "50000", "--beta", "0.6"],
        "Khai phá bộ ba mẫu khó đối lập (HIN-Guided Triplet Mining)",
    )
    if not triplet_ok:
        return False

    return run_subcommand(
        [
            sys.executable,
            "-m",
            "training.train_mrl",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ],
        "Huấn luyện biểu diễn lồng nhau VietLawBERT-MRL với Hierarchy Loss",
    )


def cmd_benchmark() -> bool:
    """Khởi tạo tập kiểm chuẩn VietLawBench và đánh giá 4 RQs."""
    bench_ok = run_subcommand(
        [sys.executable, "-m", "benchmark.build_vietlawbench", "--single", "600", "--multi", "400"],
        "Sinh tập kiểm chuẩn Ground-Truth VietLawBench (1.000 mẫu)",
    )
    if not bench_ok:
        return False

    return run_subcommand([sys.executable, "-m", "benchmark.evaluate_rqs"], "Đo lường các chỉ số khoa học RQ1-RQ4")


def cmd_ask(query: str, top_k: int = 3) -> None:
    """Truy vấn hỏi đáp trực tiếp qua động cơ LegalGenerator."""
    from rag.generator import LegalGenerator

    generator = LegalGenerator()
    result = generator.ask(query, top_k=top_k)

    print("\n" + "=" * 60)
    print(f"CÂU HỎI: {query}")
    print("=" * 60)
    print("TRẢ LỜI:")
    print(result["answer"])
    print("-" * 60)
    print(f"Model: {result.get('model_used')} | Attribution Score: {result.get('attribution_score')}")
    print("\nCĂN CỨ TRÍCH DẪN:")
    for i, ctx in enumerate(result.get("contexts", []), 1):
        print(f"[{i}] {ctx.get('hierarchy_path')} - Văn bản: {ctx.get('doc_number')}")
    print("=" * 60 + "\n")
    generator.close()


def interactive_menu():
    """Giao diện dòng lệnh thân thiện khi chạy python main.py không tham số."""
    while True:
        print("\n" + "=" * 60)
        print("      VIETLAWBERT - HỆ THỐNG ĐIỀU PHỐI QUY TRÌNH")
        print("=" * 60)
        print(" 1. Kiểm tra hạ tầng & CSDL (Docker + Health Check)")
        print(" 2. Cào dữ liệu phân đoạn (Shard Runner - Pha 1)")
        print(" 3. Bóc tách AST & Nạp Vector/Sparse (Ingest - Pha 2)")
        print(" 4. Xây dựng Đồ thị HIN & Cache Vector 128d (Pha 3)")
        print(" 5. Khai phá mẫu khó & Huấn luyện VietLawBERT-MRL (Pha 4)")
        print(" 6. Đánh giá thực nghiệm khoa học 4 RQs (VietLawBench)")
        print(" 7. Đặt câu hỏi thử nghiệm hệ thống RAG")
        print(" 0. Thoát")
        print("=" * 60)

        choice = input("Chọn chức năng (0-7): ").strip()
        if choice == "1":
            cmd_check_infrastructure()
        elif choice == "2":
            cmd_crawl()
        elif choice == "3":
            cmd_ingest()
        elif choice == "4":
            cmd_graph()
        elif choice == "5":
            cmd_train()
        elif choice == "6":
            cmd_benchmark()
        elif choice == "7":
            q = input("Nhập câu hỏi pháp lý: ").strip()
            if q:
                cmd_ask(q)
        elif choice == "0":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")


def main():
    parser = argparse.ArgumentParser(description="VietLawBERT Master Pipeline Gateway")
    parser.add_argument(
        "--phase",
        choices=["check", "crawl", "ingest", "graph", "train", "benchmark"],
        help="Thực thi phân đoạn quy trình tương ứng",
    )
    parser.add_argument("--ask", type=str, help="Kiểm thử truy vấn hỏi đáp RAG trực tiếp")
    parser.add_argument("--top-k", type=int, default=3, help="Số lượng căn cứ trích xuất khi hỏi đáp")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Thiết bị nạp embedding (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=8, help="Kích thước lô tính toán")
    parser.add_argument("--total-docs", type=int, default=160660, help="Tổng số văn bản cào")
    args = parser.parse_args()

    if args.ask:
        cmd_ask(args.ask, top_k=args.top_k)
        return

    if args.phase == "check":
        cmd_check_infrastructure()
    elif args.phase == "crawl":
        cmd_crawl(total_docs=args.total_docs)
    elif args.phase == "ingest":
        cmd_ingest(batch_size=args.batch_size, device=args.device)
    elif args.phase == "graph":
        cmd_graph()
    elif args.phase == "train":
        cmd_train(batch_size=args.batch_size)
    elif args.phase == "benchmark":
        cmd_benchmark()
    else:
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nNhận tín hiệu dừng (Ctrl+C). Thoát hệ điều phối an toàn.")
        sys.exit(0)