"""
main.py - Cổng điều phối quy trình hợp nhất VietLawBERT (Unified CLI Gateway).
Chuẩn hóa các tác vụ quản trị hạ tầng, điều phối phân đoạn, huấn luyện và kiểm thử RAG.
Hỗ trợ Real-time Unbuffered Logging, CPU Thread Throttling và tương thích 100% Cloud GPU.
"""

from __future__ import annotations

import os
import sys

# Cưỡng bức mã hóa UTF-8 cho toàn hệ thống
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import time
import socket
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from configs.paths import ROOT_DIR, RAW_SHARDS_DIR
from configs.config import config
from configs.logging_config import setup_hierarchical_logging, get_subsystem_logger

setup_hierarchical_logging()
logger = get_subsystem_logger("master", "main_gateway")


def parse_host_port(url_or_uri: str, default_host: str = "localhost", default_port: int = 80) -> tuple[str, int]:
    try:
        if "://" not in url_or_uri:
            url_or_uri = f"dummy://{url_or_uri}"
        parsed = urlparse(url_or_uri)
        host = parsed.hostname or default_host
        port = parsed.port or default_port
        return host, port
    except Exception:
        return default_host, default_port


def wait_for_service(host: str, port: int, name: str, timeout: int = 45) -> bool:
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


def run_subcommand(command: list[str], description: str, extra_env: dict[str, str] | None = None) -> bool:
    logger.info("==================================================")
    logger.info("BẮT ĐẦU: %s", description)
    logger.info("Lệnh thực thi: %s", " ".join(command))
    logger.info("==================================================")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(command, cwd=ROOT_DIR, env=env)
    if result.returncode == 0:
        logger.info("✓ HOÀN TẤT: %s", description)
        return True
    logger.error("✗ THẤT BẠI: %s (Mã lỗi: %d)", description, result.returncode)
    return False


def cmd_check_infrastructure() -> bool:
    """Khởi động Docker Compose và kiểm tra tính sẵn sàng của cụm 5 CSDL lai."""
    logger.info("Khởi động cụm CSDL phân tán qua Docker Compose...")
    if not run_subcommand(["docker", "compose", "up", "-d"], "Khởi động Docker Compose"):
        return False

    qdrant_host, qdrant_port = parse_host_port(config.QDRANT_HOST, "localhost", config.QDRANT_PORT)
    es_host, es_port = parse_host_port(config.ES_HOST, "localhost", 9200)
    neo_host, neo_port = parse_host_port(config.NEO4J_URI, "localhost", 7687)
    mongo_host, mongo_port = parse_host_port(config.MONGO_URI, "localhost", 27017)
    redis_host, redis_port = parse_host_port(config.REDIS_HOST, "localhost", config.REDIS_PORT)

    services = [
        (qdrant_host, qdrant_port, "Qdrant Vector Engine"),
        (es_host, es_port, "Elasticsearch Sparse Search"),
        (neo_host, neo_port, "Neo4j HIN Graph"),
        (mongo_host, mongo_port, "MongoDB Document Store"),
        (redis_host, redis_port, "Redis Graph Cache"),
    ]

    for host, port, name in services:
        if not wait_for_service(host, port, name):
            return False

    # Đã sửa lỗi: Trỏ chính xác vào quality.crawl_audit --databases
    return run_subcommand(
        [sys.executable, "-u", "-m", "quality.crawl_audit", "--databases"],
        "Kiểm toán tính nhất quán CSDL (Qdrant & Neo4j)",
    )


def cmd_crawl(total_docs: int = 160660, concurrency: int = 4) -> bool:
    return run_subcommand(
        [
            sys.executable,
            "-u",
            "-m",
            "crawler.shard_runner",
            "--total-documents",
            str(total_docs),
            "--concurrency",
            str(concurrency),
            "--output-dir",
            str(RAW_SHARDS_DIR),
            "--allow-upstream-missing",
            "--defer-ocr",
        ],
        f"Thu thập dữ liệu phân đoạn ({total_docs:,} văn bản)",
    )


def cmd_ingest(batch_size: int = config.EMBED_BATCH_SIZE, device: str = config.EMBED_DEVICE) -> bool:
    cpu_env = {}
    if device == "cpu":
        cpu_env = {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": "",
        }

    return run_subcommand(
        [
            sys.executable,
            "-u",
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
        extra_env=cpu_env,
    )


def cmd_graph() -> bool:
    hin_ok = run_subcommand(
        [sys.executable, "-u", "-m", "database.build_hin_graph", "--metadata", str(RAW_SHARDS_DIR)],
        "Xây dựng mạng thông tin dị thể (HIN) trên Neo4j",
    )
    if not hin_ok:
        return False

    return run_subcommand(
        [sys.executable, "-u", "-m", "training.compute_graph_embeddings", "--dim", "128"],
        "Tiền tính toán Vector Đồ thị Compile-time (Node2Vec 128d)",
    )


def cmd_train(
    epochs: int = 3,
    batch_size: int = config.EMBED_BATCH_SIZE,
    device: str = config.EMBED_DEVICE,
    max_seq_length: int = 256,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    hf_token: str | None = None,
) -> bool:
    cpu_env = {}
    if device == "cpu":
        cpu_env = {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": "",
        }

    triplet_ok = run_subcommand(
        [sys.executable, "-u", "-m", "training.generate_hin_triplets", "--limit", "50000", "--beta", "0.6"],
        "Khai phá bộ ba mẫu khó đối lập (HIN-Guided Triplet Mining)",
        extra_env=cpu_env,
    )
    if not triplet_ok:
        return False

    train_cmd = [
        sys.executable,
        "-u",
        "-m",
        "training.train_mrl",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--max-seq-length",
        str(max_seq_length),
    ]

    if push_to_hub and hub_model_id:
        train_cmd.extend(["--push-to-hub", "--hub-model-id", str(hub_model_id)])
        if hf_token:
            train_cmd.extend(["--hf-token", str(hf_token)])

    return run_subcommand(
        train_cmd,
        "Huấn luyện biểu diễn lồng nhau VietLawBERT-MRL với Hierarchy Loss",
        extra_env=cpu_env,
    )


def cmd_benchmark() -> bool:
    bench_ok = run_subcommand(
        [sys.executable, "-u", "-m", "benchmark.build_vietlawbench", "--single", "600", "--multi", "400"],
        "Sinh tập kiểm chuẩn Ground-Truth VietLawBench (1.000 mẫu)",
    )
    if not bench_ok:
        return False

    rq_ok = run_subcommand(
        [sys.executable, "-u", "-m", "benchmark.evaluate_rqs"],
        "Đo lường các chỉ số khoa học RQ1-RQ4",
    )
    if not rq_ok:
        return False

    return run_subcommand(
        [sys.executable, "-u", "-m", "benchmark.baseline_comparator"],
        "Thực nghiệm đối chứng trực diện Baseline (BM25 vs Dense vs VietLawBERT)",
    )


def cmd_run_all(
    batch_size: int = config.EMBED_BATCH_SIZE,
    device: str = config.EMBED_DEVICE,
    epochs: int = 3,
    max_seq_length: int = 256,
) -> bool:
    logger.info("=== BẮT ĐẦU CHU TRÌNH TỰ ĐỘNG TOÀN TRÌNH VIETLAWBERT ===")

    steps = [
        ("Kiểm tra hạ tầng CSDL", lambda: cmd_check_infrastructure()),
        ("Nạp dữ liệu Ingestion (Pha 2)", lambda: cmd_ingest(batch_size=batch_size, device=device)),
        ("Xây dựng Đồ thị HIN (Pha 3)", lambda: cmd_graph()),
        ("Huấn luyện VietLawBERT-MRL (Pha 4)", lambda: cmd_train(epochs=epochs, batch_size=batch_size, device=device, max_seq_length=max_seq_length)),
        ("Đánh giá thực nghiệm khoa học (Pha 5)", lambda: cmd_benchmark()),
    ]

    for name, func in steps:
        logger.info(">>> THỰC THI GIAI ĐOẠN: %s", name)
        success = func()
        if not success:
            logger.error("✗ Dừng quy trình tự động do giai đoạn '%s' thất bại.", name)
            return False

    logger.info("✓ Toàn bộ quy trình VietLawBERT đã hoàn tất mỹ mãn và chuẩn hóa thực nghiệm.")
    return True


def cmd_ask(query: str, top_k: int = 3) -> None:
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
    while True:
        print("\n" + "=" * 60)
        print("      VIETLAWBERT - HỆ THỐNG ĐIỀU PHỐI QUY TRÌNH")
        print("=" * 60)
        print(" 1. Kiểm tra hạ tầng & CSDL (Docker + Health Check)")
        print(" 2. Cào dữ liệu phân đoạn (Shard Runner - Pha 1)")
        print(" 3. Bóc tách AST & Nạp Vector/Sparse (Ingest - Pha 2)")
        print(" 4. Xây dựng Đồ thị HIN & Cache Vector 128d (Pha 3)")
        print(" 5. Khai phá mẫu khó & Huấn luyện VietLawBERT-MRL (Pha 4)")
        print(" 6. Đánh giá thực nghiệm khoa học (RQ1-RQ4 & Baseline)")
        print(" 7. Chạy TỰ ĐỘNG TOÀN TRÌNH liên hoàn (Pha 1 -> Pha 5)")
        print(" 8. Đặt câu hỏi thử nghiệm hệ thống RAG")
        print(" 0. Thoát")
        print("=" * 60)

        choice = input("Chọn chức năng (0-8): ").strip()
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
            cmd_run_all()
        elif choice == "8":
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
        choices=["check", "crawl", "ingest", "graph", "train", "benchmark", "all"],
        help="Thực thi phân đoạn quy trình tương ứng",
    )
    parser.add_argument("--ask", type=str, help="Kiểm thử truy vấn hỏi đáp RAG trực tiếp")
    parser.add_argument("--top-k", type=int, default=3, help="Số lượng căn cứ trích xuất khi hỏi đáp")
    parser.add_argument("--device", default=config.EMBED_DEVICE, choices=["cpu", "cuda"], help="Thiết bị tính toán (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=config.EMBED_BATCH_SIZE, help="Kích thước lô tính toán")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện mô hình MRL")
    parser.add_argument("--max-seq-length", type=int, default=getattr(config, "MAX_SEQ_LENGTH", 256), help="Chiều dài tối đa token")
    parser.add_argument("--total-docs", type=int, default=160660, help="Tổng số văn bản cào")

    # Bổ sung các cờ hỗ trợ Cloud GPU Orchestration
    parser.add_argument("--push-to-hub", action="store_true", help="Tự động đồng bộ weights lên Hugging Face Hub")
    parser.add_argument("--hub-model-id", type=str, default=getattr(config, "HF_MODEL_REPO_ID", None), help="Tên repo trên HF")
    parser.add_argument("--hf-token", type=str, default=getattr(config, "HF_TOKEN", None), help="Access token ghi của HF")
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
        cmd_train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            max_seq_length=args.max_seq_length,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hf_token=args.hf_token,
        )
    elif args.phase == "benchmark":
        cmd_benchmark()
    elif args.phase == "all":
        cmd_run_all(
            batch_size=args.batch_size,
            device=args.device,
            epochs=args.epochs,
            max_seq_length=args.max_seq_length,
        )
    else:
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nNhận tín hiệu dừng (Ctrl+C). Thoát hệ điều phối an toàn.")
        sys.exit(0)