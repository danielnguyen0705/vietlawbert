"""
main.py - Bộ điều phối tập trung cho toàn bộ hệ thống VietLawBERT.
Bảo toàn tính toàn vẹn giao dịch (Transactional Integrity) giữa Kafka, Neo4j, Milvus.
"""

from __future__ import annotations

import os
import sys
import time
import json
import socket
import signal
import subprocess
from pathlib import Path

from configs.logging_config import setup_hierarchical_logging, get_subsystem_logger
from configs.paths import RAW_SHARDS_DIR

setup_hierarchical_logging()
logger = get_subsystem_logger("master", "main_master")


def wait_for_port(host: str, port: int, service_name: str, timeout: int = 60) -> bool:
    start_time = time.time()
    logger.info("Đang kiểm tra kết nối cổng socket %s (%s:%d)...", service_name, host, port)
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info("✓ Dịch vụ %s đã sẵn sàng tiếp nhận kết nối.", service_name)
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(2)
    logger.error("✗ Quá thời gian chờ (%ds): Dịch vụ %s chưa sẵn sàng!", timeout, service_name)
    return False


def run_command(command: list, description: str, background: bool = False, cwd: Path = None, env: dict = None):
    logger.info("[BẮT ĐẦU] %s", description)
    current_env = os.environ.copy()
    if env:
        current_env.update(env)

    current_env["PYTHONPATH"] = str(cwd) if cwd else str(Path(__file__).resolve().parent)

    if background:
        return subprocess.Popen(command, cwd=cwd, env=current_env)

    result = subprocess.run(command, cwd=cwd, env=current_env)
    if result.returncode != 0:
        logger.error("✗ Thất bại: %s (Exit Code: %d)", description, result.returncode)
        return False
    logger.info("✓ Hoàn tất: %s", description)
    return True


def main():
    logger.info("=== HỆ THỐNG ĐIỀU PHỐI LUỒNG DỮ LIỆU LỚN VIETLAWBERT ===")
    project_root = Path(__file__).resolve().parent

    # 1. Khởi động cụm dịch vụ CSDL qua Docker Compose
    up_cmd = ["docker", "compose", "up", "-d"]
    if not run_command(up_cmd, "Khởi động cụm CSDL (MongoDB, Neo4j, Milvus, Redpanda)", cwd=project_root):
        sys.exit(1)

    # 2. Kiểm tra tính sẵn sàng cổng TCP Socket
    services_to_check = [
        ("localhost", 9092, "Redpanda Kafka Broker"),
        ("localhost", 7687, "Neo4j Bolt Protocol"),
        ("localhost", 19530, "Milvus RPC Server"),
        ("localhost", 27017, "MongoDB Document Store"),
    ]
    for host, port, name in services_to_check:
        if not wait_for_port(host, port, name, timeout=60):
            logger.error("Hạ tầng phân tán chưa sẵn sàng. Dừng kịch bản thực thi.")
            sys.exit(1)

    # 3. Kích hoạt Consumer xử lý luồng (chế độ non-blocking)
    logger.info("Khởi động Kafka Ingestion Consumer (Background Processing)...")
    consumer_cmd = [
        sys.executable,
        "-m",
        "cli.consume_embeddings",
        "--idle-exit-seconds",
        "15",
    ]
    consumer_process = run_command(
        consumer_cmd,
        "Kafka Consumer (Chế độ Non-blocking)",
        background=True,
        cwd=project_root,
    )

    try:
        # 4. Kích hoạt Web Crawler
        logger.info("Khởi động mạng nhện cào dữ liệu Scrapy...")
        crawler_cmd = ["scrapy", "crawl", "law_spider"]
        crawler_success = run_command(crawler_cmd, "Crawler Ingestion Engine", cwd=project_root)

        # 5. Đối soát trạng thái kết thúc thực tế và Shards đã lưu
        status_file = project_root / "artifacts" / "crawler_status.json"
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                finish_reason = status_data.get("finish_reason")
                scraped = status_data.get("successful_count", 0)
                scheduled = status_data.get("scheduled_count", 0)
                shards_count = status_data.get("shards_exported", 0)

                shards_on_disk = len(list(Path(RAW_SHARDS_DIR).glob("crawl_pages_*.jsonl.gz")))

                if finish_reason == "finished":
                    logger.info(
                        "✓ Crawler hoàn thành trọn vẹn danh mục (%d/%d văn bản | Đã đóng gói: %d shards đĩa cứng).",
                        scraped,
                        scheduled,
                        shards_on_disk,
                    )
                else:
                    logger.warning(
                        "⚠ Crawler dừng trước hạn! (Lý do: '%s' | Đã cào: %d văn bản | Shards bảo toàn trên đĩa: %d).",
                        finish_reason,
                        scraped,
                        shards_on_disk,
                    )
            except Exception as e:
                logger.warning("Không thể đọc trạng thái crawler (%s).", e)
        else:
            if not crawler_success:
                logger.warning("Crawler dừng lại với cảnh báo. Kiểm tra crawler.log.")

        # 6. Chờ Consumer hoàn tất xả đệm Kafka vào CSDL
        logger.info("Đang chờ Consumer xử lý nốt các bản ghi tồn đọng trong Kafka...")
        consumer_process.wait()
        logger.info("✓ Toàn bộ dữ liệu luồng đã được ghi nhất quán vào Milvus & Neo4j.")

    except KeyboardInterrupt:
        logger.warning("Nhận tín hiệu dừng từ người dùng (Ctrl+C). Đang ngắt Consumer an toàn...")
        consumer_process.send_signal(signal.SIGINT)
        try:
            consumer_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.error("Consumer không phản hồi kịp. Cưỡng chế dừng tiến trình.")
            consumer_process.kill()
        sys.exit(0)


if __name__ == "__main__":
    main()
