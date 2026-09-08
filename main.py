"""
main.py - Bộ điều phối tập trung cho toàn bộ hệ sinh thái VietLawBERT (v3 Architecture).
Bảo toàn tính sẵn sàng dịch vụ: MongoDB, Neo4j, Qdrant, Elasticsearch, Redis.
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
from configs.config import config

setup_hierarchical_logging()
logger = get_subsystem_logger("master", "main_master")


def wait_for_port(host: str, port: int, service_name: str, timeout: int = 60) -> bool:
    """Kiểm tra tính sẵn sàng của dịch vụ qua kết nối TCP Socket."""
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
    """Thực thi lệnh tiến trình con với xử lý biến môi trường chuẩn."""
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
    logger.info("=== HỆ THỐNG ĐIỀU PHỐI HẠ TẦNG & DỮ LIỆU LỚN VIETLAWBERT ===")
    project_root = Path(__file__).resolve().parent

    # 1. Khởi động cụm dịch vụ CSDL lai qua Docker Compose
    up_cmd = ["docker", "compose", "up", "-d"]
    if not run_command(up_cmd, "Khởi động cụm CSDL (MongoDB, Neo4j, Qdrant, Elasticsearch, Redis)", cwd=project_root):
        sys.exit(1)

    # 2. Kiểm tra tính sẵn sàng cổng TCP Socket tương ứng với docker-compose mới
    services_to_check = [
        (config.QDRANT_HOST, config.QDRANT_PORT, "Qdrant Vector Engine"),
        ("localhost", 9200, "Elasticsearch Sparse Search"),
        ("localhost", 7687, "Neo4j Bolt Protocol"),
        ("localhost", 27017, "MongoDB Document Store"),
        (config.REDIS_HOST, config.REDIS_PORT, "Redis Graph Cache"),
    ]
    for host, port, name in services_to_check:
        if not wait_for_port(host, port, name, timeout=60):
            logger.error("Hạ tầng CSDL chưa sẵn sàng. Dừng kịch bản thực thi.")
            sys.exit(1)

    try:
        # 3. Kích hoạt Web Crawler (Pha 1: Thu thập dữ liệu vào Raw Shards)
        logger.info("Khởi động mạng nhện cào dữ liệu Scrapy...")
        crawler_cmd = ["scrapy", "crawl", "law_spider"]
        crawler_success = run_command(crawler_cmd, "Crawler Ingestion Engine", cwd=project_root)

        # 4. Đối soát trạng thái Shards đã lưu trên đĩa cứng
        shards_on_disk = len(list(Path(RAW_SHARDS_DIR).glob("*.jsonl.gz")))
        logger.info("✓ Tổng số Shard dữ liệu thô bảo toàn trên đĩa: %d shards.", shards_on_disk)

        status_file = project_root / "artifacts" / "crawler_status.json"
        if status_file.exists():
            try:
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                finish_reason = status_data.get("finish_reason")
                scraped = status_data.get("successful_count", 0)
                logger.info("Crawler kết thúc với trạng thái: '%s' | Đã cào: %d văn bản.", finish_reason, scraped)
            except Exception as e:
                logger.warning("Không thể đọc crawler_status.json (%s).", e)

        # 5. Thông báo chuyển giao luồng trung gian ngoại tuyến
        logger.info("=" * 60)
        logger.info("Hạ tầng đã sẵn sàng cho tiến trình nạp ngoại tuyến (Phase 2)!")
        logger.info("Để kích hoạt bóc tách AST và nạp vào Qdrant/ES, chạy lệnh:")
        logger.info("  python -m pipeline.ingest_pipeline --shard-path %s", str(RAW_SHARDS_DIR))
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Nhận tín hiệu dừng từ người dùng (Ctrl+C). Đang thoát hệ thống an toàn...")
        sys.exit(0)


if __name__ == "__main__":
    main()