"""
main.py - Bộ điều phối hệ thống tối giản.
1. Khởi động hạ tầng Docker.
2. Chạy Consumer (đợi dữ liệu).
3. Chạy Crawler (đẩy dữ liệu).
"""

import subprocess
import os
import logging
import time
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] | %(message)s')
logger = logging.getLogger("VietLawBERT_Master")

def run_command(command, description, background=False, cwd=None):
    logger.info(f"[BAT DAU] {description}")
    if background:
        return subprocess.Popen(command, shell=True, cwd=cwd)
    return subprocess.run(command, shell=True, cwd=cwd).returncode == 0

def main():
    logger.info("--- VIETLAWBERT STREAMING SYSTEM ---")
    src_dir = Path(__file__).resolve().parent

    # 1. Khởi động hạ tầng
    run_command("docker compose up -d", "Khởi động Kafka, Milvus, Neo4j")
    time.sleep(10) # Chờ DB sẵn sàng

    # 2. Chạy Consumer (Background)
    logger.info("Khởi động Kafka Consumer...")
    consumer = run_command(
        f'"{sys.executable}" -m cli.consume_embeddings',
        "Consumer (Processing)",
        background=True,
        cwd=src_dir,
    )

    # 3. Chạy Crawler
    logger.info("Khởi động Crawler...")
    run_command("scrapy crawl law_spider", "Crawler (Ingestion)", cwd=src_dir)

    logger.info("Crawler đã chạy xong. Consumer sẽ tiếp tục xử lý nốt dữ liệu còn lại trong Kafka.")
    consumer.wait()

if __name__ == "__main__":
    main()
