"""
evaluator.py - Xuất chỉ số đánh giá hệ thống ra CSV.
Theo dõi throughput, độ trễ và tỷ lệ thành công của pipeline.
"""

import csv
import os
import logging
from paths import get_log_path

LOG_FILE = get_log_path("evaluator")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "metrics_report.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("Evaluator")

def export_metrics(metrics: list[dict]):
    """Xuất danh sách các chỉ số vào file CSV."""
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "doc_id", "processing_time", "status"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(metrics)
    logger.info(f"[METRICS] Đã xuất {len(metrics)} bản ghi vào {OUTPUT_CSV}")
