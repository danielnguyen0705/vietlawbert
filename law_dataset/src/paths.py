"""
paths.py - Cấu hình đường dẫn tập trung cho toàn bộ project.
"""

import os
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LOGS_DIR = os.path.join(BASE_DIR, "logs")

DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_DIR = os.path.join(DATA_DIR, "json")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MD_DIR = os.path.join(DATA_DIR, "markdown")

RAW_HTML_DIR = os.path.join(RAW_DIR, "html")
RAW_PDF_DIR = os.path.join(RAW_DIR, "pdf")
RAW_DIAGRAM_DIR = os.path.join(RAW_DIR, "diagram")

AUDIT_TRAIL_FILE = os.path.join(JSON_DIR, "audit_trail.jsonl")
METADATA_FILE = os.path.join(JSON_DIR, "metadata.jsonl")
FAILED_FILE = os.path.join(JSON_DIR, "failed_links.jsonl")
CONTEXTUAL_CHUNKS_FILE = os.path.join(JSON_DIR, "final_contextual_chunks.jsonl")
CONVERSION_REPORT_FILE = os.path.join(JSON_DIR, "conversion_report.jsonl")

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
TODAY_LOG_DIR = os.path.join(LOGS_DIR, TODAY_STR)


def get_log_path(script_name: str) -> str:
    os.makedirs(TODAY_LOG_DIR, exist_ok=True)
    return os.path.join(TODAY_LOG_DIR, f"log_{script_name}.log")


def ensure_dirs():
    for d in [
        LOGS_DIR, TODAY_LOG_DIR,
        DATA_DIR, JSON_DIR,
        RAW_DIR, RAW_HTML_DIR, RAW_PDF_DIR, RAW_DIAGRAM_DIR,
        MD_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
