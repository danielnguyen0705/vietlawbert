"""
paths.py - Quản trị tập trung toàn bộ danh mục đường dẫn của VietLawBERT.
Triệt tiêu hoàn toàn lỗi lệch thư mục gốc và chuẩn hóa phân vùng lưu trữ Big Data.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# 1. Thư mục gốc tuyệt đối của dự án (vietlawbert/)
# Đi lùi 2 cấp: vietlawbert/configs/paths.py -> vietlawbert/
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT_DIR  # Khả năng tương thích ngược (Backward Compatibility)

# 2. Thư mục gốc lưu trữ dữ liệu lớn (Ưu tiên Docker Volume /mnt/data nếu có)
DEFAULT_STORAGE_ROOT = ROOT_DIR / "data"
DATA_STORAGE_ROOT = Path(os.getenv("DATA_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT)))

# 3. Phân vùng dữ liệu thô & trung gian (Raw & Staging Layers)
RAW_SHARDS_DIR = DATA_STORAGE_ROOT / "raw_shards"
RAW_PDFS_DIR = DATA_STORAGE_ROOT / "raw_pdfs"
RAW_HTML_DIR = DATA_STORAGE_ROOT / "raw_html"
PROCESSED_DIR = DATA_STORAGE_ROOT / "processed"
JSON_DIR = DATA_STORAGE_ROOT / "json"

# 4. Phân vùng Artifacts, Checkpoints & Benchmarks
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
BENCHMARK_DIR = ROOT_DIR / "benchmark"
MODELS_DIR = ROOT_DIR / "models"

# 5. Phân vùng Quản trị Nhật ký (Logging System)
# Tự động đồng bộ vào DATA_STORAGE_ROOT/logs thay vì thư mục cache ẩn của OS
BASE_LOGS_DIR = DATA_STORAGE_ROOT / "logs"
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
DAILY_LOGS_DIR = BASE_LOGS_DIR / TODAY_STR

# Danh sách toàn bộ các thư mục vật lý cần đảm bảo sự tồn tại
ALL_DIRECTORIES = [
    DATA_STORAGE_ROOT,
    RAW_SHARDS_DIR,
    RAW_PDFS_DIR,
    RAW_HTML_DIR,
    PROCESSED_DIR,
    JSON_DIR,
    ARTIFACTS_DIR,
    BENCHMARK_DIR,
    MODELS_DIR,
    BASE_LOGS_DIR,
    DAILY_LOGS_DIR,
]


def ensure_dirs() -> None:
    """Tự động khởi tạo toàn bộ hạ tầng cây thư mục nếu chưa tồn tại."""
    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def get_log_path(script_name: str, by_date: bool = True) -> Path:
    """
    Tạo đường dẫn tệp nhật ký chuẩn hóa dạng: logs/[YYYY-MM-DD]/log_[script_name].log.
    Hỗ trợ chế độ append liên tục trong ngày để bảo toàn lịch sử truy vấn.
    """
    target_dir = DAILY_LOGS_DIR if by_date else BASE_LOGS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    clean_name = Path(script_name).stem
    return target_dir / f"log_{clean_name}.log"


# Tự động đồng bộ hạ tầng thư mục khi import module
ensure_dirs()