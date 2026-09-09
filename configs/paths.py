"""
paths.py - Quản trị tập trung toàn bộ danh mục đường dẫn của VietLawBERT.
Triệt tiêu hoàn toàn lỗi lệch thư mục gốc và chuẩn hóa phân vùng lưu trữ Big Data.
Tương thích chéo hệ điều hành (Ubuntu, Windows Native, WSL2, Docker).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# 1. Thư mục gốc tuyệt đối của dự án (vietlawbert/)
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT_DIR  # Khả năng tương thích ngược

# 2. Thư mục gốc lưu trữ dữ liệu lớn:
# - Ưu tiên 1: Đọc biến môi trường DATA_STORAGE_ROOT từ file .env
# - Ưu tiên 2: Phân vùng /mnt/data/vietlawbert_data (nếu đang chạy trên máy Ubuntu cũ)
# - Ưu tiên 3: Thư mục "data" ngay trong project (ROOT_DIR / "data")
env_storage = os.getenv("DATA_STORAGE_ROOT")
if env_storage and str(env_storage).strip():
    DATA_STORAGE_ROOT = Path(env_storage).resolve()
elif Path("/mnt/data/vietlawbert_data").exists():
    DATA_STORAGE_ROOT = Path("/mnt/data/vietlawbert_data").resolve()
else:
    DATA_STORAGE_ROOT = (ROOT_DIR / "data").resolve()

DEFAULT_STORAGE_ROOT = ROOT_DIR / "data"

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
BASE_LOGS_DIR = DATA_STORAGE_ROOT / "logs"
TODAY_STR = datetime.now().strftime("%Y-%m-%d")
DAILY_LOGS_DIR = BASE_LOGS_DIR / TODAY_STR

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
    """Tự động khởi tạo toàn bộ hạ tầng cây thư mục nếu có quyền truy cập."""
    for directory in ALL_DIRECTORIES:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def get_log_path(script_name: str, by_date: bool = True) -> Path:
    """Tạo đường dẫn tệp nhật ký chuẩn hóa dạng: logs/[YYYY-MM-DD]/log_[script_name].log."""
    target_dir = DAILY_LOGS_DIR if by_date else BASE_LOGS_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    clean_name = Path(script_name).stem
    return target_dir / f"log_{clean_name}.log"


ensure_dirs()