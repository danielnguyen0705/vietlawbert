"""
paths.py - Quản trị tập trung toàn bộ danh mục đường dẫn của VietLawBERT.
Triệt tiêu hoàn toàn lỗi lệch thư mục gốc và chuẩn hóa phân vùng lưu trữ Big Data.
Tương thích chéo hệ điều hành (Ubuntu, Windows Native, WSL2, Docker).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 1. Thư mục gốc tuyệt đối của dự án (vietlawbert/) và thư mục mã nguồn (src/)
SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
BASE_DIR = ROOT_DIR  # Khả năng tương thích ngược

# Nạp sớm tệp .env để đảm bảo biến DATA_STORAGE_ROOT sẵn sàng trước khi cấu hình đường dẫn
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

# 2. Thư mục gốc lưu trữ dữ liệu lớn:
# - Ưu tiên 1: Đọc biến môi trường DATA_STORAGE_ROOT từ file .env
# - Ưu tiên 2: Phân vùng /mnt/data/vietlawbert_data (nếu đang chạy trên máy chủ Ubuntu)
# - Ưu tiên 3: Thư mục "data" ngay trong project (ROOT_DIR / "data")
env_storage = os.getenv("DATA_STORAGE_ROOT")
_env_path = Path(env_storage).resolve() if env_storage and str(env_storage).strip() else None
# Chỉ dùng path từ .env khi thư mục CHA thật sự tồn tại (ghi được) — tránh .env của máy khác
# (vd /home/admin/... trên cloud) làm vỡ mọi lần chạy ở máy này. Không thì fallback /mnt/data → ROOT/data.
if _env_path is not None and _env_path.parent.exists():
    DATA_STORAGE_ROOT = _env_path
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

# 4. Phân vùng Outputs: Artifacts, Checkpoints, Models, Benchmarks, Reports & Flagging
OUTPUTS_DIR = ROOT_DIR / "outputs"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"
TRIPLETS_DIR = ARTIFACTS_DIR / "triplets"
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
BENCHMARK_RESULTS_DIR = BENCHMARK_DIR / "results"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EXPERIMENT_LOGS_DIR = OUTPUTS_DIR / "logs"
FLAGGING_DIR = OUTPUTS_DIR / "flagging"

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
    OUTPUTS_DIR,
    ARTIFACTS_DIR,
    TRIPLETS_DIR,
    BENCHMARK_DIR,
    BENCHMARK_RESULTS_DIR,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    EXPERIMENT_LOGS_DIR,
    FLAGGING_DIR,
    BASE_LOGS_DIR,
    DAILY_LOGS_DIR,
]


def ensure_dirs() -> None:
    """Tự động khởi tạo toàn bộ hạ tầng cây thư mục an toàn."""
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