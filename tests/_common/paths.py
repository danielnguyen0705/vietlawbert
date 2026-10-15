"""_common/paths.py - Bộ phân giải đường dẫn portable cho toàn bộ test suite VietLawBERT.

Ưu tiên biến môi trường; nếu không có thì tự dò gốc repo (thư mục chứa 'rag/' và 'preprocess/').
Nhờ vậy test chạy được trên máy trạm Windows/Ubuntu lẫn Cloud GPU mà không phải sửa hard-code.
"""
import os
from pathlib import Path


def _autodetect_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "rag").is_dir() and (p / "preprocess").is_dir():
            return p
    return Path(__file__).resolve().parents[2]


REPO_ROOT = Path(os.environ.get("VIETLAWBERT_ROOT") or _autodetect_root())
DATA_ROOT = Path(os.environ.get("VIETLAWBERT_DATA") or (REPO_ROOT / "data"))
CHECKPOINTS_DIR = Path(os.environ.get("VIETLAWBERT_CKPT") or (REPO_ROOT / "checkpoints"))

TESTS_DIR = REPO_ROOT / "tests"
HARNESS_DIR = TESTS_DIR / "rq_experiments_promptfoo"
DATASETS_DIR = HARNESS_DIR / "datasets"

RAW_SHARDS = DATA_ROOT / "raw_shards"
PROCESSED_PARQUET = DATA_ROOT / "processed_parquet"
BULK_VECTORS = DATA_ROOT / "bulk_vectors"
