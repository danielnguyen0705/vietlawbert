"""_common/paths.py - Phân giải đường dẫn cho test suite.

NGUỒN SỰ THẬT là configs/paths.py của repo (đọc DATA_STORAGE_ROOT từ .env,
tự nhận /mnt/data/vietlawbert_data trên Ubuntu). Nhờ vậy test tìm đúng dữ liệu lớn
nằm ngoài repo, không bị SKIP oan. Có fallback nếu chưa import được configs.
"""
import os
import sys
from pathlib import Path


def _autodetect_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "rag").is_dir() and (p / "preprocess").is_dir():
            return p
    return Path(__file__).resolve().parents[2]


REPO_ROOT = Path(os.environ.get("VIETLAWBERT_ROOT") or _autodetect_root())
CHECKPOINTS_DIR = Path(os.environ.get("VIETLAWBERT_CKPT") or (REPO_ROOT / "checkpoints"))

# Dùng chính configs.paths của repo làm nguồn sự thật cho gốc dữ liệu lớn
try:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from configs import paths as _P  # noqa: E402
    DATA_ROOT = Path(_P.DATA_STORAGE_ROOT)
    RAW_SHARDS = Path(_P.RAW_SHARDS_DIR)
except Exception:
    _cand = os.environ.get("VIETLAWBERT_DATA")
    if _cand:
        DATA_ROOT = Path(_cand)
    elif Path("/mnt/data/vietlawbert_data").exists():
        DATA_ROOT = Path("/mnt/data/vietlawbert_data")
    else:
        DATA_ROOT = REPO_ROOT / "data"
    RAW_SHARDS = DATA_ROOT / "raw_shards"

# Thư mục con dùng trong test (tên khớp cây dữ liệu thực tế)
PROCESSED_PARQUET = DATA_ROOT / "processed_parquet"
BULK_VECTORS = DATA_ROOT / "bulk_vectors"

TESTS_DIR = REPO_ROOT / "tests"
HARNESS_DIR = TESTS_DIR / "rq_experiments_promptfoo"
DATASETS_DIR = HARNESS_DIR / "datasets"
