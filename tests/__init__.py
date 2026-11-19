"""VietLawBERT Test Suite Package."""
import sys
from pathlib import Path

# Tự động ghim thư mục gốc dự án và src/ vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)