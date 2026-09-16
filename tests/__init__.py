"""VietLawBERT Test Suite Package."""
import sys
from pathlib import Path

# Tự động ghim thư mục gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))