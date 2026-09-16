"""
ocr.py - CLI xử lý danh sách tài liệu quét (PDF Scan) trong phân vùng cách ly (Quarantine).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
# Khóa CWD về ROOT_DIR để các tiến trình cào con định vị chính xác scrapy.cfg
bootstrap_cli_env(chdir_to_root=True)

from crawler.ocr_runner import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)