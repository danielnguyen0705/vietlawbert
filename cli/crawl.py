"""
crawl.py - CLI điều phối toàn bộ chiến dịch cào dữ liệu văn bản pháp luật phân đoạn theo Shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
# Bắt buộc chuyển CWD về ROOT_DIR để Scrapy nhận diện được scrapy.cfg
bootstrap_cli_env(chdir_to_root=True)

from crawler.shard_runner import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)