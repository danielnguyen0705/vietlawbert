"""
ingest.py - CLI chính thức khởi chạy tiến trình nạp ngoại tuyến (Phase 2 Ingestion).
Đọc Shards thô -> Bóc tách Hybrid AST -> Tiêm Metadata -> Nạp đồng thời Qdrant & ES.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

from cli.consume_embeddings import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)