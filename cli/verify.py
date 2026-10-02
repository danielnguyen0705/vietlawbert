"""
verify.py - CLI đối soát toàn vẹn dữ liệu 4 chiều End-to-End (Lineage Verification).
Kiểm tra tính nhất quán: Canonical Shards == MongoDB == Neo4j HIN == Qdrant Vector Points == ES Docs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

from quality.pipeline_verifier import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)