"""
merge_artifacts.py - CLI hợp nhất các tệp retry/overlay vào artifact chuẩn (Canonical Artifact).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

from artifacts.merge import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)