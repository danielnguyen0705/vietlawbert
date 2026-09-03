"""
inspect_database.py - CLI kiểm tra nhanh trạng thái kết nối, số lượng vector trong Milvus,
cấu trúc nút trong Neo4j và kho tài liệu MongoDB.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cli import bootstrap_cli_env
bootstrap_cli_env()

from quality.database_inspection import main as inspect_main


def main() -> int:
    # Ủy quyền toàn bộ việc parse tham số (--json, --verbose) cho hàm kiểm toán gốc
    return inspect_main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)