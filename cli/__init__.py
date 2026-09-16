"""
vietlawbert.cli
~~~~~~~~~~~~~~~
Bộ điều phối các entrypoint vận hành hệ thống VietLawBERT.
Đảm bảo tự động định tuyến đường dẫn gốc, tương thích mã hóa UTF-8 trên mọi nền tảng
và tích hợp đồng bộ với hệ thống ghi nhật ký phân cấp (Hierarchical Logging).
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# Xác định thư mục gốc tuyệt đối của dự án (vietlawbert/)
ROOT_DIR = Path(__file__).resolve().parent.parent

_BOOTSTRAPPED = False


def bootstrap_cli_env(chdir_to_root: bool = False) -> Path:
    """
    Khởi tạo môi trường CLI nhất quán:
    - Bổ sung thư mục gốc vào sys.path để triệt tiêu hoàn toàn ModuleNotFoundError.
    - Ép buộc UTF-8 toàn diện cho console và mọi tiến trình con kế thừa.
    - Điều hướng thư mục làm việc (CWD) về gốc dự án nếu cần thiết (phục vụ Scrapy).
    - Kích hoạt hệ thống ghi nhật ký phân cấp chuẩn hóa.
    """
    global _BOOTSTRAPPED

    # 1. Định tuyến sys.path
    root_str = str(ROOT_DIR)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # 2. Chuẩn hóa mã hóa ký tự UTF-8 cho terminal và tiến trình con
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # 3. Điều hướng thư mục làm việc để Scrapy định vị đúng scrapy.cfg
    if chdir_to_root:
        os.chdir(ROOT_DIR)

    # 4. Kích hoạt Logging phân cấp tập trung nếu chưa khởi tạo
    if not _BOOTSTRAPPED:
        try:
            from configs.logging_config import setup_hierarchical_logging
            setup_hierarchical_logging()
        except Exception:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
        _BOOTSTRAPPED = True

    return ROOT_DIR


# Tự động định tuyến khi import module
bootstrap_cli_env()

__all__ = ["bootstrap_cli_env", "ROOT_DIR"]