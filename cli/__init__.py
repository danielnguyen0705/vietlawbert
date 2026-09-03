"""
vietlawbert.cli
~~~~~~~~~~~~~~~
Bộ điều phối các entrypoint vận hành hệ thống VietLawBERT.
Đảm bảo tự động định tuyến đường dẫn gốc và tương thích mã hóa UTF-8 trên mọi nền tảng.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# Xác định thư mục gốc tuyệt đối của dự án (vietlawbert/)
ROOT_DIR = Path(__file__).resolve().parent.parent


def bootstrap_cli_env(chdir_to_root: bool = False) -> Path:
    """
    Khởi tạo môi trường CLI nhất quán:
    - Bổ sung thư mục gốc vào sys.path để triệt tiêu hoàn toàn ModuleNotFoundError.
    - Chuẩn hóa luồng xuất dữ liệu UTF-8 cho Windows PowerShell và Linux Terminal.
    - Điều hướng thư mục làm việc (CWD) về gốc dự án nếu được yêu cầu.
    """
    # 1. Định tuyến sys.path
    root_str = str(ROOT_DIR)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # 2. Chuẩn hóa mã hóa ký tự UTF-8 cho terminal tránh lỗi cp1252/cp1258
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # 3. Điều hướng thư mục làm việc để Scrapy định vị đúng scrapy.cfg
    if chdir_to_root:
        os.chdir(ROOT_DIR)

    # 4. Thiết lập cấu hình logging tiêu chuẩn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    return ROOT_DIR


# Tự động thực thi khi import module
bootstrap_cli_env()

__all__ = ["bootstrap_cli_env", "ROOT_DIR"]