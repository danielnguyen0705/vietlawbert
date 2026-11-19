"""
vietlawbert.configs
~~~~~~~~~~~~~~~~~~~
Phân hệ quản trị cấu hình, đường dẫn tập trung và nhật ký phân cấp của VietLawBERT.
"""

from .paths import (
    ROOT_DIR,
    BASE_DIR,
    DEFAULT_STORAGE_ROOT,
    DATA_STORAGE_ROOT,
    RAW_SHARDS_DIR,
    RAW_PDFS_DIR,
    RAW_HTML_DIR,
    PROCESSED_DIR,
    JSON_DIR,
    ARTIFACTS_DIR,
    TRIPLETS_DIR,
    BENCHMARK_DIR,
    MODELS_DIR,
    BASE_LOGS_DIR,
    DAILY_LOGS_DIR,
    TODAY_STR,
    ALL_DIRECTORIES,
    ensure_dirs,
    get_log_path,
)
from .config import Config, config
from .logging_config import setup_hierarchical_logging, get_subsystem_logger

__all__ = [
    # Đường dẫn phân vùng lưu trữ
    "ROOT_DIR",
    "BASE_DIR",
    "DEFAULT_STORAGE_ROOT",
    "DATA_STORAGE_ROOT",
    "RAW_SHARDS_DIR",
    "RAW_PDFS_DIR",
    "RAW_HTML_DIR",
    "PROCESSED_DIR",
    "JSON_DIR",
    "ARTIFACTS_DIR",
    "TRIPLETS_DIR",
    "BENCHMARK_DIR",
    "MODELS_DIR",
    "BASE_LOGS_DIR",
    "DAILY_LOGS_DIR",
    "TODAY_STR",
    "ALL_DIRECTORIES",
    "ensure_dirs",
    "get_log_path",
    # Cấu hình Singleton
    "Config",
    "config",
    # Logging tập trung
    "setup_hierarchical_logging",
    "get_subsystem_logger",
]