"""
vietlawbert.configs
~~~~~~~~~~~~~~~~~~~
Phân hệ quản trị cấu hình và phân vùng lưu trữ trung tâm của VietLawBERT:
- paths: Quản lý thư mục gốc tuyệt đối, phân vùng Big Data và sinh đường dẫn log theo ngày.
- config: Singleton nạp biến môi trường .env, đồng bộ tham số Docker, Kafka, Milvus, Neo4j, LLM.
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

__all__ = [
    # Paths & Storage
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
    "BENCHMARK_DIR",
    "MODELS_DIR",
    "BASE_LOGS_DIR",
    "DAILY_LOGS_DIR",
    "TODAY_STR",
    "ALL_DIRECTORIES",
    "ensure_dirs",
    "get_log_path",
    # Config Singleton
    "Config",
    "config",
]