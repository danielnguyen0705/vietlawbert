"""
settings.py - Cấu hình Scrapy Engine cho hệ thống VietLawBERT (Kiến trúc v3 - Lightweight).
Kích hoạt Disk-based Queue (JOBDIR) động, điều tiết tải luồng và tối ưu hóa tài nguyên RAM/CPU.
"""

from __future__ import annotations

import os
import logging
from configs.paths import DAILY_LOGS_DIR, DATA_STORAGE_ROOT

BOT_NAME = "vietlaw_crawler"

SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

ROBOTSTXT_OBEY = False

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ==============================================================================
# 1. HẠ TẦNG HÀNG ĐỢI ĐĨA CỨNG ĐỘNG (TRÁNH XUNG ĐỘT TRẠNG THÁI GIỮA CÁC SHARD)
# ==============================================================================
# Chỉ bật JOBDIR khi có biến môi trường chỉ định rõ, tránh xung đột giữa các lượt chạy phân đoạn
if os.getenv("SCRAPY_JOBDIR"):
    JOBDIR = os.getenv("SCRAPY_JOBDIR")
    SCHEDULER_DISK_QUEUE = "scrapy.squeues.FifoDiskQueue"
    SCHEDULER_MEMORY_QUEUE = "scrapy.squeues.FifoMemoryQueue"
    SCHEDULER_PRIORITY_QUEUE = "scrapy.pqueues.DownloaderAwarePriorityQueue"

# ==============================================================================
# 2. ĐIỀU TIẾT TẢI LUỒNG & CHỐNG NGHẼN CPU CORE
# ==============================================================================
CONCURRENT_REQUESTS = int(os.getenv("CRAWLER_CONCURRENCY", "4"))
CONCURRENT_REQUESTS_PER_DOMAIN = int(os.getenv("CRAWLER_CONCURRENCY", "4"))

DOWNLOAD_DELAY = float(os.getenv("CRAWLER_DOWNLOAD_DELAY", "0.15"))
RANDOMIZE_DOWNLOAD_DELAY = True

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 15.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 3.0

DOWNLOAD_TIMEOUT = 30
REACTOR_THREADPOOL_MAXSIZE = 16

RETRY_TIMES = 4
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

HTTPCACHE_ENABLED = False
COOKIES_ENABLED = True

# ==============================================================================
# 3. GIỚI HẠN BỘ NHỚ RAM BẢO VỆ TIẾN TRÌNH HỆ ĐIỀU HÀNH
# ==============================================================================
MEMUSAGE_ENABLED = True
MEMUSAGE_WARNING_MB = int(os.getenv("CRAWLER_MEMUSAGE_WARNING_MB", "3584"))
MEMUSAGE_LIMIT_MB = int(os.getenv("CRAWLER_MEMUSAGE_LIMIT_MB", "5120"))
MEMUSAGE_CHECK_INTERVAL_SECONDS = 15.0

# ==============================================================================
# 4. PIPELINE & GHI NHẬT KÝ
# ==============================================================================
ITEM_PIPELINES = {
    "crawler.pipelines.LegalOntologyMappingPipeline": 300,
}
FEED_EXPORT_ENCODING = "utf-8"

LOG_FILE = str(DAILY_LOGS_DIR / "crawler.log")
LOG_FILE_APPEND = True
LOG_LEVEL = "INFO"
LOG_STDOUT = False

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# ==============================================================================
# 5. HẠ TẦNG PLAYWRIGHT SIÊU NHẸ (HEADLESS CHROME TỐI GIẢN)
# ==============================================================================
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

PLAYWRIGHT_PROCESS_REQUEST_HEADERS = None
PLAYWRIGHT_CONTEXTS = {
    "vbpl": {
        "user_agent": USER_AGENT,
        "viewport": {"width": 1280, "height": 720},
        "locale": "vi-VN",
        "timezone_id": "Asia/Ho_Chi_Minh",
        "bypass_csp": True,
    },
}
# Loại bỏ hoàn toàn cờ --single-process để bảo đảm Chromium vận hành ổn định trên Linux
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "timeout": 20 * 1000,
    "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-setuid-sandbox",
    ],
}