"""
settings.py - Cấu hình trung tâm cho Scrapy Engine trong hệ thống VietLawBERT.
Đồng bộ các tham số điều tiết luồng, hạ tầng Playwright và giám sát RAM.
"""

from __future__ import annotations

import os
import logging
from configs.paths import DAILY_LOGS_DIR, DATA_STORAGE_ROOT
from configs.config import config

BOT_NAME = "vietlaw_crawler"

SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

ROBOTSTXT_OBEY = False

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ==========================================
# 1. ĐIỀU TIẾT LUỒNG & CHỐNG NGHẼN BỘ NHỚ
# ==========================================
CONCURRENT_REQUESTS = int(os.getenv("CRAWLER_CONCURRENCY", 8))
CONCURRENT_REQUESTS_PER_DOMAIN = int(os.getenv("CRAWLER_CONCURRENCY", 8))

DOWNLOAD_DELAY = float(os.getenv("CRAWLER_DOWNLOAD_DELAY", 0.1))
RANDOMIZE_DOWNLOAD_DELAY = True

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 30.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

DOWNLOAD_TIMEOUT = 45
REACTOR_THREADPOOL_MAXSIZE = 20

RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# ==========================================
# 2. BẢO VỆ BỘ NHỚ RAM & DUNG LƯỢNG ĐĨA CỨNG
# ==========================================
HTTPCACHE_ENABLED = False
COOKIES_ENABLED = True

MEMUSAGE_ENABLED = True
MEMUSAGE_WARNING_MB = int(os.getenv("CRAWLER_MEMUSAGE_WARNING_MB", "4096"))
MEMUSAGE_LIMIT_MB = int(os.getenv("CRAWLER_MEMUSAGE_LIMIT_MB", "7168"))
MEMUSAGE_CHECK_INTERVAL_SECONDS = 15.0

# ==========================================
# 3. QUẢN TRỊ PIPELINE & NHẬT KÝ PHÂN CẤP
# ==========================================
ITEM_PIPELINES = {
    "crawler.pipelines.LegalOntologyMappingPipeline": 300,
}
FEED_EXPORT_ENCODING = "utf-8"

# Ghi nhật ký phân cấp Scrapy ra tệp riêng, giảm tải I/O console
LOG_FILE = str(DAILY_LOGS_DIR / "crawler.log")
LOG_FILE_APPEND = True
LOG_LEVEL = "INFO"
LOG_STDOUT = False

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# ==========================================
# 4. HẠ TẦNG SCRAPY-PLAYWRIGHT
# ==========================================
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

PLAYWRIGHT_PROCESS_REQUEST_HEADERS = None
PLAYWRIGHT_CONTEXTS = {
    "vbpl": {
        "user_agent": USER_AGENT,
        "viewport": {"width": 1920, "height": 1080},
        "locale": "vi-VN",
        "timezone_id": "Asia/Ho_Chi_Minh",
        "bypass_csp": True,
    },
}
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "timeout": 30 * 1000,
}