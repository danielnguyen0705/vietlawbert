"""
settings.py - Cấu hình trung tâm cho Scrapy Engine trong hệ thống VietLawBERT.
Đồng bộ các tham số điều tiết luồng, hạ tầng Playwright và giám sát RAM.
"""

from __future__ import annotations

import os
import logging
from configs.paths import get_log_path, DATA_STORAGE_ROOT
from configs.config import config

BOT_NAME = "vietlaw_crawler"

SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

# Vô hiệu hóa robots.txt để thích ứng với cổng thông tin công
ROBOTSTXT_OBEY = False

# Chuỗi User-Agent giả lập trình duyệt hiện đại
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ==========================================
# 1. ĐIỀU TIẾT LUỒNG & CHỐNG NGHẼN MẠNG
# ==========================================
CONCURRENT_REQUESTS = int(os.getenv("CRAWLER_CONCURRENCY", 8))
CONCURRENT_REQUESTS_PER_DOMAIN = int(os.getenv("CRAWLER_CONCURRENCY", 8))

DOWNLOAD_DELAY = float(os.getenv("CRAWLER_DOWNLOAD_DELAY", 0.1))
RANDOMIZE_DOWNLOAD_DELAY = True

# Tự động điều tốc (AutoThrottle) thích ứng với trạng thái phản hồi máy chủ
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 30.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# Ngắt kết nối socket nếu quá hạn để tránh treo luồng vô hạn
DOWNLOAD_TIMEOUT = 45
REACTOR_THREADPOOL_MAXSIZE = 20

RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# ==========================================
# 2. BẢO VỆ BỘ NHỚ RAM & DUNG LƯỢNG ĐĨA CỨNG
# ==========================================
# Tắt cache đĩa mặc định để tránh tạo hàng triệu file cache nhỏ gây cạn Inodes
HTTPCACHE_ENABLED = os.getenv("HTTPCACHE_ENABLED", "0").lower() in ("1", "true", "yes")
COOKIES_ENABLED = True

# Kích hoạt giám sát bộ nhớ: cảnh báo ở mức 4GB, tự ngắt an toàn nếu vượt 6GB
MEMUSAGE_ENABLED = True
MEMUSAGE_WARNING_MB = 4096
MEMUSAGE_LIMIT_MB = 6144

# ==========================================
# 3. QUẢN TRỊ PIPELINE & XUẤT DỮ LIỆU
# ==========================================
ITEM_PIPELINES = {
    "crawler.pipelines.LegalOntologyMappingPipeline": 300,
}
FEED_EXPORT_ENCODING = "utf-8"

# Nhật ký Scrapy
LOG_LEVEL = "INFO"
LOG_STDOUT = True

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