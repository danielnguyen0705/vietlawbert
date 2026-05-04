import os
import logging

BOT_NAME = 'vietlaw_crawler'

SPIDER_MODULES = ['crawler.spiders']
NEWSPIDER_MODULE = 'crawler.spiders'

# FIX: Phải để False vì web nhà nước chặn Bot rất gắt
ROBOTSTXT_OBEY = False 

# FIX: Giả lập trình duyệt người thật
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'

# ==========================================
# CẤU HÌNH ĐIỀU TIẾT LUỒNG (CHỐNG LỖI 500)
# ==========================================

# Ép nhện đi hàng một để server không bị quá tải
CONCURRENT_REQUESTS = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# Tăng thời gian nghỉ (3 giây) và cho phép ngẫu nhiên để giống người thật hơn
DOWNLOAD_DELAY = 3.0
RANDOMIZE_DOWNLOAD_DELAY = True

# Bật AutoThrottle: Scrapy sẽ tự "nhìn sắc mặt" server để chỉnh tốc độ
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 3.0
AUTOTHROTTLE_MAX_DELAY = 60.0
# Tỉ lệ yêu cầu song song mục tiêu (giữ ở mức 1.0 cho an toàn)
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# ==========================================
# CẤU HÌNH THỬ LẠI (RETRY), TIMEOUT VÀ CACHE
# ==========================================

# [THÊM MỚI] Trị bệnh "chết lâm sàng": Quá 30s server không rep là cắt luôn, báo lỗi rồi đi tiếp!
DOWNLOAD_TIMEOUT = 30  

# [THÊM MỚI] Nới rộng hồ chứa luồng xử lý ngầm để Pipeline không bị nghẽn lúc ghi file JSONL
REACTOR_THREADPOOL_MAXSIZE = 20

# [ĐIỀU CHỈNH] Giảm số lần thử lại xuống 3. (Nếu để 5 lần x 30s timeout = kẹt 2.5 phút/link là quá lâu)
RETRY_TIMES = 3 
# Chỉ định rõ các mã lỗi cần thử lại (Đặc trị lỗi 500)
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

# Bật Cache: Đã cào link nào rồi thì lần sau chạy lại nó lấy từ máy, không hỏi server nữa
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 0 # 0 là cache vĩnh viễn
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [500, 503, 404, 403] # Không cache các trang bị lỗi để lần sau cào lại

COOKIES_ENABLED = True

ITEM_PIPELINES = {
    'crawler.pipelines.VietLawDataPipeline': 300, 
}

# ==========================================
# CẤU HÌNH GHI LOG CHO SCRAPY (HỘP ĐEN)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, '../../data/logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'scrapy_spider.log')
LOG_FILE_APPEND = False

# [ĐIỀU CHỈNH] Đổi từ DEBUG sang INFO để file log sạch sẽ, dễ đọc, chỉ hiển thị thông báo chính
LOG_LEVEL = 'INFO' 

logging.getLogger('urllib3').setLevel(logging.WARNING)