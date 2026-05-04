import scrapy
import json
import os
from datetime import datetime
from scrapy import signals

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & LOGGING ĐỘNG (GIỐNG NHỆN CHÍNH)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))

JSON_DIR = os.path.join(BASE_DIR, "json")
os.makedirs(JSON_DIR, exist_ok=True)
METADATA_FILE = os.path.join(JSON_DIR, 'metadata.jsonl')
FAILED_FILE = os.path.join(JSON_DIR, 'failed_links.jsonl')

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs", TODAY_STR)
os.makedirs(LOG_DIR, exist_ok=True)

CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = os.path.join(LOG_DIR, f"log_{CURRENT_FILENAME}.log")


class RescueSpider(scrapy.Spider):
    name = "rescue_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn"]

    custom_settings = {
        'FEEDS': {
            METADATA_FILE: { 
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False, 
            },
        },
        'LOG_FILE': LOG_FILE_PATH,
        'LOG_LEVEL': 'INFO',
        'LOG_STDOUT': True,
        
        # ==========================================
        # CHIẾN THUẬT MỚI: ĐIỀU TỐC THÔNG MINH
        # ==========================================
        'RETRY_TIMES': 2, # Chỉ thử lại 2 lần thôi, lỗi 500 cứng đầu quá thì bỏ qua để tránh kẹt
        'DOWNLOAD_DELAY': 2.0, # Delay cơ bản thấp xuống
        'AUTOTHROTTLE_ENABLED': True, # Bật tự động điều chỉnh tốc độ
        'AUTOTHROTTLE_START_DELAY': 5.0,
        'AUTOTHROTTLE_MAX_DELAY': 60.0, # Nếu server mệt, tự động giãn cách lên tới 60s
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0, 
        
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1, 
        'COOKIES_ENABLED': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    }

    def __init__(self, *args, **kwargs):
        super(RescueSpider, self).__init__(*args, **kwargs)
        
        self.failed_file = FAILED_FILE
        self.successful_ids = set()
        self.failed_items = {}

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(RescueSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        self.logger.info("[BAT DAU] Chien dich giai cuu link loi tu failed_links.jsonl")
        
        if not os.path.exists(self.failed_file):
            self.logger.info("Khong tim thay file failed_links.jsonl. He thong da sach bong loi!")
            return

        with open(self.failed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        doc_id = item.get('item_id')
                        
                        if doc_id:
                            self.failed_items[doc_id] = item
                            diagram_url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}/diagram"
                            
                            yield scrapy.Request(
                                url=diagram_url,
                                method="GET",
                                headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/'},
                                callback=self.parse_diagram,
                                errback=self.handle_error,
                                cb_kwargs={'item': item},
                                dont_filter=True 
                            )
                    except json.JSONDecodeError:
                        pass

    def parse_diagram(self, response, item):
        try:
            diagram_data = json.loads(response.text)
            item['diagram_json'] = diagram_data.get('data', {})
        except json.JSONDecodeError:
            item['diagram_json'] = None

        doc_id = item['item_id']
        html_url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/minio/buckets/vbpl/{doc_id}/{doc_id}_content_origin.html/download"
        
        yield scrapy.Request(
            url=html_url,
            method="GET",
            headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
            callback=self.parse_html,
            errback=self.handle_error,
            cb_kwargs={'item': item},
            dont_filter=True
        )

    def parse_html(self, response, item):
        item['full_text_html'] = response.text
        doc_id = item['item_id']

        self.successful_ids.add(doc_id)
        self.logger.info(f"[CUU THANH CONG] Du lieu cua van ban: {item.get('doc_number')}")
        yield item

    def handle_error(self, failure):
        request = failure.request
        item = request.cb_kwargs.get('item')
        doc_id = item.get('item_id') if item else "Unknown"
        self.logger.error(f"[THAT BAI] Bo qua ID {doc_id} sau khi retry. Server van tu choi!")

    def spider_closed(self, spider):
        self.logger.info("[DONG BO] Dang cap nhat lai danh sach loi...")
        
        remaining_failures = []
        for doc_id, item in self.failed_items.items():
            if doc_id not in self.successful_ids:
                remaining_failures.append(item)

        if remaining_failures:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                for item in remaining_failures:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"[THONG KE] Con {len(remaining_failures)} link cung dau. Hay thu chay lai sau.")
        else:
            if os.path.exists(self.failed_file):
                os.remove(self.failed_file)
            self.logger.info("DA CUU XONG TOAN BO LINK LOI! Khong con van ban nao bi sot.")