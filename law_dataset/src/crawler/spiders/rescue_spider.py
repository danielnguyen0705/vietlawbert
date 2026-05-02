import scrapy
import json
import os
from scrapy import signals

class RescueSpider(scrapy.Spider):
    name = "rescue_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn"]

    # ==========================================
    # CẤU HÌNH TỰ ĐỘNG XUẤT FILE METADATA
    # (Sử dụng chung đường dẫn với nhện chính)
    # ==========================================
    custom_settings = {
        'FEEDS': {
            'D:/Daniel_Nguyen/nckh_project/law_dataset/json/metadata.jsonl': { 
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False, # Chế độ Append nối tiếp vào file cũ
            },
        }
    }

    def __init__(self, *args, **kwargs):
        super(RescueSpider, self).__init__(*args, **kwargs)
        
        # Đường dẫn tuyệt đối trỏ thẳng vào thư mục chứa file lỗi
        self.data_dir = r"D:\Daniel_Nguyen\nckh_project\law_dataset\json"
        self.failed_file = os.path.join(self.data_dir, 'failed_links.jsonl')
        
        self.successful_ids = set()
        self.failed_items = {}

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(RescueSpider, cls).from_crawler(crawler, *args, **kwargs)
        # Bắt mạch thời điểm nhện đóng cửa để dọn dẹp file
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        self.logger.info("[BAT DAU] Chien dich giai cuu link loi tu failed_links.jsonl")
        
        if not os.path.exists(self.failed_file):
            self.logger.info("Khong tim thay file failed_links.jsonl. He thong da sach bong loi!")
            return

        # Đọc từng dòng trong file jsonl
        with open(self.failed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        doc_id = item.get('item_id')
                        
                        if doc_id:
                            self.failed_items[doc_id] = item
                            diagram_url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}/diagram"
                            
                            # Khởi tạo request lấy Lược đồ (Diagram)
                            yield scrapy.Request(
                                url=diagram_url,
                                method="GET",
                                headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/'},
                                callback=self.parse_diagram,
                                errback=self.handle_error,
                                cb_kwargs={'item': item},
                                dont_filter=True # Không lọc trùng vì ta chủ động cào lại các link đã rớt
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
        
        # Tiếp tục lấy file HTML
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
        
        # Bắn dữ liệu về pipeline để lưu y như nhện chính (sẽ tự chui vào metadata.jsonl)
        yield item

    def handle_error(self, failure):
        request = failure.request
        item = request.cb_kwargs.get('item')
        doc_id = item.get('item_id') if item else "Unknown"
        self.logger.error(f"[THAT BAI] Server van tu choi ket noi voi ID: {doc_id}")

    # ==========================================
    # CƠ CHẾ DỌN DẸP SỔ ĐEN
    # ==========================================
    def spider_closed(self, spider):
        self.logger.info("[DONG BO] Dang cap nhat lai danh sach loi...")
        
        # Tìm những ID vẫn chưa cứu được (nằm trong file failed ban đầu nhưng không có trong successful_ids)
        remaining_failures = []
        for doc_id, item in self.failed_items.items():
            if doc_id not in self.successful_ids:
                remaining_failures.append(item)

        # Ghi đè lại file failed_links.jsonl bằng danh sách những thằng ngoan cố nhất
        if remaining_failures:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                for item in remaining_failures:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"[THONG KE] Con {len(remaining_failures)} link cung dau. Hay thu chay lai cuoc giai cuu vao luc khac.")
        else:
            # Xóa sổ đen nếu cứu thành công 100%
            if os.path.exists(self.failed_file):
                os.remove(self.failed_file)
            self.logger.info("DA CUU XONG TOAN BO LINK LOI! Khong con van ban nao bi sot.")