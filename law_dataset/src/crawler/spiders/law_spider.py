import scrapy
import json
import logging
import copy
import os
from scrapy import signals
from scrapy.exceptions import DontCloseSpider

class LawSpider(scrapy.Spider):
    name = "law_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn"]
    
    SEARCH_API_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all" 

    # ==========================================
    # CẤU HÌNH TỰ ĐỘNG XUẤT FILE METADATA
    # ==========================================
    custom_settings = {
        'FEEDS': {
            # Sử dụng đường dẫn tuyệt đối để không bao giờ bị lạc file
            'D:/Daniel_Nguyen/nckh_project/law_dataset/json/metadata.jsonl': { 
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False, # Chế độ Append
            },
        }
    }

    def __init__(self, *args, **kwargs):
        super(LawSpider, self).__init__(*args, **kwargs)
        
        # Gắn chết đường dẫn tuyệt đối vào thư mục json
        self.data_dir = r"D:\Daniel_Nguyen\nckh_project\law_dataset\json"
        os.makedirs(self.data_dir, exist_ok=True)
        self.failed_file = os.path.join(self.data_dir, 'failed_links.jsonl')
        self.metadata_file = os.path.join(self.data_dir, 'metadata.jsonl')

        self.successful_ids = set()
        self.current_failed_items = {}
        self.existing_metadata_ids = set()
        
        # Biến kiểm soát: Chỉ cho phép nhện quay lại cứu 1 lần để tránh lặp vô tận nếu server chết hẳn
        self.rescue_round_done = False 

        # Đọc dữ liệu cũ để chống trùng lặp
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.existing_metadata_ids.add(data.get('item_id'))
                        except json.JSONDecodeError:
                            pass

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(LawSpider, cls).from_crawler(crawler, *args, **kwargs)
        # Bắt mạch 2 thời điểm vàng của nhện: Lúc rảnh rỗi (idle) và lúc tắt hẳn (closed)
        crawler.signals.connect(spider.spider_idle, signal=signals.spider_idle)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def start_requests(self):
        self.logger.info("[BAT DAU] Chien dich Smart Crawling VietLawBERT")
        
        payload = {
            "pageNumber": 1,
            "pageSize": 20,
            "keyword": "giao thông, đường bộ",
            "sortBy": "issueDate",
            "sortDirection": "desc",
            "groupVbpl": True,
            "docType": [
                "58bf04c0-a197-4d6e-96e9-2e51066209b5", "404b68a7-8e71-4ee5-a6c0-07e59f35f824",
                "11025e19-2dd6-4165-85ad-ab6241186a1a", "0d08b84c-7de7-4800-8760-2a68265e7890",
                "178c63a9-73ff-4fd4-9d91-18d690520090", "0e4f2bde-5ccb-4001-9e0a-b43f51cca5e8"
            ],
            "agencyIds": ["55", "56", "1", "3", "50", "473", "172", "274", "81", "9", "79", "78"],
            "agencyLevel": "TRUNG_UONG",
            "optionDoc": "all",
            "matchMode": "exact_phrase"
        }

        yield scrapy.Request(
            url=self.SEARCH_API_URL,
            method="POST",
            body=json.dumps(payload),
            headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/'},
            callback=self.parse_search_results,
            errback=self.handle_error, 
            cb_kwargs={'payload': payload, 'item': None} 
        )

    def parse_search_results(self, response, payload, item=None):
        data = json.loads(response.text)
        documents = data.get('data', {}).get('items', []) 
        
        if not documents:
            self.logger.warning("[CANH BAO] Khong tim thay van ban nao o trang nay!")
            return

        for doc in documents:
            doc_id = str(doc.get('id', ''))
            
            new_item = {
                'item_id': doc_id,
                'doc_number': doc.get('docNum', 'Unknown'),
                'metadata_api': doc,
            }
            
            diagram_url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}/diagram"
            
            yield scrapy.Request(
                url=diagram_url,
                method="GET",
                headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/'},
                callback=self.parse_diagram,
                errback=self.handle_error,
                cb_kwargs={'item': new_item}
            )

        current_page = payload['pageNumber']
        total_items = data.get('data', {}).get('total', 0)
        page_size = payload['pageSize']
        calculated_total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 1
        
        if current_page < calculated_total_pages:
            self.logger.info(f"[*] Dang chuyen sang trang {current_page + 1}/{calculated_total_pages}...")
            next_payload = copy.deepcopy(payload)
            next_payload['pageNumber'] += 1
            
            yield scrapy.Request(
                url=self.SEARCH_API_URL,
                method="POST",
                body=json.dumps(next_payload),
                headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/'},
                callback=self.parse_search_results,
                errback=self.handle_error,
                cb_kwargs={'payload': next_payload, 'item': None}
            )

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
            cb_kwargs={'item': item}
        )

    def parse_html(self, response, item):
        item['full_text_html'] = response.text
        doc_id = item['item_id']

        self.successful_ids.add(doc_id)
        self.logger.info(f"[HOAN THANH] Du lieu cua van ban: {item.get('doc_number')}")
        
        if doc_id not in self.existing_metadata_ids:
            self.existing_metadata_ids.add(doc_id)
            yield item
            
    # ==========================================
    # CƠ CHẾ 1: GHI NÓNG VÀO FILE KHI LỖI
    # ==========================================
    def handle_error(self, failure):
        request = failure.request
        item = request.cb_kwargs.get('item')
        
        if item and 'item_id' in item:
            doc_id = item['item_id']
            self.current_failed_items[doc_id] = item
            
            self.logger.error(f"[GHI NONG] Luu ngay item loi vao file: {doc_id} - URL: {request.url}")
            # Ghi chèn (append) ngay lập tức vào failed_links.jsonl
            with open(self.failed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ==========================================
    # CƠ CHẾ 2: VÒNG GIẢI CỨU SAU KHI QUÉT XONG
    # ==========================================
    def spider_idle(self, spider):
        # Hàm này tự động gọi khi nhện hết việc làm (chuẩn bị đóng)
        # Kiểm tra xem có lỗi không và CHƯA từng chạy giải cứu
        pending_failures = [doc_id for doc_id in self.current_failed_items if doc_id not in self.successful_ids]
        
        if pending_failures and not self.rescue_round_done:
            self.rescue_round_done = True # Đánh dấu đã giải cứu để không lặp vô tận
            self.logger.info("\n" + "="*60)
            self.logger.info(f"KICH HOAT VONG GIAI CUU: TU DONG CAO LAI {len(pending_failures)} LINK LOI!")
            self.logger.info("="*60 + "\n")
            
            for doc_id in pending_failures:
                item = self.current_failed_items[doc_id]
                diagram_url = f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}/diagram"
                
                # Ném lại Request vào hàng đợi để nhện cào tiếp
                yield scrapy.Request(
                    url=diagram_url,
                    method="GET",
                    headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/'},
                    callback=self.parse_diagram,
                    errback=self.handle_error,
                    cb_kwargs={'item': item},
                    dont_filter=True # Bắt buộc phải có cái này để bỏ qua bộ lọc trùng lặp của Scrapy
                )
            
            # Cấm nhện đóng cửa, bắt nó ở lại làm việc tiếp!
            raise DontCloseSpider("Chưa xong đâu nhện ơi, còn link lỗi kìa!")

    # ==========================================
    # CƠ CHẾ 3: TỔNG VỆ SINH FILE SAU CÙNG
    # ==========================================
    def spider_closed(self, spider):
        self.logger.info("[DONG BO] Nhon dang tong ve sinh file failed_links.jsonl...")
        existing_failures = {}
        
        # Đọc lại toàn bộ file lỗi (lúc này có thể bị trùng do append)
        if os.path.exists(self.failed_file):
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            existing_failures[data['item_id']] = data
                        except json.JSONDecodeError:
                            pass

        # Gạt bỏ những thằng đã cào thành công (kể cả thành công ở vòng giải cứu)
        for success_id in self.successful_ids:
            if success_id in existing_failures:
                del existing_failures[success_id]
                self.logger.info(f"Da cuu thanh cong va don khoi blacklist: {success_id}")

        # Ghi đè file sạch sẽ cuối cùng
        if existing_failures:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                for failed_item in existing_failures.values():
                    f.write(json.dumps(failed_item, ensure_ascii=False) + '\n')
            self.logger.info(f"[THONG KE] Con lai {len(existing_failures)} ca chua cuu duoc. Giu lai de lan sau cao tiep!")
        else:
            if os.path.exists(self.failed_file):
                os.remove(self.failed_file)
            self.logger.info("TUYET VOI! Da clear hoan toan danh sach loi, metadata no ne!")