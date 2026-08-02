import scrapy
import json
import os
import re
from scrapy import signals
from bs4 import BeautifulSoup
from paths import METADATA_FILE, FAILED_FILE, get_log_path
from crawler.items import HTMLStatus
from crawler.pipelines import LegalOntologyMappingPipeline

CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = get_log_path(CURRENT_FILENAME)

SEARCH_ACTION = "c529d164f28418e5898a834422629e64c6816af1"
SEARCH_API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
HOME_URL = "https://vbpl.vn/"
ROUTER_TREE = '%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%2Ctrue%5D'

def is_numeric_id(doc_id):
    return bool(re.match(r'^\d+$', str(doc_id)))

class LawSpider(scrapy.Spider):
    name = "law_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn"]

    custom_settings = {
        'LOG_FILE': LOG_FILE_PATH,
        'LOG_LEVEL': 'INFO',
        'LOG_STDOUT': False,
        'DOWNLOAD_DELAY': 3.0,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'COOKIES_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'DOWNLOAD_TIMEOUT': 45,
        'HTTPCACHE_ENABLED': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept-Language': 'vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5',
            'sec-ch-ua': '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_file = FAILED_FILE
        self.metadata_file = METADATA_FILE
        self.successful_ids = set()
        self.current_failed_items = {}
        self.existing_metadata_ids = set()
        # THREAD-SAFE: In-memory buffer for dynamic key discovery (single-threaded Scrapy)
        self.dynamic_map = {}  # {raw_key: {edge_type, direction, graph_layer}}
        self.ontology_pipeline = LegalOntologyMappingPipeline(self.dynamic_map)
        
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
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    async def start(self):
        self.logger.info("[BAT DAU] GET vbpl.vn de lay Cookie WAF...")
        yield scrapy.Request(
            url=HOME_URL,
            method="GET",
            headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
            },
            callback=self.do_search,
            dont_filter=True
        )

    def make_search_body(self, page_number, page_size=10):
        item = {
            "pageNumber": page_number,
            "pageSize": page_size,
            "keyword": "giao thông",
            "groupVbpl": "$undefined",
            "documentName": "$undefined",
            "docNum": "$undefined",
            "docType": "$undefined",
            "majorTypeIds": "$undefined",
            "fieldTypeIds": "$undefined",
            "agencyIds": ["3", "1"],
            "effStatus": "$undefined",
            "status": "$undefined",
            "issueDateFrom": "$undefined",
            "issueDateTo": "$undefined",
            "effToFrom": "$undefined",
            "effToEnd": "$undefined",
            "administrativeUnit": "$undefined",
            "agencyLevel": "$undefined",
            "documentType": "$undefined",
            "optionDoc": "title",
            "matchMode": "all_words",
            "effFromBegin": "$undefined",
            "effFromEnd": "$undefined"
        }
        return json.dumps([item], ensure_ascii=False)

    def do_search(self, response):
        self.logger.info("[OK] Da co Cookie! Goi Next.js server action de tim kiem...")

        body = self.make_search_body(1)
        yield scrapy.Request(
            url=HOME_URL,
            method="POST",
            body=body.encode('utf-8'),
            headers={
                'accept': 'text/x-component',
                'content-type': 'text/plain;charset=UTF-8',
                'next-action': SEARCH_ACTION,
                'next-router-state-tree': ROUTER_TREE,
                'origin': 'https://vbpl.vn',
                'referer': 'https://vbpl.vn/',
            },
            callback=self.parse_search,
            cb_kwargs={'page_number': 1, 'total_pages': None},
            dont_filter=True
        )

    def parse_search(self, response, page_number, total_pages):
        text = response.text
        json_part = text[text.index('{"total'):] if '{"total' in text else text
        data = json.loads(json_part)

        documents = data.get('items', [])
        if not documents:
            self.logger.warning("[WARN] Khong co van ban nao o trang nay!")
            return

        self.logger.info(f"[TIM THAY] {len(documents)} van ban o trang {page_number}")

        for doc in documents:
            doc_id = str(doc.get('id', ''))
            if not doc_id:
                continue
            if doc_id in self.existing_metadata_ids:
                self.logger.info(f"[BO QUA] {doc_id} da co san")
                continue

            new_item = {
                'item_id': doc_id,
                'doc_number': doc.get('docNum', 'Unknown'),
                'metadata_api': doc,
            }

            detail_url = f"{SEARCH_API}/{doc_id}"
            yield scrapy.Request(
                url=detail_url,
                method="GET",
                headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
                callback=self.parse_detail,
                errback=self.handle_failure,
                cb_kwargs={'item': new_item}
            )

        total = data.get('total', 0)
        if total_pages is None:
            total_pages = (total + 9) // 10

        if page_number < total_pages:
            self.logger.info(f"[PHAN TRANG] Trang {page_number + 1}/{total_pages}...")
            body = self.make_search_body(page_number + 1)
            yield scrapy.Request(
                url=HOME_URL,
                method="POST",
                body=body.encode('utf-8'),
                headers={
                    'accept': 'text/x-component',
                    'content-type': 'text/plain;charset=UTF-8',
                    'next-action': SEARCH_ACTION,
                    'next-router-state-tree': ROUTER_TREE,
                    'origin': 'https://vbpl.vn',
                    'referer': 'https://vbpl.vn/',
                },
                callback=self.parse_search,
                cb_kwargs={'page_number': page_number + 1, 'total_pages': total_pages},
                dont_filter=True
            )

    def parse_detail(self, response, item):
        try:
            data = json.loads(response.text)
            doc_data = data.get('data', {})
        except Exception:
            self.logger.error(f"[LOI] Khong parse duoc JSON cho {item['item_id']}")
            self.handle_failure_internal(item)
            return

        doc_content = doc_data.get('documentContent')
        html_raw = (doc_content or {}).get('content', '')

        if not html_raw or len(html_raw.strip()) < 100:
            self.logger.warning(f"[THIEU HTML] Van ban {item['item_id']} khong co HTML content")
            item['html_status'] = HTMLStatus.EMPTY
            item['html_raw'] = ""
            html_dom = None
        else:
            item['html_status'] = HTMLStatus.VALID
            item['html_raw'] = html_raw
            # REQUEST CHAINING: Parse DOM here to pass to next request via meta
            html_dom = BeautifulSoup(html_raw, 'html.parser')

        item['metadata_detail'] = doc_data

        doc_id = item['item_id']
        diagram_url = f"{SEARCH_API}/{doc_id}/diagram"
        
        # Pass DOM via meta to diagram request (Zero Network Overhead for fallback)
        yield scrapy.Request(
            url=diagram_url,
            method="GET",
            headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
            callback=self.parse_diagram,
            errback=self.handle_failure,
            cb_kwargs={'item': item},
            meta={'html_dom': html_dom}
        )

    def parse_diagram(self, response, item):
        try:
            diagram_data = json.loads(response.text)
            item['diagram_json'] = diagram_data.get('data', {})
        except json.JSONDecodeError:
            item['diagram_json'] = None

        # Pass to Ontology Pipeline for processing
        html_dom = response.meta.get('html_dom')
        item = self.ontology_pipeline.process_diagram(item, html_dom)

        doc_id = item['item_id']
        self.successful_ids.add(doc_id)
        self.logger.info(f"[HOAN THANH] {item.get('doc_number')} (ID: {doc_id}) - Status: {item.get('diagram_status')}")

        if doc_id not in self.existing_metadata_ids:
            self.existing_metadata_ids.add(doc_id)
            yield item

    def handle_failure(self, failure):
        item = failure.request.cb_kwargs.get('item')
        if item and 'item_id' in item:
            self.handle_failure_internal(item)

    def handle_failure_internal(self, item):
        doc_id = item['item_id']
        self.current_failed_items[doc_id] = item
        self.logger.error(f"[THAT BAI] ID: {doc_id}")
        with open(self.failed_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def spider_closed(self, spider):
        self.logger.info("[DON DEP] Ve sinh failed_links.jsonl...")
        
        # SINGLE DISK I/O: Persist discovered dynamic mappings to system ontology
        if self.dynamic_map:
            self.logger.info(f"[ONTOLOGY] Writing {len(self.dynamic_map)} new mappings to disk...")
            self.ontology_pipeline.save_dynamic_mappings()

        existing_failures = {}
        if os.path.exists(self.failed_file):
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            existing_failures[data['item_id']] = data
                        except json.JSONDecodeError:
                            pass

        for sid in self.successful_ids:
            if sid in existing_failures:
                del existing_failures[sid]

        if existing_failures:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                for fi in existing_failures.values():
                    f.write(json.dumps(fi, ensure_ascii=False) + '\n')
            self.logger.info(f"[CON LAI] {len(existing_failures)} item loi. Hay chay rescue_spider!")
        else:
            if os.path.exists(self.failed_file):
                os.remove(self.failed_file)
            self.logger.info("[SACH] Khong con item loi nao!")
