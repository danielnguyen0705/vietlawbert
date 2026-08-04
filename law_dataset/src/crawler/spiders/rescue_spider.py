import scrapy
import json
import os
import re
from datetime import datetime
from scrapy import signals
from typing import Optional
from paths import get_log_path
from crawler.items import HTMLStatus, PDFStatus
from crawler.pipelines import LegalOntologyMappingPipeline

CURRENT_FILENAME = os.path.basename(__file__).split('.')[0]
LOG_FILE_PATH = get_log_path(CURRENT_FILENAME)

SEARCH_API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"


class RescueSpider(scrapy.Spider):
    name = "rescue_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn"]

    custom_settings = {
        'LOG_FILE': LOG_FILE_PATH,
        'LOG_LEVEL': 'INFO',
        'LOG_STDOUT': True,
        'RETRY_TIMES': 3,
        'DOWNLOAD_DELAY': 3.0,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 8.0,
        'AUTOTHROTTLE_MAX_DELAY': 90.0,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'COOKIES_ENABLED': False,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'DOWNLOAD_TIMEOUT': 60,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_ids = set()
        self.pending_items = {}
        self.dynamic_map = {}
        self.ontology_pipeline = LegalOntologyMappingPipeline(self.dynamic_map)

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    async def start(self):
        self.logger.info("[WARN] Rescue spider cần được refactor để đọc từ Kafka thay vì audit_trail.jsonl")
        return

    def _extract_pdf_filename(self, doc_data: dict) -> Optional[str]:
        # 1. attachments
        attachments = (
            doc_data.get("attachments")
            or doc_data.get("documentContent", {}).get("attachments")
            or []
        )
        for att in attachments:
            file_name = att.get("fileName", "") or ""
            if file_name.lower().endswith(".pdf"):
                return file_name

        # 2. documentRelatedList (for VBHN)
        related = doc_data.get("documentRelatedList") or []
        for att in related:
            file_name = att.get("fileName", "") or ""
            if file_name.lower().endswith(".pdf"):
                return file_name

        # 3. Try infer from doc_number for VBHN
        meta = doc_data.get("metadata_api") or {}
        doc_num = meta.get("docNum", "") or doc_data.get("docNum", "")
        if doc_num and "VBHN" in doc_num:
            safe = doc_num.replace("/", "_").replace("\\", "_")
            return f"{safe}.pdf"

        return None

    def _build_nextjs_body(self, doc_id: str, file_name: str) -> str:
        return json.dumps([{"bucketName": "vbpl", "folderName": doc_id, "objectName": file_name, "preview": None}])

    def _extract_pdf_url_from_api(self, doc_data: dict) -> Optional[str]:
        """Trích xuất trực tiếp link PDF từ các field metadata của API gateway."""
        # Check attachments ở các cấp
        attachments = (
            doc_data.get("attachments")
            or doc_data.get("documentContent", {}).get("attachments")
            or []
        )
        for att in attachments:
            url = att.get("fileUrl") or att.get("url")
            if url and (".pdf" in url.lower() or att.get("fileName", "").lower().endswith(".pdf")):
                return url

        # Check documentRelatedList (Văn bản hợp nhất hay đính kèm ở đây)
        related = doc_data.get("documentRelatedList") or []
        for att in related:
            url = att.get("fileUrl") or att.get("url")
            if url and (".pdf" in url.lower() or att.get("fileName", "").lower().endswith(".pdf")):
                return url
        return None

    def parse_detail(self, response, record):
        try:
            data = json.loads(response.text)
            doc_data = data.get('data', {})
        except Exception:
            doc_id = record.get('item_id')
            self.logger.error(f"[LOI] Khong parse duoc JSON cho {doc_id}")
            return

        doc_content = doc_data.get('documentContent')
        html_raw = (doc_content or {}).get('content', '')

        item = dict(record)  # copy
        item['metadata_detail'] = doc_data

        if html_raw and len(html_raw.strip()) > 100:
            item['html_status'] = HTMLStatus.VALID.value
            item['html_raw'] = html_raw
            item['pdf_status'] = PDFStatus.NOT_FOUND.value
            self.logger.info(f"[HTML CO SAN] {item.get('item_id')} - lay diagram")
            yield self.yield_diagram(item)
            return

        # HTML empty, need PDF
        item['html_status'] = HTMLStatus.EMPTY.value
        item['html_raw'] = ""
        doc_id = item.get('item_id', '')

        # THỬ LẤY TRỰC TIẾP URL TỪ API GATEWAY (ƯU TIÊN HÀNG ĐẦU)
        direct_pdf_url = self._extract_pdf_url_from_api(doc_data)
        if direct_pdf_url:
            self.logger.info(f"[DIRECT PDF URL] Tim thay direct link tu API cho {doc_id}: {direct_pdf_url}")
            item['pdf_path'] = direct_pdf_url
            yield scrapy.Request(
                url=direct_pdf_url,
                method="GET",
                callback=self.parse_pdf,
                errback=self.handle_error,
                cb_kwargs={'item': item, 'doc_data': doc_data},
                dont_filter=True
            )
            return

        # Nếu không có direct link, thử tải qua endpoint download chung của moj
        self.logger.warning(f"[FALLBACK DOWNLOAD] {doc_id} - khong co direct link, thu tải qua /download")
        download_url = f"{SEARCH_API}/{doc_id}/download"
        yield scrapy.Request(
            url=download_url,
            method="GET",
            headers={
                'Origin': 'https://vbpl.vn',
                'Referer': f'https://vbpl.vn/van-ban/{doc_id}',
                'Accept': 'application/pdf,*/*',
            },
            callback=self.parse_pdf,
            errback=self.handle_error,
            cb_kwargs={'item': item, 'doc_data': doc_data},
            dont_filter=True
        )

    def yield_diagram(self, item):
        diagram_url = f"{SEARCH_API}/{item.get('item_id')}/diagram"
        return scrapy.Request(
            url=diagram_url,
            method="GET",
            headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
            callback=self.parse_diagram,
            errback=self.handle_error,
            cb_kwargs={'item': item},
            dont_filter=True
        )

    def parse_pdf(self, response, item, doc_data):
        doc_id = item.get('item_id', 'unknown')
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', errors='ignore')
        body = response.body

        if (
            response.status == 200
            and body
            and len(body) > 5000
            and ('pdf' in content_type.lower() or response.url.endswith('.pdf'))
        ):
            pdf_path = os.path.join(RAW_PDF_DIR, f"{doc_id}.pdf")
            os.makedirs(RAW_PDF_DIR, exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(body)

            import fitz
            try:
                doc = fitz.open(pdf_path)
                text = "".join([page.get_text() for page in doc])
                doc.close()

                if len(text.strip()) > 500:
                    item['pdf_status'] = PDFStatus.DIGITAL_TEXT.value
                    self.logger.info(f"[PDF OK] {doc_id} - DIGITAL_TEXT ({len(body)} bytes, {len(text.strip())} chars)")
                else:
                    item['pdf_status'] = PDFStatus.SCANNED_OR_CORRUPTED.value
                    self.logger.warning(f"[PDF SCAN] {doc_id} - SCANNED_OR_CORRUPTED ({len(body)} bytes, {len(text.strip())} chars)")
            except Exception as e:
                item['pdf_status'] = PDFStatus.SCANNED_OR_CORRUPTED.value
                self.logger.warning(f"[PDF ERROR] {doc_id} - parse failed: {e}")

            item['pdf_local_path'] = pdf_path
            self.logger.info(f"[PDF LUU] {doc_id} -> {pdf_path}")
        else:
            item['pdf_status'] = PDFStatus.NOT_FOUND.value
            self.logger.warning(f"[PDF LOI] {doc_id} - response {response.status}, content-type: {content_type}")

        item['metadata_detail'] = doc_data
        diagram_url = f"{SEARCH_API}/{doc_id}/diagram"
        yield scrapy.Request(
            url=diagram_url,
            method="GET",
            headers={
                'Origin': 'https://vbpl.vn',
                'Referer': 'https://vbpl.vn/',
                'Accept': 'application/json'
            },
            callback=self.parse_diagram,
            errback=self.handle_error,
            cb_kwargs={'item': item},
            dont_filter=True
        )

    def parse_diagram(self, response, item):
        try:
            diagram_data = json.loads(response.text)
            item['diagram_json'] = diagram_data.get('data', {})
        except json.JSONDecodeError:
            item['diagram_json'] = None

        item = self.ontology_pipeline.process_diagram(item)

        doc_id = item['item_id']
        if doc_id not in self.successful_ids:
            self.successful_ids.add(doc_id)
            self.logger.info(f"[CUU THANH CONG] {item.get('doc_number')} (ID: {doc_id}) [Status: {item.get('diagram_status')}]")
            yield item

    def handle_error(self, failure):
        request = failure.request
        item = request.cb_kwargs.get('item')
        doc_id = item.get('item_id') if item else "Unknown"
        self.logger.error(f"[THAT BAI] Bo qua ID {doc_id} sau khi retry.")

    def spider_closed(self, spider):
        if self.dynamic_map:
            self.ontology_pipeline.save_dynamic_mappings()
        self.logger.info("[DONG BO] Cap nhat audit_trail.jsonl...")

        # Read current audit, update successful items
        updated_records = {}
        if os.path.exists(self.audit_file):
            with open(self.audit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            updated_records[rec['item_id']] = rec
                        except json.JSONDecodeError:
                            pass

        for sid in self.successful_ids:
            if sid in updated_records:
                # Update status from our processed item
                # We'll just mark it - the pipeline will write fresh records
                pass

        self.logger.info(f"Da xu ly xong {len(self.successful_ids)} items.")