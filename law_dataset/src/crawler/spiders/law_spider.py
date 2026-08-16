import asyncio
import scrapy
import html as html_lib
import io
import json
import os
import re
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from scrapy import signals
from bs4 import BeautifulSoup
from crawler.items import HTMLStatus

SEARCH_API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
HOME_URL = "https://vbpl.vn/"
ROUTER_TREE = '%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%2Ctrue%5D'
FILES_ACTION = os.getenv("VBPL_FILES_ACTION", "7b29f485c8e43b71a94bfc11b54459d7e27293e6")

# Public `LoaiVanBan` catalogue exposed by vbpl.vn. The search action expects
# UUID values (not the short codes found in result metadata). Keeping the
# catalogue here lets the server exclude translations before pagination; the
# client-side code check below remains a defensive guard.
OFFICIAL_DOC_TYPE_IDS = {
    "HP": "58bf04c0-a197-4d6e-96e9-2e51066209b5",
    "BL": "404b68a7-8e71-4ee5-a6c0-07e59f35f824",
    "LU": "11025e19-2dd6-4165-85ad-ab6241186a1a",
    "PL": "1cd0d144-ccb5-4196-8b56-9a3f599c9341",
    "NĐ": "0d08b84c-7de7-4800-8760-2a68265e7890",
    "TT": "178c63a9-73ff-4fd4-9d91-18d690520090",
    "TTLT_TW": "0e4f2bde-5ccb-4001-9e0a-b43f51cca5e8",
    "QĐ": "0a5362e8-cdca-436e-96cd-979598df3b16",
    "L-CTN": "43323400-7d47-418e-80ef-a912a349d4e3",
    "NQ": "044d941c-40de-45b9-ae84-51f5a730bfe0",
    "NQLT": "048dd409-3441-4c13-a5ae-80236ad3ce68",
    "VBHN": "26b8a9ff-1b59-4c57-9605-f2ad4ed7c324",
    "VBHC": "25fad6ae-78f6-4acd-a70e-da19032729af",
    "BD": "9979b9a4-9e4c-4a14-a990-6d86e11b75c5",
    "CTHI": "0045710b-eb54-4ce5-b511-76aa23f3021b",
    "VBHTH": "c1e77420-856e-4741-973b-82e41c3783e5",
    "CHUA_XAC_DINH": "9f3a2c7e-4b6d-4f1a-a9d8-2c8e5b7a41f3",
    "LVBM-CV": "ecc6bbe0-5d0c-4916-836b-bb2f8146249c",
    "QD": "608e6760-929a-44b0-9a67-ea41c903e6b9",
    "LVBM-SL": "276cbd73-5b42-4b90-a6a7-ee1bca492b7d",
    "LVBM-TB": "ed938c9b-e50d-42f8-80b9-e945f207bd04",
    "LVBM-CU": "956a1482-c8b2-4b1b-8b50-31df8582ae6e",
    "LVB-VBK": "39a344bd-d086-4fd9-880c-9ecefdb19a40",
    "LVB-VBLQ": "e154c97d-d4d8-4968-9b72-feffa38924d5",
    "LVBM-TTLB": "88516734-9f43-4d32-8f54-256109edfbd4",
    "LVB-SLE": "1e0386d2-9395-4a0f-9d04-18a9e01f0ca7",
}

def is_numeric_id(doc_id):
    return bool(re.match(r'^\d+$', str(doc_id)))

class LawSpider(scrapy.Spider):
    name = "law_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn", "fptcloud.com"]

    handle_httpstatus_list = [403]

    custom_settings = {
        'LOG_LEVEL': 'INFO',
        'LOG_STDOUT': True,
        'DOWNLOAD_DELAY': float(os.getenv('CRAWLER_DOWNLOAD_DELAY', '0.1')),
        'CONCURRENT_REQUESTS': int(os.getenv('CRAWLER_CONCURRENCY', '8')),
        'CONCURRENT_REQUESTS_PER_DOMAIN': int(os.getenv('CRAWLER_CONCURRENCY', '8')),
        'COOKIES_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'DOWNLOAD_TIMEOUT': 45,
        'HTTPCACHE_ENABLED': False,
    }

    def __init__(
        self,
        start_page=1,
        pages=None,
        limit=None,
        page_size=10,
        keyword="",
        agency_ids="",
        doc_ids="",
        group_vbpl="true",
        exclude_doc_type_codes="BD",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_page = max(1, int(start_page))
        self.max_pages = max(1, int(pages)) if pages else None
        self.max_items = max(1, int(limit)) if limit else None
        self.page_size = max(1, min(int(page_size), 100))
        self.keyword = keyword.strip()
        self.agency_ids = [value.strip() for value in agency_ids.split(',') if value.strip()]
        self.requested_doc_ids = [value.strip() for value in doc_ids.split(',') if value.strip()]
        self.group_vbpl = str(group_vbpl).strip().lower() not in {"0", "false", "no"}
        self.excluded_doc_type_codes = {
            value.strip().upper()
            for value in exclude_doc_type_codes.split(',')
            if value.strip()
        }
        self.server_doc_type_ids = [
            doc_type_id
            for code, doc_type_id in OFFICIAL_DOC_TYPE_IDS.items()
            if code not in self.excluded_doc_type_codes
        ]
        self.skipped_doc_types = {}
        self.successful_ids = set()
        self.scheduled_ids = set()
        self.search_action = None
        self.action_tokens = []
        artifacts_dir = os.getenv('CRAWLER_ARTIFACTS_DIR', os.path.join(os.getcwd(), 'artifacts'))
        os.makedirs(artifacts_dir, exist_ok=True)
        self.failed_file = os.path.join(artifacts_dir, 'crawl_failures.jsonl')
        self.current_failed_items = {}
        self._ocr_semaphore = None

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    async def start(self):
        if self.requested_doc_ids:
            self.logger.info("[TARGETED] Tai truc tiep %d document ID.", len(self.requested_doc_ids))
            for doc_id in self.requested_doc_ids:
                self.scheduled_ids.add(doc_id)
                yield scrapy.Request(
                    url=f"{SEARCH_API}/{doc_id}",
                    method="GET",
                    headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
                    callback=self.parse_detail,
                    errback=self.handle_failure,
                    cb_kwargs={'item': {'item_id': doc_id, 'doc_number': doc_id, 'metadata_api': {}}},
                )
            return
        self.logger.info("[BAT DAU] Mo vbpl.vn bang Playwright...")
        yield scrapy.Request(
            url=HOME_URL,
            meta={
                "playwright": True,
                "playwright_context": "vbpl",
                "playwright_include_page": True,
                "playwright_page_event_handlers": {
                    "request": self.capture_playwright_request,
                },
            },
            callback=self.extract_tokens_and_search,
            dont_filter=True,
        )

    def capture_playwright_request(self, request):
        action_token = request.headers.get("next-action")
        if action_token and action_token not in self.action_tokens:
            self.action_tokens.append(action_token)
            self.logger.info("[PLAYWRIGHT] Bat next-action tu browser context")

    async def extract_tokens_and_search(self, response):
        page = response.meta.get("playwright_page")
        try:
            if response.status == 403:
                self.logger.error("[403] vbpl.vn tu choi browser request. Dung crawl va luu log de kiem tra.")
                return

            await page.wait_for_timeout(500)

            if not self.action_tokens:
                self.logger.warning("[PLAYWRIGHT] Khong thay token prefetch, thu tim element...")
                search_input = page.locator("input[type='text']")
                if await search_input.count() > 0:
                    await search_input.first.fill("giao thông")
                    await page.keyboard.press("Enter")
                    await page.wait_for_timeout(1000)

            if not self.action_tokens:
                self.logger.error("[PLAYWRIGHT] Khong bat duoc next-action tu network. Dung fallback.")
                self.action_tokens.append("c529d164f28418e5898a834422629e64c6816af1")

            self.search_action = self.action_tokens[0]

            # Tất cả trang search chạy trong cùng browser context để giữ cookie WAF.
            page_number = self.start_page
            total_pages = None
            while total_pages is None or page_number <= total_pages:
                self.logger.info("[SEARCH] Trang %d%s", page_number, f"/{total_pages}" if total_pages else "")
                result_text = await page.evaluate(
                    """async ({action, routerTree, body}) => {
                        const res = await fetch("https://vbpl.vn/", {
                            method: "POST",
                            headers: {
                                "accept": "text/x-component",
                                "content-type": "text/plain;charset=UTF-8",
                                "next-action": action,
                                "next-router-state-tree": routerTree
                            },
                            body
                        });
                        const text = await res.text();
                        if (!res.ok) throw new Error(`search HTTP ${res.status}: ${text.slice(0, 200)}`);
                        return text;
                    }""",
                    {
                        "action": self.search_action,
                        "routerTree": ROUTER_TREE,
                        "body": self.make_search_body(page_number),
                    },
                )
                data = self._decode_search_payload(result_text)
                documents = data.get('items', [])
                if not documents:
                    self.logger.warning("[SEARCH] Không còn văn bản ở trang %d.", page_number)
                    break

                for request in self._build_detail_requests(documents):
                    yield request

                total = int(data.get('total') or 0)
                total_pages = max(1, (total + self.page_size - 1) // self.page_size)
                if self.max_pages:
                    total_pages = min(total_pages, self.start_page + self.max_pages - 1)
                if self.max_items and len(self.scheduled_ids) >= self.max_items:
                    break
                page_number += 1

        finally:
            if page is not None:
                await page.close()

    def make_search_body(self, page_number, page_size=10):
        page_size = self.page_size
        item = {
            "pageNumber": page_number,
            "pageSize": page_size,
            "keyword": self.keyword or "$undefined",
            "groupVbpl": self.group_vbpl,
            "documentName": "$undefined",
            "docNum": "$undefined",
            "docType": self.server_doc_type_ids or "$undefined",
            "majorTypeIds": "$undefined",
            "fieldTypeIds": "$undefined",
            "agencyIds": self.agency_ids or "$undefined",
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

    @staticmethod
    def _decode_search_payload(text):
        json_part = text[text.index('{"total'):] if '{"total' in text else text
        return json.loads(json_part)

    @staticmethod
    def _decode_action_value(text):
        for line in text.splitlines():
            if line.startswith("1:"):
                return json.loads(line[2:])
        raise ValueError("Server action response không có payload 1:")

    @staticmethod
    def _prepare_html(html_raw):
        html_dom = BeautifulSoup(html_raw, 'html.parser') if html_raw else None
        if html_dom is not None:
            for embedded in html_dom.find_all(src=re.compile(r'^data:', re.I)):
                embedded.decompose()
            for embedded in html_dom.find_all(data=re.compile(r'^data:', re.I)):
                embedded.decompose()
        text_content = html_dom.get_text(" ", strip=True) if html_dom is not None else ""
        if len(text_content) < 100:
            return HTMLStatus.EMPTY, "", None
        return HTMLStatus.VALID, str(html_dom), html_dom

    @staticmethod
    def _extract_docx_text(body):
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        paragraphs = []
        for paragraph in root.iter():
            if not paragraph.tag.endswith("}p"):
                continue
            text = "".join(
                node.text or ""
                for node in paragraph.iter()
                if node.tag.endswith("}t")
            ).strip()
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)

    @staticmethod
    def _extract_pdf_text(body):
        import pymupdf

        document = pymupdf.open(stream=body, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in document)
        finally:
            document.close()

    async def _ocr_pdf_async(self, body):
        if self._ocr_semaphore is None:
            self._ocr_semaphore = asyncio.Semaphore(
                max(1, int(os.getenv("OCR_CONCURRENCY", "2")))
            )
        async with self._ocr_semaphore:
            return await asyncio.to_thread(self._ocr_pdf, body)

    @staticmethod
    def _ocr_pdf(body):
        executable = shutil.which("tesseract")
        if not executable:
            return "", "OCR_ENGINE_UNAVAILABLE"
        import pymupdf

        language = os.getenv("OCR_LANG", "vie+eng")
        dpi = max(100, int(os.getenv("OCR_DPI", "200")))
        max_pages = max(1, int(os.getenv("OCR_MAX_PAGES", "200")))
        timeout = max(10, int(os.getenv("OCR_PAGE_TIMEOUT_SECONDS", "60")))
        document = pymupdf.open(stream=body, filetype="pdf")
        texts = []
        try:
            for page_number, page in enumerate(document):
                if page_number >= max_pages:
                    return "\n".join(texts), "OCR_PAGE_LIMIT"
                scale = dpi / 72
                image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
                result = subprocess.run(
                    [executable, "stdin", "stdout", "-l", language, "--psm", "6"],
                    input=image,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                if result.returncode != 0:
                    error = result.stderr.decode("utf-8", errors="replace")[:300]
                    return "\n".join(texts), f"OCR_ERROR:{error}"
                texts.append(result.stdout.decode("utf-8", errors="replace"))
        finally:
            document.close()
        return "\n".join(texts), "OCR_COMPLETED"

    @staticmethod
    def _fallback_file_priority(file_info):
        name = str(file_info.get("fileName") or "").lower()
        if name.endswith("_content.html") and "origin" not in name:
            return 0
        if name.endswith(".html") and "origin" not in name:
            return 1
        if name.endswith(".docx"):
            return 2
        if name.endswith(".pdf"):
            return 3
        return 99

    def _build_detail_requests(self, documents):
        for doc in documents:
            if self.max_items and len(self.scheduled_ids) >= self.max_items:
                return
            doc_id = str(doc.get('id', '')).strip()
            doc_type_code = str((doc.get('docType') or {}).get('code') or '').strip().upper()
            if doc_type_code in self.excluded_doc_type_codes:
                self.skipped_doc_types[doc_type_code] = self.skipped_doc_types.get(doc_type_code, 0) + 1
                continue
            if not doc_id or doc_id in self.scheduled_ids:
                continue
            self.scheduled_ids.add(doc_id)
            item = {
                'item_id': doc_id,
                'doc_number': doc.get('docNum', 'Unknown'),
                'metadata_api': doc,
            }
            yield scrapy.Request(
                url=f"{SEARCH_API}/{doc_id}",
                method="GET",
                headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
                callback=self.parse_detail,
                errback=self.handle_failure,
                cb_kwargs={'item': item},
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
        if not item.get('metadata_api'):
            item['metadata_api'] = {
                key: doc_data.get(key)
                for key in (
                    'id', 'title', 'docNum', 'docType', 'issueDate', 'effFrom', 'effTo',
                    'effStatus', 'agencyIds', 'agencyName', 'isLw'
                )
            }
            item['doc_number'] = doc_data.get('docNum') or item.get('doc_number') or item['item_id']

        html_status, prepared_html, html_dom = self._prepare_html(html_raw)
        if html_status != HTMLStatus.VALID:
            self.logger.warning(f"[THIEU HTML] Van ban {item['item_id']} khong co HTML content")
            item['html_status'] = HTMLStatus.EMPTY
            item['html_raw'] = ""
        else:
            item['html_status'] = HTMLStatus.VALID
            item['html_raw'] = prepared_html
            item['content_source'] = "detail_html"

        # `html_raw` is the canonical content copy. Avoid duplicating it inside
        # metadata_detail, which otherwise doubles every Kafka message.
        metadata_detail = dict(doc_data)
        if isinstance(doc_content, dict):
            metadata_detail['documentContent'] = {
                key: value for key, value in doc_content.items() if key != 'content'
            }
        item['metadata_detail'] = metadata_detail

        if item['html_status'] != HTMLStatus.VALID:
            if str(item['item_id']).isdigit() and os.getenv("LEGACY_FALLBACK_ENABLED", "0") == "1":
                yield self._legacy_print_request(item)
            else:
                yield self._rescue_browser_request(item)
            return

        yield self._diagram_request(item, html_dom)

    @staticmethod
    def _playwright_rescue_meta():
        timeout_ms = int(os.getenv("RESCUE_NAVIGATION_TIMEOUT_MS", "45000"))
        return {
            "playwright": True,
            "playwright_context": "vbpl",
            "playwright_include_page": True,
            "playwright_page_goto_kwargs": {
                "wait_until": "domcontentloaded",
                "timeout": timeout_ms,
            },
            "playwright_page_init_callback": LawSpider._init_rescue_page,
            "max_retry_times": 0,
            "download_timeout": max(50, timeout_ms // 1000 + 5),
        }

    @staticmethod
    async def _init_rescue_page(page, request):
        blocked_types = {"image", "font", "media", "stylesheet"}
        if os.getenv("RESCUE_BLOCK_SCRIPTS", "1") != "0":
            blocked_types.add("script")

        async def route_resource(route):
            if route.request.resource_type in blocked_types:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", route_resource)

    def _legacy_print_request(self, item):
        return scrapy.Request(
            url=f"https://vbpl.vn/tw/Pages/vbpq-print.aspx?ItemID={item['item_id']}",
            meta=self._playwright_rescue_meta(),
            callback=self.parse_legacy_print,
            errback=self.handle_legacy_print_failure,
            cb_kwargs={'item': item},
            dont_filter=True,
        )

    async def parse_legacy_print(self, response, item):
        page = response.meta.get("playwright_page")
        try:
            content_html = response.css("#content").get() or ""
            status, prepared_html, html_dom = self._prepare_html(content_html)
            if status == HTMLStatus.VALID:
                item['html_status'] = status
                item['html_raw'] = prepared_html
                item['content_source'] = "legacy_print_html"
                item['rescue_status'] = "LEGACY_HTML_RECOVERED"
                self.logger.info("[RESCUE LEGACY HTML] %s", item['item_id'])
                yield self._diagram_request(item, html_dom)
                return
            self.logger.warning("[RESCUE LEGACY EMPTY] %s", item['item_id'])
            yield self._rescue_browser_request(item)
        finally:
            if page is not None:
                await page.close()

    async def handle_legacy_print_failure(self, failure):
        item = failure.request.cb_kwargs['item']
        page = failure.request.meta.get("playwright_page")
        if page is not None:
            try:
                await page.close()
            except Exception as close_exc:
                self.logger.debug("Khong dong duoc legacy page: %s", close_exc)
        self.logger.warning("[RESCUE LEGACY FAILED] %s: %s", item['item_id'], failure.value)
        return self._rescue_browser_request(item)

    def _rescue_browser_request(self, item, attempt=1):
        return scrapy.Request(
            url=HOME_URL,
            # Chỉ cần DOM/cookie để gọi server action bằng fetch; không chờ
            # toàn bộ ảnh/font/script phụ tải xong. RetryMiddleware không biết
            # đóng Playwright page nên retry được quản lý ở errback.
            meta=self._playwright_rescue_meta(),
            callback=self.fetch_file_list_in_browser,
            errback=self.handle_file_list_failure,
            cb_kwargs={'item': item, 'rescue_attempt': attempt},
            dont_filter=True,
        )

    def _diagram_request(self, item, html_dom=None):
        doc_id = item['item_id']
        return scrapy.Request(
            url=f"{SEARCH_API}/{doc_id}/diagram",
            method="GET",
            headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
            callback=self.parse_diagram,
            errback=self.handle_failure,
            cb_kwargs={'item': item},
            meta={'html_dom': html_dom},
        )

    def _next_file_request(self, item, files):
        if not files:
            item['rescue_status'] = item.get('rescue_status') or "FILE_NOT_FOUND"
            return self._diagram_request(item)
        current, *remaining = files
        item['rescue_file'] = {
            "fileName": current.get("fileName"),
            "size": current.get("size"),
            "relatedType": current.get("relatedType"),
        }
        # VBPL dùng đúng file trắng này cho record chưa công bố nội dung. Không
        # tải/render/OCR lặp lại một placeholder đã biết; thử candidate kế tiếp
        # nếu có, nếu không thì quarantine có giải thích.
        if (
            str(current.get("fileName") or "").lower() == "template.pdf"
            and int(current.get("size") or 0) == 32052
        ):
            item['rescue_status'] = "UPSTREAM_TEMPLATE"
            item['upstream_content_unavailable'] = True
            self.logger.warning("[RESCUE PLACEHOLDER] %s Template.pdf", item['item_id'])
            return self._next_file_request(item, remaining) if remaining else self._diagram_request(item)
        return scrapy.Request(
            url=current['presignedUrl'],
            method="GET",
            callback=self.parse_fallback_file,
            errback=self.handle_fallback_file_failure,
            cb_kwargs={'item': item, 'file_info': current, 'remaining_files': remaining},
            dont_filter=True,
        )

    def parse_file_list(self, response, item):
        try:
            files = self._decode_action_value(response.text)
            candidates = sorted(
                [
                    file_info for file_info in files
                    if file_info.get("presignedUrl") and self._fallback_file_priority(file_info) < 99
                ],
                key=self._fallback_file_priority,
            )
        except Exception as exc:
            self.logger.warning("[RESCUE FILE LIST] %s: %s", item['item_id'], exc)
            item['rescue_status'] = "FILE_LIST_ERROR"
            yield self._diagram_request(item)
            return
        item['rescue_files_found'] = len(candidates)
        yield self._next_file_request(item, candidates)

    async def fetch_file_list_in_browser(self, response, item, rescue_attempt=1):
        page = response.meta.get("playwright_page")
        try:
            result_text = await page.evaluate(
                """async ({action, routerTree, body}) => {
                    const res = await fetch("https://vbpl.vn/", {
                        method: "POST",
                        headers: {
                            "accept": "text/x-component",
                            "content-type": "text/plain;charset=UTF-8",
                            "next-action": action,
                            "next-router-state-tree": routerTree
                        },
                        body
                    });
                    const text = await res.text();
                    if (!res.ok) throw new Error(`files HTTP ${res.status}: ${text.slice(0, 200)}`);
                    return text;
                }""",
                {
                    "action": FILES_ACTION,
                    "routerTree": ROUTER_TREE,
                    "body": json.dumps([item['item_id'], None]),
                },
            )
            files = self._decode_action_value(result_text)
            candidates = sorted(
                [
                    file_info for file_info in files
                    if file_info.get("presignedUrl") and self._fallback_file_priority(file_info) < 99
                ],
                key=self._fallback_file_priority,
            )
            item['rescue_files_found'] = len(candidates)
            yield self._next_file_request(item, candidates)
        except Exception as exc:
            item['rescue_status'] = "FILE_LIST_ERROR"
            self.logger.warning("[RESCUE FILE LIST BROWSER] %s: %s", item['item_id'], exc)
            yield self._diagram_request(item)
        finally:
            if page is not None:
                await page.close()

    async def parse_fallback_file(self, response, item, file_info, remaining_files):
        name = str(file_info.get("fileName") or "").lower()
        extracted_text = ""
        try:
            if name.endswith(".html"):
                raw_html = response.body.decode("utf-8", errors="replace")
                status, prepared_html, html_dom = self._prepare_html(raw_html)
                if status == HTMLStatus.VALID:
                    item['html_status'] = status
                    item['html_raw'] = prepared_html
                    item['content_source'] = "fallback_html"
                    item['rescue_status'] = "HTML_RECOVERED"
                    self.logger.info("[RESCUE HTML] %s tu %s", item['item_id'], file_info.get('fileName'))
                    yield self._diagram_request(item, html_dom)
                    return
            elif name.endswith(".docx"):
                extracted_text = self._extract_docx_text(response.body)
                source = "fallback_docx"
                status_name = "DOCX_TEXT_RECOVERED"
            elif name.endswith(".pdf"):
                extracted_text = await asyncio.to_thread(self._extract_pdf_text, response.body)
                source = "fallback_pdf"
                status_name = "PDF_TEXT_RECOVERED"
                if len(extracted_text.strip()) < 100 and os.getenv("OCR_ENABLED", "1") != "0":
                    if os.getenv("OCR_INLINE_ENABLED", "1") == "0":
                        item['ocr_status'] = "OCR_PENDING"
                        item['rescue_status'] = "OCR_PENDING"
                    else:
                        extracted_text, ocr_status = await self._ocr_pdf_async(response.body)
                        if ocr_status == "OCR_COMPLETED" and len(extracted_text.strip()) < 100:
                            ocr_status = "OCR_EMPTY"
                        item['ocr_status'] = ocr_status
                        if len(extracted_text.strip()) >= 100:
                            source = "fallback_pdf_ocr"
                            status_name = "PDF_OCR_RECOVERED"
                        else:
                            item['rescue_status'] = ocr_status

            if len(extracted_text.strip()) >= 100:
                recovered_html = "<html><body><pre>" + html_lib.escape(extracted_text) + "</pre></body></html>"
                item['html_status'] = HTMLStatus.VALID
                item['html_raw'] = recovered_html
                item['content_source'] = source
                item['rescue_status'] = status_name
                self.logger.info("[RESCUE TEXT] %s tu %s", item['item_id'], file_info.get('fileName'))
                yield self._diagram_request(item, BeautifulSoup(recovered_html, 'html.parser'))
                return
        except Exception as exc:
            self.logger.warning("[RESCUE FILE ERROR] %s / %s: %s", item['item_id'], file_info.get('fileName'), exc)

        if name.endswith(".pdf") and not remaining_files:
            item['rescue_status'] = item.get('rescue_status') or "OCR_REQUIRED"
        yield self._next_file_request(item, remaining_files)

    async def handle_file_list_failure(self, failure):
        item = failure.request.cb_kwargs['item']
        attempt = int(failure.request.cb_kwargs.get('rescue_attempt', 1))
        page = failure.request.meta.get("playwright_page")
        if page is not None:
            try:
                await page.close()
            except Exception as close_exc:
                self.logger.debug("Khong dong duoc rescue page: %s", close_exc)
        max_attempts = max(1, int(os.getenv("RESCUE_BROWSER_ATTEMPTS", "3")))
        if attempt < max_attempts:
            self.logger.warning(
                "[RESCUE FILE LIST RETRY] %s lan %d/%d: %s",
                item['item_id'], attempt, max_attempts, failure.value,
            )
            return self._rescue_browser_request(item, attempt + 1)
        item['rescue_status'] = "FILE_LIST_REQUEST_FAILED"
        self.logger.warning(
            "[RESCUE FILE LIST FAILED] %s sau %d lan: %s",
            item['item_id'], attempt, failure.value,
        )
        return self._diagram_request(item)

    def handle_fallback_file_failure(self, failure):
        item = failure.request.cb_kwargs['item']
        remaining_files = failure.request.cb_kwargs['remaining_files']
        self.logger.warning("[RESCUE DOWNLOAD FAILED] %s: %s", item['item_id'], failure.value)
        return self._next_file_request(item, remaining_files)

    def parse_diagram(self, response, item):
        try:
            diagram_data = json.loads(response.text)
            item['diagram_json'] = diagram_data.get('data', {})
        except json.JSONDecodeError:
            item['diagram_json'] = None

        doc_id = item['item_id']
        self.successful_ids.add(doc_id)
        self.logger.info(f"[HOAN THANH] {item.get('doc_number')} (ID: {doc_id})")

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
        self.logger.info(
            "Da xu ly xong %d/%d items; that bai=%d.",
            len(self.successful_ids),
            len(self.scheduled_ids),
            len(self.current_failed_items),
        )
        if self.skipped_doc_types:
            self.logger.info("Da loai theo docType: %s", self.skipped_doc_types)
