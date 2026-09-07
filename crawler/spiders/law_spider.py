"""
law_spider.py - Con nhện thu thập toàn diện văn bản pháp luật Việt Nam (VBPL).
Tích hợp Next.js Server Action, Playwright Context, Page Refresh định kỳ và Disk Shards Staging.
"""

from __future__ import annotations

import io
import os
import re
import json
import gzip
import shutil
import asyncio
import zipfile
import subprocess
import html as html_lib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

import scrapy
from scrapy import signals
from bs4 import BeautifulSoup

from configs.paths import RAW_SHARDS_DIR, ARTIFACTS_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("crawler", "crawler")

try:
    from crawler.items import HTMLStatus
except ImportError:
    class HTMLStatus:
        VALID = "VALID"
        EMPTY = "EMPTY"
        ERROR = "ERROR"

SEARCH_API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
HOME_URL = "https://vbpl.vn/"
ROUTER_TREE = '%5B%22%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%2Ctrue%5D'
FILES_ACTION = os.getenv("VBPL_FILES_ACTION", "7b29f485c8e43b71a94bfc11b54459d7e27293e6")

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


class LawSpider(scrapy.Spider):
    name = "law_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn", "fptcloud.com"]
    handle_httpstatus_list = [403]

    def __init__(
        self,
        start_page: int = 1,
        pages: Optional[int] = None,
        limit: Optional[int] = None,
        page_size: int = 10,
        keyword: str = "",
        agency_ids: str = "",
        doc_ids: str = "",
        group_vbpl: str = "true",
        exclude_doc_type_codes: str = "BD",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_page = max(1, int(start_page))
        self.max_pages = max(1, int(pages)) if pages else None
        self.max_items = max(1, int(limit)) if limit else None
        self.page_size = max(1, min(int(page_size), 100))
        self.keyword = keyword.strip()
        self.agency_ids = [v.strip() for v in agency_ids.split(",") if v.strip()]
        self.requested_doc_ids = [v.strip() for v in doc_ids.split(",") if v.strip()]
        self.group_vbpl = str(group_vbpl).strip().lower() not in {"0", "false", "no"}

        self.excluded_doc_type_codes = {
            v.strip().upper() for v in exclude_doc_type_codes.split(",") if v.strip()
        }
        self.server_doc_type_ids = [
            doc_type_id
            for code, doc_type_id in OFFICIAL_DOC_TYPE_IDS.items()
            if code not in self.excluded_doc_type_codes
        ]

        self.skipped_doc_types: Dict[str, int] = {}
        self.successful_ids: Set[str] = set()
        self.scheduled_ids: Set[str] = set()
        self.current_failed_ids: Set[str] = set()

        self.search_action: Optional[str] = os.getenv("VBPL_SEARCH_ACTION")
        self.action_tokens: List[str] = []

        self.artifacts_dir = Path(ARTIFACTS_DIR)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.failed_file = self.artifacts_dir / "crawl_failures.jsonl"

        # Khởi tạo phân vùng Staging Shards (1.000 văn bản / shard)
        self.raw_shards_dir = Path(RAW_SHARDS_DIR)
        self.raw_shards_dir.mkdir(parents=True, exist_ok=True)
        self.current_shard_records: List[Dict[str, Any]] = []
        self.shard_counter = 1

        self._ocr_semaphore: Optional[asyncio.Semaphore] = None

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def _flush_shard_to_disk(self):
        """Ghi đệm Shard hiện tại xuống tệp nén Gzip (.jsonl.gz) phục vụ Pipeline Staging."""
        if not self.current_shard_records:
            return

        shard_filename = self.raw_shards_dir / f"crawl_pages_{self.shard_counter:05d}.jsonl.gz"
        try:
            with gzip.open(shard_filename, "wt", encoding="utf-8") as f:
                for record in self.current_shard_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.logger.info(
                "[STAGING SHARD] Đã ghi thành công %d bản ghi vào %s",
                len(self.current_shard_records),
                shard_filename.name,
            )
            self.current_shard_records.clear()
            self.shard_counter += 1
        except Exception as exc:
            self.logger.error("Lỗi ghi Shard xuống đĩa: %s", exc, exc_info=True)

    def _get_in_flight_count(self) -> int:
        if hasattr(self, "crawler") and self.crawler.stats:
            enqueued = self.crawler.stats.get_value("scheduler/enqueued", 0) or 0
            dequeued = self.crawler.stats.get_value("scheduler/dequeued", 0) or 0
            diff = enqueued - dequeued
            if diff > 0:
                return diff

        try:
            engine = getattr(self.crawler, "engine", None)
            if engine and engine.slot:
                sched = getattr(engine.slot, "scheduler", None)
                down = getattr(engine.slot, "downloader", None)
                s_len = len(sched) if sched is not None else 0
                d_len = len(down.active) if down and hasattr(down, "active") else 0
                return s_len + d_len
        except Exception:
            pass

        return 0

    async def start(self):
        if self.requested_doc_ids:
            self.logger.info("[MỤC TIÊU] Tải trực tiếp %d Document ID chỉ định.", len(self.requested_doc_ids))
            for doc_id in self.requested_doc_ids:
                self.scheduled_ids.add(doc_id)
                yield scrapy.Request(
                    url=f"{SEARCH_API}/{doc_id}",
                    method="GET",
                    headers={"Origin": "https://vbpl.vn", "Referer": "https://vbpl.vn/", "Accept": "application/json"},
                    callback=self.parse_detail,
                    errback=self.handle_failure,
                    cb_kwargs={"item": {"item_id": doc_id, "doc_number": doc_id, "metadata_api": {}}},
                )
            return

        self.logger.info("[KHỞI TẠO] Mở phiên trình duyệt Playwright kết nối vbpl.vn...")
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
            self.logger.info("[PLAYWRIGHT] Thu nhận next-action token từ network headers.")

    async def extract_tokens_and_search(self, response):
        page = response.meta.get("playwright_page")
        try:
            if response.status == 403:
                self.logger.error("[CHẶN 403] vbpl.vn từ chối yêu cầu. Kiểm tra WAF hoặc Proxy.")
                return

            await page.wait_for_timeout(500)

            if not self.action_tokens and not self.search_action:
                self.logger.warning("[PLAYWRIGHT] Chưa thấy Action Token, kích hoạt sự kiện DOM...")
                search_input = page.locator("input[type='text']")
                if await search_input.count() > 0:
                    await search_input.first.fill("luật")
                    await page.keyboard.press("Enter")
                    await page.wait_for_timeout(1000)

            if not self.search_action:
                if self.action_tokens:
                    self.search_action = self.action_tokens[0]
                else:
                    self.logger.warning("[PLAYWRIGHT] Dùng fallback action token định sẵn.")
                    self.search_action = "c529d164f28418e5898a834422629e64c6816af1"

            page_number = self.start_page
            total_pages = None

            while total_pages is None or page_number <= total_pages:
                # =========================================================================
                # 1. BẢO VỆ CHỐNG RÒ RỈ BỘ NHỚ CHROMIUM (Periodic Page Refresh mỗi 500 trang)
                # =========================================================================
                if page_number > self.start_page and (page_number - self.start_page) % 500 == 0:
                    self.logger.info("[PLAYWRIGHT REFRESH] Tái tạo Page Chromium để xả V8 Heap sau 500 trang...")
                    browser_context = page.context
                    await page.close()
                    page = await browser_context.new_page()
                    await self._init_rescue_page(page, None)

                # =========================================================================
                # 2. CƠ CHẾ BACKPRESSURE CHỦ ĐỘNG (Giữ hàng đợi engine < 120 requests)
                # =========================================================================
                while self._get_in_flight_count() >= 120:
                    await asyncio.sleep(1.5)

                self.logger.info("[TÌM KIẾM] Quét trang %d%s", page_number, f"/{total_pages}" if total_pages else "")
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
                documents = data.get("items", [])
                if not documents:
                    self.logger.warning("[TÌM KIẾM] Không còn văn bản ở trang %d. Kết thúc tìm kiếm.", page_number)
                    break

                for request in self._build_detail_requests(documents):
                    yield request

                total = int(data.get("total") or 0)
                total_pages = max(1, (total + self.page_size - 1) // self.page_size)
                if self.max_pages:
                    total_pages = min(total_pages, self.start_page + self.max_pages - 1)
                if self.max_items and len(self.scheduled_ids) >= self.max_items:
                    break
                page_number += 1

        finally:
            if page is not None:
                await page.close()

    def make_search_body(self, page_number: int) -> str:
        item = {
            "pageNumber": page_number,
            "pageSize": self.page_size,
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
            "effFromEnd": "$undefined",
        }
        return json.dumps([item], ensure_ascii=False)

    @staticmethod
    def _decode_search_payload(text: str) -> dict:
        json_part = text[text.index('{"total'):] if '{"total' in text else text
        return json.loads(json_part)

    @staticmethod
    def _decode_action_value(text: str) -> Any:
        for line in text.splitlines():
            if line.startswith("1:"):
                return json.loads(line[2:])
        raise ValueError("Server action response không có payload định dạng 1:")

    @staticmethod
    def _prepare_html(html_raw: Optional[str]) -> Tuple[str, str, Optional[BeautifulSoup]]:
        if not html_raw:
            return HTMLStatus.EMPTY, "", None

        html_dom = BeautifulSoup(html_raw, "html.parser")
        for embedded in html_dom.find_all(src=re.compile(r"^data:", re.I)):
            embedded.decompose()
        for embedded in html_dom.find_all(data=re.compile(r"^data:", re.I)):
            embedded.decompose()

        text_content = html_dom.get_text(" ", strip=True)
        if len(text_content) < 100:
            return HTMLStatus.EMPTY, "", None
        return HTMLStatus.VALID, str(html_dom), html_dom

    @staticmethod
    def _extract_docx_text(body: bytes) -> str:
        if not body or len(body) < 10:
            return ""
        try:
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
        except Exception:
            return ""

    @staticmethod
    def _extract_pdf_text(body: bytes) -> str:
        if not body or len(body) < 10:
            return ""
        try:
            import pymupdf
            document = pymupdf.open(stream=body, filetype="pdf")
            try:
                return "\n".join(page.get_text() for page in document)
            finally:
                document.close()
        except Exception:
            return ""

    async def _ocr_pdf_async(self, body: bytes) -> Tuple[str, str]:
        if self._ocr_semaphore is None:
            max_concurrency = max(1, int(getattr(config, "OCR_CONCURRENCY", 2)))
            self._ocr_semaphore = asyncio.Semaphore(max_concurrency)
        async with self._ocr_semaphore:
            return await asyncio.to_thread(self._ocr_pdf, body)

    @staticmethod
    def _ocr_pdf(body: bytes) -> Tuple[str, str]:
        if not body or len(body) < 10:
            return "", "OCR_EMPTY_BODY"

        executable = shutil.which("tesseract")
        if not executable:
            for fallback_path in ("/usr/bin/tesseract", "/usr/local/bin/tesseract", "/bin/tesseract"):
                if os.path.isfile(fallback_path) and os.access(fallback_path, os.X_OK):
                    executable = fallback_path
                    break

        if not executable:
            return "", "OCR_ENGINE_UNAVAILABLE"

        try:
            import pymupdf
            language = getattr(config, "OCR_LANG", "vie+eng")
            dpi = max(100, int(getattr(config, "OCR_DPI", 200)))
            max_pages = max(1, int(getattr(config, "OCR_MAX_PAGES", 200)))
            timeout = max(10, int(os.getenv("OCR_PAGE_TIMEOUT_SECONDS", "60")))

            document = pymupdf.open(stream=body, filetype="pdf")
            texts = []
            try:
                for page_number, page in enumerate(document):
                    if page_number >= max_pages:
                        return "\n".join(texts), "OCR_PAGE_LIMIT"
                    scale = dpi / 72
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    image = pixmap.tobytes("png")
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
        except Exception as exc:
            return "", f"OCR_EXCEPTION:{exc}"

    @staticmethod
    def _fallback_file_priority(file_info: dict) -> int:
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

    def _build_detail_requests(self, documents: List[dict]):
        for doc in documents:
            if self.max_items and len(self.scheduled_ids) >= self.max_items:
                return
            doc_id = str(doc.get("id", "")).strip()
            doc_type_code = str((doc.get("docType") or {}).get("code") or "").strip().upper()
            if doc_type_code in self.excluded_doc_type_codes:
                self.skipped_doc_types[doc_type_code] = self.skipped_doc_types.get(doc_type_code, 0) + 1
                continue
            if not doc_id or doc_id in self.scheduled_ids:
                continue

            self.scheduled_ids.add(doc_id)
            item = {
                "item_id": doc_id,
                "doc_number": doc.get("docNum", "Unknown"),
                "metadata_api": doc,
            }
            yield scrapy.Request(
                url=f"{SEARCH_API}/{doc_id}",
                method="GET",
                headers={"Origin": "https://vbpl.vn", "Referer": "https://vbpl.vn/", "Accept": "application/json"},
                callback=self.parse_detail,
                errback=self.handle_failure,
                cb_kwargs={"item": item},
            )

    def parse_detail(self, response, item: dict):
        try:
            data = json.loads(response.text)
            doc_data = data.get("data", {})
        except Exception:
            self.logger.error("[LỖI] Không parse được JSON cho Document %s", item.get("item_id"))
            self.handle_failure_internal(item)
            return

        doc_content = doc_data.get("documentContent")
        html_raw = (doc_content or {}).get("content", "")
        if not item.get("metadata_api"):
            item["metadata_api"] = {
                key: doc_data.get(key)
                for key in (
                    "id", "title", "docNum", "docType", "issueDate", "effFrom", "effTo",
                    "effStatus", "agencyIds", "agencyName", "isLw"
                )
            }
            item["doc_number"] = doc_data.get("docNum") or item.get("doc_number") or item["item_id"]

        html_status, prepared_html, html_dom = self._prepare_html(html_raw)
        if html_status != HTMLStatus.VALID:
            self.logger.warning("[THIẾU HTML] Văn bản %s không có nội dung HTML sẵn.", item["item_id"])
            item["html_status"] = HTMLStatus.EMPTY
            item["html_raw"] = ""
        else:
            item["html_status"] = HTMLStatus.VALID
            item["html_raw"] = prepared_html
            item["content_source"] = "detail_html"

        metadata_detail = dict(doc_data)
        if isinstance(doc_content, dict):
            metadata_detail["documentContent"] = {
                key: value for key, value in doc_content.items() if key != "content"
            }
        item["metadata_detail"] = metadata_detail

        if item["html_status"] != HTMLStatus.VALID:
            if str(item["item_id"]).isdigit() and os.getenv("LEGACY_FALLBACK_ENABLED", "0") == "1":
                yield self._legacy_print_request(item)
            else:
                yield self._rescue_browser_request(item)
            return

        yield self._diagram_request(item, html_dom)

    @staticmethod
    def _playwright_rescue_meta() -> dict:
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

    def _legacy_print_request(self, item: dict):
        return scrapy.Request(
            url=f"https://vbpl.vn/tw/Pages/vbpq-print.aspx?ItemID={item['item_id']}",
            meta=self._playwright_rescue_meta(),
            callback=self.parse_legacy_print,
            errback=self.handle_legacy_print_failure,
            cb_kwargs={"item": item},
            dont_filter=True,
        )

    async def parse_legacy_print(self, response, item: dict):
        page = response.meta.get("playwright_page")
        try:
            content_html = response.css("#content").get() or ""
            status, prepared_html, html_dom = self._prepare_html(content_html)
            if status == HTMLStatus.VALID:
                item["html_status"] = status
                item["html_raw"] = prepared_html
                item["content_source"] = "legacy_print_html"
                item["rescue_status"] = "LEGACY_HTML_RECOVERED"
                self.logger.info("[CỨU HỘ LEGACY] %s thành công.", item["item_id"])
                yield self._diagram_request(item, html_dom)
                return
            yield self._rescue_browser_request(item)
        finally:
            if page is not None:
                await page.close()

    async def handle_legacy_print_failure(self, failure):
        item = failure.request.cb_kwargs["item"]
        page = failure.request.meta.get("playwright_page")
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass
        return self._rescue_browser_request(item)

    def _rescue_browser_request(self, item: dict, attempt: int = 1):
        return scrapy.Request(
            url=HOME_URL,
            meta=self._playwright_rescue_meta(),
            callback=self.fetch_file_list_in_browser,
            errback=self.handle_file_list_failure,
            cb_kwargs={"item": item, "rescue_attempt": attempt},
            dont_filter=True,
        )

    def _diagram_request(self, item: dict, html_dom=None):
        doc_id = item["item_id"]
        return scrapy.Request(
            url=f"{SEARCH_API}/{doc_id}/diagram",
            method="GET",
            headers={"Origin": "https://vbpl.vn", "Referer": "https://vbpl.vn/", "Accept": "application/json"},
            callback=self.parse_diagram,
            errback=self.handle_failure,
            cb_kwargs={"item": item},
            meta={"html_dom": html_dom},
        )

    def _next_file_request(self, item: dict, files: list):
        if not files:
            item["rescue_status"] = item.get("rescue_status") or "FILE_NOT_FOUND"
            return self._diagram_request(item)

        current, *remaining = files
        item["rescue_file"] = {
            "fileName": current.get("fileName"),
            "size": current.get("size"),
            "relatedType": current.get("relatedType"),
        }

        file_name = str(current.get("fileName") or "").lower()
        if "template.pdf" in file_name:
            item["rescue_status"] = "UPSTREAM_TEMPLATE"
            item["upstream_content_unavailable"] = True
            return self._next_file_request(item, remaining) if remaining else self._diagram_request(item)

        return scrapy.Request(
            url=current["presignedUrl"],
            method="GET",
            callback=self.parse_fallback_file,
            errback=self.handle_fallback_file_failure,
            cb_kwargs={"item": item, "file_info": current, "remaining_files": remaining},
            dont_filter=True,
        )

    async def fetch_file_list_in_browser(self, response, item: dict, rescue_attempt: int = 1):
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
                    "body": json.dumps([item["item_id"], None]),
                },
            )
            files = self._decode_action_value(result_text)
            candidates = sorted(
                [
                    f for f in files
                    if f.get("presignedUrl") and self._fallback_file_priority(f) < 99
                ],
                key=self._fallback_file_priority,
            )
            item["rescue_files_found"] = len(candidates)
            yield self._next_file_request(item, candidates)
        except Exception as exc:
            item["rescue_status"] = "FILE_LIST_ERROR"
            self.logger.warning("[CỨU HỘ FILE LIST BROWSER] %s thất bại: %s", item["item_id"], exc)
            yield self._diagram_request(item)
        finally:
            if page is not None:
                await page.close()

    async def parse_fallback_file(self, response, item: dict, file_info: dict, remaining_files: list):
        name = str(file_info.get("fileName") or "").lower()
        extracted_text = ""
        try:
            if name.endswith(".html"):
                raw_html = response.body.decode("utf-8", errors="replace")
                status, prepared_html, html_dom = self._prepare_html(raw_html)
                if status == HTMLStatus.VALID:
                    item["html_status"] = status
                    item["html_raw"] = prepared_html
                    item["content_source"] = "fallback_html"
                    item["rescue_status"] = "HTML_RECOVERED"
                    self.logger.info("[CỨU HỘ HTML THÀNH CÔNG] %s từ %s", item["item_id"], file_info.get("fileName"))
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

                ocr_enabled = getattr(config, "OCR_ENABLED", True)
                inline_ocr = getattr(config, "OCR_INLINE_ENABLED", True)

                if len(extracted_text.strip()) < 100 and ocr_enabled:
                    if not inline_ocr:
                        item["ocr_status"] = "OCR_PENDING"
                        item["rescue_status"] = "OCR_PENDING"
                    else:
                        extracted_text, ocr_status = await self._ocr_pdf_async(response.body)
                        if ocr_status == "OCR_COMPLETED" and len(extracted_text.strip()) < 100:
                            ocr_status = "OCR_EMPTY"

                        if ocr_status == "OCR_ENGINE_UNAVAILABLE":
                            item["ocr_status"] = "OCR_PENDING"
                            item["rescue_status"] = "OCR_PENDING"
                        else:
                            item["ocr_status"] = ocr_status
                            if len(extracted_text.strip()) >= 100:
                                source = "fallback_pdf_ocr"
                                status_name = "PDF_OCR_RECOVERED"
                            else:
                                item["rescue_status"] = ocr_status

            if len(extracted_text.strip()) >= 100:
                recovered_html = "<html><body><pre>" + html_lib.escape(extracted_text) + "</pre></body></html>"
                item["html_status"] = HTMLStatus.VALID
                item["html_raw"] = recovered_html
                item["content_source"] = source
                item["rescue_status"] = status_name
                self.logger.info("[CỨU HỘ VĂN BẢN THÀNH CÔNG] %s từ %s", item["item_id"], file_info.get("fileName"))
                yield self._diagram_request(item, BeautifulSoup(recovered_html, "html.parser"))
                return
        except Exception as exc:
            self.logger.warning("[LỖI XỬ LÝ FILE ĐÍNH KÈM] %s: %s", item["item_id"], exc)

        if name.endswith(".pdf") and not remaining_files:
            item["rescue_status"] = item.get("rescue_status") or "OCR_REQUIRED"
        yield self._next_file_request(item, remaining_files)

    async def handle_file_list_failure(self, failure):
        item = failure.request.cb_kwargs["item"]
        attempt = int(failure.request.cb_kwargs.get("rescue_attempt", 1))
        page = failure.request.meta.get("playwright_page")
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass

        max_attempts = max(1, int(os.getenv("RESCUE_BROWSER_ATTEMPTS", "3")))
        if attempt < max_attempts:
            return self._rescue_browser_request(item, attempt + 1)

        item["rescue_status"] = "FILE_LIST_REQUEST_FAILED"
        return self._diagram_request(item)

    def handle_fallback_file_failure(self, failure):
        item = failure.request.cb_kwargs["item"]
        remaining_files = failure.request.cb_kwargs["remaining_files"]
        return self._next_file_request(item, remaining_files)

    def parse_diagram(self, response, item: dict):
        try:
            diagram_data = json.loads(response.text)
            item["diagram_json"] = diagram_data.get("data", {})
        except json.JSONDecodeError:
            item["diagram_json"] = None

        doc_id = item["item_id"]
        self.successful_ids.add(doc_id)

        # =========================================================================
        # 3. DUAL-WRITE STAGING SHARDS: Ghi song song bản ghi chuẩn vào Shard đĩa cứng
        # =========================================================================
        self.current_shard_records.append(item)
        if len(self.current_shard_records) >= 1000:
            self._flush_shard_to_disk()

        self.logger.info("[HOÀN TẤT VĂN BẢN] %s (ID: %s)", item.get("doc_number"), doc_id)
        yield item

    def handle_failure(self, failure):
        item = failure.request.cb_kwargs.get("item")
        if item and "item_id" in item:
            self.handle_failure_internal(item)

    def handle_failure_internal(self, item: dict):
        doc_id = str(item["item_id"])
        self.current_failed_ids.add(doc_id)
        self.logger.error("[THẤT BẠI] Document ID: %s", doc_id)
        with open(self.failed_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def spider_closed(self, spider):
        # Xả nốt phần Shard còn dở dang cuối cùng xuống đĩa
        self._flush_shard_to_disk()

        finish_reason = spider.crawler.stats.get_value("finish_reason", "unknown")
        stats_data = {
            "finish_reason": finish_reason,
            "successful_count": len(self.successful_ids),
            "scheduled_count": len(self.scheduled_ids),
            "failed_count": len(self.current_failed_ids),
            "pages_crawled": spider.crawler.stats.get_value("response_received_count", 0),
            "shards_exported": self.shard_counter - 1,
        }

        status_path = self.artifacts_dir / "crawler_status.json"
        try:
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        self.logger.info(
            "Chiến dịch kết thúc: Lý do=%s | Đã thu thập: %d/%d văn bản | Xuất: %d shards (.jsonl.gz) | Thất bại: %d.",
            finish_reason,
            len(self.successful_ids),
            len(self.scheduled_ids),
            self.shard_counter - 1,
            len(self.current_failed_ids),
        )
        if self.skipped_doc_types:
            self.logger.info("Thống kê phân loại đã lọc bỏ: %s", self.skipped_doc_types)
