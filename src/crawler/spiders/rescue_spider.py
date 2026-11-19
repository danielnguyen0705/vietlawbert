"""
rescue_spider.py - Con nhện chuyên trách cứu hộ và tái xử lý văn bản lỗi/quarantine.
Kế thừa toàn bộ động cơ phân tích của LawSpider, tích hợp cơ chế cập nhật thất bại nguyên tử.
"""

from __future__ import annotations

import os
import json
import scrapy
from pathlib import Path
from typing import Set, Dict, Any, List

from configs.paths import ARTIFACTS_DIR, JSON_DIR
from .law_spider import LawSpider, SEARCH_API


class RescueSpider(LawSpider):
    name = "rescue_spider"
    allowed_domains = ["vbpl.vn", "moj.gov.vn", "vbpl-bientap-gateway.moj.gov.vn", "fptcloud.com"]

    custom_settings = {
        'LOG_LEVEL': 'INFO',
        'LOG_STDOUT': True,
        'DOWNLOAD_DELAY': float(os.getenv('RESCUE_DOWNLOAD_DELAY', '0.5')),
        'CONCURRENT_REQUESTS': int(os.getenv('RESCUE_CONCURRENCY', '2')),
        'CONCURRENT_REQUESTS_PER_DOMAIN': int(os.getenv('RESCUE_CONCURRENCY', '2')),
        'COOKIES_ENABLED': True,
        'RETRY_TIMES': 3,
        'DOWNLOAD_TIMEOUT': 60,
    }

    def __init__(self, input_file: str = "", doc_ids: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if input_file:
            self.rescue_source_file = Path(input_file)
        else:
            primary_fail = Path(ARTIFACTS_DIR) / 'crawl_failures.jsonl'
            secondary_fail = Path(JSON_DIR) / 'failed_links.jsonl'
            self.rescue_source_file = primary_fail if primary_fail.exists() else secondary_fail

        self.cli_doc_ids = [v.strip() for v in doc_ids.split(',') if v.strip()]
        self.pending_rescue_records: Dict[str, dict] = {}

    def start_requests(self):
        self.logger.info("=== BẮT ĐẦU CHIẾN DỊCH CỨU HỘ VĂN BẢN PHÁP LUẬT (VIETLAWBERT RESCUE) ===")

        if self.cli_doc_ids:
            self.logger.info("[CHỈ ĐỊNH] Đang cứu hộ %d Document IDs từ CLI...", len(self.cli_doc_ids))
            for doc_id in self.cli_doc_ids:
                self.scheduled_ids.add(doc_id)
                item = {
                    'item_id': doc_id,
                    'doc_id': doc_id,
                    'doc_number': doc_id,
                    'metadata_api': {},
                }
                self.pending_rescue_records[doc_id] = item
                yield scrapy.Request(
                    url=f"{SEARCH_API}/{doc_id}",
                    method="GET",
                    headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
                    callback=self.parse_detail,
                    errback=self.handle_failure,
                    cb_kwargs={'item': item},
                    dont_filter=True,
                )
            return

        if not self.rescue_source_file.exists():
            self.logger.info("✓ Không tìm thấy tệp lỗi tại: %s. Hệ thống không có văn bản tồn đọng!", self.rescue_source_file)
            return

        self.logger.info("[NẠP TỆP LỖI] Đang đọc danh sách cứu hộ từ: %s", self.rescue_source_file)
        count = 0
        with open(self.rescue_source_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                    doc_id = str(record.get('item_id') or record.get('doc_id') or record.get('id') or '').strip()
                    if doc_id and doc_id not in self.scheduled_ids:
                        self.scheduled_ids.add(doc_id)
                        record['doc_id'] = doc_id
                        record['item_id'] = doc_id
                        self.pending_rescue_records[doc_id] = record
                        count += 1
                        yield scrapy.Request(
                            url=f"{SEARCH_API}/{doc_id}",
                            method="GET",
                            headers={'Origin': 'https://vbpl.vn', 'Referer': 'https://vbpl.vn/', 'Accept': 'application/json'},
                            callback=self.parse_detail,
                            errback=self.handle_failure,
                            cb_kwargs={'item': record},
                            dont_filter=True,
                        )
                except json.JSONDecodeError as exc:
                    self.logger.warning("Lỗi cú pháp JSON tại dòng %d: %s", line_no, exc)

        self.logger.info("✓ Đã nạp thành công %d văn bản lỗi vào hàng đợi tái xử lý.", count)

    def spider_closed(self, spider):
        rescued_count = len(self.successful_ids)
        total_scheduled = len(self.scheduled_ids)
        self.logger.info("=== KẾT THÚC CỨU HỘ: Đã cứu %d/%d văn bản thành công ===", rescued_count, total_scheduled)

        # Cập nhật thống kê crawler_status.json từ lớp cha
        super().spider_closed(spider)

        if not self.rescue_source_file.exists() and not self.pending_rescue_records:
            return

        remaining_failures = [
            record for doc_id, record in self.pending_rescue_records.items()
            if doc_id not in self.successful_ids
        ]

        if remaining_failures:
            temp_file = self.rescue_source_file.with_name(f"{self.rescue_source_file.name}.tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                for item in remaining_failures:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file.replace(self.rescue_source_file)
            self.logger.info("! Đã cập nhật tệp lỗi: Còn lại %d văn bản chưa thể khắc phục.", len(remaining_failures))
        else:
            if self.rescue_source_file.exists():
                self.rescue_source_file.unlink()
            self.logger.info("Đã giải cứu toàn bộ văn bản lỗi! Tệp %s đã được xóa sạch.", self.rescue_source_file.name)