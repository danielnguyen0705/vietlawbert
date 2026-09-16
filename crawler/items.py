"""
items.py - Định nghĩa Schema dữ liệu Scrapy Item cho hệ thống VietLawBERT.
Bảo toàn tính tương thích trường dữ liệu và chống lỗi tuần tự hóa JSON.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any
import scrapy


class HTMLStatus(str, Enum):
    VALID = "VALID"
    EMPTY = "EMPTY"
    ERROR = "ERROR"
    NOT_FOUND = "NOT_FOUND"


class PDFStatus(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    DIGITAL_TEXT = "DIGITAL_TEXT"
    SCANNED_OR_CORRUPTED = "SCANNED_OR_CORRUPTED"
    OCR_PENDING = "OCR_PENDING"
    OCR_COMPLETED = "OCR_COMPLETED"
    OCR_EMPTY = "OCR_EMPTY"


class VietLawItem(scrapy.Item):
    # Khóa định danh & siêu dữ liệu cốt lõi phục vụ AST Parser
    item_id = scrapy.Field()
    doc_id = scrapy.Field()
    doc_number = scrapy.Field()
    title = scrapy.Field()
    effective_date = scrapy.Field()
    issue_date = scrapy.Field()
    status = scrapy.Field()
    co_quan = scrapy.Field()
    org = scrapy.Field()
    signer = scrapy.Field()
    doc_type = scrapy.Field()

    # Nội dung văn bản phục vụ AST Parser & Embedding
    text = scrapy.Field()
    full_text = scrapy.Field()

    # Trạng thái và nội dung HTML
    html_status = scrapy.Field()
    html_raw = scrapy.Field()
    html_path = scrapy.Field()
    content_source = scrapy.Field()

    # Trạng thái PDF & Quá trình OCR
    pdf_status = scrapy.Field()
    pdf_path = scrapy.Field()
    pdf_local_path = scrapy.Field()
    ocr_status = scrapy.Field()

    # Cơ chế cứu hộ (Rescue System)
    rescue_status = scrapy.Field()
    rescue_file = scrapy.Field()
    rescue_files_found = scrapy.Field()
    upstream_content_unavailable = scrapy.Field()

    # Cấu trúc Đồ thị & Lược đồ quan hệ (22 Quan hệ HIN)
    diagram_json = scrapy.Field()
    html_dom = scrapy.Field()  # Tạm thời trên RAM, loại bỏ khi lưu đĩa
    relationships = scrapy.Field()
    diagram_status = scrapy.Field()
    diagram_unresolved_keys = scrapy.Field()

    # Siêu dữ liệu hệ thống (Metadata API)
    metadata_api = scrapy.Field()
    metadata_detail = scrapy.Field()

    def to_clean_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi Item thành dictionary sạch:
        - Loại bỏ thuộc tính html_dom (BeautifulSoup object) tránh gây lỗi tuần tự hóa JSON.
        - Chuyển đổi toàn bộ các Enum sang chuỗi thuần túy (str).
        - Đồng bộ doc_id = item_id nếu thiếu.
        """
        clean_dict = {}
        for key, val in self.items():
            if key == "html_dom":
                continue
            if isinstance(val, Enum):
                clean_dict[key] = val.value
            else:
                clean_dict[key] = val

        if "doc_id" not in clean_dict and "item_id" in clean_dict:
            clean_dict["doc_id"] = clean_dict["item_id"]

        return clean_dict