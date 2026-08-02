import scrapy
from enum import Enum


class HTMLStatus(str, Enum):
    VALID = "VALID"
    EMPTY = "EMPTY"
    NOT_FOUND = "NOT_FOUND"


class PDFStatus(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    DIGITAL_TEXT = "DIGITAL_TEXT"
    SCANNED_OR_CORRUPTED = "SCANNED_OR_CORRUPTED"


class VietLawItem(scrapy.Item):
    item_id = scrapy.Field()
    doc_number = scrapy.Field()

    html_status = scrapy.Field()
    html_raw = scrapy.Field()
    html_path = scrapy.Field()
    pdf_status = scrapy.Field()
    pdf_path = scrapy.Field()
    pdf_local_path = scrapy.Field()

    diagram_json = scrapy.Field()
    html_dom = scrapy.Field()

    metadata_api = scrapy.Field()
    metadata_detail = scrapy.Field()

    relationships = scrapy.Field()
    diagram_status = scrapy.Field()
    diagram_unresolved_keys = scrapy.Field()
