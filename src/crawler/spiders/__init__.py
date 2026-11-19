"""
vietlawbert.crawler.spiders
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gói điều phối các Scrapy Spiders phục vụ dự án VietLawBERT:
- LawSpider: Thu thập diện rộng kho văn bản quy phạm pháp luật quốc gia.
- RescueSpider: Tái xử lý và cứu hộ các bản ghi lỗi/quarantine.
"""

from .law_spider import LawSpider, SEARCH_API, HOME_URL
from .rescue_spider import RescueSpider

__all__ = [
    "LawSpider",
    "RescueSpider",
    "SEARCH_API",
    "HOME_URL",
]