"""
vietlawbert.crawler.spiders
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gói điều phối các Scrapy Spiders phục vụ dự án VietLawBERT:
- LawSpider: Nhện thu thập diện rộng toàn bộ danh mục văn bản pháp luật quốc gia.
- RescueSpider: Nhện cứu hộ tái thu thập các bản ghi lỗi/quarantine theo kiến trúc Microservice.
"""

from .law_spider import LawSpider
from .rescue_spider import RescueSpider

__all__ = ["LawSpider", "RescueSpider"]