"""
vietlawbert.rag
~~~~~~~~~~~~~~~
Hạ tầng truy xuất lai đa tầng (3-Layer Hybrid RRF) và suy luận ngữ cảnh pháp luật.
- LegalRetriever: Tích hợp Dense MRL Vector + Sparse BM25 + Neo4j Graph + Cross-Encoder Re-ranker.
- LegalGenerator: Động cơ sinh câu trả lời có căn cứ pháp lý từ mô hình ngôn ngữ lớn.
"""

from .retriever import LegalRetriever, LegalReranker
from .generator import LegalGenerator

__all__ = [
    "LegalRetriever",
    "LegalReranker",
    "LegalGenerator",
]