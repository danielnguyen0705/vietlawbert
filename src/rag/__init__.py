"""
vietlawbert.rag
~~~~~~~~~~~~~~~
Hạ tầng truy xuất lai đa tầng và suy luận pháp lý có căn cứ (Grounded Legal Reasoning - v3):
- LegalHybridRetriever / LegalRetriever: Hợp nhất Qdrant Dense (d=256) + ES BM25 + Compile-time Graph Scoring.
- LegalElasticsearchRetriever: Động cơ Sparse Retrieval vận hành Custom Vietnamese Legal Analyzer.
- LegalGenerator / LegalAnswerGenerator: Động cơ sinh câu trả lời pháp lý có trích dẫn và đo Attribution Score (RQ4).
"""

from .retriever import LegalHybridRetriever, LegalRetriever
from .es_retriever import LegalElasticsearchRetriever
from .generator import LegalGenerator, LegalAnswerGenerator

__all__ = [
    "LegalHybridRetriever",
    "LegalRetriever",
    "LegalElasticsearchRetriever",
    "LegalGenerator",
    "LegalAnswerGenerator",
]