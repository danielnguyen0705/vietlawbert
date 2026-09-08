"""
es_retriever.py - Động cơ truy xuất từ khóa thưa (Sparse Retrieval Engine) cho văn bản pháp luật.
Sử dụng Elasticsearch 8.x với Custom Vietnamese Legal Analyzer và Shingle N-grams.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger("VietLawBERT_ESRetriever")


class LegalElasticsearchRetriever:
    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        index_name: str = "vietlaw_sparse_idx"
    ):
        self.hosts = hosts or ["http://localhost:9200"]
        self.index_name = index_name
        self.client = Elasticsearch(self.hosts, request_timeout=60)
        self._ensure_index_and_analyzer()

    def _ensure_index_and_analyzer(self) -> None:
        """Cấu hình bộ phân tích tùy chỉnh tiếng Việt pháp lý tối ưu hóa truy vấn số hiệu và thuật ngữ."""
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Elasticsearch index [{self.index_name}] đã sẵn sàng.")
            return

        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "filter": {
                        "vietnamese_stop": {
                            "type": "stop",
                            "stopwords": ["và", "hoặc", "của", "tại", "theo", "về", "các", "những"]
                        },
                        "legal_shingle": {
                            "type": "shingle",
                            "min_shingle_size": 2,
                            "max_shingle_size": 3,
                            "output_unigrams": True
                        }
                    },
                    "analyzer": {
                        "vietnamese_legal_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",
                                "vietnamese_stop",
                                "legal_shingle"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "doc_number": {
                        "type": "text",
                        "analyzer": "vietnamese_legal_analyzer",
                        "fields": {"raw": {"type": "keyword"}}
                    },
                    "hierarchy_path": {
                        "type": "text",
                        "analyzer": "vietnamese_legal_analyzer"
                    },
                    "macro_label": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "vietnamese_legal_analyzer"
                    },
                    "is_effective": {"type": "boolean"}
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=index_settings)
        logger.info(f"Đã tạo mới chỉ mục Elasticsearch [{self.index_name}] với Legal Analyzer thành công.")

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """Nạp theo lô hàng loạt chunk pháp lý vào Elasticsearch."""
        actions = []
        for ch in chunks:
            action = {
                "_index": self.index_name,
                "_id": ch["chunk_id"],
                "_source": {
                    "chunk_id": ch["chunk_id"],
                    "doc_id": ch.get("doc_id", ""),
                    "doc_number": ch.get("doc_number", "N/A"),
                    "hierarchy_path": ch.get("hierarchy_path", ""),
                    "macro_label": ch.get("macro_label", "CHUNG"),
                    "content": ch.get("text", "") or ch.get("content", ""),
                    "is_effective": ch.get("is_effective", True)
                }
            }
            actions.append(action)

        success_count, failed = helpers.bulk(self.client, actions, chunk_size=batch_size, stats_only=True)
        if failed:
            logger.warning(f"Có {failed} tài liệu gặp sự cố khi nạp vào Elasticsearch.")
        logger.info(f"Đã nạp thành công {success_count} văn bản vào Elasticsearch index [{self.index_name}].")
        self.client.indices.refresh(index=self.index_name)
        return success_count

    def search_sparse(
        self,
        query: str,
        top_k: int = 50,
        must_be_effective: bool = True
    ) -> List[Dict[str, Any]]:
        """Truy vấn kết hợp tăng cường trọng số cho Phân cấp và Số hiệu văn bản."""
        filter_clauses = []
        if must_be_effective:
            filter_clauses.append({"term": {"is_effective": True}})

        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "doc_number^4",
                                "hierarchy_path^3",
                                "content^1.5"
                            ],
                            "type": "best_fields",
                            "tie_breaker": 0.3
                        }
                    }
                ],
                "filter": filter_clauses
            }
        }

        try:
            response = self.client.search(
                index=self.index_name,
                query=es_query,
                size=top_k
            )
            results = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                src["score"] = float(hit["_score"])
                results.append(src)
            return results
        except Exception as exc:
            logger.error(f"Lỗi truy vấn Elasticsearch: {exc}")
            return []