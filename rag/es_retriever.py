"""
es_retriever.py - Động cơ truy xuất từ khóa thưa (Sparse Retrieval Engine) cho văn bản pháp luật.
Sử dụng Elasticsearch 8.x với Custom Vietnamese Legal Analyzer và chuẩn hóa thứ tự Token Filter.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger("VietLawBERT_ESRetriever")


class LegalElasticsearchRetriever:
    def __init__(
        self,
        hosts: Optional[List[str] | str] = None,
        index_name: str = "vietlaw_sparse_idx",
    ):
        if hosts is None:
            raw_hosts = ["http://localhost:9200"]
        elif isinstance(hosts, str):
            raw_hosts = [hosts]
        else:
            raw_hosts = list(hosts)

        self.hosts = raw_hosts
        self.index_name = index_name
        self.client = Elasticsearch(hosts=self.hosts, request_timeout=60.0)
        self._ensure_index_and_analyzer()

    def _ensure_index_and_analyzer(self) -> None:
        """Cấu hình Legal Analyzer chuẩn hóa thứ tự: lowercase -> vietnamese_stop -> asciifolding -> shingle."""
        if self.client.indices.exists(index=self.index_name):
            logger.info("Elasticsearch index [%s] đã sẵn sàng.", self.index_name)
            return

        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "vietnamese_stop": {
                        "type": "stop",
                        "stopwords": ["và", "hoặc", "của", "tại", "theo", "về", "các", "những", "căn", "cứ", "quy", "định"],
                    },
                    "legal_shingle": {
                        "type": "shingle",
                        "min_shingle_size": 2,
                        "max_shingle_size": 3,
                        "output_unigrams": True,
                    },
                },
                "analyzer": {
                    "vietnamese_legal_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "vietnamese_stop",   # Lọc từ dừng có dấu trước
                            "asciifolding",      # Sau đó mới bóc tách dấu thanh
                            "legal_shingle",
                        ],
                    }
                },
            },
        }

        mappings = {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "doc_number": {
                    "type": "text",
                    "analyzer": "vietnamese_legal_analyzer",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "hierarchy_path": {
                    "type": "text",
                    "analyzer": "vietnamese_legal_analyzer",
                },
                "macro_label": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "vietnamese_legal_analyzer",
                },
                "is_effective": {"type": "boolean"},
            }
        }

        # ES 8.x: Sử dụng tham số settings và mappings trực tiếp thay vì body
        self.client.indices.create(index=self.index_name, settings=settings, mappings=mappings)
        logger.info("Đã tạo mới chỉ mục Elasticsearch [%s] với Legal Analyzer thành công.", self.index_name)

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 500) -> int:
        actions = []
        for ch in chunks:
            meta = ch.get("metadata", {})
            chunk_id = str(ch.get("chunk_id") or meta.get("chunk_id") or "")
            doc_id = str(ch.get("doc_id") or meta.get("doc_id") or "")
            doc_number = str(ch.get("doc_number") or meta.get("doc_number") or "N/A")
            hierarchy_path = str(ch.get("hierarchy_path") or meta.get("hierarchy_path") or "")
            macro_label = str(ch.get("macro_label") or meta.get("macro_label") or "CHUNG")
            content = str(ch.get("content") or ch.get("text") or "")
            is_effective = bool(ch.get("is_effective", True))

            if not chunk_id:
                continue

            action = {
                "_index": self.index_name,
                "_id": chunk_id,
                "_source": {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_number": doc_number,
                    "hierarchy_path": hierarchy_path,
                    "macro_label": macro_label,
                    "content": content,
                    "is_effective": is_effective,
                },
            }
            actions.append(action)

        success_count, failed = helpers.bulk(self.client, actions, chunk_size=batch_size, stats_only=True)
        if failed:
            logger.warning("Có %s tài liệu gặp sự cố khi nạp vào Elasticsearch.", failed)
        logger.info("Đã nạp thành công %d văn bản vào Elasticsearch index [%s].", success_count, self.index_name)
        return success_count

    def search_sparse(
        self,
        query: str,
        top_k: int = 50,
        must_be_effective: bool = True,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

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
                                "content^1.5",
                            ],
                            "type": "best_fields",
                            "tie_breaker": 0.3,
                        }
                    }
                ],
                "filter": filter_clauses,
            }
        }

        try:
            response = self.client.search(
                index=self.index_name,
                query=es_query,
                size=top_k,
            )
            results = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                src["score"] = float(hit.get("_score") or 0.0)
                src["id"] = str(hit["_id"])
                src["chunk_id"] = str(src.get("chunk_id") or hit["_id"])
                results.append(src)
            return results
        except Exception as exc:
            logger.error("Lỗi truy vấn Elasticsearch: %s", exc)
            return []

    def search(
        self,
        query: str,
        top_k: int = 10,
        must_be_effective: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Giao diện chuẩn hóa tương thích cho các bộ evaluator và comparator."""
        return self.search_sparse(query=query, top_k=top_k, must_be_effective=must_be_effective)

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        must_be_effective: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Alias tương thích ngược trỏ về search_sparse."""
        return self.search_sparse(query=query, top_k=top_k, must_be_effective=must_be_effective)
