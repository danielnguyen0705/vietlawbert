"""
test_ingestion_stabilization.py - Kiểm thử tích hợp tính ổn định của Spider, Pipeline và Ingest Worker (v3).
"""

import json
import gzip
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crawler.spiders.law_spider import LawSpider, OFFICIAL_DOC_TYPE_IDS
from crawler.pipelines import LegalOntologyMappingPipeline
from pipeline.ingest_pipeline import IngestPipelineWorker


class SpiderIntegrationTests(unittest.TestCase):
    def test_spider_filters_translated_documents_and_respects_limit(self):
        spider = LawSpider(limit="2", page_size="10")
        documents = [
            {"id": "trans_1", "docType": {"code": "BD"}},
            {"id": "doc_1", "docType": {"code": "LU"}},
            {"id": "doc_2", "docType": {"code": "NĐ"}},
            {"id": "doc_3", "docType": {"code": "TT"}},
        ]
        requests = list(spider._build_detail_requests(documents))
        self.assertEqual(2, len(requests))
        self.assertEqual({"doc_1", "doc_2"}, spider.scheduled_ids)
        self.assertEqual(1, spider.skipped_doc_types.get("BD"))

    def test_search_body_excludes_translation_doc_type(self):
        spider = LawSpider(limit="100")
        body = json.loads(spider.make_search_body(1))[0]
        self.assertEqual(len(OFFICIAL_DOC_TYPE_IDS) - 1, len(body["docType"]))
        self.assertNotIn(OFFICIAL_DOC_TYPE_IDS["BD"], body["docType"])
        self.assertIn(OFFICIAL_DOC_TYPE_IDS["LU"], body["docType"])


class OntologyMappingPipelineTests(unittest.TestCase):
    def test_diagram_mapping_extracts_hin_relations(self):
        pipeline = LegalOntologyMappingPipeline()
        item = {
            "item_id": "DOC_2026_01",
            "doc_number": "15/2026/NĐ-CP",
            "diagram_json": {
                "documentNamesByType": {"12": [{"id": "DOC_OLD_01", "name": "Nghị định cũ"}]},
                "documentNamesBySource": {"10": [{"id": "DOC_NEW_01", "name": "Nghị định mới"}]},
            },
        }
        processed = pipeline.process_diagram(item)
        self.assertEqual("VALID", processed["diagram_status"])
        rel_set = {(r["target_id"], r["edge_type"], r["direction"]) for r in processed["relationships"]}
        self.assertIn(("DOC_OLD_01", "THAY_THE", "OUTGOING"), rel_set)
        self.assertIn(("DOC_NEW_01", "SUA_DOI_BO_SUNG", "INCOMING"), rel_set)


class IngestPipelineWorkerTests(unittest.TestCase):
    @patch("pipeline.ingest_pipeline.SentenceTransformer")
    @patch("pipeline.ingest_pipeline.QdrantClientWrapper")
    @patch("pipeline.ingest_pipeline.LegalElasticsearchRetriever")
    def test_worker_processes_shard_with_html_fallback(self, mock_es, mock_qdrant, mock_encoder, tmp_path):
        # Giả lập encoder trả về vector float
        encoder_instance = Mock()
        encoder_instance.encode.return_value = Mock(tolist=lambda: [[0.05] * 256])
        mock_encoder.return_value = encoder_instance

        worker = IngestPipelineWorker(
            qdrant_host="localhost",
            qdrant_port=6333,
            es_host="http://localhost:9200",
            model_name_or_path="mock-bge-m3",
            vector_dim=256,
            batch_size=10,
            device="cpu"
        )

        shard_file = tmp_path / "crawl_pages_00001.jsonl.gz"
        doc_record = {
            "doc_id": "TEST_DOC_FALLBACK",
            "doc_number": "01/2026/TT-BTP",
            "html_raw": "<html><body><h1>Điều 1. Quy định</h1><p>Nội dung quy định chi tiết kiểm thử pipeline trung gian.</p></body></html>",
            "metadata_detail": {"title": "Thông tư mẫu"}
        }

        with gzip.open(shard_file, "wt", encoding="utf-8") as f:
            f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

        total_chunks = worker.process_raw_shard(shard_file)
        self.assertGreaterEqual(total_chunks, 1)
        self.assertTrue(worker.qdrant.upsert_batch.called)
        self.assertTrue(worker.es.bulk_index_chunks.called)


if __name__ == "__main__":
    unittest.main()