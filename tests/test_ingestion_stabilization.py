import importlib.util
import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "law_dataset" / "src"
sys.path.insert(0, str(SRC))


class SpiderLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from crawler.spiders.law_spider import LawSpider
        except ModuleNotFoundError as exc:
            if exc.name != "scrapy":
                raise
            scrapy = types.ModuleType("scrapy")
            scrapy.Spider = type("Spider", (), {"__init__": lambda self, *args, **kwargs: None})
            scrapy.Item = dict
            scrapy.Field = lambda: None
            scrapy.Request = lambda **kwargs: types.SimpleNamespace(**kwargs)
            scrapy.signals = types.SimpleNamespace(spider_closed=object())
            sys.modules["scrapy"] = scrapy
            from crawler.spiders.law_spider import LawSpider
        cls.spider_class = LawSpider

    def test_limit_is_exact_even_when_search_page_is_larger(self):
        spider = self.spider_class(limit="3", page_size="10")
        documents = [{"id": index, "docNum": f"DOC-{index}"} for index in range(10)]

        requests = list(spider._build_detail_requests(documents))

        self.assertEqual(3, len(requests))
        self.assertEqual({"0", "1", "2"}, spider.scheduled_ids)

    def test_translated_documents_are_excluded_before_limit(self):
        spider = self.spider_class(limit="2")
        documents = [
            {"id": "translation", "docType": {"code": "BD"}},
            {"id": "law-1", "docType": {"code": "Lu"}},
            {"id": "law-2", "docType": {"code": "NĐ"}},
        ]

        requests = list(spider._build_detail_requests(documents))

        self.assertEqual(2, len(requests))
        self.assertEqual({"law-1", "law-2"}, spider.scheduled_ids)
        self.assertEqual({"BD": 1}, spider.skipped_doc_types)

    def test_empty_filters_are_sent_as_undefined(self):
        spider = self.spider_class(limit="100", keyword="", agency_ids="")

        body = json.loads(spider.make_search_body(1))[0]

        self.assertEqual("$undefined", body["keyword"])
        self.assertEqual("$undefined", body["agencyIds"])

    def test_translations_are_excluded_in_server_search_body(self):
        from crawler.spiders.law_spider import OFFICIAL_DOC_TYPE_IDS

        spider = self.spider_class(limit="100")
        body = json.loads(spider.make_search_body(1))[0]

        self.assertEqual(len(OFFICIAL_DOC_TYPE_IDS) - 1, len(body["docType"]))
        self.assertNotIn(OFFICIAL_DOC_TYPE_IDS["BD"], body["docType"])
        self.assertIn(OFFICIAL_DOC_TYPE_IDS["HP"], body["docType"])
        self.assertTrue(body["groupVbpl"])

    def test_pages_argument_is_honoured(self):
        spider = self.spider_class(pages="5")
        self.assertEqual(5, spider.max_pages)

    def test_start_page_and_page_count_define_checkpoint_range(self):
        spider = self.spider_class(start_page="101", pages="100")
        self.assertEqual(101, spider.start_page)
        self.assertEqual(100, spider.max_pages)

    def test_targeted_document_ids_are_parsed_for_rescue_retries(self):
        spider = self.spider_class(doc_ids="doc-1, doc-2")
        self.assertEqual(["doc-1", "doc-2"], spider.requested_doc_ids)

    def test_embedded_binary_is_removed_and_detail_content_is_not_duplicated(self):
        spider = self.spider_class(limit="1")
        html = '<p>' + ('Nội dung pháp luật ' * 10) + '</p><img src="data:image/png;base64,AAAA">'
        response = types.SimpleNamespace(
            text=json.dumps({"data": {"documentContent": {"content": html, "fileName": "law.pdf"}}})
        )

        request = next(spider.parse_detail(response, {"item_id": "law-1"}))
        item = request.cb_kwargs["item"]

        self.assertNotIn("base64", item["html_raw"])
        self.assertNotIn("content", item["metadata_detail"]["documentContent"])
        self.assertEqual("law.pdf", item["metadata_detail"]["documentContent"]["fileName"])

    def test_file_action_payload_and_fallback_priority(self):
        spider = self.spider_class(limit="1")
        payload = '0:["$@1"]\n1:[{"fileName":"law.pdf"},{"fileName":"law_content.html"}]\n'

        files = spider._decode_action_value(payload)
        ordered = sorted(files, key=spider._fallback_file_priority)

        self.assertEqual("law_content.html", ordered[0]["fileName"])
        self.assertEqual("law.pdf", ordered[1]["fileName"])

    def test_known_upstream_template_skips_download_and_ocr(self):
        spider = self.spider_class(limit="1")
        spider.logger = Mock()
        item = {"item_id": "187551"}

        request = spider._next_file_request(
            item,
            [{"fileName": "Template.pdf", "size": 32052, "presignedUrl": "https://example.invalid"}],
        )

        self.assertEqual("UPSTREAM_TEMPLATE", item["rescue_status"])
        self.assertTrue(item["upstream_content_unavailable"])
        self.assertIn("diagram", request.url)

    def test_ocr_reports_missing_engine_instead_of_silently_dropping(self):
        spider = self.spider_class(limit="1")
        with patch("crawler.spiders.law_spider.shutil.which", return_value=None):
            text, status = spider._ocr_pdf(b"not-opened-without-engine")
        self.assertEqual("", text)
        self.assertEqual("OCR_ENGINE_UNAVAILABLE", status)

    def test_rescue_browser_request_uses_bounded_fast_navigation(self):
        spider = self.spider_class(limit="1")

        request = spider._rescue_browser_request({"item_id": "law-1"})

        self.assertEqual("domcontentloaded", request.meta["playwright_page_goto_kwargs"]["wait_until"])
        self.assertEqual(0, request.meta["max_retry_times"])
        self.assertTrue(callable(request.meta["playwright_page_init_callback"]))
        self.assertEqual(1, request.cb_kwargs["rescue_attempt"])

    def test_numeric_legacy_record_uses_official_print_fallback_first(self):
        spider = self.spider_class(limit="1")

        request = spider._legacy_print_request({"item_id": "187551"})

        self.assertIn("vbpq-print.aspx?ItemID=187551", request.url)
        self.assertEqual("domcontentloaded", request.meta["playwright_page_goto_kwargs"]["wait_until"])

    def test_rescue_failure_closes_page_before_retry(self):
        spider = self.spider_class(limit="1")
        spider.logger = Mock()
        request = spider._rescue_browser_request({"item_id": "law-1"})
        page = AsyncMock()
        request.meta["playwright_page"] = page
        failure = types.SimpleNamespace(request=request, value=TimeoutError("navigation"))

        retry = asyncio.run(spider.handle_file_list_failure(failure))

        page.close.assert_awaited_once()
        self.assertEqual(2, retry.cb_kwargs["rescue_attempt"])


def load_consumer_with_stubs():
    kafka = types.ModuleType("confluent_kafka")
    kafka.Consumer = Mock
    kafka.Producer = Mock
    kafka.KafkaError = types.SimpleNamespace(_PARTITION_EOF=-191)
    kafka.TopicPartition = lambda topic, partition, offset: (topic, partition, offset)
    sys.modules["confluent_kafka"] = kafka

    chunker = types.ModuleType("preprocess.legal_chunker")
    chunker.chunk_legal_document = Mock(return_value=[])
    sys.modules["preprocess.legal_chunker"] = chunker

    cleaner = types.ModuleType("preprocess.text_cleaner")
    cleaner.clean_boilerplate = lambda text: text
    cleaner.extract_doc_type = lambda text: ""
    cleaner.extract_doc_number = lambda text: ""
    cleaner.extract_effective_date = lambda text: ""
    sys.modules["preprocess.text_cleaner"] = cleaner

    milvus = types.ModuleType("database.milvus_client")
    milvus.load_encoder = Mock(return_value=(Mock(), Mock()))
    milvus.encode_texts = Mock(side_effect=lambda tokenizer, model, texts: [[0.0]] * len(texts))
    milvus.setup_milvus = Mock()
    sys.modules["database.milvus_client"] = milvus

    neo4j = types.ModuleType("database.neo4j_client")
    neo4j.Neo4jManager = Mock
    sys.modules["database.neo4j_client"] = neo4j

    module_name = "consumer_stabilization_test_target"
    spec = importlib.util.spec_from_file_location(
        module_name, SRC / "ingestion" / "embedding_consumer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_real_chunker():
    module_name = "legal_chunker_stabilization_test_target"
    spec = importlib.util.spec_from_file_location(
        module_name, SRC / "preprocess" / "legal_chunker.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ConsumerSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.consumer_module = load_consumer_with_stubs()

    def test_processing_error_is_not_swallowed_or_marked_complete(self):
        consumer = self.consumer_module.LawEventConsumer.__new__(
            self.consumer_module.LawEventConsumer
        )
        consumer._process_raw_document = Mock(side_effect=ValueError("bad record"))
        consumer.pending_messages = []

        with self.assertRaisesRegex(ValueError, "bad record"):
            consumer.process_message(Mock(), {"item_id": "broken"})

        self.assertEqual([], consumer.pending_messages)

    def test_chunk_flush_is_bounded_and_removes_only_written_rows(self):
        consumer = self.consumer_module.LawEventConsumer.__new__(
            self.consumer_module.LawEventConsumer
        )
        consumer.tokenizer = Mock()
        consumer.model = Mock()
        consumer.milvus_client = Mock()
        consumer.neo_manager = Mock()
        consumer.pending_milvus_rows = [{"chunk_id": str(i)} for i in range(3)]
        consumer.pending_milvus_texts = [f"text-{i}" for i in range(3)]
        consumer.pending_neo_batch = [{"chunk_id": str(i)} for i in range(3)]
        consumer.chunks_written = 0

        consumer._flush_chunk_storage(2)

        self.assertEqual(1, len(consumer.pending_milvus_rows))
        self.assertEqual(1, len(consumer.pending_neo_batch))
        self.assertEqual(2, consumer.chunks_written)
        self.assertEqual(
            2,
            len(consumer.milvus_client.upsert.call_args.kwargs["data"]),
        )

    def test_commit_tracks_highest_offset_for_each_partition(self):
        consumer = self.consumer_module.LawEventConsumer.__new__(
            self.consumer_module.LawEventConsumer
        )
        consumer.consumer = Mock()

        def message(partition, offset):
            result = Mock()
            result.topic.return_value = "law-documents"
            result.partition.return_value = partition
            result.offset.return_value = offset
            return result

        consumer.pending_messages = [message(0, 3), message(1, 7), message(0, 5)]
        consumer._commit_pending_messages()

        self.assertEqual(
            {("law-documents", 0, 6), ("law-documents", 1, 8)},
            set(consumer.consumer.commit.call_args.kwargs["offsets"]),
        )
        self.assertFalse(consumer.consumer.commit.call_args.kwargs["asynchronous"])


class LegalChunkerHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunker = load_real_chunker()

    def test_markdown_bold_and_curly_quote_article_is_detected(self):
        text = (
            '**“Điều 31. Thẩm quyền của Thanh tra**\n\n'
            '1\\. Thanh tra viên có quyền xử phạt theo quy định của pháp luật. '
            'Nội dung bổ sung để đoạn thử nghiệm đủ dài và không bị xem là tiêu đề rỗng.'
        )

        chunks = self.chunker.chunk_legal_document(text, "doc-31")

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.hierarchy.dieu == "Điều 31" for chunk in chunks))
        self.assertNotIn("_full", chunks[0].chunk_id)

    def test_short_preamble_is_not_merged_into_first_article(self):
        chunks = self.chunker.chunk_by_dieu(
            "Lời mở đầu.\nĐiều 1. Phạm vi điều chỉnh", "doc-1"
        )
        merged = self.chunker.merge_short_chunks(chunks, min_size=200)

        self.assertEqual(2, len(merged))
        self.assertEqual("preamble", merged[0].metadata.get("type"))
        self.assertEqual("Điều 1", merged[1].hierarchy.dieu)

    def test_oversized_atomic_unit_is_bounded_without_losing_hierarchy(self):
        text = "Điều 1. " + ("Nội dung quy định rất dài; " * 120)

        chunks = self.chunker.chunk_legal_document(text, "doc-long", max_chunk_size=500)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 500 for chunk in chunks))
        self.assertTrue(all(chunk.hierarchy.dieu == "Điều 1" for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("oversized_segment_count") == len(chunks) for chunk in chunks))


class OntologyQuarantineTests(unittest.TestCase):
    def test_unresolved_relation_types_are_not_emitted_to_graph_batch(self):
        from crawler.pipelines import LegalOntologyMappingPipeline

        pipeline = LegalOntologyMappingPipeline.__new__(LegalOntologyMappingPipeline)
        pipeline.static_mapping = {}
        pipeline.edge_templates = {}
        pipeline.dynamic_maps = {}
        item = {
            "item_id": "doc-1",
            "diagram_json": {
                "documentNamesByType": {
                    "999": [{"id": "doc-2", "name": "Văn bản chưa phân loại"}]
                }
            },
        }

        result = pipeline.process_diagram(item)

        self.assertEqual("INCONSISTENT", result["diagram_status"])
        self.assertEqual([], result["relationships"])
        self.assertEqual("999", result["diagram_unresolved_keys"][0]["key"])

    def test_official_numeric_mapping_and_group_direction(self):
        from crawler.pipelines import LegalOntologyMappingPipeline

        pipeline = LegalOntologyMappingPipeline()
        item = {
            "item_id": "current",
            "diagram_json": {
                "documentNamesByType": {"12": [{"id": "old", "name": "Old"}]},
                "documentNamesBySource": {"10": [{"id": "new", "name": "New"}]},
            },
        }

        result = pipeline.process_diagram(item)

        self.assertEqual("VALID", result["diagram_status"])
        relations = {(rel["target_id"], rel["edge_type"], rel["direction"]) for rel in result["relationships"]}
        self.assertEqual(
            {("old", "THAY_THE", "OUTGOING"), ("new", "SUA_DOI_BO_SUNG", "INCOMING")},
            relations,
        )

    def test_empty_content_is_not_published_to_kafka(self):
        from crawler.pipelines import LegalOntologyMappingPipeline

        pipeline = LegalOntologyMappingPipeline.__new__(LegalOntologyMappingPipeline)
        pipeline.process_diagram = Mock(side_effect=lambda item, html_dom: item)
        pipeline.publish_empty = False
        pipeline.kafka_enabled = True
        pipeline.kafka_producer = Mock()
        pipeline.logger = Mock()
        pipeline.crawler = Mock()

        result = pipeline.process_item({"item_id": "empty", "html_status": "EMPTY"})

        self.assertEqual("empty", result["item_id"])
        pipeline.kafka_producer.produce.assert_not_called()
        pipeline.crawler.stats.inc_value.assert_called_once_with("kafka/quarantined_empty")


if __name__ == "__main__":
    unittest.main()
