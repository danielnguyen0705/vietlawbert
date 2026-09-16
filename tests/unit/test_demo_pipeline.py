import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pipeline.ingest_pipeline import IngestPipelineWorker
from cli.demo import prepare, render_report, storage_snapshot
from preprocess.validity import is_currently_effective
from rag.generator import LegalGenerator
from datetime import date


def worker_stub():
    worker = IngestPipelineWorker.__new__(IngestPipelineWorker)
    worker.skip_existing = False
    worker.batch_size = 2
    worker.vector_dim = 2
    worker.qdrant = Mock()
    worker.es = Mock()
    worker.encoder = Mock()
    worker.encoder.encode.return_value = np.ones((1, 2))
    return worker


def test_partial_dense_write_fails_before_sparse_write():
    worker = worker_stub()
    worker.qdrant.upsert_batch.return_value = 0
    with pytest.raises(RuntimeError):
        worker._flush_chunks_to_storage([{"chunk_id": "a", "text": "source"}])
    worker.es.bulk_index_chunks.assert_not_called()


def test_partial_sparse_write_is_not_success():
    worker = worker_stub()
    worker.qdrant.upsert_batch.return_value = 1
    worker.es.bulk_index_chunks.return_value = 0
    with pytest.raises(RuntimeError):
        worker._flush_chunks_to_storage([{"chunk_id": "a", "text": "source"}])


def test_shard_storage_failure_propagates(tmp_path):
    worker = worker_stub()
    worker.parser = Mock()
    worker.parser.parse_document.return_value = [{"chunk_id": "a", "text": "source"}]
    worker._flush_chunks_to_storage = Mock(side_effect=RuntimeError("storage offline"))
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"doc_id": "d1", "text": "source " * 20}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="storage offline"):
        worker.process_raw_shard(path)
    worker.qdrant.doc_exists.assert_not_called()


def test_snapshot_checks_ids_not_only_counts(monkeypatch):
    worker = worker_stub()
    worker.qdrant.client.scroll.return_value = ([SimpleNamespace(payload={"chunk_id": "a", "content": "same"})], None)
    monkeypatch.setattr("elasticsearch.helpers.scan", lambda *a, **k: [{"_source": {"chunk_id": "b", "content": "same"}}])
    with pytest.raises(RuntimeError, match="Storage gate failed"):
        storage_snapshot(worker)


def test_repeated_article_ids_preserve_both_segments(tmp_path):
    worker = worker_stub()
    worker.parser = Mock()
    worker.parser.parse_document.side_effect = lambda *a: [
        {"chunk_id": "d1_art_1", "text": "first article"},
        {"chunk_id": "d1_art_1", "text": "quoted article"},
    ]
    captured = []
    worker._flush_chunks_to_storage = lambda chunks: captured.extend(dict(c) for c in chunks)
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"doc_id": "d1", "text": "source " * 20}), encoding="utf-8")
    assert worker.process_raw_shard(path) == 2
    first_ids = [c["chunk_id"] for c in captured]
    assert len(set(first_ids)) == 2
    assert [c["text"] for c in captured] == ["first article", "quoted article"]
    captured.clear()
    worker.process_raw_shard(path)
    assert [c["chunk_id"] for c in captured] == first_ids


def test_report_escapes_source_html(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "chat.json").write_text(json.dumps({"answer": "<script>alert(1)</script>"}), encoding="utf-8")
    render_report(tmp_path)
    text = (reports / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


@pytest.mark.parametrize("status,effective,expected", [
    ("Còn hiệu lực", "2026-09-10T00:00:00", True),
    ("Còn hiệu lực", "2026-09-20T00:00:00", False),
    ("Chưa có hiệu lực", "2026-09-10", False),
    ("Hết hiệu lực toàn bộ", "2026-09-10", False),
    (None, "2026-09-10", False),
    ("Còn hiệu lực", "Chưa xác định", False),
])
def test_effectiveness_is_not_inferred_from_absence_of_expiry(status, effective, expected):
    assert is_currently_effective({"status": status, "effective_date": effective}, date(2026, 9, 15)) is expected


def test_attribution_normalizes_unicode_dashes_and_spaces():
    generator = LegalGenerator.__new__(LegalGenerator)
    score = generator._compute_attribution_score(
        "Căn cứ 35/2026/NQ‑HĐND, tại Điều\u202f3 quy định nội dung sau.",
        [{"doc_number": "35/2026/NQ-HĐND", "hierarchy_path": "Điều 3 > Khoản 1"}],
    )
    assert score == 1.0


def test_prepare_rechecks_stale_successful_audit_before_storage(tmp_path, monkeypatch):
    from artifacts.canonical import write_jsonl
    raw = tmp_path / "raw_shards"
    raw.mkdir()
    shard = raw / "crawl_pages_00001_00001.jsonl.gz"
    write_jsonl(shard, [{"item_id": "invalid", "html_status": "VALID", "html_raw": ""}])
    audit = raw / "crawl_pages_00001_00001.audit.json"
    audit.write_text(json.dumps({"records": 1, "passed": True}), encoding="utf-8")
    constructor = Mock()
    monkeypatch.setattr("pipeline.ingest_pipeline.IngestPipelineWorker", constructor)
    with pytest.raises(RuntimeError, match="Failed current crawl audit"):
        prepare(SimpleNamespace(), tmp_path)
    constructor.assert_not_called()
    assert not json.loads(audit.read_text(encoding="utf-8"))["passed"]
