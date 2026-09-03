"""
test_merge_crawl_artifacts.py - Kiểm thử tích hợp tính toàn vẹn Shard, Gzip Envelope và Hợp nhất Artifacts.
"""

import json
import sys
from pathlib import Path

# Tự động ghim thư mục gốc dự án
ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "integration" else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.merge import merge_records_streaming
from artifacts.canonical import (
    read_jsonl,
    write_jsonl,
    artifact_manifest,
    canonical_artifacts,
    decode_kafka_envelope,
    encode_kafka_envelope,
    encode_payload,
)
from quality.crawl_audit import audit_crawl
from crawler.shard_runner import build_shards, write_content_quarantine
from crawler.ocr_runner import base_artifact_for, pending_ids, rescued_artifact_for


def test_overlay_replaces_in_place_and_appends_new_ids(tmp_path):
    base_file = tmp_path / "base.jsonl"
    overlay_file = tmp_path / "overlay.jsonl"
    out_file = tmp_path / "merged.jsonl"

    write_jsonl(base_file, [{"item_id": "a", "value": 1}, {"item_id": "b", "value": 2}])
    write_jsonl(overlay_file, [{"item_id": "b", "value": 20}, {"item_id": "c", "value": 3}])

    metrics = merge_records_streaming(base_file, [overlay_file], out_file)
    merged = list(read_jsonl(out_file))

    assert merged == [
        {"item_id": "a", "value": 1},
        {"item_id": "b", "value": 20},
        {"item_id": "c", "value": 3},
    ]
    assert metrics["overwritten_records"] == 1
    assert metrics["appended_records"] == 1


def test_jsonl_round_trip_unicode(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"item_id": "a", "title": "Văn bản quy phạm pháp luật Việt Nam"}]
    write_jsonl(path, records)

    assert list(read_jsonl(path)) == records
    assert "Văn bản" in path.read_text(encoding="utf-8")


def test_read_jsonl_rejects_missing_item_id(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"title": "missing"}) + "\n", encoding="utf-8")

    from artifacts.canonical import extract_item_id
    records = list(read_jsonl(path))
    try:
        extract_item_id(records[0])
    except KeyError as exc:
        assert "không chứa trường định danh" in str(exc)
    else:
        raise AssertionError("Phải báo lỗi khi bản ghi thiếu item_id")


def test_audit_rejects_valid_status_without_real_content(tmp_path):
    path = tmp_path / "false_valid.jsonl"
    write_jsonl(path, [{"item_id": "a", "html_status": "VALID", "html_raw": ""}])
    result = audit_crawl(path, expected_documents=1)
    assert result["html_invalid"] == 1
    assert result["passed"] is False


def test_audit_can_quarantine_explicit_upstream_placeholder(tmp_path):
    path = tmp_path / "upstream_missing.jsonl"
    write_jsonl(
        path,
        [{
            "item_id": "a",
            "html_status": "EMPTY",
            "html_raw": "",
            "rescue_status": "OCR_EMPTY",
            "upstream_content_unavailable": True,
            "rescue_file": {"fileName": "Template.pdf", "size": 32052},
            "metadata_detail": {
                "hasContent": False,
                "documentContentFileName": "Template.pdf",
            },
        }],
    )

    strict = audit_crawl(path, expected_documents=1)
    full_crawl = audit_crawl(path, expected_documents=1, allow_upstream_missing=True)

    assert strict["passed"] is False
    assert full_crawl["passed"] is True
    assert full_crawl["upstream_content_unavailable"] == 1
    assert full_crawl["html_invalid_unexplained"] == 0


def test_audit_can_defer_scanned_pdf_to_separate_ocr_phase(tmp_path):
    path = tmp_path / "ocr_pending.jsonl"
    write_jsonl(
        path,
        [{
            "item_id": "scan",
            "html_status": "EMPTY",
            "html_raw": "",
            "ocr_status": "OCR_PENDING",
            "rescue_status": "OCR_PENDING",
        }],
    )

    strict = audit_crawl(path, expected_documents=1)
    deferred = audit_crawl(path, expected_documents=1, allow_ocr_pending=True)

    assert strict["passed"] is False
    assert deferred["passed"] is True
    assert deferred["ocr_pending"] == 1


def test_full_crawl_shards_handle_partial_last_page():
    shards = build_shards(total_documents=160_660, page_size=100, pages_per_shard=10)
    assert len(shards) == 161
    assert shards[0].start_page == 1
    assert shards[0].expected_documents == 1_000
    assert shards[-1].start_page == 1601
    assert shards[-1].pages == 7
    assert shards[-1].expected_documents == 660


def test_content_quarantine_contains_only_invalid_records(tmp_path):
    artifact = tmp_path / "shard.jsonl.gz"
    write_jsonl(
        artifact,
        [
            {"item_id": "ok", "html_status": "VALID", "html_raw": "x" * 100},
            {"item_id": "missing", "html_status": "EMPTY", "html_raw": ""},
        ],
    )

    count = write_content_quarantine(artifact)
    quarantine_file = artifact.with_name(artifact.name.replace(".jsonl.gz", ".quarantine.jsonl"))
    quarantined = list(read_jsonl(quarantine_file))

    assert count == 1
    assert [record["item_id"] for record in quarantined] == ["missing"]


def test_ocr_runner_selects_pending_and_resolves_base_path(tmp_path):
    quarantine = tmp_path / "crawl_pages_00001_00010.quarantine.jsonl"
    write_jsonl(
        quarantine,
        [
            {"item_id": "scan", "ocr_status": "OCR_PENDING"},
            {"item_id": "template", "rescue_status": "UPSTREAM_TEMPLATE"},
        ],
    )

    assert pending_ids(quarantine) == ["scan"]
    assert base_artifact_for(quarantine).name == "crawl_pages_00001_00010.jsonl.gz"
    assert rescued_artifact_for(quarantine).name == "crawl_pages_00001_00010.rescued.jsonl.gz"


def test_canonical_artifacts_prefer_rescued_and_hash_exact_payload(tmp_path):
    base = tmp_path / "crawl_pages_00001_00010.jsonl.gz"
    rescued = tmp_path / "crawl_pages_00001_00010.rescued.jsonl.gz"
    write_jsonl(base, [{"item_id": "a", "value": 1}])
    write_jsonl(rescued, [{"item_id": "a", "value": 2}])

    selected = canonical_artifacts(tmp_path, expected_shards=1)
    manifest = artifact_manifest(selected)

    assert selected == [rescued]
    assert len(manifest["a"]) == 64
    assert encode_payload({"item_id": "a"}).startswith(b'{"item_id"')


def test_kafka_gzip_envelope_round_trip_and_compresses_verbose_html():
    record = {"item_id": "large", "html_raw": "<span style='x'>Luật</span>" * 10000}
    envelope = encode_kafka_envelope(record)
    assert decode_kafka_envelope(envelope) == encode_payload(record)
    assert len(envelope) < len(encode_payload(record)) / 10