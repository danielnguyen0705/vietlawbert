"""
test_merge_crawl_artifacts.py - Kiểm thử tích hợp tính toàn vẹn Shard, Gzip Envelope và Hợp nhất Artifacts.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.merge import merge_records_streaming
from artifacts.canonical import (
    read_jsonl,
    write_jsonl,
    artifact_manifest,
    canonical_artifacts,
    extract_item_id,
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

    records = list(read_jsonl(path, require_item_id=False))
    try:
        extract_item_id(records[0])
    except KeyError as exc:
        assert "thiếu item_id" in str(exc).lower()
    else:
        raise AssertionError("Phải báo lỗi khi bản ghi thiếu item_id")


def test_audit_rejects_valid_status_without_real_content(tmp_path):
    path = tmp_path / "false_valid.jsonl"
    write_jsonl(path, [{"item_id": "a", "html_status": "VALID", "html_raw": ""}])
    result = audit_crawl(path, expected_documents=1)
    assert result["html_invalid"] == 1
    assert result["passed"] is False


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