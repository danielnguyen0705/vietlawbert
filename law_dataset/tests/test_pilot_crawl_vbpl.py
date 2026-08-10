import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.pilot_crawl_vbpl import (  # noqa: E402
    compact_metadata,
    evaluate,
    refresh_record_classification,
    stratified_sample,
)


class PilotCrawlTests(unittest.TestCase):
    def test_stratification_covers_observed_three_axis_groups(self):
        rows = [
            {"document_id": "1", "scope": "central", "id_type": "numeric", "document_group": "A"},
            {"document_id": "2", "scope": "central", "id_type": "uuid", "document_group": "A"},
            {"document_id": "3", "scope": "local", "id_type": "numeric", "document_group": "B"},
            {"document_id": "4", "scope": "local", "id_type": "numeric", "document_group": "B"},
        ]
        selected = stratified_sample(rows, 3)
        strata = {(r["scope"], r["id_type"], r["document_group"]) for r in selected}
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(strata), 3)

    def test_content_is_not_duplicated_inside_metadata(self):
        value = compact_metadata({
            "docNum": "1/2026",
            "documentContent": {"content": "large"},
            "documentContentEn": {"content": "large-en"},
        })
        self.assertEqual(value, {"docNum": "1/2026"})

    def test_quality_gate_rejects_incomplete_pilot(self):
        records = [{
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
        }]
        result = evaluate(records, requested=2, stopped_early=False)
        self.assertEqual(result["decision"], "REVIEW_AND_REPEAT")
        self.assertFalse(result["gates"]["completed_requested_sample"])

    def test_short_placeholders_are_not_counted_as_corpus_duplicates(self):
        records = [
            {
                "status": "CONTENT_TOO_SHORT",
                "official_group_code": "VBHN",
                "content_sha256": "same-placeholder",
            },
            {
                "status": "CONTENT_TOO_SHORT",
                "official_group_code": "VBHN",
                "content_sha256": "same-placeholder",
            },
        ]
        result = evaluate(records, requested=2, stopped_early=False)
        self.assertEqual(result["metrics"]["duplicate_content_records"], 0)

    def test_short_html_with_pdf_is_counted_as_recoverable_source(self):
        records = [{
            "status": "CONTENT_TOO_SHORT",
            "official_group_code": "VBHN",
            "content_sha256": "placeholder",
            "files": [{"fileName": "original.PDF"}],
            "warnings": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["metrics"]["content_or_pdf"], 1)
        self.assertEqual(result["metrics"]["content_or_pdf_ratio"], 1.0)

    def test_checkpoint_classification_can_be_refreshed_from_detail_metadata(self):
        record = {
            "metadata": {
                "docNum": "15/2017/QĐ-UBND",
                "docType": {"code": "QĐ", "name": "Quyết định", "parentCode": "VBQPPL"},
            },
            "warnings": ["OFFICIAL_GROUP_NOT_FOUND"],
        }
        refreshed = refresh_record_classification(record)
        self.assertEqual(refreshed["official_group_code"], "VBQPPL")
        self.assertNotIn("OFFICIAL_GROUP_NOT_FOUND", refreshed["warnings"])


if __name__ == "__main__":
    unittest.main()
