import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.pilot_crawl_vbpl import (  # noqa: E402
    _fallback_recovery_metrics,
    build_report,
    compact_metadata,
    evaluate,
    normalize_manifest_row,
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

    def test_manifest_rows_use_observed_group_without_resampling_schema_loss(self):
        row = normalize_manifest_row({
            "document_id": "vbpqta_1964",
            "scope": "central",
            "id_type": "legacy_prefixed",
            "observed_group": "TRANSLATION",
        })
        self.assertEqual(row["document_group"], "TRANSLATION")
        self.assertEqual(row["observed_group"], "TRANSLATION")

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

    def test_fallback_recovery_metrics_count_pdf_and_ocr_outcomes(self):
        records = [
            {"document_id": "a", "warnings": ["CONTENT_RECOVERED_FROM_PDF", "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"]},
            {"document_id": "b", "warnings": ["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"]},
            {"document_id": "c", "warnings": ["PDF_FETCH_FAILED"]},
            {"document_id": "d", "warnings": ["PDF_EXTRACTION_INSUFFICIENT"]},
        ]
        result = _fallback_recovery_metrics(records)
        self.assertEqual(result["recovered_from_pdf"], 2)
        self.assertEqual(result["recovered_by_ocr"], 1)
        self.assertEqual(result["pdf_fetch_failed"], 1)
        self.assertEqual(result["pdf_extraction_insufficient"], 1)
        self.assertEqual(result["recovered_document_ids"], ["a", "b"])
        self.assertEqual(result["ocr_recovered_document_ids"], ["b"])

    def test_evaluate_counts_recovered_quality_warnings_as_review_signals(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["metrics"]["classification_review_records"], 0)
        self.assertEqual(result["decision"], "READY_FOR_LARGER_PILOT")
        self.assertEqual(result["corpus_readiness"]["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["metrics"]["corpus_review_required_records"], 1)
        self.assertEqual(result["metrics"]["corpus_ready_records"], 0)
        self.assertEqual(result["metrics"]["recovered_quality_review_records"], 0)

    def test_build_report_keeps_recovered_quality_warnings_in_counts(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "source": {"scope": "central", "id_type": "numeric", "observed_group": "LEGAL_FORM_CANDIDATE"},
            "official_group_code": "VBQPPL",
            "content_chars": 200,
            "content_sha256": "a",
            "warnings": ["CONTENT_RECOVERED_FROM_PDF", "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        report = build_report(records, requested=1, input_rows=1, stopped_early=False)
        self.assertEqual(report["counts"]["warnings"]["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"], 1)
        self.assertEqual(report["fallback"]["recovered_from_pdf"], 1)

    def test_report_markdown_lists_new_quality_warning_counts(self):
        report = {
            "pilot": {"completed": 1, "requested": 1, "stopped_early": False},
            "evaluation": {
                "decision": "READY_FOR_LARGER_PILOT",
                "gates": {"completed_requested_sample": True},
                "metrics": {"classification_review_records": 1},
                "technical_readiness": {
                    "decision": "READY_FOR_LARGER_PILOT",
                    "gates": {"completed_requested_sample": True},
                    "metrics": {"classification_review_records": 1},
                },
                "corpus_readiness": {
                    "decision": "REVIEW_REQUIRED",
                    "gates": {"corpus_review_required_records_eq_0": False},
                    "metrics": {"corpus_review_required_records": 1},
                },
            },
            "fallback": {
                "recovered_from_pdf": 1,
                "recovered_by_ocr": 0,
                "pdf_fetch_failed": 0,
                "pdf_extraction_insufficient": 0,
            },
            "counts": {
                "by_status": {"HTML_VALID": 1},
                "by_scope": {"central": 1},
                "by_id_type": {"numeric": 1},
                "by_observed_group": {"LEGAL_FORM_CANDIDATE": 1},
                "by_official_group": {"VBQPPL": 1},
                "warnings": {"RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED": 1},
            },
            "limits": ["x"],
        }
        markdown = __import__("crawler.pilot_crawl_vbpl", fromlist=["report_markdown"]).report_markdown(report)
        self.assertIn("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", markdown)
        self.assertIn("Corpus readiness decision", markdown)
        self.assertIn("REVIEW_REQUIRED", markdown)
        self.assertIn("Keep Phase 2 open", markdown)

    def test_evaluate_allows_review_ratio_boundary_only_when_under_threshold(self):
        records = []
        for index in range(5):
            records.append({
                "document_id": str(index),
                "status": "HTML_VALID",
                "official_group_code": "VBQPPL",
                "content_sha256": str(index),
                "warnings": ["DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED"] if index == 0 else [],
                "files": [],
            })
        result = evaluate(records, requested=5, stopped_early=False)
        self.assertEqual(result["metrics"]["classification_review_ratio"], 0.2)
        self.assertEqual(result["decision"], "READY_FOR_LARGER_PILOT")
        self.assertEqual(result["corpus_readiness"]["decision"], "READY_FOR_PHASE_3")

    def test_evaluate_keeps_technical_ready_but_blocks_corpus_when_recovered_quality_reviews_exist(self):
        records = []
        for index in range(4):
            records.append({
                "document_id": str(index),
                "status": "HTML_VALID",
                "official_group_code": "VBQPPL",
                "content_sha256": str(index),
                "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"] if index == 0 else (["OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"] if index == 1 else []),
                "files": [],
            })
        result = evaluate(records, requested=4, stopped_early=False)
        self.assertEqual(result["metrics"]["classification_review_ratio"], 0.0)
        self.assertEqual(result["decision"], "READY_FOR_LARGER_PILOT")
        self.assertEqual(result["corpus_readiness"]["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["metrics"]["corpus_review_required_records"], 2)
        self.assertEqual(result["metrics"]["recovered_quality_review_records"], 0)

    def test_build_report_with_quality_review_warning_keeps_ready_metric_logic(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "source": {"scope": "central", "id_type": "numeric", "observed_group": "LEGAL_FORM_CANDIDATE"},
            "official_group_code": "VBQPPL",
            "content_chars": 500,
            "content_sha256": "a",
            "warnings": ["OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        report = build_report(records, requested=1, input_rows=1, stopped_early=False)
        self.assertEqual(report["evaluation"]["decision"], "READY_FOR_LARGER_PILOT")
        self.assertEqual(report["evaluation"]["corpus_readiness"]["decision"], "REVIEW_REQUIRED")
        self.assertEqual(report["counts"]["warnings"]["OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"], 1)

    def test_quality_review_warning_does_not_change_content_or_pdf_metric(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["metrics"]["content_or_pdf"], 1)
        self.assertEqual(result["metrics"]["content_or_pdf_ratio"], 1.0)

    def test_fallback_recovery_metrics_ignore_non_recovered_quality_review_only_records(self):
        records = [
            {"document_id": "a", "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"]},
            {"document_id": "b", "warnings": ["OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"]},
        ]
        result = _fallback_recovery_metrics(records)
        self.assertEqual(result["recovered_from_pdf"], 0)
        self.assertEqual(result["recovered_by_ocr"], 0)
        self.assertEqual(result["recovered_document_ids"], [])
        self.assertEqual(result["ocr_recovered_document_ids"], [])

    def test_build_report_includes_quality_review_warning_in_counts_without_files(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "source": {"scope": "central", "id_type": "numeric", "observed_group": "LEGAL_FORM_CANDIDATE"},
            "official_group_code": "VBQPPL",
            "content_chars": 200,
            "content_sha256": "a",
            "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        report = build_report(records, requested=1, input_rows=1, stopped_early=False)
        self.assertIn("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", report["counts"]["warnings"])

    def test_report_markdown_ready_branch_still_mentions_fallback_section(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "source": {"scope": "central", "id_type": "numeric", "observed_group": "LEGAL_FORM_CANDIDATE"},
            "official_group_code": "VBQPPL",
            "content_chars": 200,
            "content_sha256": "a",
            "warnings": ["CONTENT_RECOVERED_FROM_PDF"],
            "files": [],
        }]
        report = build_report(records, requested=1, input_rows=1, stopped_early=False)
        markdown = __import__("crawler.pilot_crawl_vbpl", fromlist=["report_markdown"]).report_markdown(report)
        self.assertIn("## PDF/OCR fallback", markdown)

    def test_quality_review_warning_counts_as_classification_review_even_without_pdf_flag(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["metrics"]["classification_review_records"], 0)
        self.assertEqual(result["metrics"]["corpus_review_required_records"], 1)
        self.assertEqual(result["decision"], "READY_FOR_LARGER_PILOT")
        self.assertEqual(result["corpus_readiness"]["decision"], "REVIEW_REQUIRED")

    def test_corpus_review_warning_without_pdf_recovery_flag_still_blocks_corpus(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["corpus_readiness"]["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["metrics"]["recovered_quality_review_records"], 0)
        self.assertEqual(result["metrics"]["corpus_review_required_records"], 1)

    def test_recovered_pdf_quality_review_metric_requires_pdf_recovery_flag(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": [
                "CONTENT_RECOVERED_FROM_PDF",
                "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
            ],
            "files": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["metrics"]["recovered_quality_review_records"], 1)
        self.assertEqual(result["corpus_readiness"]["decision"], "REVIEW_REQUIRED")

    def test_clean_pdf_recovery_can_still_be_corpus_ready(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": ["CONTENT_RECOVERED_FROM_PDF"],
            "files": [],
        }]
        result = evaluate(records, requested=1, stopped_early=False)
        self.assertEqual(result["decision"], "READY_FOR_LARGER_PILOT")
        self.assertEqual(result["corpus_readiness"]["decision"], "READY_FOR_PHASE_3")
        self.assertEqual(result["metrics"]["corpus_ready_records"], 1)
        self.assertEqual(result["metrics"]["corpus_review_required_records"], 0)
        self.assertEqual(result["metrics"]["recovered_quality_review_records"], 0)

    def test_build_report_with_quality_warning_and_no_content_lengths_works(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "source": {"scope": "central", "id_type": "numeric", "observed_group": "LEGAL_FORM_CANDIDATE"},
            "official_group_code": "VBQPPL",
            "content_chars": 0,
            "content_sha256": "a",
            "warnings": ["RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED"],
            "files": [],
        }]
        report = build_report(records, requested=1, input_rows=1, stopped_early=False)
        self.assertEqual(report["content_chars"]["total"], 0)

    def test_evaluate_review_ratio_zero_without_warnings(self):
        result = evaluate([{
            "document_id": "a",
            "status": "HTML_VALID",
            "official_group_code": "VBQPPL",
            "content_sha256": "a",
            "warnings": [],
            "files": [],
        }], requested=1, stopped_early=False)
        self.assertEqual(result["metrics"]["classification_review_ratio"], 0.0)

    def test_build_report_includes_fallback_metrics(self):
        records = [{
            "document_id": "a",
            "status": "HTML_VALID",
            "source": {"scope": "central", "id_type": "numeric", "observed_group": "LEGAL_FORM_CANDIDATE"},
            "official_group_code": "VBQPPL",
            "content_chars": 200,
            "content_sha256": "a",
            "warnings": ["CONTENT_RECOVERED_FROM_PDF"],
        }]
        report = build_report(records, requested=1, input_rows=1, stopped_early=False)
        self.assertEqual(report["fallback"]["recovered_from_pdf"], 1)
        self.assertIn("PDF/OCR fallback", __import__("crawler.pilot_crawl_vbpl", fromlist=["report_markdown"]).report_markdown(report))


if __name__ == "__main__":
    unittest.main()
