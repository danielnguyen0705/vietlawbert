from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.adapters.vbpl_record_adapter import adapt_vbpl_record, is_vbpl_native_record
from harness.checks.validate_crawl import validate_document_content, vietnamese_legal_score
from harness.checks.validate_duplicates import find_duplicates
from harness.checks.validate_jsonl import validate_jsonl_file
from harness.checks.validate_metadata import validate_metadata
from harness.checks.validate_pipeline import run_validation
from harness.config import HarnessConfig

VALID_TEXT = (
    "Căn cứ Luật ban hành văn bản quy phạm pháp luật. Điều 1. Phạm vi điều chỉnh. "
    "Văn bản này quy định nguyên tắc quản lý, kiểm tra và bảo quản dữ liệu pháp luật Việt Nam. "
    "Điều 2. Tổ chức thực hiện. Cơ quan, tổ chức có trách nhiệm lưu nguồn, kiểm tra trùng lặp "
    "và lập báo cáo chất lượng dữ liệu."
)


def valid_record(**overrides):
    record = {
        "document_id": "sample-001",
        "source_url": "https://vbpl.vn/van-ban/chi-tiet/sample-001",
        "title": "Quyết định mẫu",
        "status": "HTML_VALID",
        "official_group_code": "VBQPPL",
        "content": VALID_TEXT,
    }
    record.update(overrides)
    return record


class HarnessValidationTests(unittest.TestCase):
    def test_valid_vietnamese_legal_document_passes(self):
        result = validate_document_content(valid_record())
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.vietnamese_score, 0.25)

    def test_cloudflare_page_is_rejected(self):
        record = valid_record(content="Performing security verification. Ray ID: abc123. Cloudflare")
        result = validate_document_content(record, HarnessConfig(minimum_content_chars=10))
        self.assertFalse(result.passed)
        self.assertIn("security_challenge", result.reasons)

    def test_valid_document_with_breadcrumb_trang_chu_is_not_rejected_as_non_document(self):
        content = "<html><body>Trang chủ / Văn bản / Chi tiết. Căn cứ Luật ban hành văn bản quy phạm pháp luật. Điều 1. Phạm vi điều chỉnh. Văn bản này quy định quản lý dữ liệu pháp luật Việt Nam với nội dung đầy đủ và hợp lệ.</body></html>"
        result = validate_document_content(valid_record(content=content), HarnessConfig(minimum_content_chars=50))
        self.assertNotIn("non_document_page", result.reasons)

    def test_empty_document_is_rejected(self):
        result = validate_document_content(valid_record(content=""))
        self.assertFalse(result.passed)
        self.assertIn("empty_content", result.reasons)

    def test_duplicate_url_and_content_are_detected(self):
        records = [valid_record(), valid_record(document_id="sample-002")]
        findings = find_duplicates(records)
        kinds = {finding.kind for finding in findings}
        self.assertIn("duplicate_source_url", kinds)
        self.assertIn("duplicate_content", kinds)

    def test_duplicate_document_id_is_detected(self):
        records = [valid_record(), valid_record(source_url="https://vbpl.vn/van-ban/chi-tiet/sample-002")]
        findings = find_duplicates(records)
        self.assertIn("duplicate_document_id", {finding.kind for finding in findings})

    def test_broken_jsonl_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_text('{"document_id":"ok"}\n{"document_id":\n', encoding="utf-8")
            _, errors = validate_jsonl_file(path)
        self.assertEqual(errors[0].line_number, 2)
        self.assertTrue(errors[0].reason.startswith("invalid_json"))

    def test_missing_optional_title_can_pass(self):
        result = validate_metadata(valid_record(title=""), HarnessConfig(require_title=False))
        self.assertTrue(result.passed)

    def test_warning_codes_must_be_list_when_present(self):
        result = validate_metadata(valid_record(warnings="MISSING_OFFICIAL_PARENT_GROUP"))
        self.assertFalse(result.passed)
        self.assertIn("warning_codes_not_list", result.reasons)

    def test_unknown_warning_code_is_rejected(self):
        result = validate_metadata(valid_record(warnings=["CUSTOM_WARNING_CODE"]))
        self.assertFalse(result.passed)
        self.assertIn("unknown_warning_code:CUSTOM_WARNING_CODE", result.reasons)

    def test_recovered_text_quality_warning_codes_are_known_but_require_corpus_review(self):
        result = validate_metadata(valid_record(warnings=[
            "CONTENT_RECOVERED_FROM_PDF",
            "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
            "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
            "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
            "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED",
            "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
        ]))
        self.assertFalse(result.passed)
        self.assertFalse(result.completeness["unknown_warning_code"])
        self.assertIn("corpus_review_required:RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", result.reasons)
        self.assertIn("corpus_review_required:OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", result.reasons)
        self.assertIn("corpus_review_required:RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED", result.reasons)
        self.assertIn("corpus_review_required:RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED", result.reasons)
        self.assertIn("corpus_review_required:RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED", result.reasons)

    def test_clean_pdf_text_layer_recovery_can_pass_corpus_metadata_gate(self):
        result = validate_metadata(valid_record(warnings=["CONTENT_RECOVERED_FROM_PDF"]))
        self.assertTrue(result.passed)

    def test_ocr_recovered_content_requires_corpus_review_even_without_other_quality_warning(self):
        result = validate_metadata(valid_record(warnings=[
            "CONTENT_RECOVERED_FROM_PDF",
            "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
        ]))
        self.assertFalse(result.passed)
        self.assertIn("corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.reasons)

    def test_manual_corpus_review_can_override_specific_warning(self):
        result = validate_metadata(valid_record(
            warnings=["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            manual_corpus_review={
                "status": "approved",
                "reviewer": "cuong",
                "reviewed_at": "2026-08-13T10:00:00+07:00",
                "notes": "Checked against original PDF",
                "override_warning_codes": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            },
        ))
        self.assertTrue(result.passed)
        self.assertNotIn("corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.reasons)

    def test_manual_corpus_review_does_not_override_unlisted_warning(self):
        result = validate_metadata(valid_record(
            warnings=[
                "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
                "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
            ],
            manual_corpus_review={
                "status": "approved",
                "reviewer": "cuong",
                "reviewed_at": "2026-08-13T10:00:00+07:00",
                "notes": "Checked against original PDF",
                "override_warning_codes": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            },
        ))
        self.assertFalse(result.passed)
        self.assertNotIn("corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.reasons)
        self.assertIn("corpus_review_required:RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", result.reasons)

    def test_invalid_manual_corpus_review_is_rejected(self):
        result = validate_metadata(valid_record(
            warnings=["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            manual_corpus_review={
                "status": "approved",
                "reviewer": "",
                "reviewed_at": "bad-date",
                "notes": "",
                "override_warning_codes": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            },
        ))
        self.assertFalse(result.passed)
        self.assertIn("invalid_manual_corpus_review", result.reasons)
        self.assertIn("corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.reasons)

    def test_pipeline_accepts_manually_promoted_ocr_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.jsonl"
            native = {
                "source": {
                    "url": "https://vbpl.vn/van-ban/chi-tiet/133328",
                    "scope": "local",
                    "id_type": "numeric",
                    "observed_group": "LEGAL_FORM_CANDIDATE",
                },
                "document_id": "133328",
                "status": "HTML_VALID",
                "metadata": {"name": "Nghị quyết mẫu"},
                "content_html": VALID_TEXT,
                "content_sha256": "ocr-a",
                "content_chars": len(VALID_TEXT),
                "warnings": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
                "manual_corpus_review": {
                    "status": "approved",
                    "reviewer": "cuong",
                    "reviewed_at": "2026-08-13T10:00:00+07:00",
                    "notes": "Checked against original PDF",
                    "override_warning_codes": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
                },
                "files": [],
            }
            path.write_text(__import__("json").dumps(native, ensure_ascii=False) + "\n", encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["rejected_documents"], 0)

    def test_recovered_quality_warning_does_not_block_review_only_status_twice(self):
        result = validate_metadata(valid_record(
            status="PDF_ONLY",
            warnings=["OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"],
        ))
        self.assertFalse(result.passed)
        self.assertIn("non_accepted_status:PDF_ONLY", result.reasons)
        self.assertNotIn("corpus_review_required:OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", result.reasons)

    def test_warning_codes_alias_enforces_corpus_review_gate(self):
        result = validate_metadata(valid_record(
            warnings=[],
            warning_codes=["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
        ))
        self.assertFalse(result.passed)
        self.assertIn("corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.reasons)

    def test_multiple_known_corpus_review_warnings_are_each_reported(self):
        result = validate_metadata(valid_record(warnings=[
            "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
            "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
            "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
        ]))
        self.assertFalse(result.passed)
        self.assertEqual(
            sorted(reason for reason in result.reasons if reason.startswith("corpus_review_required:")),
            sorted([
                "corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
                "corpus_review_required:RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
                "corpus_review_required:RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
            ]),
        )

    def test_pipeline_rejects_new_recovered_quality_warning_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.jsonl"
            native = {
                "source": {
                    "url": "https://vbpl.vn/van-ban/chi-tiet/138353",
                    "scope": "central",
                    "id_type": "numeric",
                    "observed_group": "CONSOLIDATED",
                },
                "document_id": "138353",
                "status": "HTML_VALID",
                "metadata": {"name": "Văn bản hợp nhất mẫu"},
                "content_html": VALID_TEXT,
                "content_sha256": "ocr-b",
                "content_chars": len(VALID_TEXT),
                "warnings": ["RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED"],
                "files": [],
            }
            path.write_text(__import__("json").dumps(native, ensure_ascii=False) + "\n", encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "corpus_review_required:RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
            report["record_results"][0]["reasons"],
        )

    def test_native_adapter_keeps_recovered_text_metrics_fields(self):
        native = {
            "source": {
                "url": "https://vbpl.vn/van-ban/chi-tiet/138353",
                "scope": "central",
                "id_type": "numeric",
                "observed_group": "CONSOLIDATED",
            },
            "document_id": "138353",
            "status": "HTML_VALID",
            "metadata": {"name": "Văn bản hợp nhất mẫu"},
            "content_html": VALID_TEXT,
            "content_chars": len(VALID_TEXT),
            "warnings": ["CONTENT_RECOVERED_FROM_PDF"],
            "recovered_text_normalized": VALID_TEXT,
            "recovered_text_normalization": {"flags": ["whitespace_normalized"]},
            "recovered_text_metrics": {"total_chars": len(VALID_TEXT), "trailing_fragment_suspected": False},
            "files": [],
        }
        adapted = adapt_vbpl_record(native)
        self.assertEqual(adapted["recovered_text_normalized"], VALID_TEXT)
        self.assertEqual(adapted["recovered_text_normalization"]["flags"], ["whitespace_normalized"])
        self.assertFalse(adapted["recovered_text_metrics"]["trailing_fragment_suspected"])
        self.assertEqual(adapted["recovered_text_metrics"]["total_chars"], len(VALID_TEXT))

    def test_native_adapter_drops_invalid_recovered_text_metrics_fields(self):
        native = {
            "source": {
                "url": "https://vbpl.vn/van-ban/chi-tiet/138353",
                "scope": "central",
                "id_type": "numeric",
                "observed_group": "CONSOLIDATED",
            },
            "document_id": "138353",
            "status": "HTML_VALID",
            "metadata": {"name": "Văn bản hợp nhất mẫu"},
            "content_html": VALID_TEXT,
            "content_chars": len(VALID_TEXT),
            "warnings": ["CONTENT_RECOVERED_FROM_PDF"],
            "recovered_text_normalized": 123,
            "recovered_text_normalization": "bad",
            "recovered_text_metrics": [],
            "files": [],
        }
        adapted = adapt_vbpl_record(native)
        self.assertIsNone(adapted["recovered_text_normalized"])
        self.assertIsNone(adapted["recovered_text_normalization"])
        self.assertIsNone(adapted["recovered_text_metrics"])

    def test_missing_contract_recommended_metadata_is_review_signal_not_fail(self):
        result = validate_metadata(valid_record(official_group_code="", official_form_name="", warnings=[]))
        self.assertTrue(result.passed)
        self.assertTrue(result.completeness["missing_official_group_code"])
        self.assertTrue(result.completeness["missing_official_form"])

    def test_known_scope_and_id_type_are_tracked_in_completeness(self):
        result = validate_metadata(valid_record(scope="central", id_type="numeric", observed_group="LEGAL_FORM_CANDIDATE"))
        self.assertFalse(result.completeness["unknown_scope"])
        self.assertFalse(result.completeness["unknown_id_type"])
        self.assertFalse(result.completeness["unknown_observed_group"])

    def test_unknown_scope_and_id_type_are_review_signals(self):
        result = validate_metadata(valid_record(scope="regional", id_type="slug", observed_group="OTHER"))
        self.assertTrue(result.passed)
        self.assertTrue(result.completeness["unknown_scope"])
        self.assertTrue(result.completeness["unknown_id_type"])
        self.assertTrue(result.completeness["unknown_observed_group"])

    def test_unknown_official_group_code_is_review_signal(self):
        result = validate_metadata(valid_record(official_group_code="CUSTOM"))
        self.assertTrue(result.passed)
        self.assertTrue(result.completeness["unknown_official_group_code"])

    def test_document_number_can_come_from_metadata_docnum(self):
        record = valid_record(document_number="", metadata={"docNum": "01/2026/NĐ-CP"})
        result = validate_metadata(record)
        self.assertFalse(result.completeness["missing_document_number"])

    def test_warning_codes_alias_is_accepted(self):
        result = validate_metadata(valid_record(warning_codes=["MISSING_OFFICIAL_PARENT_GROUP"], warnings=[]))
        self.assertTrue(result.passed)
        self.assertTrue(result.completeness["warning_codes"])

    def test_malformed_crawl_timestamp_fails(self):
        result = validate_metadata(valid_record(crawl_timestamp="2026/08/12 10:00:00"))
        self.assertFalse(result.passed)
        self.assertIn("malformed_date:crawl_timestamp", result.reasons)

    def test_pipeline_keeps_canonical_warning_codes_and_content_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.json"
            native = {
                "source": {
                    "url": "https://vbpl.vn/van-ban/chi-tiet/175440",
                    "scope": "central",
                    "id_type": "numeric",
                    "observed_group": "LEGAL_FORM_CANDIDATE",
                },
                "document_id": "175440",
                "status": "HTML_VALID",
                "metadata": {"name": "Luật ban hành văn bản quy phạm pháp luật"},
                "content_html": VALID_TEXT,
                "content_sha256": "abc123",
                "content_chars": len(VALID_TEXT),
                "warnings": ["DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED"],
                "files": [],
            }
            path.write_text(__import__("json").dumps([native], ensure_ascii=False), encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["record_results"][0]["native_status"], "HTML_VALID")
        adapted = adapt_vbpl_record(native)
        self.assertEqual(adapted["warning_codes"], ["DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED"])
        self.assertEqual(adapted["content_sha256"], "abc123")
        self.assertEqual(adapted["content_chars"], len(VALID_TEXT))

    def test_pipeline_rejects_corpus_review_required_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.jsonl"
            native = {
                "source": {
                    "url": "https://vbpl.vn/van-ban/chi-tiet/133328",
                    "scope": "local",
                    "id_type": "numeric",
                    "observed_group": "LEGAL_FORM_CANDIDATE",
                },
                "document_id": "133328",
                "status": "HTML_VALID",
                "metadata": {"name": "Nghị quyết mẫu"},
                "content_html": VALID_TEXT,
                "content_sha256": "ocr-a",
                "content_chars": len(VALID_TEXT),
                "warnings": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
                "files": [],
            }
            path.write_text(__import__("json").dumps(native, ensure_ascii=False) + "\n", encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["rejected_documents"], 1)
        self.assertIn(
            "corpus_review_required:CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
            report["record_results"][0]["reasons"],
        )

    def test_missing_required_metadata_fails(self):
        result = validate_metadata(valid_record(document_id=""))
        self.assertFalse(result.passed)
        self.assertIn("missing_document_id", result.reasons)

    def test_missing_status_fails(self):
        result = validate_metadata(valid_record(status=""))
        self.assertFalse(result.passed)
        self.assertIn("missing_status", result.reasons)

    def test_unknown_status_fails(self):
        result = validate_metadata(valid_record(status="HTML_OK"))
        self.assertFalse(result.passed)
        self.assertIn("unknown_status:HTML_OK", result.reasons)

    def test_pdf_only_status_is_not_accepted_as_corpus_record(self):
        result = validate_metadata(valid_record(status="PDF_ONLY"))
        self.assertFalse(result.passed)
        self.assertIn("non_accepted_status:PDF_ONLY", result.reasons)

    def test_pipeline_report_fails_on_duplicate_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            first = valid_record()
            second = valid_record(document_id="sample-002")
            path.write_text(
                f"{__import__('json').dumps(first, ensure_ascii=False)}\n{__import__('json').dumps(second, ensure_ascii=False)}\n",
                encoding="utf-8",
            )
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "FAIL")
        self.assertGreaterEqual(report["duplicate_count"], 1)
        self.assertEqual(report["duplicate_record_count"], 2)
        self.assertEqual(report["valid_documents"], 0)
        self.assertEqual(report["rejected_documents"], 2)

    def test_short_placeholder_duplicates_are_not_counted_for_non_html_valid_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            placeholder = "<!DOCTYPE html><html><head><meta charset=\"UTF-8\"></head><body></body></html>"
            first = valid_record(document_id="a", status="CONTENT_TOO_SHORT", content=placeholder)
            second = valid_record(document_id="b", source_url="https://vbpl.vn/van-ban/chi-tiet/b", status="CONTENT_TOO_SHORT", content=placeholder)
            path.write_text(
                f"{__import__('json').dumps(first, ensure_ascii=False)}\n{__import__('json').dumps(second, ensure_ascii=False)}\n",
                encoding="utf-8",
            )
            report = run_validation(path, HarnessConfig(minimum_content_chars=200))
        self.assertEqual(report["duplicate_count"], 0)

    def test_json_list_with_non_object_record_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.json"
            path.write_text('[{"document_id":"ok","source_url":"https://vbpl.vn/a","status":"HTML_VALID","content":"Điều 1. Văn bản này quy định quản lý dữ liệu pháp luật Việt Nam đủ dài để qua ngưỡng kiểm tra."}, 123]', encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["json_errors"][0]["reason"], "record_not_object")

    def test_vietnamese_score_is_meaningful_not_binary(self):
        score = vietnamese_legal_score(VALID_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.25)

    def test_native_vbpl_adapter_preserves_missing_values(self):
        adapted = adapt_vbpl_record({
            "source": {},
            "document_id": None,
            "status": "HTML_VALID",
            "metadata": {},
            "content_html": "",
        })
        self.assertIsNone(adapted["document_id"])
        self.assertIsNone(adapted["source_url"])
        self.assertEqual(adapted["content"], "")
        self.assertEqual(adapted["native_status"], "HTML_VALID")


if __name__ == "__main__":
    unittest.main()
