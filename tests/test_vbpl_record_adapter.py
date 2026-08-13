from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from harness.adapters.vbpl_record_adapter import adapt_vbpl_record, is_vbpl_native_record
from harness.checks.validate_jsonl import validate_jsonl_file
from harness.checks.validate_pipeline import run_validation
from harness.config import HarnessConfig


class VBPLRecordAdapterTests(unittest.TestCase):
    def native_record(self, **overrides):
        record = {
            "source": {
                "url": "https://vbpl.vn/van-ban/chi-tiet/175440",
                "scope": "central",
                "id_type": "numeric",
                "observed_group": "LEGAL_FORM_CANDIDATE",
            },
            "document_id": "175440",
            "status": "HTML_VALID",
            "metadata": {
                "name": "Luật ban hành văn bản quy phạm pháp luật",
                "docNum": "64/2025/QH15",
            },
            "content_html": "Căn cứ Luật ban hành văn bản quy phạm pháp luật. Điều 1. Phạm vi điều chỉnh. Văn bản này quy định việc quản lý, kiểm tra và bảo quản dữ liệu pháp luật Việt Nam. Điều 2. Cơ quan có trách nhiệm lưu nguồn và lập báo cáo chất lượng dữ liệu.",
            "official_group_code": "VBQPPL",
            "official_form_name": "Luật",
            "files": [],
            "warnings": [],
            "fetched_at": "2026-08-12T10:00:00+00:00",
            "extraction_method": None,
            "extraction_source_file": None,
            "extraction_confidence": None,
            "extraction_note": None,
            "ocr_attempted": False,
            "pdf_url": None,
            "pdf_text_chars": None,
            "pdf_ocr_chars": None,
            "pdf_fetch_error": None,
        }
        record.update(overrides)
        return record

    def test_detects_native_vbpl_record(self):
        self.assertTrue(is_vbpl_native_record(self.native_record()))
        self.assertFalse(is_vbpl_native_record({"document_id": "x"}))

    def test_adapter_maps_fields_without_fabrication(self):
        adapted = adapt_vbpl_record(self.native_record())
        self.assertEqual(adapted["document_id"], "175440")
        self.assertEqual(adapted["source_url"], "https://vbpl.vn/van-ban/chi-tiet/175440")
        self.assertEqual(adapted["title"], "Luật ban hành văn bản quy phạm pháp luật")
        self.assertEqual(adapted["status"], "HTML_VALID")
        self.assertEqual(adapted["native_status"], "HTML_VALID")
        self.assertEqual(adapted["content"], self.native_record()["content_html"])
        self.assertEqual(adapted["content_chars"], len(self.native_record()["content_html"]))
        self.assertIsNone(adapted["content_sha256"])
        self.assertEqual(adapted["warning_codes"], [])
        self.assertIsNone(adapted["extraction_method"])
        self.assertFalse(adapted["ocr_attempted"])
        self.assertIsNone(adapted["pdf_url"])
        self.assertIsNone(adapted["pdf_text_chars"])
        self.assertIsNone(adapted["pdf_ocr_chars"])
        self.assertIsNone(adapted["pdf_fetch_error"])
        self.assertEqual(adapted["scope"], "central")
        self.assertEqual(adapted["id_type"], "numeric")
        self.assertEqual(adapted["observed_group"], "LEGAL_FORM_CANDIDATE")

    def test_adapter_preserves_pdf_fallback_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            extraction_method="PDF_OCR",
            extraction_source_file="scan.pdf",
            extraction_confidence=0.7,
            extraction_note="ocr_used",
            ocr_attempted=True,
            pdf_url=" https://example.test/scan.pdf ",
            pdf_text_chars=15,
            pdf_ocr_chars=240,
            pdf_fetch_error=" temporary network issue ",
        ))
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")
        self.assertEqual(adapted["extraction_source_file"], "scan.pdf")
        self.assertEqual(adapted["extraction_confidence"], 0.7)
        self.assertEqual(adapted["extraction_note"], "ocr_used")
        self.assertTrue(adapted["ocr_attempted"])
        self.assertEqual(adapted["pdf_url"], "https://example.test/scan.pdf")
        self.assertEqual(adapted["pdf_text_chars"], 15)
        self.assertEqual(adapted["pdf_ocr_chars"], 240)
        self.assertEqual(adapted["pdf_fetch_error"], "temporary network issue")

    def test_adapter_normalizes_pdf_fallback_field_types(self):
        adapted = adapt_vbpl_record(self.native_record(
            extraction_confidence="0.9",
            ocr_attempted=None,
            pdf_text_chars="15",
            pdf_ocr_chars="240",
        ))
        self.assertIsNone(adapted["extraction_confidence"])
        self.assertFalse(adapted["ocr_attempted"])
        self.assertIsNone(adapted["pdf_text_chars"])
        self.assertIsNone(adapted["pdf_ocr_chars"])

    def test_adapter_truthy_ocr_attempted_is_preserved(self):
        adapted = adapt_vbpl_record(self.native_record(ocr_attempted=True))
        self.assertTrue(adapted["ocr_attempted"])

    def test_adapter_false_ocr_attempted_is_preserved(self):
        adapted = adapt_vbpl_record(self.native_record(ocr_attempted=False))
        self.assertFalse(adapted["ocr_attempted"])

    def test_adapter_trims_extraction_textual_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            extraction_method=" PDF_TEXT_LAYER ",
            extraction_source_file=" original.pdf ",
            extraction_note=" recovered from text layer ",
            pdf_fetch_error=" fetch failed ",
        ))
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["extraction_source_file"], "original.pdf")
        self.assertEqual(adapted["extraction_note"], "recovered from text layer")
        self.assertEqual(adapted["pdf_fetch_error"], "fetch failed")

    def test_adapter_accepts_zero_pdf_char_counts(self):
        adapted = adapt_vbpl_record(self.native_record(pdf_text_chars=0, pdf_ocr_chars=0))
        self.assertEqual(adapted["pdf_text_chars"], 0)
        self.assertEqual(adapted["pdf_ocr_chars"], 0)

    def test_adapter_accepts_int_extraction_confidence(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_confidence=1))
        self.assertEqual(adapted["extraction_confidence"], 1)

    def test_adapter_preserves_html_extraction_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method="HTML", extraction_confidence=1.0))
        self.assertEqual(adapted["extraction_method"], "HTML")
        self.assertEqual(adapted["extraction_confidence"], 1.0)

    def test_adapter_missing_pdf_fields_stay_missing(self):
        adapted = adapt_vbpl_record(self.native_record(
            extraction_method=None,
            extraction_source_file=None,
            extraction_confidence=None,
            extraction_note=None,
            pdf_url=None,
            pdf_fetch_error=None,
            pdf_text_chars=None,
            pdf_ocr_chars=None,
        ))
        self.assertIsNone(adapted["extraction_method"])
        self.assertIsNone(adapted["extraction_source_file"])
        self.assertIsNone(adapted["extraction_confidence"])
        self.assertIsNone(adapted["extraction_note"])
        self.assertIsNone(adapted["pdf_url"])
        self.assertIsNone(adapted["pdf_fetch_error"])
        self.assertIsNone(adapted["pdf_text_chars"])
        self.assertIsNone(adapted["pdf_ocr_chars"])

    def test_adapter_preserves_recovery_warning_codes(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=["CONTENT_RECOVERED_FROM_PDF", "PDF_FETCH_FAILED"]))
        self.assertEqual(adapted["warning_codes"], ["CONTENT_RECOVERED_FROM_PDF", "PDF_FETCH_FAILED"])

    def test_adapter_preserves_recovered_text_quality_warning_codes(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=[
            "CONTENT_RECOVERED_FROM_PDF",
            "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
            "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
        ]))
        self.assertEqual(
            adapted["warning_codes"],
            [
                "CONTENT_RECOVERED_FROM_PDF",
                "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
                "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
            ],
        )

    def test_adapter_preserves_recovered_text_remediation_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            recovered_text_normalized="Điều 1.",
            recovered_text_normalization={"flags": ["whitespace_normalized"]},
            recovered_text_metrics={"total_chars": 8, "suspect_char_density": 0.0},
        ))
        self.assertEqual(adapted["recovered_text_normalized"], "Điều 1.")
        self.assertEqual(adapted["recovered_text_normalization"]["flags"], ["whitespace_normalized"])
        self.assertEqual(adapted["recovered_text_metrics"]["total_chars"], 8)

    def test_adapter_does_not_fabricate_invalid_recovered_text_remediation_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            recovered_text_normalized=123,
            recovered_text_normalization="bad",
            recovered_text_metrics=[]
        ))
        self.assertIsNone(adapted["recovered_text_normalized"])
        self.assertIsNone(adapted["recovered_text_normalization"])
        self.assertIsNone(adapted["recovered_text_metrics"])

    def test_adapter_preserves_selection_and_manual_review_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            recovered_text_selection={"selected_method": "PDF_OCR", "reason": "ocr_selected_over_noisy_text_layer"},
            manual_corpus_review={
                "status": "approved",
                "reviewer": "cuong",
                "reviewed_at": "2026-08-13T10:00:00+07:00",
                "notes": "Page-level review passed",
                "override_warning_codes": ["CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            },
        ))
        self.assertEqual(adapted["recovered_text_selection"]["selected_method"], "PDF_OCR")
        self.assertEqual(adapted["manual_corpus_review"]["status"], "approved")

    def test_adapter_drops_invalid_selection_and_manual_review_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            recovered_text_selection=[],
            manual_corpus_review="bad",
        ))
        self.assertIsNone(adapted["recovered_text_selection"])
        self.assertIsNone(adapted["manual_corpus_review"])

    def test_adapter_preserves_content_sha_even_with_pdf_recovery_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            content_sha256="deadbeef",
            extraction_method="PDF_TEXT_LAYER",
            pdf_text_chars=123,
        ))
        self.assertEqual(adapted["content_sha256"], "deadbeef")
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["pdf_text_chars"], 123)

    def test_adapter_preserves_backend_and_pdf_fields_together(self):
        adapted = adapt_vbpl_record(self.native_record(
            backend="server_action",
            extraction_method="PDF_OCR",
            pdf_url="https://example.test/a.pdf",
        ))
        self.assertEqual(adapted["backend"], "server_action")
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")
        self.assertEqual(adapted["pdf_url"], "https://example.test/a.pdf")

    def test_adapter_preserves_error_and_pdf_fetch_error_separately(self):
        adapted = adapt_vbpl_record(self.native_record(
            error="adapter failed once",
            pdf_fetch_error="download failed",
        ))
        self.assertEqual(adapted["error"], "adapter failed once")
        self.assertEqual(adapted["pdf_fetch_error"], "download failed")

    def test_adapter_preserves_recovered_record_for_harness(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="HTML_VALID",
            warnings=["CONTENT_RECOVERED_FROM_PDF"],
            extraction_method="PDF_TEXT_LAYER",
            pdf_text_chars=250,
        ))
        self.assertEqual(adapted["status"], "HTML_VALID")
        self.assertEqual(adapted["warning_codes"], ["CONTENT_RECOVERED_FROM_PDF"])
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["pdf_text_chars"], 250)

    def test_adapter_handles_recovered_ocr_record_for_harness(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="HTML_VALID",
            warnings=["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"],
            extraction_method="PDF_OCR",
            ocr_attempted=True,
            pdf_text_chars=12,
            pdf_ocr_chars=260,
        ))
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")
        self.assertTrue(adapted["ocr_attempted"])
        self.assertEqual(adapted["pdf_text_chars"], 12)
        self.assertEqual(adapted["pdf_ocr_chars"], 260)
        self.assertEqual(adapted["warning_codes"], ["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED"])

    def test_adapter_handles_pdf_only_with_fetch_failure(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="PDF_ONLY",
            warnings=["PDF_FETCH_FAILED"],
            extraction_method=None,
            pdf_fetch_error="VBPLHTTPError: timeout",
        ))
        self.assertEqual(adapted["status"], "PDF_ONLY")
        self.assertEqual(adapted["warning_codes"], ["PDF_FETCH_FAILED"])
        self.assertEqual(adapted["pdf_fetch_error"], "VBPLHTTPError: timeout")

    def test_adapter_handles_pdf_only_with_insufficient_extraction(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="PDF_ONLY",
            warnings=["PDF_EXTRACTION_INSUFFICIENT"],
            extraction_method="PDF_EXTRACTION_INSUFFICIENT",
            pdf_text_chars=10,
            pdf_ocr_chars=20,
        ))
        self.assertEqual(adapted["status"], "PDF_ONLY")
        self.assertEqual(adapted["extraction_method"], "PDF_EXTRACTION_INSUFFICIENT")
        self.assertEqual(adapted["pdf_text_chars"], 10)
        self.assertEqual(adapted["pdf_ocr_chars"], 20)

    def test_adapter_handles_content_too_short_with_pdf_attempt(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="CONTENT_TOO_SHORT",
            warnings=["PDF_EXTRACTION_INSUFFICIENT"],
            extraction_method="PDF_EXTRACTION_INSUFFICIENT",
            pdf_text_chars=8,
            pdf_ocr_chars=15,
        ))
        self.assertEqual(adapted["status"], "CONTENT_TOO_SHORT")
        self.assertEqual(adapted["extraction_method"], "PDF_EXTRACTION_INSUFFICIENT")
        self.assertEqual(adapted["warning_codes"], ["PDF_EXTRACTION_INSUFFICIENT"])

    def test_adapter_preserves_pdf_url_with_queryless_trim(self):
        adapted = adapt_vbpl_record(self.native_record(pdf_url="  https://example.test/file.pdf  "))
        self.assertEqual(adapted["pdf_url"], "https://example.test/file.pdf")

    def test_adapter_keeps_numeric_zero_confidence(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_confidence=0.0))
        self.assertEqual(adapted["extraction_confidence"], 0.0)

    def test_adapter_keeps_falsey_but_valid_pdf_char_counts(self):
        adapted = adapt_vbpl_record(self.native_record(pdf_text_chars=0, pdf_ocr_chars=1))
        self.assertEqual(adapted["pdf_text_chars"], 0)
        self.assertEqual(adapted["pdf_ocr_chars"], 1)

    def test_adapter_keeps_recovery_metadata_when_status_is_html_valid(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="HTML_VALID",
            extraction_method="PDF_TEXT_LAYER",
            extraction_confidence=0.95,
            pdf_url="https://example.test/r.pdf",
            pdf_text_chars=320,
        ))
        self.assertEqual(adapted["status"], "HTML_VALID")
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["extraction_confidence"], 0.95)
        self.assertEqual(adapted["pdf_url"], "https://example.test/r.pdf")
        self.assertEqual(adapted["pdf_text_chars"], 320)

    def test_adapter_keeps_recovery_metadata_when_status_is_pdf_only(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="PDF_ONLY",
            extraction_method="PDF_EXTRACTION_INSUFFICIENT",
            pdf_url="https://example.test/r.pdf",
            pdf_text_chars=10,
            pdf_ocr_chars=20,
        ))
        self.assertEqual(adapted["status"], "PDF_ONLY")
        self.assertEqual(adapted["extraction_method"], "PDF_EXTRACTION_INSUFFICIENT")
        self.assertEqual(adapted["pdf_url"], "https://example.test/r.pdf")
        self.assertEqual(adapted["pdf_text_chars"], 10)
        self.assertEqual(adapted["pdf_ocr_chars"], 20)

    def test_adapter_keeps_recovery_metadata_when_status_is_content_too_short(self):
        adapted = adapt_vbpl_record(self.native_record(
            status="CONTENT_TOO_SHORT",
            extraction_method="PDF_EXTRACTION_INSUFFICIENT",
            pdf_text_chars=5,
            pdf_ocr_chars=8,
        ))
        self.assertEqual(adapted["status"], "CONTENT_TOO_SHORT")
        self.assertEqual(adapted["extraction_method"], "PDF_EXTRACTION_INSUFFICIENT")
        self.assertEqual(adapted["pdf_text_chars"], 5)
        self.assertEqual(adapted["pdf_ocr_chars"], 8)

    def test_adapter_keeps_recovery_note(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_note="insufficient_pdf_text"))
        self.assertEqual(adapted["extraction_note"], "insufficient_pdf_text")

    def test_adapter_keeps_source_file_for_recovery(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_source_file="original.pdf"))
        self.assertEqual(adapted["extraction_source_file"], "original.pdf")

    def test_adapter_keeps_pdf_fetch_error_without_extraction_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method=None, pdf_fetch_error="timeout"))
        self.assertIsNone(adapted["extraction_method"])
        self.assertEqual(adapted["pdf_fetch_error"], "timeout")

    def test_adapter_keeps_pdf_url_without_extraction_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method=None, pdf_url="https://example.test/p.pdf"))
        self.assertIsNone(adapted["extraction_method"])
        self.assertEqual(adapted["pdf_url"], "https://example.test/p.pdf")

    def test_adapter_keeps_ocr_attempted_true_without_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method=None, ocr_attempted=True))
        self.assertTrue(adapted["ocr_attempted"])

    def test_adapter_keeps_pdf_counts_without_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method=None, pdf_text_chars=2, pdf_ocr_chars=3))
        self.assertEqual(adapted["pdf_text_chars"], 2)
        self.assertEqual(adapted["pdf_ocr_chars"], 3)

    def test_adapter_keeps_zero_confidence_without_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method=None, extraction_confidence=0.0))
        self.assertEqual(adapted["extraction_confidence"], 0.0)

    def test_adapter_keeps_zero_counts_without_method(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method=None, pdf_text_chars=0, pdf_ocr_chars=0))
        self.assertEqual(adapted["pdf_text_chars"], 0)
        self.assertEqual(adapted["pdf_ocr_chars"], 0)

    def test_adapter_keeps_empty_recovery_warning_list(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=[]))
        self.assertEqual(adapted["warning_codes"], [])

    def test_adapter_keeps_recovery_warning_order(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", "PDF_FETCH_FAILED"]))
        self.assertEqual(adapted["warning_codes"], ["CONTENT_RECOVERED_FROM_PDF", "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", "PDF_FETCH_FAILED"])

    def test_adapter_keeps_pdf_fields_when_source_missing(self):
        adapted = adapt_vbpl_record(self.native_record(source={}, extraction_method="PDF_OCR", pdf_url="https://example.test/a.pdf"))
        self.assertIsNone(adapted["source_url"])
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")
        self.assertEqual(adapted["pdf_url"], "https://example.test/a.pdf")

    def test_adapter_keeps_pdf_fields_when_document_id_missing(self):
        adapted = adapt_vbpl_record(self.native_record(document_id=None, extraction_method="PDF_OCR", pdf_url="https://example.test/a.pdf"))
        self.assertIsNone(adapted["document_id"])
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")
        self.assertEqual(adapted["pdf_url"], "https://example.test/a.pdf")

    def test_adapter_keeps_pdf_fields_with_unknown_status(self):
        adapted = adapt_vbpl_record(self.native_record(status="WEIRD_STATUS", extraction_method="PDF_OCR", pdf_url="https://example.test/a.pdf"))
        self.assertEqual(adapted["status"], "WEIRD_STATUS")
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")
        self.assertEqual(adapted["pdf_url"], "https://example.test/a.pdf")

    def test_adapter_keeps_pdf_fields_with_warning_alias(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=["CONTENT_RECOVERED_FROM_PDF"], extraction_method="PDF_TEXT_LAYER", pdf_text_chars=100))
        self.assertEqual(adapted["warning_codes"], ["CONTENT_RECOVERED_FROM_PDF"])
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["pdf_text_chars"], 100)

    def test_adapter_keeps_pdf_fetch_error_trimmed(self):
        adapted = adapt_vbpl_record(self.native_record(pdf_fetch_error="  timeout  "))
        self.assertEqual(adapted["pdf_fetch_error"], "timeout")

    def test_adapter_keeps_pdf_url_trimmed(self):
        adapted = adapt_vbpl_record(self.native_record(pdf_url="  https://example.test/z.pdf  "))
        self.assertEqual(adapted["pdf_url"], "https://example.test/z.pdf")

    def test_adapter_keeps_source_file_trimmed(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_source_file="  z.pdf  "))
        self.assertEqual(adapted["extraction_source_file"], "z.pdf")

    def test_adapter_keeps_method_trimmed(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_method="  PDF_OCR  "))
        self.assertEqual(adapted["extraction_method"], "PDF_OCR")

    def test_adapter_keeps_note_trimmed(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_note="  note  "))
        self.assertEqual(adapted["extraction_note"], "note")

    def test_adapter_keeps_confidence_float(self):
        adapted = adapt_vbpl_record(self.native_record(extraction_confidence=0.55))
        self.assertEqual(adapted["extraction_confidence"], 0.55)

    def test_adapter_keeps_counts_int(self):
        adapted = adapt_vbpl_record(self.native_record(pdf_text_chars=7, pdf_ocr_chars=9))
        self.assertEqual(adapted["pdf_text_chars"], 7)
        self.assertEqual(adapted["pdf_ocr_chars"], 9)

    def test_adapter_keeps_pdf_error_even_if_error_missing(self):
        adapted = adapt_vbpl_record(self.native_record(error=None, pdf_fetch_error="timeout"))
        self.assertIsNone(adapted["error"])
        self.assertEqual(adapted["pdf_fetch_error"], "timeout")

    def test_adapter_keeps_backend_even_if_pdf_fields_missing(self):
        adapted = adapt_vbpl_record(self.native_record(backend="server_action", extraction_method=None))
        self.assertEqual(adapted["backend"], "server_action")
        self.assertIsNone(adapted["extraction_method"])

    def test_adapter_keeps_warning_alias_for_recovery_codes(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=["PDF_EXTRACTION_INSUFFICIENT", "PDF_FETCH_FAILED"]))
        self.assertEqual(adapted["warning_codes"], ["PDF_EXTRACTION_INSUFFICIENT", "PDF_FETCH_FAILED"])

    def test_adapter_keeps_pdf_recovery_fields_with_empty_warnings(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=[], extraction_method="PDF_TEXT_LAYER", pdf_text_chars=123))
        self.assertEqual(adapted["warning_codes"], [])
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["pdf_text_chars"], 123)

    def test_adapter_keeps_recovery_fields_with_nonempty_warnings(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=["CONTENT_RECOVERED_FROM_PDF"], extraction_method="PDF_TEXT_LAYER", pdf_text_chars=123))
        self.assertEqual(adapted["warning_codes"], ["CONTENT_RECOVERED_FROM_PDF"])
        self.assertEqual(adapted["extraction_method"], "PDF_TEXT_LAYER")
        self.assertEqual(adapted["pdf_text_chars"], 123)

    def test_adapter_normalizes_files_and_warning_codes(self):
        adapted = adapt_vbpl_record(self.native_record(
            files=[{"fileName": "original.pdf"}, "bad-item"],
            warnings=[" MISSING_OFFICIAL_PARENT_GROUP ", "", None],
            content_chars=999,
            content_sha256=" abc123 ",
        ))
        self.assertEqual(adapted["files"], [{"fileName": "original.pdf"}])
        self.assertEqual(adapted["warnings"], ["MISSING_OFFICIAL_PARENT_GROUP"])
        self.assertEqual(adapted["warning_codes"], ["MISSING_OFFICIAL_PARENT_GROUP"])
        self.assertEqual(adapted["content_chars"], 999)
        self.assertEqual(adapted["content_sha256"], "abc123")

    def test_adapter_derives_content_chars_when_missing(self):
        content = self.native_record()["content_html"]
        adapted = adapt_vbpl_record(self.native_record(content_chars=None, content_html=content))
        self.assertEqual(adapted["content_chars"], len(content))

    def test_adapter_preserves_unknown_warning_code_as_data(self):
        adapted = adapt_vbpl_record(self.native_record(warnings=["CUSTOM_WARNING_CODE"]))
        self.assertEqual(adapted["warning_codes"], ["CUSTOM_WARNING_CODE"])

    def test_adapter_uppercases_status_and_trims_textual_fields(self):
        adapted = adapt_vbpl_record(self.native_record(
            status=" html_valid ",
            backend=" server_action ",
            error=" temporary parse issue ",
            source={
                "url": "  https://vbpl.vn/van-ban/chi-tiet/175440  ",
                "scope": " central ",
                "id_type": " numeric ",
                "observed_group": " LEGAL_FORM_CANDIDATE ",
            },
        ))
        self.assertEqual(adapted["status"], "HTML_VALID")
        self.assertEqual(adapted["source_url"], "https://vbpl.vn/van-ban/chi-tiet/175440")
        self.assertEqual(adapted["scope"], "central")
        self.assertEqual(adapted["id_type"], "numeric")
        self.assertEqual(adapted["observed_group"], "LEGAL_FORM_CANDIDATE")
        self.assertEqual(adapted["backend"], "server_action")
        self.assertEqual(adapted["error"], "temporary parse issue")

    def test_adapter_keeps_unknown_status_for_harness_to_flag(self):
        adapted = adapt_vbpl_record(self.native_record(status=" weird_status "))
        self.assertEqual(adapted["status"], "WEIRD_STATUS")
        self.assertEqual(adapted["native_status"], "WEIRD_STATUS")

    def test_missing_native_field_stays_missing(self):
        adapted = adapt_vbpl_record(self.native_record(document_id=None, source={}))
        self.assertIsNone(adapted["document_id"])
        self.assertIsNone(adapted["source_url"])

    def test_unknown_status_surfaces_in_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.json"
            path.write_text(json.dumps([self.native_record(status="WEIRD_STATUS")], ensure_ascii=False), encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("unknown_status:WEIRD_STATUS", report["rejection_reasons"])

    def test_native_record_can_be_adapted_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.json"
            adapted = adapt_vbpl_record(self.native_record())
            path.write_text(json.dumps([adapted], ensure_ascii=False), encoding="utf-8")
            report = run_validation(path, HarnessConfig())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["documents_scanned"], 1)
        self.assertEqual(report["valid_documents"], 1)

    def test_valid_jsonl_gz_passes_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"document_id": "1"}, ensure_ascii=False) + "\n")
            records, errors = validate_jsonl_file(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(errors, [])

    def test_malformed_jsonl_gz_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('{"document_id":"1"}\n')
                stream.write('{"document_id":\n')
            _, errors = validate_jsonl_file(path)
        self.assertEqual(errors[0].line_number, 2)
        self.assertTrue(errors[0].reason.startswith("invalid_json"))

    def test_non_object_json_record_inside_gz_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "badshape.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('123\n')
            _, errors = validate_jsonl_file(path)
        self.assertEqual(errors[0].reason, "record_not_object")


if __name__ == "__main__":
    unittest.main()
