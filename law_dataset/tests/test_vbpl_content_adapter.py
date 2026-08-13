import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.vbpl_content_adapter import (  # noqa: E402
    ActionContract,
    ContentStatus,
    NextActionClient,
    PDFExtractionResult,
    PDFExtractor,
    VBPLContentAdapter,
    VBPLHTTPError,
    VBPLResponseParseError,
    choose_recovered_pdf_extraction,
    classify_official_doc_type,
    recovered_text_quality_metrics,
    remediate_recovered_text,
)


class FakeClient:
    def detail(self, document_id):
        return {
            "id": document_id,
            "docNum": "01/2026/NĐ-CP",
            "defaultLanguage": "VN",
            "documentContent": {"content": "<p>Điều 1.</p>" * 20},
        }

    def files(self, document_id):
        return [{"fileName": "original.pdf"}]

    def download_pdf(self, document_id, file_name=None):
        return b"%PDF-1.4 fake", "https://example.test/original.pdf"

    def search_number(self, number, page_size=50):
        return {
            "items": [
                {
                    "id": "abc",
                    "docType": {
                        "name": "Nghị định",
                        "code": "NĐ",
                        "parentCode": "VBQPPL",
                        "parentName": "",
                    },
                }
            ]
        }

    def diagram(self, document_id):
        return {"relations": []}


class StubPDFExtractor:
    def __init__(self, result):
        self.result = result

    def extract(self, pdf_bytes, source_file=None):
        return self.result


class PDFExtractorTests(unittest.TestCase):
    def test_clean_text_compacts_blank_segments(self):
        text = PDFExtractor._clean_text(["", " Điều 1 ", "\n", "Điều 2"])
        self.assertEqual(text, "Điều 1\n\nĐiều 2")

    def test_usable_threshold(self):
        result = PDFExtractionResult(text="Điều 1. " * 30, method="PDF_TEXT_LAYER", text_chars=len("Điều 1. " * 30))
        self.assertTrue(result.usable)

    def test_extract_returns_insufficient_when_no_methods_available(self):
        class NoDepsExtractor(PDFExtractor):
            @staticmethod
            def _module(name):
                return None
        result = NoDepsExtractor(minimum_text_chars=50).extract(b"%PDF-1.4 fake", "original.pdf")
        self.assertEqual(result.method, "PDF_EXTRACTION_INSUFFICIENT")
        self.assertIn("pdfplumber_missing", result.note)

    def test_extract_prefers_text_layer_when_sufficient(self):
        class TextOnlyExtractor(PDFExtractor):
            def extract_text_layer(self, pdf_bytes, source_file=None):
                return PDFExtractionResult(
                    text="Điều 1. " * 30,
                    method="PDF_TEXT_LAYER",
                    source_file=source_file,
                    confidence=0.95,
                    text_chars=len("Điều 1. " * 30),
                )

            def extract_ocr(self, pdf_bytes, source_file=None):
                raise AssertionError("OCR should not run when text layer is enough")
        result = TextOnlyExtractor(minimum_text_chars=50).extract(b"%PDF-1.4 fake", "original.pdf")
        self.assertEqual(result.method, "PDF_TEXT_LAYER")
        self.assertTrue(result.usable)

    def test_extract_falls_back_to_ocr_when_text_layer_short(self):
        class OCRExtractor(PDFExtractor):
            def extract_text_layer(self, pdf_bytes, source_file=None):
                return PDFExtractionResult(
                    text="short",
                    method="PDF_TEXT_LAYER",
                    source_file=source_file,
                    confidence=0.2,
                    note="empty_text_layer",
                    text_chars=5,
                )

            def extract_ocr(self, pdf_bytes, source_file=None):
                return PDFExtractionResult(
                    text="Điều 1. " * 30,
                    method="PDF_OCR",
                    source_file=source_file,
                    confidence=0.7,
                    ocr_attempted=True,
                    ocr_chars=len("Điều 1. " * 30),
                )
        result = OCRExtractor(minimum_text_chars=50).extract(b"%PDF-1.4 fake", "scan.pdf")
        self.assertEqual(result.method, "PDF_OCR")
        self.assertTrue(result.ocr_attempted)
        self.assertTrue(result.usable)
        self.assertEqual(result.text_chars, 5)

    def test_choose_recovered_pdf_extraction_prefers_clean_ocr_over_noisy_text_layer(self):
        text_layer = PDFExtractionResult(
            text="(cid:12) " * 200 + "CQNG 110A XA 1101 CHU NGHIA VIVI' NAM",
            method="PDF_TEXT_LAYER",
            source_file="original.pdf",
            confidence=0.95,
            text_chars=len("(cid:12) " * 200 + "CQNG 110A XA 1101 CHU NGHIA VIVI' NAM"),
        )
        ocr_text = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM. THÔNG TƯ. Căn cứ Luật. Điều 1. " * 40
        ocr = PDFExtractionResult(
            text=ocr_text,
            method="PDF_OCR",
            source_file="original.pdf",
            confidence=0.7,
            ocr_attempted=True,
            ocr_chars=len(ocr_text),
        )
        selected, decision = choose_recovered_pdf_extraction(text_layer, ocr, "original.pdf")
        self.assertEqual(selected.method, "PDF_OCR")
        self.assertEqual(decision["selected_method"], "PDF_OCR")
        self.assertEqual(decision["reason"], "ocr_selected_over_noisy_text_layer")

    def test_choose_recovered_pdf_extraction_keeps_text_layer_when_ocr_still_weak(self):
        text_layer = PDFExtractionResult(
            text="(cid:12) " * 200 + "CQNG 110A XA 1101 CHU NGHIA VIVI' NAM",
            method="PDF_TEXT_LAYER",
            source_file="original.pdf",
            confidence=0.95,
            text_chars=len("(cid:12) " * 200 + "CQNG 110A XA 1101 CHU NGHIA VIVI' NAM"),
        )
        ocr = PDFExtractionResult(
            text="Điều 1 ngắn",
            method="PDF_OCR",
            source_file="original.pdf",
            confidence=0.7,
            ocr_attempted=True,
            ocr_chars=len("Điều 1 ngắn"),
        )
        selected, decision = choose_recovered_pdf_extraction(text_layer, ocr, "original.pdf")
        self.assertEqual(selected.method, "PDF_TEXT_LAYER")
        self.assertEqual(decision["selected_method"], "PDF_TEXT_LAYER")


class ContentAdapterTests(unittest.TestCase):
    def test_action_contract_loads(self):
        path = Path(__file__).resolve().parents[1] / "config" / "vbpl_action_contract.json"
        contract = ActionContract.load(path)
        self.assertFalse(contract.requires_cookie)
        self.assertEqual(len(contract.detail), 40)

    def test_valid_html_and_official_group_are_separate(self):
        adapter = VBPLContentAdapter(FakeClient())
        result = adapter.fetch("abc", include_diagram=True)
        self.assertEqual(result.status, ContentStatus.HTML_VALID)
        self.assertEqual(result.official_group_code, "VBQPPL")
        self.assertEqual(result.official_group_name, "Văn bản quy phạm pháp luật")
        self.assertEqual(result.official_form_code, "NĐ")
        self.assertEqual(result.official_form_name, "Nghị định")
        self.assertEqual(result.diagram, {"relations": []})

    def test_short_html_is_not_valid(self):
        class ShortClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = "short"
                return value

        result = VBPLContentAdapter(ShortClient()).fetch("abc")
        self.assertEqual(result.status, ContentStatus.CONTENT_TOO_SHORT)

    def test_pdf_only_without_fallback_stays_pdf_only(self):
        class PDFOnlyClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

        result = VBPLContentAdapter(PDFOnlyClient(), enable_pdf_fallback=False).fetch("abc")
        self.assertEqual(result.status, ContentStatus.PDF_ONLY)

    def test_pdf_fallback_can_recover_content_as_html_valid(self):
        class PDFOnlyClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

        extractor = StubPDFExtractor(PDFExtractionResult(
            text="Điều 1. " * 30,
            method="PDF_TEXT_LAYER",
            source_file="original.pdf",
            confidence=0.95,
            text_chars=len("Điều 1. " * 30),
        ))
        result = VBPLContentAdapter(PDFOnlyClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.HTML_VALID)
        self.assertEqual(result.extraction_method, "PDF_TEXT_LAYER")
        self.assertEqual(result.extraction_source_file, "original.pdf")
        self.assertIn("CONTENT_RECOVERED_FROM_PDF", result.warnings)
        self.assertEqual(result.pdf_url, "https://example.test/original.pdf")
        self.assertGreater(len(result.content_html), 100)

    def test_pdf_fallback_marks_ocr_recovery_for_review(self):
        class PDFOnlyClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

        extractor = StubPDFExtractor(PDFExtractionResult(
            text="Điều 1. " * 30,
            method="PDF_OCR",
            source_file="scan.pdf",
            confidence=0.7,
            ocr_attempted=True,
            ocr_chars=len("Điều 1. " * 30),
        ))
        result = VBPLContentAdapter(PDFOnlyClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.HTML_VALID)
        self.assertIn("CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.warnings)
        self.assertIn("OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", result.warnings)
        self.assertTrue(result.ocr_attempted)
        self.assertEqual(result.pdf_ocr_chars, len("Điều 1. " * 30))

    def test_pdf_text_layer_with_encoding_artifacts_is_marked_for_review(self):
        class PDFOnlyClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

        extractor = StubPDFExtractor(PDFExtractionResult(
            text="Điều 1 (cid:12) phạm vi điều chỉnh. " * 10,
            method="PDF_TEXT_LAYER",
            source_file="original.pdf",
            confidence=0.95,
            text_chars=len("Điều 1 (cid:12) phạm vi điều chỉnh. " * 10),
        ))
        result = VBPLContentAdapter(PDFOnlyClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.HTML_VALID)
        self.assertIn("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", result.warnings)

    def test_pdf_ocr_with_longer_text_does_not_add_short_review_warning(self):
        class PDFOnlyClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

        long_text = "Điều 1. Phạm vi điều chỉnh. " * 130
        extractor = StubPDFExtractor(PDFExtractionResult(
            text=long_text,
            method="PDF_OCR",
            source_file="scan.pdf",
            confidence=0.7,
            ocr_attempted=True,
            ocr_chars=len(long_text),
        ))
        result = VBPLContentAdapter(PDFOnlyClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.HTML_VALID)
        self.assertNotIn("OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", result.warnings)
        self.assertIn("CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED", result.warnings)

    def test_quality_warning_helper_ignores_clean_pdf_text_layer(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Điều 1. Phạm vi điều chỉnh. Điều 2. Tổ chức thực hiện.",
            PDFExtractionResult(text="Điều 1. Phạm vi điều chỉnh.", method="PDF_TEXT_LAYER", text_chars=50),
        )
        self.assertEqual(warnings, [])

    def test_recovered_text_quality_metrics_counts_structure_and_suspect_tokens(self):
        text = "Điều 1 (cid:12) phạm vi. Căn cứ Luật. Khoản 1."
        metrics = recovered_text_quality_metrics(text)
        self.assertEqual(metrics.suspect_token_count, 1)
        self.assertGreater(metrics.suspect_char_density, 0)
        self.assertGreaterEqual(metrics.structure_marker_count, 3)
        self.assertGreater(metrics.vietnamese_char_ratio, 0.05)

    def test_remediate_recovered_text_preserves_raw_by_returning_separate_normalized_text(self):
        raw = "Điều 1.   Phạm vi\r\n\r\n\r\nĐiều 2."
        normalized, metadata = remediate_recovered_text(
            raw,
            PDFExtractionResult(text=raw, method="PDF_TEXT_LAYER", text_chars=len(raw)),
        )
        self.assertEqual(normalized, "Điều 1. Phạm vi\n\nĐiều 2.")
        self.assertEqual(metadata["method"], "PDF_TEXT_LAYER")
        self.assertIn("whitespace_normalized", metadata["flags"])

    def test_structure_markers_accept_ocr_text_without_diacritics(self):
        metrics = recovered_text_quality_metrics("Dieu 1. Can cu Luat. Quyet nghi.")
        self.assertGreaterEqual(metrics.structure_marker_count, 3)

    def test_trailing_hyphen_in_complete_ocr_text_is_not_marked_truncated(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Noi nhan: Luu VP-THHD.- Ch.200 -",
            PDFExtractionResult(text="Noi nhan: Luu VP-THHD.- Ch.200 -", method="PDF_OCR", ocr_attempted=True, ocr_chars=35),
        )
        self.assertNotIn("RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED", warnings)

    def test_quality_warning_helper_deduplicates_low_vietnamese_ratio_warning(self):
        text = "A" * 2100
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            text,
            PDFExtractionResult(text=text, method="PDF_OCR", ocr_attempted=True, ocr_chars=len(text)),
        )
        self.assertEqual(warnings.count("RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED"), 1)

    def test_quality_warning_helper_detects_short_ocr_and_encoding_markers(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Điều 1 (cid:12) abc",
            PDFExtractionResult(text="Điều 1 (cid:12) abc", method="PDF_OCR", ocr_attempted=True, ocr_chars=20),
        )
        self.assertIn("OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", warnings)
        self.assertIn("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", warnings)

    def test_quality_warning_helper_ignores_empty_text(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "   ",
            PDFExtractionResult(text="", method="PDF_OCR", ocr_attempted=True, ocr_chars=0),
        )
        self.assertEqual(warnings, [])

    def test_quality_warning_helper_only_adds_encoding_warning_for_text_layer(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Điều 1 � phạm vi điều chỉnh",
            PDFExtractionResult(text="Điều 1 � phạm vi điều chỉnh", method="PDF_TEXT_LAYER", text_chars=30),
        )
        self.assertEqual(
            warnings,
            [
                "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
                "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
            ],
        )

    def test_quality_warning_helper_only_adds_short_warning_for_clean_short_ocr(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Điều 1. Ngắn.",
            PDFExtractionResult(text="Điều 1. Ngắn.", method="PDF_OCR", ocr_attempted=True, ocr_chars=14),
        )
        self.assertEqual(warnings, ["OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED"])

    def test_quality_warning_helper_does_not_mark_non_recovered_methods(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Điều 1 (cid:12)",
            PDFExtractionResult(text="Điều 1 (cid:12)", method="PDF_EXTRACTION_INSUFFICIENT", text_chars=14),
        )
        self.assertEqual(warnings, [])

    def test_quality_warning_helper_detects_null_byte_encoding_artifact(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "Điều 1\x00 phạm vi điều chỉnh",
            PDFExtractionResult(text="Điều 1\x00 phạm vi điều chỉnh", method="PDF_TEXT_LAYER", text_chars=28),
        )
        self.assertIn("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", warnings)

    def test_quality_warning_helper_uses_trimmed_length_for_short_ocr_warning(self):
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            "   Điều 1. Ngắn.   ",
            PDFExtractionResult(text="   Điều 1. Ngắn.   ", method="PDF_OCR", ocr_attempted=True, ocr_chars=20),
        )
        self.assertIn("OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED", warnings)

    def test_quality_warning_helper_does_not_duplicate_clean_long_ocr_warning(self):
        long_text = "Điều 1. Phạm vi điều chỉnh. " * 130
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            long_text,
            PDFExtractionResult(text=long_text, method="PDF_OCR", ocr_attempted=True, ocr_chars=len(long_text)),
        )
        self.assertEqual(warnings, [])

    def test_quality_warning_helper_detects_both_warnings_for_short_dirty_ocr(self):
        dirty_text = "Điều 1 (cid:9) ngắn"
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            dirty_text,
            PDFExtractionResult(text=dirty_text, method="PDF_OCR", ocr_attempted=True, ocr_chars=len(dirty_text)),
        )
        self.assertEqual(
            sorted(warnings),
            sorted([
                "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
                "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
                "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
            ]),
        )

    def test_quality_warning_helper_handles_replacement_char_encoding_artifact(self):
        dirty_text = "Điều 1 � abc"
        warnings = __import__("crawler.vbpl_content_adapter", fromlist=["recovered_text_quality_warnings"]).recovered_text_quality_warnings(
            dirty_text,
            PDFExtractionResult(text=dirty_text, method="PDF_OCR", ocr_attempted=True, ocr_chars=len(dirty_text)),
        )
        self.assertIn("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED", warnings)

    def test_pdf_fallback_insufficient_keeps_pdf_only(self):
        class PDFOnlyClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

        extractor = StubPDFExtractor(PDFExtractionResult(
            text="short",
            method="PDF_EXTRACTION_INSUFFICIENT",
            source_file="original.pdf",
            confidence=0.2,
            note="insufficient_pdf_text",
            ocr_attempted=True,
            text_chars=5,
            ocr_chars=0,
        ))
        result = VBPLContentAdapter(PDFOnlyClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.PDF_ONLY)
        self.assertIn("PDF_EXTRACTION_INSUFFICIENT", result.warnings)
        self.assertEqual(result.extraction_method, "PDF_EXTRACTION_INSUFFICIENT")

    def test_pdf_fetch_failure_is_explicit(self):
        class BrokenDownloadClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = ""
                return value

            def download_pdf(self, document_id, file_name=None):
                raise VBPLHTTPError("download failed")

        extractor = StubPDFExtractor(PDFExtractionResult(
            text="",
            method="PDF_EXTRACTION_INSUFFICIENT",
            source_file="original.pdf",
            confidence=0.0,
            note="insufficient_pdf_text",
            text_chars=0,
            ocr_chars=0,
        ))
        result = VBPLContentAdapter(BrokenDownloadClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.PDF_ONLY)
        self.assertIn("PDF_FETCH_FAILED", result.warnings)
        self.assertIn("VBPLHTTPError", result.pdf_fetch_error)

    def test_content_too_short_can_be_recovered_from_pdf(self):
        class ShortClient(FakeClient):
            def detail(self, document_id):
                value = super().detail(document_id)
                value["documentContent"]["content"] = "short"
                return value

        extractor = StubPDFExtractor(PDFExtractionResult(
            text="Điều 1. " * 30,
            method="PDF_TEXT_LAYER",
            source_file="original.pdf",
            confidence=0.95,
            text_chars=len("Điều 1. " * 30),
        ))
        result = VBPLContentAdapter(ShortClient(), enable_pdf_fallback=True, pdf_extractor=extractor).fetch("abc")
        self.assertEqual(result.status, ContentStatus.HTML_VALID)
        self.assertIn("CONTENT_RECOVERED_FROM_PDF", result.warnings)

    def test_pdf_file_name_detection_prefers_pdf_suffix(self):
        name = VBPLContentAdapter._pdf_file_name([
            {"fileName": "note.txt"},
            {"objectName": "scan.PDF"},
        ])
        self.assertEqual(name, "scan.PDF")

    def test_extract_pdf_fallback_reports_missing_downloader(self):
        class NoDownloaderClient:
            def detail(self, document_id):
                return {"docNum": "01/2026/NĐ-CP", "documentContent": {"content": ""}}

            def files(self, document_id):
                return [{"fileName": "original.pdf"}]

            def search_number(self, number, page_size=50):
                return {"items": []}

        adapter = VBPLContentAdapter(NoDownloaderClient(), enable_pdf_fallback=True)
        extraction, pdf_url, pdf_error, selection = adapter._extract_pdf_fallback("abc", [{"fileName": "original.pdf"}])
        self.assertIsNone(extraction)
        self.assertIsNone(pdf_url)
        self.assertEqual(pdf_error, "PDF_DOWNLOADER_UNAVAILABLE")
        self.assertIsNone(selection)

    def test_next_action_client_download_pdf_rejects_non_pdf_http_200(self):
        class FakeHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        class FakeResponse:
            def __init__(self, body, content_type, status=200):
                self._body = body
                self.status = status
                self.headers = FakeHeaders({"content-type": content_type})

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        contract = ActionContract(detail="d" * 40, diagram="g" * 40, files="f" * 40, search="s" * 40, verified_at="2026-08-12", source_chunk="chunk")
        client = NextActionClient(contract, gateway_url="https://gateway.test")
        import urllib.request
        original = urllib.request.urlopen
        urllib.request.urlopen = lambda request, timeout=0: FakeResponse(b"<html>not pdf</html>", "text/html")
        try:
            with self.assertRaises(Exception) as ctx:
                client.download_pdf("abc", "original.pdf")
            self.assertIn("not PDF", str(ctx.exception))
        finally:
            urllib.request.urlopen = original

    def test_next_action_client_download_pdf_accepts_pdf_signature(self):
        class FakeHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        class FakeResponse:
            def __init__(self, body, content_type, status=200):
                self._body = body
                self.status = status
                self.headers = FakeHeaders({"content-type": content_type})

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        contract = ActionContract(detail="d" * 40, diagram="g" * 40, files="f" * 40, search="s" * 40, verified_at="2026-08-12", source_chunk="chunk")
        client = NextActionClient(contract, gateway_url="https://gateway.test")
        import urllib.request
        original = urllib.request.urlopen
        urllib.request.urlopen = lambda request, timeout=0: FakeResponse(b"%PDF-1.4 binary", "application/pdf")
        try:
            body, url = client.download_pdf("abc", "original.pdf")
            self.assertTrue(body.startswith(b"%PDF"))
            self.assertIn("original.pdf", url)
        finally:
            urllib.request.urlopen = original

    def test_transport_and_parse_failures_are_distinct(self):
        class ErrorClient(FakeClient):
            error_type = VBPLHTTPError

            def detail(self, document_id):
                raise self.error_type("failure")

        http_result = VBPLContentAdapter(ErrorClient()).fetch("abc")
        self.assertEqual(http_result.status, ContentStatus.HTTP_ERROR)

        ErrorClient.error_type = VBPLResponseParseError
        parse_result = VBPLContentAdapter(ErrorClient()).fetch("abc")
        self.assertEqual(parse_result.status, ContentStatus.PARSE_ERROR)

    def test_detail_doc_type_is_preferred_and_form_is_not_invented_as_group(self):
        classification = classify_official_doc_type({
            "docType": {"code": "LVB-SLe", "name": "Sắc lệnh", "parentCode": None}
        })
        self.assertEqual(classification[0], None)
        self.assertEqual(classification[2], "LVB-SLe")
        self.assertIn("MISSING_OFFICIAL_PARENT_GROUP", classification[4])

    def test_detail_parent_group_avoids_fragile_search_lookup(self):
        classification = classify_official_doc_type({
            "docNum": "15/2017/QĐ-UBND",
            "docType": {"code": "QĐ", "name": "Quyết định", "parentCode": "VBQPPL"},
        })
        self.assertEqual(classification[0], "VBQPPL")
        self.assertNotIn("DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED", classification[4])

    def test_decision_without_year_pattern_is_sent_to_review(self):
        classification = classify_official_doc_type({
            "docNum": "3410/QĐ-UBND",
            "docType": {"code": "QĐ", "name": "Quyết định", "parentCode": "VBQPPL"},
        })
        self.assertIn("DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED", classification[4])


if __name__ == "__main__":
    unittest.main()
