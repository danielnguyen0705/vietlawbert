import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.vbpl_content_adapter import ActionContract, AdapterResult, ContentStatus  # noqa: E402
from crawler.verify_vbpl_contract import (  # noqa: E402
    compact_record,
    make_report,
    parse_documents,
    report_markdown,
)


class VerifyContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = ActionContract(
            detail="a" * 40,
            diagram="b" * 40,
            files="c" * 40,
            search="d" * 40,
            verified_at="2026-08-10",
            source_chunk="chunk.js",
        )

    def test_compact_record_does_not_store_raw_content(self):
        result = AdapterResult(
            document_id="1",
            status=ContentStatus.HTML_VALID,
            content_html="<p>legal text</p>",
        )
        record = compact_record(result, "fixture")
        self.assertNotIn("content_html", record)
        self.assertEqual(record["content_chars"], 17)
        self.assertEqual(len(record["content_sha256"]), 64)

    def test_contract_failure_fails_report(self):
        record = compact_record(
            AdapterResult(document_id="1", status=ContentStatus.CONTRACT_ERROR, error="changed"),
            "fixture",
        )
        self.assertFalse(make_report([record], self.contract)["passed"])

    def test_markdown_formats_warning_without_python_list_syntax(self):
        record = compact_record(
            AdapterResult(
                document_id="1",
                status=ContentStatus.HTML_VALID,
                warnings=["REVIEW_ME"],
            ),
            "fixture",
        )
        markdown = report_markdown(make_report([record], self.contract))
        self.assertIn("| REVIEW_ME |", markdown)
        self.assertNotIn("['REVIEW_ME']", markdown)

    def test_custom_document_label(self):
        self.assertEqual(parse_documents(["123:law", "abc"]), [("123", "law"), ("abc", "custom")])


if __name__ == "__main__":
    unittest.main()
