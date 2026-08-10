import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.audit_vbpl import (  # noqa: E402
    classify_document_group,
    metadata_disposition,
    quality_warnings,
)


class AuditVBPLTests(unittest.TestCase):
    def test_document_group_classification(self):
        self.assertEqual(
            classify_document_group({"legislationType": "Luật"}, "", "175440"),
            "LEGAL_FORM_CANDIDATE",
        )
        self.assertEqual(
            classify_document_group({"legislationType": "Văn bản hợp nhất"}, "", "112171"),
            "CONSOLIDATED",
        )
        self.assertEqual(
            classify_document_group({}, "Văn bản hành chính liên quan", "vbpqdinhchinh_75"),
            "CORRECTION",
        )

    def test_quality_warnings_do_not_treat_http_200_as_complete(self):
        page = {
            "ok": True,
            "url": "https://vbpl.vn/van-ban/chi-tiet/1",
            "canonical": "https://vbpl.vn/van-ban/chi-tiet/1",
            "json_ld_errors": 0,
            "legislation": {"legislationType": "Luật"},
        }
        warnings = quality_warnings(page)
        self.assertIn("MISSING_NAME", warnings)
        self.assertIn("MISSING_IDENTIFIER", warnings)
        self.assertIn("MISSING_DATE", warnings)
        self.assertIn("MISSING_AGENCY", warnings)

    def test_disposition_requires_content_check(self):
        self.assertEqual(
            metadata_disposition("LEGAL_FORM_CANDIDATE", []),
            "GROUP_AND_CONTENT_CHECK_REQUIRED",
        )
        self.assertEqual(
            metadata_disposition("UNKNOWN", []),
            "MANUAL_REVIEW",
        )


if __name__ == "__main__":
    unittest.main()
