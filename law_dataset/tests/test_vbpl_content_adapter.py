import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.vbpl_content_adapter import (  # noqa: E402
    ActionContract,
    ContentStatus,
    NextActionClient,
    VBPLContentAdapter,
    VBPLHTTPError,
    VBPLResponseParseError,
    classify_official_doc_type,
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
