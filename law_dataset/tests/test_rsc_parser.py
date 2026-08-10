import json
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.rsc_parser import RSCParseError, parse_action_result, parse_rsc_records  # noqa: E402


class RSCParserTests(unittest.TestCase):
    def test_resolves_utf8_text_record_by_byte_length(self):
        html = "<p>Điều 1. Nội dung</p>"
        encoded = html.encode("utf-8")
        result = {"documentContent": {"content": "$2"}}
        payload = (
            b'0:["$@1",["token",null]]\n'
            + f"2:T{len(encoded):x},".encode()
            + encoded
            + b"\n1:"
            + json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        parsed = parse_action_result(payload)
        self.assertEqual(parsed["documentContent"]["content"], html)

    def test_rejects_truncated_text_record(self):
        with self.assertRaises(RSCParseError):
            parse_rsc_records(b"2:Tff,short")
