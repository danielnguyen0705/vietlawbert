import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from crawler.probe_vbpl import (  # noqa: E402
    PageMetadataParser,
    classify_document_id,
    extract_document_id,
    legislation_from,
    parse_sitemap_index,
)


class ProbeVBPLTests(unittest.TestCase):
    def test_sitemap_comments_define_scope(self):
        xml = b"""<?xml version='1.0'?>
        <sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
          <!-- Trang tinh -->
          <sitemap><loc>https://vbpl.vn/sitemap/0.xml</loc></sitemap>
          <!-- Trung uong -->
          <sitemap><loc>https://vbpl.vn/sitemap/1.xml</loc></sitemap>
          <!-- Dia phuong -->
          <sitemap><loc>https://vbpl.vn/sitemap/13.xml</loc></sitemap>
        </sitemapindex>"""
        # ASCII fixture intentionally omits accents, so only static is classified.
        refs = parse_sitemap_index(xml)
        self.assertEqual([ref.url for ref in refs], [
            "https://vbpl.vn/sitemap/0.xml",
            "https://vbpl.vn/sitemap/1.xml",
            "https://vbpl.vn/sitemap/13.xml",
        ])

    def test_realistic_accented_scope_comments(self):
        xml = """<sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
          <!-- Trung ương -->
          <sitemap><loc>https://vbpl.vn/sitemap/1.xml</loc></sitemap>
          <!-- Địa phương -->
          <sitemap><loc>https://vbpl.vn/sitemap/13.xml</loc></sitemap>
        </sitemapindex>""".encode()
        refs = parse_sitemap_index(xml)
        self.assertEqual([ref.scope for ref in refs], ["central", "local"])

    def test_identifier_formats(self):
        self.assertEqual(classify_document_id("175440"), "numeric")
        self.assertEqual(
            classify_document_id("83021d50-94a3-11f1-8d4c-051a25b7b85b"),
            "uuid",
        )
        self.assertEqual(classify_document_id("hhtquyetdinh_228"), "legacy_prefixed")

    def test_identifier_extraction_supports_legacy_urls_without_separator(self):
        self.assertEqual(
            extract_document_id("https://vbpl.vn/van-ban/chi-tiet/vbpqdinhchinh_75"),
            "vbpqdinhchinh_75",
        )
        self.assertEqual(
            extract_document_id("https://vbpl.vn/van-ban/chi-tiet/112171"),
            "112171",
        )
        self.assertEqual(
            extract_document_id("https://vbpl.vn/van-ban/chi-tiet/example--abc_12"),
            "abc_12",
        )

    def test_legislation_json_ld(self):
        parser = PageMetadataParser()
        parser.feed(
            """<html><head><title>Example</title>
            <script type='application/ld+json'>
            {"@context":"https://schema.org","@type":"Legislation",
             "legislationIdentifier":"64/2025/QH15"}
            </script></head></html>"""
        )
        self.assertEqual(parser.title, "Example")
        legislation = legislation_from(parser)
        self.assertIsNotNone(legislation)
        self.assertEqual(legislation["legislationIdentifier"], "64/2025/QH15")


if __name__ == "__main__":
    unittest.main()
