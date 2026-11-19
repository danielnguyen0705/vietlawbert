# -*- coding: utf-8 -*-
"""Unit-test appendix_download: chọn đúng zip, tải (fetch giả), set appendix_zip_path; biên không-zip/rỗng."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from crawler.appendix_download import appendix_zip_candidates, save_appendix_zip  # noqa: E402


def test_chon_va_tai_zip_set_path(tmp_path):
    files = [
        {"fileName": "TT_content.html", "presignedUrl": "http://x/html"},
        {"fileName": "TT_168_Phụ lục.zip", "presignedUrl": "http://x/zip", "fileOrder": 1},
    ]
    item = {"item_id": "DOC1"}
    saved = save_appendix_zip(files, item, str(tmp_path), fetch=lambda u: b"PK\x03\x04" + b"x" * 200)
    assert saved and os.path.exists(saved)
    assert item["appendix_zip_path"] == saved
    assert item["appendix_file_name"] == "TT_168_Phụ lục.zip"
    assert open(saved, "rb").read().startswith(b"PK")       # đúng nội dung đã tải


def test_khong_co_zip_tra_none(tmp_path):
    files = [{"fileName": "a.html", "presignedUrl": "http://x/h"}]
    item = {"item_id": "D"}
    assert save_appendix_zip(files, item, str(tmp_path), fetch=lambda u: b"x" * 500) is None
    assert "appendix_zip_path" not in item


def test_zip_rong_khong_set_path(tmp_path):
    files = [{"fileName": "p.zip", "presignedUrl": "http://x/z"}]
    item = {"item_id": "D"}
    assert save_appendix_zip(files, item, str(tmp_path), fetch=lambda u: b"") is None
    assert "appendix_zip_path" not in item


def test_candidates_bo_zip_thieu_url():
    files = [{"fileName": "p.zip"}, {"fileName": "q.zip", "presignedUrl": "http://x"}]
    cands = appendix_zip_candidates(files)
    assert len(cands) == 1 and cands[0]["fileName"] == "q.zip"
