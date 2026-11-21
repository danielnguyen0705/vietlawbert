# -*- coding: utf-8 -*-
"""Unit-test Document Store (Mongo): build_doc_record trích đủ metadata (gồm NGƯỜI KÝ) + round-trip.

build_doc_record là hàm THUẦN (không cần service) → test luôn chạy.
Phần MongoDocStore round-trip chỉ chạy khi có Mongo (skip nếu không kết nối được).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from database.mongo_client import build_doc_record, MongoDocStore  # noqa: E402

_REC = {
    "doc_id": "D1",
    "doc_number": "105/2026/QĐ-UBND",
    "title": "Quyết định ban hành quy định ...",
    "metadata_detail": {
        "docType": {"name": "Quyết định"},
        "docGroup": "VBQPPL",
        "agencyName": "UBND Thành phố Cần Thơ",
        "documentIssues": [{"personName": "Trần Chí Hùng", "jobTitleName": "Phó Chủ tịch"}],
        "issueDate": "2026-11-09T00:00:00",
        "effFrom": "2026-11-20T00:00:00",
        "effTo": None,
        "effStatus": {"name": "Còn hiệu lực"},
        "documentFields": [{"name": "Bảo vệ môi trường"}],
        "documentMajors": [{"name": "Nông nghiệp và Môi trường"}],
    },
}


def test_build_doc_record_trich_du_truong_va_nguoi_ky():
    r = build_doc_record(_REC, "D1", "TOÀN VĂN ...")
    assert len(r) == 15
    assert r["signer"] == "Trần Chí Hùng" and r["signer_title"] == "Phó Chủ tịch"
    assert r["doc_type"] == "Quyết định"
    assert r["issuer"] == "UBND Thành phố Cần Thơ"
    assert r["eff_status"] == "Còn hiệu lực"
    assert r["legal_fields"] == ["Bảo vệ môi trường"]
    assert r["issue_date"] == "2026-11-09" and r["effective_from"] == "2026-11-20"
    assert r["effective_to"] is None   # null CÓ nghĩa = vô thời hạn, không bịa
    assert r["full_text"] == "TOÀN VĂN ..."


def test_build_doc_record_gop_space_thua_trong_nguoi_ky():
    rec = dict(_REC)
    rec["metadata_detail"] = dict(_REC["metadata_detail"])
    rec["metadata_detail"]["documentIssues"] = [{"personName": "Lâm  Văn Bi", "jobTitleName": "Phó  Chủ tịch"}]
    r = build_doc_record(rec, "D3", "")
    assert r["signer"] == "Lâm Văn Bi" and r["signer_title"] == "Phó Chủ tịch"  # gộp space thừa


def test_build_doc_record_truong_thieu_thi_null_khong_bia():
    bare = {"doc_id": "D2", "doc_number": "02/2026", "metadata_detail": {}}
    r = build_doc_record(bare, "D2", "")
    assert r["signer"] is None and r["doc_type"] is None and r["legal_fields"] == []
    assert r["doc_number"] == "02/2026"


def _mongo_or_skip():
    try:
        s = MongoDocStore(collection="documents_pytest")
        s.col.database.client.admin.command("ping")
        return s
    except Exception as e:
        pytest.skip(f"Mongo không sẵn sàng: {str(e)[:60]}")


def test_mongo_upsert_get_roundtrip():
    store = _mongo_or_skip()
    try:
        store.wipe()
        rec = build_doc_record(_REC, "D1", "toàn văn")
        store.upsert_doc(rec)
        store.upsert_doc(rec)   # idempotent: upsert lại không nhân đôi
        assert store.count() == 1
        got = store.get_doc("D1")
        assert got["signer"] == "Trần Chí Hùng" and got["full_text"] == "toàn văn"
        assert store.get_doc("KHONG_TON_TAI") is None
    finally:
        store.wipe()
        store.close()
