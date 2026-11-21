"""
mongo_client.py - Document Store (MongoDB) cho VietLawBERT.
Lưu TOÀN VĂN + bộ metadata hành chính theo doc_id, để tầng tạo sinh bốc tách căn cứ
chính xác (số hiệu, cơ quan, NGƯỜI KÝ, tình trạng hiệu lực, lĩnh vực) khi trả lời.
Khóa doc_id dùng chung với Qdrant/Neo4j → 4 CSDL join được qua 1 định danh.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from configs.config import config
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("database", "mongo_client")


def _name(v: Any) -> str:
    """Lấy nhãn người-đọc từ field vbpl có thể là dict {'name':...} HOẶC chuỗi thẳng."""
    if isinstance(v, dict):
        return str(v.get("name") or v.get("value") or "").strip()
    return str(v or "").strip()


def _names(v: Any) -> List[str]:
    """Danh sách nhãn từ field list-of-dict (documentFields/documentMajors)."""
    if isinstance(v, list):
        return [n for n in (_name(x) for x in v) if n]
    n = _name(v)
    return [n] if n else []


def build_doc_record(doc_record: Dict[str, Any], doc_id: str, full_text: str) -> Dict[str, Any]:
    """Dựng 1 document Mongo = toàn văn + metadata hành chính, trích từ metadata_detail (fallback _api).
    Ô trống → null (CÓ nghĩa, vd effective_to null = vô thời hạn), không bịa."""
    md = doc_record.get("metadata_detail") or {}
    ma = doc_record.get("metadata_api") or {}

    issues = md.get("documentIssues") or []
    first_issue = issues[0] if isinstance(issues, list) and issues else {}
    _clean = lambda s: re.sub(r"\s+", " ", str(s or "")).strip()   # gộp space thừa ("Phó  Chủ tịch")
    signer = _clean(first_issue.get("personName")) or None
    signer_title = _clean(first_issue.get("jobTitleName")) or None

    def _date(*vals):
        for v in vals:
            s = str(v or "").strip()
            if s:
                return s[:10] if "T" in s else s   # ISO 'YYYY-MM-DDT..' → 'YYYY-MM-DD'
        return None

    return {
        "doc_id": doc_id,
        "doc_number": str(doc_record.get("doc_number") or md.get("docNum") or ma.get("docNum") or "N/A"),
        "doc_type": _name(md.get("docType") or ma.get("docType")) or None,
        "doc_group": str(md.get("docGroup") or "").strip() or None,
        "title": str(doc_record.get("title") or md.get("title") or ma.get("title") or "").strip() or None,
        "issuer": str(md.get("agencyName") or ma.get("agencyName") or "").strip() or None,
        "signer": signer,
        "signer_title": signer_title,
        "issue_date": _date(md.get("issueDate"), ma.get("issueDate")),
        "effective_from": _date(md.get("effFrom"), ma.get("effFrom")),
        "effective_to": _date(md.get("effTo"), ma.get("effTo")),          # null = vô thời hạn
        "eff_status": _name(md.get("effStatus") or ma.get("effStatus")) or None,
        "legal_fields": _names(md.get("documentFields")),
        "legal_majors": _names(md.get("documentMajors")),
        "full_text": full_text,
    }


class MongoDocStore:
    """Tầng truy cập Document Store. Chịu lỗi: nếu Mongo chết, loader/generator vẫn chạy (degrade mềm)."""

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None, collection: str = "documents"):
        self.client = MongoClient(uri or config.MONGO_URI, serverSelectionTimeoutMS=5000)
        self.col = self.client[db_name or config.MONGO_DB_NAME][collection]
        self.col.create_index("doc_id", unique=True)
        logger.info("MongoDocStore sẵn sàng: %s.%s", db_name or config.MONGO_DB_NAME, collection)

    def upsert_doc(self, record: Dict[str, Any]) -> None:
        self.col.replace_one({"doc_id": record["doc_id"]}, record, upsert=True)

    def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if not doc_id:
            return None
        return self.col.find_one({"doc_id": doc_id}, {"_id": 0})

    def count(self) -> int:
        return self.col.estimated_document_count()

    def wipe(self) -> None:
        self.col.delete_many({})
        logger.warning("Đã XÓA SẠCH collection Mongo [%s].", self.col.name)

    def close(self) -> None:
        self.client.close()
