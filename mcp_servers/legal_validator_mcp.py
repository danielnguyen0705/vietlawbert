"""legal_validator_mcp.py — MCP Tool: thẩm định hiệu lực thời gian (check_document_status)."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from configs.global_id import classify_doc_number

_SUPERSEDE = {"THAY_THE", "BAI_BO"}
_AMEND = {"SUA_DOI_BO_SUNG"}


class CheckStatusInput(BaseModel):
    doc_number: str = Field(..., min_length=1)
    target_date: Optional[str] = Field(None, description="Mốc thời gian cần xét (YYYY-MM-DD)")


class CheckStatusOutput(BaseModel):
    doc_number: str
    status: str          # CON_HIEU_LUC | HET_HIEU_LUC | SUA_DOI | KHONG_XAC_DINH
    superseded_by: Optional[str] = None
    reason: str = ""


def check_document_status(inp: CheckStatusInput, run_query=None) -> CheckStatusOutput:
    """Thẩm định hiệu lực: số hiệu bẩn → KHONG_XAC_DINH; có THAY_THE/BAI_BO → HET_HIEU_LUC;
    có SUA_DOI_BO_SUNG → SUA_DOI; ngược lại CON_HIEU_LUC. run_query trả quan hệ 'bị tác động'."""
    status, _ = classify_doc_number(inp.doc_number)
    if status != "VALID":
        return CheckStatusOutput(doc_number=inp.doc_number, status="KHONG_XAC_DINH",
                                 reason="số hiệu không hợp lệ/không xác định")
    rows = (run_query(doc_number=inp.doc_number) if run_query else []) or []
    amend_by = None
    for r in rows:
        rt = str(r.get("rel_type") or "")
        by = str(r.get("by_doc") or r.get("neighbor_doc_number") or "")
        if rt in _SUPERSEDE:
            return CheckStatusOutput(doc_number=inp.doc_number, status="HET_HIEU_LUC",
                                     superseded_by=by, reason=f"bị {rt} bởi {by}")
        if rt in _AMEND:
            amend_by = by
    if amend_by:
        return CheckStatusOutput(doc_number=inp.doc_number, status="SUA_DOI",
                                 superseded_by=amend_by, reason=f"bị sửa đổi/bổ sung bởi {amend_by}")
    return CheckStatusOutput(doc_number=inp.doc_number, status="CON_HIEU_LUC")


def build_server():  # pragma: no cover
    from mcp.server.fastmcp import FastMCP
    from database.neo4j_client import Neo4jClient
    server = FastMCP("legal-validator")
    _c = Neo4jClient()

    def _run(doc_number: str):
        cy = ("MATCH (d:LawDocument {doc_number:$dn})<-[r:LEGAL_RELATION]-(n:LawDocument) "
              "WHERE r.type IN ['THAY_THE','BAI_BO','SUA_DOI_BO_SUNG'] "
              "RETURN r.type AS rel_type, n.doc_number AS neighbor_doc_number LIMIT 50")
        with _c.driver.session() as s:
            return [dict(x) for x in s.run(cy, dn=doc_number)]

    @server.tool()
    def check_status(doc_number: str, target_date: str = None) -> dict:
        return check_document_status(CheckStatusInput(doc_number=doc_number, target_date=target_date), _run).model_dump()
    return server


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
