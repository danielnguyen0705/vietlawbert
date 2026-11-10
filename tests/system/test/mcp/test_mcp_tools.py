"""Tier-2/§V: 3 MCP tool — schema Pydantic hợp lệ + logic thuần đúng (mock backend)."""
import pytest
from pydantic import ValidationError
from mcp_servers.legal_retrieval_mcp import SearchLegalClausesInput, search_legal_clauses
from mcp_servers.legal_graph_mcp import TraceLineageInput, trace_legal_lineage
from mcp_servers.legal_validator_mcp import CheckStatusInput, check_document_status


def test_retrieval_tool_maps_results():
    class _R:
        def retrieve(self, q, top_k=5):
            return [{"chunk_id": "D1_art_5", "doc_number": "88/2025/QĐ-UBND",
                     "hierarchy_path": "Điều 5", "content": "x" * 2000, "final_rerank_score": 0.9}]
    out = search_legal_clauses(SearchLegalClausesInput(query="phạt vượt đèn đỏ", top_k=3), _R())
    assert len(out.hits) == 1 and out.hits[0].doc_number == "88/2025/QĐ-UBND"
    assert len(out.hits[0].content) <= 1000 and out.hits[0].score == 0.9


def test_retrieval_schema_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchLegalClausesInput(query="")


def test_lineage_tool():
    rows = [{"rel_type": "DAN_CHIEU", "neighbor_doc_number": "100/2020/NĐ-CP", "direction": "out"},
            {"rel_type": "", "neighbor_doc_number": "bỏ", "direction": "in"}]  # dòng rỗng bị loại
    out = trace_legal_lineage(TraceLineageInput(doc_number="88/2025/QĐ-UBND"), lambda doc_number: rows)
    assert len(out.relations) == 1 and out.relations[0].rel_type == "DAN_CHIEU"


def test_validator_status_logic():
    # hết hiệu lực
    o1 = check_document_status(CheckStatusInput(doc_number="10/2010/QĐ-UBND"),
                              lambda doc_number: [{"rel_type": "THAY_THE", "neighbor_doc_number": "88/2025/QĐ-UBND"}])
    assert o1.status == "HET_HIEU_LUC" and o1.superseded_by == "88/2025/QĐ-UBND"
    # sửa đổi
    o2 = check_document_status(CheckStatusInput(doc_number="10/2010/QĐ-UBND"),
                              lambda doc_number: [{"rel_type": "SUA_DOI_BO_SUNG", "neighbor_doc_number": "20/2015/QĐ-UBND"}])
    assert o2.status == "SUA_DOI"
    # còn hiệu lực
    o3 = check_document_status(CheckStatusInput(doc_number="88/2025/QĐ-UBND"), lambda doc_number: [])
    assert o3.status == "CON_HIEU_LUC"
    # số hiệu bẩn → không xác định
    o4 = check_document_status(CheckStatusInput(doc_number="Không số"), lambda doc_number: [])
    assert o4.status == "KHONG_XAC_DINH"
