"""legal_retrieval_mcp.py — MCP Tool: truy xuất ngữ cảnh pháp lý (search_legal_clauses)."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class SearchLegalClausesInput(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi/nhu cầu pháp lý")
    domain: Optional[str] = Field(None, description="Lĩnh vực (giao thông, thuế...)")
    mrl_dim: int = Field(256, ge=64, le=1024, description="Số chiều MRL")
    top_k: int = Field(5, ge=1, le=50)


class ClauseHit(BaseModel):
    chunk_id: str
    doc_number: str
    hierarchy_path: str
    content: str
    score: float


class SearchLegalClausesOutput(BaseModel):
    query: str
    hits: List[ClauseHit]


def search_legal_clauses(inp: SearchLegalClausesInput, retriever) -> SearchLegalClausesOutput:
    """Logic thuần: gọi retriever thật, chuẩn hoá kết quả sang schema MCP."""
    res = retriever.retrieve(inp.query, top_k=inp.top_k) or []
    hits = [ClauseHit(
        chunk_id=str(r.get("chunk_id") or r.get("id") or ""),
        doc_number=str(r.get("doc_number") or ""),
        hierarchy_path=str(r.get("hierarchy_path") or ""),
        content=str(r.get("content") or r.get("text") or "")[:1000],
        score=float(r.get("final_rerank_score") or r.get("rrf_score") or r.get("score") or 0.0),
    ) for r in res]
    return SearchLegalClausesOutput(query=inp.query, hits=hits)


def build_server():  # pragma: no cover — chỉ chạy khi có thư viện mcp
    from mcp.server.fastmcp import FastMCP
    from rag.retriever import LegalHybridRetriever
    server = FastMCP("legal-retrieval")
    _r = LegalHybridRetriever()

    @server.tool()
    def search_clauses(query: str, domain: str = None, mrl_dim: int = 256, top_k: int = 5) -> dict:
        return search_legal_clauses(
            SearchLegalClausesInput(query=query, domain=domain, mrl_dim=mrl_dim, top_k=top_k), _r
        ).model_dump()
    return server


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
