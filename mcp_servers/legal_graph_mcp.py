"""legal_graph_mcp.py — MCP Tool: truy vấn phả hệ & quan hệ văn bản (trace_legal_lineage)."""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class TraceLineageInput(BaseModel):
    doc_number: str = Field(..., min_length=1, description="Số hiệu văn bản cần truy phả hệ")
    max_hops: int = Field(1, ge=1, le=3)


class LegalRelation(BaseModel):
    rel_type: str
    neighbor_doc_number: str
    direction: str  # 'out' (văn bản này → đích) | 'in' (nguồn → văn bản này)


class TraceLineageOutput(BaseModel):
    doc_number: str
    relations: List[LegalRelation]


def trace_legal_lineage(inp: TraceLineageInput, run_query) -> TraceLineageOutput:
    """Logic thuần: run_query(doc_number) -> [{rel_type, neighbor_doc_number, direction}]."""
    rows = run_query(doc_number=inp.doc_number) or []
    rels = [LegalRelation(
        rel_type=str(r.get("rel_type") or ""),
        neighbor_doc_number=str(r.get("neighbor_doc_number") or ""),
        direction=str(r.get("direction") or "out"),
    ) for r in rows if r.get("rel_type")]
    return TraceLineageOutput(doc_number=inp.doc_number, relations=rels)


def build_server():  # pragma: no cover
    from mcp.server.fastmcp import FastMCP
    from database.neo4j_client import Neo4jClient
    server = FastMCP("legal-graph")
    _c = Neo4jClient()

    def _run(doc_number: str):
        cy = ("MATCH (d:LawDocument {doc_number:$dn})-[r:LEGAL_RELATION]-(n:LawDocument) "
              "RETURN r.type AS rel_type, n.doc_number AS neighbor_doc_number, "
              "CASE WHEN startNode(r)=d THEN 'out' ELSE 'in' END AS direction LIMIT 200")
        with _c.driver.session() as s:
            return [dict(x) for x in s.run(cy, dn=doc_number)]

    @server.tool()
    def trace_lineage(doc_number: str, max_hops: int = 1) -> dict:
        return trace_legal_lineage(TraceLineageInput(doc_number=doc_number, max_hops=max_hops), _run).model_dump()
    return server


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
