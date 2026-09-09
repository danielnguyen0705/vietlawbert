"""
cypher_templates.py - Thư viện truy vấn Cypher chuẩn hóa cho mạng thông tin dị thể (HIN).
Khớp chính xác với cấu trúc: LawDocument -> HAS_CHUNK -> Chunk và 22 loại quan hệ pháp lý.
Tối ưu hóa tránh tích Descartes và hỗ trợ khai phá bộ ba mẫu khó đối lập (GG-SLM Triplet Mining).
"""

from typing import Tuple, Dict, Any

# ==============================================================================
# Kịch bản 1: Truy vết Căn cứ Ban hành (Hierarchical Root Tracing)
# ==============================================================================
TRACE_HIERARCHICAL_ROOT = """
MATCH path = (sub_doc:LawDocument {doc_id: $doc_id})<-[:LEGAL_RELATION*1..3]-(root_doc:LawDocument)
WHERE ALL(r IN relationships(path) WHERE r.type IN ["CAN_CU_BAN_HANH", "CAN_CU"])
RETURN [node in nodes(path) | coalesce(node.doc_number, node.title, node.doc_id)] AS Hierarchical_Chain,
       [node in nodes(path) | node.doc_id] AS Hierarchical_IDs,
       length(path) AS Depth
ORDER BY Depth DESC
"""

# ==============================================================================
# Kịch bản 2: Hiệu ứng Domino Hủy bỏ Hiệu lực (Temporal Cascade Impact)
# ==============================================================================
FIND_CASCADE_IMPACT = """
MATCH (luat_moi:LawDocument)-[r:LEGAL_RELATION]->(luat_cu:LawDocument)
WHERE r.type IN ["THAY_THE", "BAI_BO", "SUA_DOI_BO_SUNG", "SUA_DOI", "BO_SUNG"]
OPTIONAL MATCH (affected_doc:LawDocument)-[r_aff:LEGAL_RELATION]->(luat_cu)
WHERE r_aff.type IN ["HUONG_DAN_CHI_TIET", "HUONG_DAN", "DAN_CHIEU"]
RETURN coalesce(luat_moi.doc_number, luat_moi.title, luat_moi.doc_id) AS Replacement_Doc,
       luat_moi.doc_id AS Replacement_Doc_ID,
       r.type AS Impact_Type,
       coalesce(luat_cu.doc_number, luat_cu.title, luat_cu.doc_id) AS Target_Doc,
       luat_cu.doc_id AS Target_Doc_ID,
       collect(DISTINCT coalesce(affected_doc.doc_number, affected_doc.title, affected_doc.doc_id)) AS At_Risk_Documents,
       count(DISTINCT affected_doc) AS Total_Affected_Count
ORDER BY Total_Affected_Count DESC
LIMIT $limit
"""

# ==============================================================================
# Kịch bản 3: Mở rộng Đồ thị 1-hop xung quanh Văn bản (Graph Context Expansion)
# ==============================================================================
GET_NEIGHBORS = """
MATCH (d:LawDocument {doc_id: $doc_id})-[r:LEGAL_RELATION]-(n:LawDocument)
RETURN r.type AS rel_type,
       n.doc_id AS neighbor_id,
       coalesce(n.doc_number, "N/A") AS neighbor_doc_number,
       coalesce(n.title, "") AS neighbor_title
LIMIT $limit
"""

# ==============================================================================
# Kịch bản 4: Khai phá Mẫu Khó HIN-Guided Contrastive Mining (Chuẩn Reviewer Q1)
# Tối ưu hóa WITH-LIMIT chặn tắc nghẽn bộ nhớ khi đồ thị vượt triệu đỉnh
# ==============================================================================
HIN_TRIPLET_MINING = """
MATCH (d_anchor:LawDocument)-[r:LEGAL_RELATION]->(d_pos:LawDocument)
MATCH (d_anchor)-[:HAS_CHUNK]->(ck_anchor:Chunk)
MATCH (d_pos)-[:HAS_CHUNK]->(ck_pos:Chunk)
WITH d_anchor, d_pos, ck_anchor, ck_pos
LIMIT $limit
MATCH (d_neg:LawDocument)-[:HAS_CHUNK]->(ck_neg:Chunk {macro_label: ck_anchor.macro_label})
WHERE d_neg.doc_id <> d_anchor.doc_id 
  AND d_neg.doc_id <> d_pos.doc_id
  AND NOT (d_anchor)-[:LEGAL_RELATION]-(d_neg)
RETURN ck_anchor.chunk_id AS anchor_id,
       coalesce(ck_anchor.content, ck_anchor.text, "") AS anchor_text,
       ck_pos.chunk_id AS positive_id,
       coalesce(ck_pos.content, ck_pos.text, "") AS positive_text,
       ck_neg.chunk_id AS negative_id,
       coalesce(ck_neg.content, ck_neg.text, "") AS negative_text,
       ck_anchor.macro_label AS hierarchy_label
LIMIT $limit
"""


def get_trace_hierarchical_root(doc_id: str) -> Tuple[str, Dict[str, Any]]:
    return TRACE_HIERARCHICAL_ROOT, {"doc_id": str(doc_id)}


def get_find_cascade_impact(limit: int = 50) -> Tuple[str, Dict[str, Any]]:
    return FIND_CASCADE_IMPACT, {"limit": int(limit)}


def get_neighbors(doc_id: str, limit: int = 50) -> Tuple[str, Dict[str, Any]]:
    return GET_NEIGHBORS, {"doc_id": str(doc_id), "limit": int(limit)}


def get_hin_triplets(limit: int = 10000) -> Tuple[str, Dict[str, Any]]:
    return HIN_TRIPLET_MINING, {"limit": int(limit)}