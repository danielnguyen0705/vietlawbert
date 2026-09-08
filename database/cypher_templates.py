"""
cypher_templates.py - Thư viện truy vấn Cypher chuẩn hóa cho mạng thông tin dị thể (HIN).
Khớp chính xác với cấu trúc: LawDocument -> HAS_CHUNK -> Chunk và 22 loại quan hệ pháp lý.
"""

from typing import Tuple, Dict, Any

# ==============================================================================
# Kịch bản 1: Truy vết Căn cứ Ban hành (Hierarchical Root Tracing)
# ==============================================================================
TRACE_HIERARCHICAL_ROOT = """
MATCH path = (sub_doc:LawDocument {doc_id: $doc_id})<-[:LEGAL_RELATION {type: "CAN_CU_BAN_HANH"}*1..3]-(root_doc:LawDocument)
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
WHERE r.type IN ["THAY_THE", "BAI_BO", "SUA_DOI_BO_SUNG"]
OPTIONAL MATCH (affected_doc:LawDocument)-[:LEGAL_RELATION {type: "HUONG_DAN_CHI_TIET"}]->(luat_cu)
RETURN coalesce(luat_moi.doc_number, luat_moi.title) AS Replacement_Doc,
       luat_moi.doc_id AS Replacement_Doc_ID,
       r.type AS Impact_Type,
       coalesce(luat_cu.doc_number, luat_cu.title) AS Target_Doc,
       luat_cu.doc_id AS Target_Doc_ID,
       collect(DISTINCT coalesce(affected_doc.doc_number, affected_doc.title)) AS At_Risk_Documents,
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
# Kịch bản 4: Khai phá Mẫu Khó HIN-Guided Contrastive Mining (RQ1 Benchmark)
# Anchor và Positive liên kết qua quan hệ pháp lý, Negative chung từ vựng nhưng khác nhánh HIN
# ==============================================================================
HIN_TRIPLET_MINING = """
MATCH (d_anchor:LawDocument)-[r:LEGAL_RELATION]->(d_pos:LawDocument)
MATCH (d_anchor)-[:HAS_CHUNK]->(ck_anchor:Chunk)
MATCH (d_pos)-[:HAS_CHUNK]->(ck_pos:Chunk)
MATCH (d_neg:LawDocument)-[:HAS_CHUNK]->(ck_neg:Chunk)
WHERE d_neg.doc_id <> d_anchor.doc_id 
  AND d_neg.doc_id <> d_pos.doc_id
  AND NOT (d_anchor)-[:LEGAL_RELATION]-(d_neg)
  AND ck_anchor.macro_label = ck_neg.macro_label
RETURN ck_anchor.chunk_id AS anchor_id,
       ck_anchor.content AS anchor_text,
       ck_pos.chunk_id AS positive_id,
       ck_pos.content AS positive_text,
       ck_neg.chunk_id AS negative_id,
       ck_neg.content AS negative_text,
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