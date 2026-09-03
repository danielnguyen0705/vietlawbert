"""
cypher_templates.py - Thư viện truy vấn Cypher chuẩn hóa cho VietLawBERT.
Hỗ trợ Multi-hop Reasoning, truy vết hiệu lực thời gian và khai phá bộ ba GG-SLM.
"""

from typing import Tuple, Dict, Any

# ==============================================================================
# Kịch bản 1: Truy vết Đạo luật Gốc (Hierarchical Root Tracing)
# Đi ngược hướng INCOMING của quan hệ căn cứ ban hành từ 1 đến 3 chặng
# ==============================================================================
TRACE_HIERARCHICAL_ROOT = """
MATCH path = (sub_doc:LawDocument {doc_id: $doc_id})<-[:CAN_CU_BAN_HANH|CAN_CO_BAN_HANH*1..3]-(root_doc:LawDocument)
RETURN [node in nodes(path) | coalesce(node.name, node.title, node.doc_number)] AS Hierarchical_Chain,
       [node in nodes(path) | node.doc_id] AS Hierarchical_IDs,
       length(path) AS Depth
ORDER BY Depth DESC
"""

# ==============================================================================
# Kịch bản 2: Hiệu ứng Domino Hủy bỏ Hiệu lực (Temporal-Operational Impact)
# Phát hiện các văn bản có nguy cơ mất căn cứ pháp lý khi văn bản cha bị thay thế/bãi bỏ
# ==============================================================================
FIND_CASCADE_IMPACT = """
MATCH (luat_moi:LawDocument)-[:THAY_THE|BAI_BO|VAN_BAN_DUOC_THAY_THE]->(luat_cu:LawDocument)
OPTIONAL MATCH (affected_doc:LawDocument)-[:CAN_CU_BAN_HANH|HUONG_DAN_AP_DUNG|VAN_BAN_AP_DUNG]->(luat_cu)
RETURN coalesce(luat_moi.name, luat_moi.doc_number) AS Replacement_Doc,
       luat_moi.doc_id AS Replacement_Doc_ID,
       coalesce(luat_cu.name, luat_cu.doc_number) AS Expired_Doc,
       luat_cu.doc_id AS Expired_Doc_ID,
       collect(DISTINCT coalesce(affected_doc.name, affected_doc.doc_number)) AS At_Risk_Documents,
       count(DISTINCT affected_doc) AS Total_Affected_Count
ORDER BY Total_Affected_Count DESC
"""

# ==============================================================================
# Kịch bản 3: Mở rộng Đồ thị 1-hop xung quanh Văn bản (Graph Expansion cho RAG)
# Thu thập toàn bộ ngữ cảnh quan hệ đa tầng phục vụ Reciprocal Rank Fusion
# ==============================================================================
GET_NEIGHBORS = """
MATCH (d:LawDocument {doc_id: $doc_id})-[r]-(n:LawDocument)
RETURN type(r) AS edge_type,
       coalesce(r.graph_layer, "Operational") AS graph_layer,
       coalesce(r.direction, "UNDIRECTED") AS direction,
       n.doc_id AS neighbor_id,
       coalesce(n.name, n.title, n.doc_number, "") AS neighbor_name,
       coalesce(n.doc_number, "") AS neighbor_doc_number
LIMIT $limit
"""

# ==============================================================================
# Kịch bản 4: Khai phá Mẫu Khó GG-SLM (Graph-Guided Sentence-Law Mining)
# Đóng góp cốt lõi của bài báo: Khai thác Anchor, Positive và Hard Negative
# từ cùng cấu trúc cây phân cấp (Cùng Chương/Cùng Luật nhưng khác Điều)
# ==============================================================================
GG_SLM_TRIPLET_MINING = """
MATCH (doc:LawDocument)-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(art_pos:Article)-[:HAS_CHUNK]->(ck_pos:Chunk)
MATCH (art_pos)-[:REFERENCES]->(art_target:Article)-[:HAS_CHUNK]->(ck_anchor:Chunk)
MATCH (ch)-[:HAS_ARTICLE]->(art_neg:Article)-[:HAS_CHUNK]->(ck_neg:Chunk)
WHERE art_pos <> art_neg
  AND ck_pos.text IS NOT NULL 
  AND ck_anchor.text IS NOT NULL 
  AND ck_neg.text IS NOT NULL
RETURN ck_anchor.chunk_id AS anchor_id,
       ck_anchor.text AS anchor_text,
       ck_pos.chunk_id AS positive_id,
       ck_pos.text AS positive_text,
       ck_neg.chunk_id AS negative_id,
       ck_neg.text AS negative_text,
       doc.doc_id AS doc_id,
       ch.name AS chapter_name,
       art_pos.name AS positive_article,
       art_neg.name AS negative_article
LIMIT $limit
"""


def get_trace_hierarchical_root(doc_id: str) -> Tuple[str, Dict[str, Any]]:
    return TRACE_HIERARCHICAL_ROOT, {"doc_id": str(doc_id)}


def get_find_cascade_impact() -> Tuple[str, Dict[str, Any]]:
    return FIND_CASCADE_IMPACT, {}


def get_neighbors(doc_id: str, limit: int = 50) -> Tuple[str, Dict[str, Any]]:
    return GET_NEIGHBORS, {"doc_id": str(doc_id), "limit": int(limit)}


def get_gg_slm_triplets(limit: int = 10000) -> Tuple[str, Dict[str, Any]]:
    return GG_SLM_TRIPLET_MINING, {"limit": int(limit)}