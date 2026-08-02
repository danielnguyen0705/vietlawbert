"""
cypher_templates.py - Thư viện câu truy vấn Cypher chuẩn cho Multi-hop Reasoning.

Mỗi hàm trả về (query_string, parameters_dict) để dễ test và tái sử dụng.
"""

# Kịch bản 1: Truy vết Đạo luật Gốc (Hierarchical Root Tracing)
# Đi ngược hướng INCOMING của CAN_CO_BAN_HANH từ 1-3 chặng
TRACE_HIERARCHICAL_ROOT = """
MATCH path = (thong_tu:LawDocument {doc_id: $doc_id})<-[:CAN_CO_BAN_HANH*1..3]-(luat_goc:LawDocument)
RETURN [node in nodes(path) | node.name] AS Hierarchical_Chain,
       [node in nodes(path) | node.doc_id] AS Hierarchical_IDs
"""


# Kịch bản 2: Hiệu ứng Domino Hủy bỏ Hiệu lực (Temporal-Operational Impact)
# Tìm: Văn bản MỚI thay thế văn bản CŨ + liệt kê văn bản đang áp dụng văn bản cũ
# VAN_BAN_AP_DUNG là INCOMING: (van_ban_bi_anh_huong)-[:VAN_BAN_AP_DUNG]->(luat_cu)
FIND_CASCADE_IMPACT = """
MATCH (luat_moi:LawDocument)-[:VAN_BAN_DUOC_THAY_THE]->(luat_cu:LawDocument)
MATCH (van_ban_bi_anh_huong:LawDocument)-[:VAN_BAN_AP_DUNG]->(luat_cu)
RETURN luat_moi.name AS Luat_Thay_The,
       luat_moi.doc_id AS Luat_Moi_ID,
       luat_cu.name AS Luat_Cu_Da_Chet,
       luat_cu.doc_id AS Luat_Cu_ID,
       collect(van_ban_bi_anh_huong.name) AS Danh_Sach_Nguy_Co_Mat_Hieu_Luc,
       count(van_ban_bi_anh_huong) AS So_VB_Anh_Huong
"""


# Kịch bản bổ sung: Toàn bộ quan hệ 1-hop quanh 1 văn bản
GET_NEIGHBORS = """
MATCH (d:LawDocument {doc_id: $doc_id})-[r]-(n:LawDocument)
RETURN type(r) AS edge_type,
       r.graph_layer AS graph_layer,
       r.direction AS direction,
       n.doc_id AS neighbor_id,
       n.name AS neighbor_name
"""


def get_trace_hierarchical_root(doc_id: str):
    return TRACE_HIERARCHICAL_ROOT, {"doc_id": str(doc_id)}


def get_find_cascade_impact():
    return FIND_CASCADE_IMPACT, {}


def get_neighbors(doc_id: str):
    return GET_NEIGHBORS, {"doc_id": str(doc_id)}
