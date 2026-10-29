"""Tier 2 - Phổ quan hệ hiệu lực/dẫn chiếu trên đồ thị HIN (Neo4j).

Danh mục quan hệ hợp lệ NẠP ĐỘNG từ crawler/system_ontology_map.json (nguồn sự thật duy nhất),
nên test không bao giờ lệch tên so với ontology thực tế.
  1. test_relation_types_are_recognized: mọi loại quan hệ trong đồ thị phải thuộc danh mục ontology
     (bắt quan hệ lạ/nhiễu).
  2. test_relation_type_coverage: đồ thị phải có đủ các quan hệ hiệu lực TRỌNG YẾU.
Cần Neo4j (nếu không sẽ SKIP). Đặc tả: Giai đoạn 1 > Đồ thị HIN 22 quan hệ.
"""
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.tier2, pytest.mark.requires_neo4j]

# Nạp danh mục edge_type hợp lệ từ ontology của hệ thống
from _common.paths import REPO_ROOT
_ONTO = REPO_ROOT / "crawler" / "system_ontology_map.json"
try:
    _data = json.loads(_ONTO.read_text(encoding="utf-8"))
    KNOWN_RELATIONS = {
        str(v).upper()
        for g in ("documentNamesByType", "documentNamesBySource")
        for v in (_data.get(g) or {}).values()
    }
except Exception:
    KNOWN_RELATIONS = {
        "BAI_BO", "BAN_DICH", "CAN_CU_BAN_HANH", "DAN_CHIEU", "DINH_CHI_THI_HANH",
        "DINH_CHINH", "HOP_NHAT", "HUONG_DAN_AP_DUNG", "QUY_DINH_CHI_TIET_HUONG_DAN",
        "SUA_DOI_BO_SUNG", "TAM_NGUNG_HIEU_LUC", "THAY_THE", "BO_SUNG", "GIAI_THICH", "CONG_BO",
    }

# Các quan hệ hiệu lực trọng yếu BẮT BUỘC có mặt (dùng đúng tên canonical của ontology)
CORE_RELATIONS = {
    "CAN_CU_BAN_HANH", "DAN_CHIEU", "THAY_THE", "BAI_BO",
    "SUA_DOI_BO_SUNG", "HUONG_DAN_AP_DUNG", "QUY_DINH_CHI_TIET_HUONG_DAN",
}


def _relation_histogram(driver) -> dict:
    q = """
    MATCH ()-[r]->()
    WHERE r.type IS NOT NULL
    RETURN toUpper(r.type) AS rtype, count(*) AS n
    ORDER BY n DESC
    """
    with driver.session() as s:
        return {row["rtype"]: row["n"] for row in s.run(q)}


def test_relation_types_are_recognized(neo4j_driver):
    """Bất biến: không có loại quan hệ lạ ngoài danh mục ontology (cho phép REL_* tạm chưa map)."""
    hist = _relation_histogram(neo4j_driver)
    if not hist:
        pytest.skip("Đồ thị chưa có cạnh mang thuộc tính r.type để kiểm đếm.")
    print("\n[HIN] Phổ quan hệ hiện có:", {k: hist[k] for k in sorted(hist, key=lambda x: -hist[x])})
    unknown = {k for k in hist if k not in KNOWN_RELATIONS and not k.startswith("REL_")}
    assert not unknown, f"Xuất hiện loại quan hệ lạ ngoài ontology: {unknown}"


def test_relation_type_coverage(neo4j_driver):
    """Đồ thị phải có đủ các quan hệ hiệu lực trọng yếu."""
    hist = _relation_histogram(neo4j_driver)
    if not hist:
        pytest.skip("Đồ thị chưa có cạnh mang thuộc tính r.type.")
    missing = CORE_RELATIONS - set(hist)
    assert not missing, f"Thiếu các quan hệ hiệu lực trọng yếu: {missing}"
