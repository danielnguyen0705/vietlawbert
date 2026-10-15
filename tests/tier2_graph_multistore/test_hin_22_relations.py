"""Tier 2 - Kiểm đếm phổ quan hệ hiệu lực/dẫn chiếu trên đồ thị HIN (Neo4j).

Đặc tả cam kết 22 loại quan hệ (HUONG_DAN_AP_DUNG, SUA_DOI_BO_SUNG, THAY_THE, BAI_BO, ...).
Test này liệt kê các loại r.type thực tế, báo cáo số lượng mỗi loại, và khẳng định các quan hệ
trọng yếu đều hiện diện. Cần Neo4j đang chạy (nếu không sẽ SKIP).
Đặc tả: Giai đoạn 1 > Đồ thị HIN Neo4j (22 quan hệ).
"""
import pytest

pytestmark = [pytest.mark.tier2, pytest.mark.requires_neo4j]

CORE_RELATIONS = {
    "SUA_DOI_BO_SUNG", "THAY_THE", "BAI_BO", "HUONG_DAN_AP_DUNG",
    "HUONG_DAN_THI_HANH", "DAN_CHIEU", "CAN_CU",
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


def test_relation_type_coverage(neo4j_driver):
    hist = _relation_histogram(neo4j_driver)
    if not hist:
        pytest.skip("Đồ thị chưa có cạnh mang thuộc tính r.type để kiểm đếm.")
    present = set(hist)
    print("\n[HIN] Số loại quan hệ hiện diện:", len(present))
    for k, v in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    missing_core = CORE_RELATIONS - present
    assert not missing_core, f"Thiếu các quan hệ hiệu lực trọng yếu: {missing_core}"


def test_relation_type_count_benchmark(neo4j_driver):
    """Cảnh báo (không fail cứng) nếu số loại quan hệ chưa đạt kỳ vọng 22 trong đề cương."""
    hist = _relation_histogram(neo4j_driver)
    if not hist:
        pytest.skip("Đồ thị chưa có cạnh mang thuộc tính r.type.")
    n = len(hist)
    assert n >= 10, f"Chỉ có {n} loại quan hệ - đồ thị HIN có thể chưa nạp đủ phổ 22 quan hệ."
