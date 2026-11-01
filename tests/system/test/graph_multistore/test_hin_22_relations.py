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
    unknown = {k for k in hist if k not in KNOWN_RELATIONS and not k.startswith("REL_") and k != "UNMAPPED"}
    assert not unknown, f"Xuất hiện loại quan hệ lạ ngoài ontology: {unknown}"

    # REL_* / UNMAPPED = quan hệ CHƯA map được về 22 loại canonical → phải là thiểu số,
    # không được tràn lan (nếu nhiều → lỗi pipelines._to_standard_relation hoặc ontology thiếu key).
    total = sum(hist.values())
    unmapped = sum(n for k, n in hist.items() if k.startswith("REL_") or k == "UNMAPPED")
    if total:
        frac = unmapped / total
        print(f"[HIN] Cạnh chưa map (REL_*/UNMAPPED): {unmapped}/{total} = {frac:.1%}")
        assert frac <= 0.05, (
            f"Quá nhiều cạnh chưa map đúng 22 quan hệ: {frac:.1%} (>5%). "
            f"Kiểm tra pipelines._to_standard_relation + system_ontology_map.json."
        )


def test_relation_type_coverage(neo4j_driver):
    """Đồ thị phải có đủ các quan hệ hiệu lực trọng yếu."""
    hist = _relation_histogram(neo4j_driver)
    if not hist:
        pytest.skip("Đồ thị chưa có cạnh mang thuộc tính r.type.")
    missing = CORE_RELATIONS - set(hist)
    assert not missing, f"Thiếu các quan hệ hiệu lực trọng yếu: {missing}"


# ─── WP2.2: Recall cạnh vs GOLD (relationships cào sẵn trong shard) ──────────
def _load_shard_gold(max_docs: int = 200):
    """Lấy gold quan hệ từ item['relationships'] trong shard thô (do crawler cào từ Lược đồ vbpl)."""
    import gzip, glob
    try:
        from configs.config import config
        shard_dir = Path(config.STORAGE_ROOT) / "raw_shards"
    except Exception:
        return None
    shards = sorted(glob.glob(str(shard_dir / "*.jsonl*")))
    shards = [f for f in shards if not f.endswith((".quarantine.jsonl", ".corrupted"))]
    if not shards:
        return None
    gold, docs = [], 0
    for f in shards:
        op = gzip.open(f, "rt", encoding="utf-8") if f.endswith(".gz") else open(f, encoding="utf-8")
        with op as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rels = d.get("relationships") or []
                if not rels:
                    continue
                src = str(d.get("doc_id") or d.get("item_id") or "")
                for rel in rels:
                    tgt = str(rel.get("target_id") or "")
                    et = str(rel.get("edge_type") or "").upper()
                    if src and tgt and et and not et.startswith("REL_"):
                        gold.append((src, tgt, et))
                docs += 1
                if docs >= max_docs:
                    return gold
    return gold


def test_edge_recall_vs_shard_gold(neo4j_driver):
    """WP2.2: so cạnh Neo4j vs gold shard → recall tổng + theo loại. Thiếu cạnh = lỗi dựng (WP2.5)."""
    gold = _load_shard_gold(max_docs=200)
    if not gold:
        pytest.skip("Không có shard raw để lấy gold quan hệ (chạy trên máy có data).")
    gold_set = {(frozenset({s, t}), ty) for s, t, ty in gold}
    src_ids = list({s for s, _, _ in gold})
    q = """
    MATCH (s:LawDocument)-[r:LEGAL_RELATION]-(t:LawDocument)
    WHERE s.doc_id IN $ids
    RETURN s.doc_id AS s, t.doc_id AS t, toUpper(r.type) AS ty
    """
    neo = set()
    with neo4j_driver.session() as sess:
        for rec in sess.run(q, ids=src_ids):
            neo.add((frozenset({str(rec["s"]), str(rec["t"])}), str(rec["ty"])))
    if not neo:
        pytest.skip("Neo4j chưa có cạnh cho các doc nguồn trong gold.")

    from collections import Counter
    tot_by, hit_by = Counter(), Counter()
    hit = 0
    for g in gold_set:
        tot_by[g[1]] += 1
        if g in neo:
            hit += 1
            hit_by[g[1]] += 1
    recall = hit / len(gold_set) if gold_set else 0.0
    print(f"\n[EDGE RECALL vs gold] {hit}/{len(gold_set)} = {recall:.1%}")
    for ty in sorted(tot_by, key=lambda x: -tot_by[x]):
        print(f"   {ty:28} {hit_by[ty]}/{tot_by[ty]} = {100*hit_by[ty]/max(1,tot_by[ty]):.0f}%")
    assert recall >= 0.6, (
        f"Recall cạnh vs gold quá thấp: {recall:.1%} (<60%) → logic dựng cạnh thiếu/sai (WP2.5). "
        f"Xem recall theo loại ở trên để biết quan hệ nào rớt."
    )
