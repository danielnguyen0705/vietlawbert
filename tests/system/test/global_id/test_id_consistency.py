"""
Tầng-2 — Bất biến KHÓA TOÀN CỤC (chạy SAU khi dựng lại kho; cổng "done").
Mục tiêu: chống trùng node + doc_id tất định + đồng nhất Neo4j↔ES.
Test SKIP nếu không kết nối được service; FAIL (đúng ý đồ) nếu service sống mà dữ liệu CHƯA dựng lại.

Chạy:  pytest -m tier2 tests/system/test/global_id/test_id_consistency.py -s
"""
import importlib.util, pathlib
import pytest

pytestmark = pytest.mark.tier2
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_g = importlib.util.spec_from_file_location("global_id", _ROOT / "configs" / "global_id.py")
G = importlib.util.module_from_spec(_g); _g.loader.exec_module(G)


def _neo4j():
    try:
        from configs.config import config
        from neo4j import GraphDatabase
        d = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                                 connection_acquisition_timeout=8.0)
        d.verify_connectivity(); return d
    except Exception as e:
        pytest.skip(f"Neo4j không kết nối: {e}")


def test_no_duplicate_nodes_per_doc_number():
    """Sau rebuild: số node ≈ số doc_number phân biệt (không nhân bản). Tỷ lệ >1.05 là còn trùng."""
    d = _neo4j()
    with d.session() as s:
        r = s.run("MATCH (x:LawDocument) WHERE x.doc_number IS NOT NULL AND x.doc_number <> 'N/A' "
                  "RETURN count(x) AS n, count(DISTINCT x.doc_number) AS dn").single()
    d.close()
    n, dn = r["n"], r["dn"]
    ratio = n / max(dn, 1)
    print(f"\n[no-dup] node={n} distinct doc_number={dn} ratio={ratio:.3f}")
    assert ratio <= 1.05, f"Còn node trùng: {n} node / {dn} doc_number (ratio {ratio:.2f}); cần dựng lại bằng khóa canonical"


def test_docid_is_canonical_of_docnumber():
    """doc_id trên node phải khớp canonical_doc_id(doc_number)."""
    d = _neo4j()
    with d.session() as s:
        rows = list(s.run("MATCH (x:LawDocument) WHERE x.doc_number IS NOT NULL AND x.doc_number <> 'N/A' "
                          "RETURN x.doc_id AS id, x.doc_number AS dn LIMIT 300"))
    d.close()
    if not rows:
        pytest.skip("Chưa có LawDocument.")
    bad = [(r["id"], r["dn"]) for r in rows if r["id"] != G.canonical_doc_id(r["dn"])]
    frac = len(bad) / len(rows)
    print(f"\n[canonical] lệch {len(bad)}/{len(rows)} ({frac:.1%}); ví dụ: {bad[:3]}")
    assert frac <= 0.02, "doc_id chưa canonical theo doc_number (cần dựng lại Neo4j)"


def test_crossstore_docid_overlap_neo4j_es():
    """doc_id của Neo4j phải tồn tại trong ES (đồng nhất khóa xuyên kho)."""
    try:
        from configs.config import config
        from elasticsearch import Elasticsearch
        es = Elasticsearch(config.ES_HOST)
        if not es.ping(): pytest.skip("ES không ping được.")
    except Exception as e:
        pytest.skip(f"ES không kết nối: {e}")
    d = _neo4j()
    with d.session() as s:
        ids = [r["id"] for r in s.run(
            "MATCH (x:LawDocument) WHERE x.doc_number IS NOT NULL AND x.doc_number<>'N/A' "
            "RETURN x.doc_id AS id LIMIT 100")]
    d.close()
    if not ids: pytest.skip("Chưa có LawDocument.")
    hit = sum(1 for i in ids
              if es.count(index=config.ES_INDEX_NAME, query={"term": {"doc_id": i}})["count"] > 0)
    frac = hit / len(ids)
    print(f"\n[cross-store] doc_id Neo4j có trong ES: {hit}/{len(ids)} ({frac:.1%})")
    # Lưu ý: còn phụ thuộc độ phủ ES; ngưỡng thấp để chỉ bắt lỗi KHÓA, không phải độ phủ
    assert frac >= 0.30, "Khóa doc_id Neo4j↔ES vẫn lệch (không phải do độ phủ): cần dựng lại cùng canonical"
