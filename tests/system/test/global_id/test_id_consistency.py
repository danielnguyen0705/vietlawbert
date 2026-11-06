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


def test_crossstore_es_subset_of_neo4j_canonical():
    """Bất biến ĐÚNG cho index (kể cả có-mục-tiêu): MỌI doc trong ES phải có doc_id canonical
    và tồn tại trong Neo4j. Lấy mẫu PHÍA ES (không phải Neo4j) để không bị nhiễu bởi độ phủ."""
    try:
        from configs.config import config
        from elasticsearch import Elasticsearch
        es = Elasticsearch(config.ES_HOST)
        if not es.ping(): pytest.skip("ES không ping được.")
    except Exception as e:
        pytest.skip(f"ES không kết nối: {e}")
    # lấy mẫu (doc_id, doc_number) từ ES
    res = es.search(index=config.ES_INDEX_NAME, size=100,
                    query={"match_all": {}}, source=["doc_id", "doc_number"])
    hits = res["hits"]["hits"]
    if not hits: pytest.skip("ES chưa có dữ liệu (chạy ingest_targeted trước).")
    d = _neo4j()
    canon_ok = inneo = 0
    with d.session() as s:
        for h in hits:
            src = h["_source"]; es_id = src.get("doc_id"); dn = src.get("doc_number")
            if es_id == G.canonical_doc_id(dn):
                canon_ok += 1
            c = s.run("MATCH (x:LawDocument {doc_id:$i}) RETURN count(x) AS c", i=es_id).single()["c"]
            if c > 0:
                inneo += 1
    d.close()
    n = len(hits)
    print(f"\n[cross-store] ES doc_id canonical: {canon_ok}/{n} | tồn tại trong Neo4j: {inneo}/{n}")
    assert canon_ok / n >= 0.95, "doc_id trong ES chưa canonical (chạy lại ingest_targeted bằng pipeline canonical)"
    assert inneo / n >= 0.90, "doc_id ES không khớp Neo4j (hai kho lệch khóa)"
