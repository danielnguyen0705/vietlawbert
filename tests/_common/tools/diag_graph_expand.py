"""
diag_graph_expand.py — Chẩn đoán 3 mắt xích graph-expansion, tách bạch điểm đứt (CPU).
Chạy:  python -m tests._common.tools.diag_graph_expand --n 50

Lọc cặp (A,B) đều-trong-index NGAY trong Cypher (WHERE doc_number IN $idx) nên tìm trúng
các cặp hiếm, thay vì bốc ngẫu nhiên rồi lọc (xác suất ~0).
  [M1] Traversal : GET_NEIGHBORS(A.doc_id) có trả B.doc_id với rel ∈ BO_TRO?
  [M2] Fetch eff=True  : _search_sparse_in_docs([B.doc_id], must_be_effective=True) ra chunk B?
  [M3] Fetch eff=False : như trên, bỏ lọc is_effective (cô lập ảnh hưởng cờ hiệu lực).
"""
from __future__ import annotations
import argparse
from configs.config import config
from neo4j import GraphDatabase
from database.cypher_templates import GET_NEIGHBORS
from rag.retriever import LegalHybridRetriever

BO_TRO = ["DAN_CHIEU","CAN_CU_BAN_HANH","QUY_DINH_CHI_TIET_HUONG_DAN","HUONG_DAN_AP_DUNG","GIAI_THICH"]

MINE = """
MATCH (a:LawDocument)-[r:LEGAL_RELATION]->(b:LawDocument)
WHERE toUpper(r.type) IN $bo_tro AND a <> b
  AND a.doc_number IN $idx AND b.doc_number IN $idx
WITH a,b,toUpper(r.type) AS rel LIMIT $cand
MATCH (b)-[:HAS_CHUNK]->(cb:Chunk) WHERE size(coalesce(cb.content,''))>120
RETURN a.doc_id AS a_id, a.doc_number AS a_dn,
       b.doc_id AS b_id, b.doc_number AS b_dn, collect(cb.content)[0] AS b_text, rel
"""

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=50)
    ap.add_argument("--indexed",default="tests/_common/datasets/indexed_docs.txt"); a=ap.parse_args()
    r=LegalHybridRetriever()
    # Lấy danh sách doc_number có trong ES TRỰC TIẾP qua client (khỏi phụ thuộc file hay bị git clean xoá)
    idx=[]
    try:
        res=r.es.search(index=config.ES_INDEX_NAME, size=0,
                        aggs={"docs":{"terms":{"field":"doc_number.keyword","size":5000}}})
        idx=[b["key"] for b in res["aggregations"]["docs"]["buckets"]]
    except Exception as exc:
        print("Agg ES lỗi (%s) -> thử đọc file." % exc)
        try: idx=[l.strip() for l in open(a.indexed) if l.strip()]
        except Exception: idx=[]
    print(f"Số doc_number trong index nạp vào bộ lọc: {len(idx)} | mẫu: {idx[:3]}")
    drv=GraphDatabase.driver(config.NEO4J_URI,auth=(config.NEO4J_USER,config.NEO4J_PASSWORD))
    pairs=[]
    with drv.session() as s:
        for rec in s.run(MINE, bo_tro=BO_TRO, idx=idx, cand=a.n*5):
            if len(pairs)>=a.n: break
            if rec["a_id"] and rec["b_id"] and rec["b_dn"]:
                pairs.append((str(rec["a_id"]),rec["a_dn"],str(rec["b_id"]),rec["b_dn"],(rec["b_text"] or "")[:160]))
    print(f"Số cặp (A,B) đều-trong-index: {len(pairs)}")
    if not pairs:
        print("⚠ Vẫn 0 — doc_number trong Neo4j và ES có thể khác định dạng (dấu/hoa-thường). "
              "Gửi tôi vài dòng mẫu của indexed_docs.txt và output: "
              "cypher 'MATCH (b:LawDocument) RETURN b.doc_number LIMIT 10'.")
        drv.close(); return
    m1=m2=m3=0
    with drv.session() as s:
        for a_id,a_dn,b_id,b_dn,b_text in pairs:
            nbrs={str(x["neighbor_id"]) for x in s.run(GET_NEIGHBORS,doc_id=a_id,limit=50)
                  if str(x.get("rel_type") or "").upper() in r.BO_TRO_RELATIONS}
            q=b_text or b_dn
            f_eff=[h for h in r._search_sparse_in_docs(q,[b_id],top_k=20,must_be_effective=True)  if str(h.get("doc_id"))==b_id]
            f_all=[h for h in r._search_sparse_in_docs(q,[b_id],top_k=20,must_be_effective=False) if str(h.get("doc_id"))==b_id]
            m1+=(b_id in nbrs); m2+=bool(f_eff); m3+=bool(f_all)
    n=len(pairs)
    print(f"[M1] Traversal  A->B qua BO_TRO : {m1}/{n} ({100*m1//n}%)")
    print(f"[M2] Fetch eff=True  ra chunk B: {m2}/{n} ({100*m2//n}%)")
    print(f"[M3] Fetch eff=False ra chunk B: {m3}/{n} ({100*m3//n}%)")
    print("\nSuy luận: M1≈0 -> doc_id lệch Qdrant/ES<->Neo4j (WP5.1). "
          "M2<<M3 -> lọc is_effective loại đích B. M1&M2 cao -> cơ chế OK, lỗi fusion/top-k.")
    drv.close()

if __name__=="__main__":
    main()
