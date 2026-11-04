"""
diag_graph_expand.py — Chẩn đoán 3 mắt xích của graph-expansion, tách bạch điểm đứt.
Chạy (Ubuntu, cần Neo4j+ES sống):  python -m tests._common.tools.diag_graph_expand --n 50

Với mỗi cặp bổ-trợ (A)-[BO_TRO]->(B) mà CẢ HAI doc_number đều có trong index ES:
  [M1] Traversal : GET_NEIGHBORS(A.doc_id) có trả B.doc_id với rel ∈ BO_TRO không?
  [M2] Fetch eff=True  : _search_sparse_in_docs([B.doc_id], must_be_effective=True) có ra chunk của B?
  [M3] Fetch eff=False : như trên nhưng bỏ lọc is_effective (cô lập ảnh hưởng của cờ hiệu lực).
Thống kê tỷ lệ qua từng mắt → biết đứt ở doc_id (M1/M2 cùng 0), ở cờ hiệu lực (M2≪M3), hay ở đâu.
"""
from __future__ import annotations
import argparse, unicodedata, re, collections
from configs.config import config
from neo4j import GraphDatabase
from database.cypher_templates import GET_NEIGHBORS
from rag.retriever import get_retriever

BO_TRO = ["DAN_CHIEU","CAN_CU_BAN_HANH","QUY_DINH_CHI_TIET_HUONG_DAN","HUONG_DAN_AP_DUNG","GIAI_THICH"]

def nd(s):
    s=unicodedata.normalize("NFKD",str(s or "")); s=re.sub(r"[̀-ͯ]","",s)
    return s.lower().replace("đ","d").replace(" ","")

MINE = """
MATCH (a:LawDocument)-[r:LEGAL_RELATION]->(b:LawDocument)
WHERE toUpper(r.type) IN $bo_tro AND a <> b
WITH a,b,toUpper(r.type) AS rel, rand() AS rnd ORDER BY rnd LIMIT $cand
MATCH (b)-[:HAS_CHUNK]->(cb:Chunk) WHERE size(coalesce(cb.content,''))>120
RETURN a.doc_id AS a_id, a.doc_number AS a_dn,
       b.doc_id AS b_id, b.doc_number AS b_dn, collect(cb.content)[0] AS b_text, rel
"""

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=50)
    ap.add_argument("--indexed",default="tests/_common/datasets/indexed_docs.txt"); a=ap.parse_args()
    idx=set(nd(l) for l in open(a.indexed)) if a.indexed else None
    r=get_retriever()
    drv=GraphDatabase.driver(config.NEO4J_URI,auth=(config.NEO4J_USER,config.NEO4J_PASSWORD))
    pairs=[]
    with drv.session() as s:
        for rec in s.run(MINE, bo_tro=BO_TRO, cand=a.n*40):
            if len(pairs)>=a.n: break
            a_id,a_dn,b_id,b_dn=rec["a_id"],rec["a_dn"],rec["b_id"],rec["b_dn"]
            if not (a_id and b_id and b_dn): continue
            if idx is not None and (nd(a_dn) not in idx or nd(b_dn) not in idx): continue
            pairs.append((str(a_id),a_dn,str(b_id),b_dn,(rec["b_text"] or "")[:160]))
    print(f"Số cặp (A,B) đều-trong-index để chẩn đoán: {len(pairs)}")
    if not pairs: 
        print("⚠ Không có cặp nào — nới --indexed hoặc bỏ lọc."); drv.close(); return
    m1=m2=m3=0
    with drv.session() as s:
        for a_id,a_dn,b_id,b_dn,b_text in pairs:
            nbrs={str(x["neighbor_id"]) for x in s.run(GET_NEIGHBORS,doc_id=a_id,limit=50)
                  if str(x.get("rel_type") or "").upper() in r.BO_TRO_RELATIONS}
            hit1 = b_id in nbrs
            q=b_text or b_dn
            f_eff =[h for h in r._search_sparse_in_docs(q,[b_id],top_k=20,must_be_effective=True)  if str(h.get("doc_id"))==b_id]
            f_all =[h for h in r._search_sparse_in_docs(q,[b_id],top_k=20,must_be_effective=False) if str(h.get("doc_id"))==b_id]
            m1+=hit1; m2+=bool(f_eff); m3+=bool(f_all)
    n=len(pairs)
    print(f"[M1] Traversal  A→B qua BO_TRO : {m1}/{n} ({100*m1//n}%)")
    print(f"[M2] Fetch eff=True  ra chunk B: {m2}/{n} ({100*m2//n}%)")
    print(f"[M3] Fetch eff=False ra chunk B: {m3}/{n} ({100*m3//n}%)")
    print("\nSuy luận: M1≈0 → doc_id lệch Qdrant/ES↔Neo4j (WP5.1). "
          "M2≪M3 → bộ lọc is_effective loại đích B. M1 cao & M2 cao → cơ chế OK, lỗi ở fusion/top-k.")
    drv.close()

if __name__=="__main__":
    main()
