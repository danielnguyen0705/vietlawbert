import os
from pymilvus import MilvusClient
from neo4j import GraphDatabase

print("==================================================")
print("       KIEM TRA CO SO DU LIEU VIETLAWBERT        ")
print("==================================================")

# 1. KIEM TRA MILVUS
print("\n[1] VECTOR DATABASE (Milvus):")
try:
    client = MilvusClient(uri="http://localhost:19530")
    if client.has_collection("vietlaw_chunks"):
        client.flush(collection_name="vietlaw_chunks")
        client.load_collection("vietlaw_chunks")
        stats = client.get_collection_stats("vietlaw_chunks")
        row_count = stats.get("row_count", "N/A")
        print(f" -> Trang thai: HOAT DONG")
        print(f" -> Tong so doan luat (Chunks da embed): {row_count}")
        
        # In thu 2 chunk
        samples = client.query(
            collection_name="vietlaw_chunks", 
            filter="chunk_id != ''", 
            output_fields=["chunk_id", "source_doc", "hierarchy", "original_text"], 
            limit=2
        )
        print("\n [Mau Chunk trong VectorDB]:")
        for s in samples:
            print(f"   * Chunker ID : {s.get('chunk_id')}")
            print(f"     Nguon      : {s.get('source_doc')}")
            print(f"     Phan cap   : {s.get('hierarchy')}")
            print(f"     Noi dung   : {s.get('original_text', '')[:100]}...\n")
    else:
        print(" -> Collection 'vietlaw_chunks' chua duoc tao.")
except Exception as e:
    print(f" -> Loi Milvus: {e}")

# 2. KIEM TRA NEO4J
print("[2] KNOWLEDGE GRAPH DATABASE (Neo4j):")
try:
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "vietlawbert"))
    with driver.session() as session:
        doc_count = session.run("MATCH (n:LawDocument) RETURN count(n) AS c").single()["c"]
        chap_count = session.run("MATCH (n:Chapter) RETURN count(n) AS c").single()["c"]
        art_count = session.run("MATCH (n:Article) RETURN count(n) AS c").single()["c"]
        chunk_count = session.run("MATCH (n:Chunk) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        print(f" -> So Van Ban (LawDocument)     : {doc_count}")
        print(f" -> So Chuong (Chapter)          : {chap_count}")
        print(f" -> So Dieu luat (Article)       : {art_count}")
        print(f" -> So Doan trich (Chunk)        : {chunk_count}")
        print(f" -> Tong so Moi quan he (Edges)  : {rel_count}")

        # In thu quan he giua cac van ban
        print("\n [Mau Quan he do thi (Knowledge Graph)]:")
        records = session.run("""
            MATCH (s:LawDocument)-[r]->(t:LawDocument)
            RETURN s.name AS Source, type(r) AS Relation, t.name AS Target
            LIMIT 3
        """)
        found = False
        for rec in records:
            found = True
            print(f"   * [{rec['Source']}] --({rec['Relation']})--> [{rec['Target']}]")
        if not found:
            print("   * Chua co quan he lien ket van ban nao hoac dang su dung cau truc phan cap.")

except Exception as e:
    print(f" -> Loi Neo4j: {e}")
print("==================================================")


