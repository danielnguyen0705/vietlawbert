"""Kiểm tra nhanh Milvus và Neo4j bằng cấu hình từ environment."""

from __future__ import annotations

import os


def main() -> int:
    from neo4j import GraphDatabase
    from pymilvus import MilvusClient
    errors = 0
    collection = os.getenv("MILVUS_COLLECTION", "vietlaw_chunks")
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "vietlawbert")

    print("=== MILVUS ===")
    try:
        client = MilvusClient(uri=milvus_uri)
        if client.has_collection(collection):
            client.flush(collection_name=collection)
            stats = client.get_collection_stats(collection)
            print(f"collection={collection} rows={stats.get('row_count', 'N/A')}")
        else:
            print(f"collection={collection} chưa tồn tại")
    except Exception as exc:
        errors += 1
        print(f"Milvus error: {exc}")

    print("=== NEO4J ===")
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver:
            with driver.session() as session:
                counts = session.run(
                    "MATCH (n) RETURN labels(n) AS labels, count(*) AS total "
                    "ORDER BY total DESC"
                )
                for record in counts:
                    print(f"labels={record['labels']} total={record['total']}")
                relations = session.run("MATCH ()-[r]->() RETURN count(r) AS total").single()["total"]
                print(f"relationships={relations}")
    except Exception as exc:
        errors += 1
        print(f"Neo4j error: {exc}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
