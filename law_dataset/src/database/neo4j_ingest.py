"""
neo4j_ingest.py - CLI tool để nạp relationships từ metadata.jsonl vào Neo4j.

Usage:
    python -m database.neo4j_ingest --drop
    python -m database.neo4j_ingest --query cascade
"""

import sys
import os
import argparse
import logging

from database.neo4j_client import Neo4jManager
from database.cypher_templates import (
    get_trace_hierarchical_root,
    get_find_cascade_impact,
    get_neighbors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("Neo4jIngest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="Xóa DB trước khi nạp")
    parser.add_argument("--query", choices=["hierarchical", "cascade", "neighbors"],
                        help="Chạy query test (yêu cầu --doc-id cho hierarchical/neighbors)")
    parser.add_argument("--doc-id", help="ID văn bản cho query")
    args = parser.parse_args()

    neo = Neo4jManager(drop_existing=args.drop)

    if args.query:
        if args.query in ("hierarchical", "neighbors") and not args.doc_id:
            logger.error(f"--doc-id required for {args.query}")
            sys.exit(1)

        if args.query == "hierarchical":
            q, p = get_trace_hierarchical_root(args.doc_id)
        elif args.query == "cascade":
            q, p = get_find_cascade_impact()
        else:
            q, p = get_neighbors(args.doc_id)

        with neo.driver.session() as session:
            result = session.run(q, **p)
            for record in result:
                print(dict(record))
    else:
        # 1. Build structural graph (LawDocument -> Chapter -> Article -> Chunk)
        neo.build_structural_graph()
        
        # 2. Build semantic relationships
        items = neo.load_items_from_metadata()
        if items:
            neo.insert_semantic_relations_batch(items)

    neo.close()


if __name__ == "__main__":
    main()
