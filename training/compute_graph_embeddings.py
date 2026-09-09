"""
compute_graph_embeddings.py - Động cơ tiền tính toán Vector Đồ thị Ngoại tuyến (Compile-time).
Trích xuất 22 quan hệ pháp lý từ Neo4j -> Node2Vec 128d (Biased Random Walk) -> Lưu Parquet & Redis Cache.
"""

from __future__ import annotations

import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import pandas as pd
from neo4j import GraphDatabase
from gensim.models import Word2Vec

from configs.config import config
from configs.paths import ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_GraphEmbedder")


class Neo4jGraphExtractor:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password), connection_acquisition_timeout=10.0)
        self.driver.verify_connectivity()
        logger.info("Kết nối Neo4j thành công tại %s.", self.uri)

    def close(self):
        self.driver.close()

    def fetch_legal_edges(self) -> List[Tuple[str, str, str]]:
        """Trích xuất các cạnh phân cấp HAS_CHUNK và 22 quan hệ liên văn bản."""
        cypher = """
        MATCH (s)-[r]->(t)
        WHERE (s:LawDocument OR s:Chunk) AND (t:LawDocument OR t:Chunk)
        RETURN coalesce(s.chunk_id, s.doc_id) AS u,
               coalesce(r.type, type(r)) AS rel_type,
               coalesce(t.chunk_id, t.doc_id) AS v
        LIMIT 500000
        """
        edges = []
        with self.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                u, v, rel = record["u"], record["v"], record["rel_type"]
                if u and v and u != v:
                    edges.append((str(u), str(v), str(rel)))

        logger.info("✓ Đã trích xuất %d cạnh quan hệ pháp lý từ Neo4j.", len(edges))
        return edges


class Node2VecRandomWalker:
    """Hiện thực hóa giải thuật duyệt ngẫu nhiên bậc 2 có trọng số Node2Vec (Grover & Leskovec, 2016)."""

    def __init__(self, edges: List[Tuple[str, str, str]], p: float = 1.0, q: float = 0.5):
        self.adj = defaultdict(list)
        for u, v, _ in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        # Loại bỏ các đỉnh trùng lặp trong danh sách kề
        for node in self.adj:
            self.adj[node] = list(set(self.adj[node]))

        self.nodes = list(self.adj.keys())
        self.p = p  # Return parameter
        self.q = q  # In-out parameter (q=0.5 kích thích duyệt DFS sâu vào đồ thị)
        logger.info("Khởi tạo cấu trúc đồ thị: %d nút phân biệt.", len(self.nodes))

    def _get_next_step(self, prev_node: str, curr_node: str) -> str:
        neighbors = self.adj.get(curr_node, [])
        if not neighbors:
            return curr_node

        if len(neighbors) == 1:
            return neighbors[0]

        prev_neighbors = set(self.adj.get(prev_node, []))
        weights = []

        for nbr in neighbors:
            if nbr == prev_node:
                weights.append(1.0 / self.p)
            elif nbr in prev_neighbors:
                weights.append(1.0)
            else:
                weights.append(1.0 / self.q)

        return random.choices(neighbors, weights=weights, k=1)[0]

    def generate_walks(self, num_walks: int = 10, walk_length: int = 40) -> List[List[str]]:
        walks = []
        for walk_iter in range(num_walks):
            random.shuffle(self.nodes)
            for node in self.nodes:
                walk = [node]
                curr_neighbors = self.adj.get(node, [])
                if not curr_neighbors:
                    continue
                walk.append(random.choice(curr_neighbors))

                while len(walk) < walk_length:
                    next_node = self._get_next_step(walk[-2], walk[-1])
                    if next_node == walk[-1]:
                        break
                    walk.append(next_node)
                walks.append(walk)

        return walks


def sync_embeddings_to_redis(records: List[Dict[str, Any]]) -> None:
    """Nạp vector nhị phân vào Redis để phục vụ truy vấn thời gian thực dưới 500ms."""
    try:
        import redis
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, socket_timeout=5.0)
        pipe = r.pipeline(transaction=False)
        for item in records:
            pipe.set(f"graph_emb:{item['chunk_id']}", json.dumps(item["graph_embedding"]))
            if len(pipe) >= 5000:
                pipe.execute()
        pipe.execute()
        r.close()
        logger.info("✓ Đã nạp thành công %d vector đồ thị vào Redis In-Memory Cache.", len(records))
    except Exception as exc:
        logger.warning("Bỏ qua đồng bộ Redis (%s). Hệ thống vẫn lưu trữ qua tệp Parquet.", exc)


def compute_and_export_embeddings(
    output_path: str,
    dimensions: int = 128,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    extractor = Neo4jGraphExtractor(uri=uri, user=user, password=password)
    edges = extractor.fetch_legal_edges()
    extractor.close()

    if not edges:
        logger.warning("Đồ thị Neo4j chưa có cạnh. Tạo khung cạnh giả lập để bảo toàn luồng...")
        edges = [(f"doc_{i}", f"chunk_{i}", "HAS_CHUNK") for i in range(100)]

    walker = Node2VecRandomWalker(edges, p=1.0, q=0.5)
    walks = walker.generate_walks(num_walks=10, walk_length=40)

    logger.info("Huấn luyện Word2Vec Skip-Gram trên %d đường đi ngẫu nhiên...", len(walks))
    w2v = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=5,
        min_count=1,
        sg=1,
        workers=2,
        epochs=5,
    )

    records = []
    for node in walker.nodes:
        if node in w2v.wv:
            vec = w2v.wv[node].tolist()
            records.append({"chunk_id": node, "graph_embedding": vec})

    df = pd.DataFrame(records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, engine="pyarrow", compression="snappy")
    logger.info("✓ Đã lưu %d vector đồ thị (%dd) vào Parquet: %s", len(df), dimensions, out_file.resolve())

    # Đồng bộ sang Redis Cache
    sync_embeddings_to_redis(records)


def main():
    parser = argparse.ArgumentParser(description="Tiền tính toán Vector Đồ thị Ngoại tuyến (Node2Vec 128d)")
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "graph_embeddings_128d.parquet"))
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--uri", default=config.NEO4J_URI)
    parser.add_argument("--user", default=config.NEO4J_USER)
    parser.add_argument("--password", default=config.NEO4J_PASSWORD)
    args = parser.parse_args()

    compute_and_export_embeddings(
        output_path=args.output,
        dimensions=args.dim,
        uri=args.uri,
        user=args.user,
        password=args.password,
    )


if __name__ == "__main__":
    main()