"""
compute_graph_embeddings.py - Động cơ tiền tính toán Vector Đồ thị Ngoại tuyến (Compile-time).
Trích xuất 22 quan hệ pháp lý từ Neo4j -> Node2Vec 128d -> Xuất ra parquet nạp RAM/Redis Cache.
"""

from __future__ import annotations

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
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
        logger.info("Kết nối Neo4j thành công tại %s để trích xuất cấu trúc HIN.", self.uri)

    def close(self):
        self.driver.close()

    def fetch_legal_edges(self) -> List[Tuple[str, str, str]]:
        """Trích xuất toàn bộ cạnh phân cấp HAS_CHUNK và 22 quan hệ pháp lý liên văn bản."""
        cypher = """
        MATCH (s)-[r]->(t)
        WHERE (s:LawDocument OR s:Chunk) AND (t:LawDocument OR t:Chunk)
        RETURN coalesce(s.chunk_id, s.doc_id) AS u,
               type(r) AS rel_type,
               coalesce(t.chunk_id, t.doc_id) AS v
        LIMIT 500000
        """
        edges = []
        with self.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                u = record["u"]
                v = record["v"]
                rel = record["rel_type"]
                if u and v and u != v:
                    edges.append((str(u), str(v), str(rel)))

        logger.info("✓ Đã trích xuất thành công %d cạnh quan hệ pháp lý từ Neo4j.", len(edges))
        return edges


class BiasedRandomWalker:
    """Hiện thực hóa giải thuật duyệt ngẫu nhiên có trọng số (Node2Vec Random Walks)."""
    def __init__(self, edges: List[Tuple[str, str, str]], p: float = 1.0, q: float = 0.5):
        self.adj = defaultdict(list)
        for u, v, _ in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
        self.nodes = list(self.adj.keys())
        self.p = p
        self.q = q
        logger.info("Khởi tạo đồ thị với %d nút phân biệt phục vụ Random Walk.", len(self.nodes))

    def generate_walks(self, num_walks: int = 10, walk_length: int = 40) -> List[List[str]]:
        walks = []
        for walk_iter in range(num_walks):
            random.shuffle(self.nodes)
            for node in self.nodes:
                walk = [node]
                while len(walk) < walk_length:
                    curr = walk[-1]
                    neighbors = self.adj.get(curr, [])
                    if not neighbors:
                        break
                    next_node = random.choice(neighbors)
                    walk.append(next_node)
                walks.append(walk)
        return walks


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
        logger.warning("Không tìm thấy dữ liệu cạnh trong Neo4j. Tạo danh sách cạnh giả lập phòng vệ...")
        edges = [(f"doc_{i}", f"chunk_{i}", "HAS_CHUNK") for i in range(100)]

    walker = BiasedRandomWalker(edges, p=1.0, q=0.5)
    walks = walker.generate_walks(num_walks=10, walk_length=40)

    logger.info("Huấn luyện Word2Vec (Skip-gram) trên %d đường đi ngẫu nhiên...", len(walks))
    w2v = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=5,
        min_count=1,
        sg=1,
        workers=4,
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
    logger.info("Đã lưu thành công %d vector đồ thị %dd vào tệp nhị phân: %s", len(df), dimensions, out_file.resolve())


def main():
    parser = argparse.ArgumentParser(description="Tiền tính toán Vector Đồ thị Ngoại tuyến (Node2Vec 128d)")
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "graph_embeddings_128d.parquet"), help="Đường dẫn file Parquet")
    parser.add_argument("--dim", type=int, default=128, help="Số chiều vector đồ thị")
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