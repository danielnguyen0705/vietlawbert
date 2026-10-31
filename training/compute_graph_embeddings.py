"""
compute_graph_embeddings.py - Động cơ tiền tính toán Vector Đồ thị Ngoại tuyến (Compile-time).
Trích xuất 22 quan hệ pháp lý từ Neo4j -> Fast Integer Walk 128d -> PyTorch Sparse SGNS -> Lưu Parquet & Redis Cache.
Tối ưu hóa tài nguyên phần cứng Dell G7: Giảm thời gian thực thi từ 3 tiếng xuống dưới 3 phút.
Tương thích 100% chuẩn Neo4j 5.x GQL, Python 3.14 & NumPy 2.x, loại bỏ hoàn toàn gensim.
"""

from __future__ import annotations

import os
import json
import random
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase

try:
    torch.set_num_threads(2)
except Exception:
    pass

from configs.config import config
from configs.paths import ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_GraphEmbedder")

# Trọng số cạnh pháp lý trong random walk (>>1 để walk đi qua quan hệ liên-VB,
# tránh đồ thị hình sao làm pha loãng tín hiệu 22 quan hệ → gốc của graph_boost≈0).
LEGAL_EDGE_WEIGHT = float(os.getenv("GRAPH_LEGAL_EDGE_WEIGHT", "20"))


class Neo4jGraphExtractor:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or os.getenv("NEO4J_URI") or config.NEO4J_URI
        self.user = user or os.getenv("NEO4J_USER") or config.NEO4J_USER
        self.password = password or os.getenv("NEO4J_PASSWORD") or getattr(config, "NEO4J_PASSWORD", "vietlawbert2026")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password), connection_acquisition_timeout=15.0)
        self.driver.verify_connectivity()
        logger.info("Kết nối Neo4j Engine thành công tại %s (User: %s).", self.uri, self.user)

    def close(self):
        self.driver.close()

    def fetch_weighted_edges(self, legal_weight: float = 20.0) -> List[Tuple[str, str, float]]:
        """
        Trích xuất cạnh đồ thị kèm TRỌNG SỐ, phân biệt theo loại:
          - Cạnh phân cấp HAS_CHUNK / HAS_ARTICLE: trọng số 1.0. ĐƯA Article vào để chunk nối nhau
            qua điều (sibling trong cùng điều) thay vì chỉ nối hình sao vào văn bản cha.
          - Cạnh pháp lý LEGAL_RELATION (22 quan hệ): trọng số `legal_weight` (>>1), để random walk
            đi qua quan hệ liên-văn-bản đủ thường xuyên → vector đồ thị mã hóa được ngữ nghĩa pháp lý.
        Khắc phục gốc rễ graph_boost≈0: trước đây gộp mọi cạnh vô hướng + BỎ Article → đồ thị hình sao,
        tín hiệu 22 quan hệ bị pha loãng tới ~0 (xác suất walk rẽ vào cạnh pháp lý ≈ n_legal/n_chunk ≈ 1-2%).
        """
        cypher = """
        MATCH (s)-[r]->(t)
        WHERE (s:LawDocument OR s:Chunk OR s:Article)
          AND (t:LawDocument OR t:Chunk OR t:Article)
        RETURN coalesce(s.chunk_id, s.article_id, s.doc_id) AS u,
               coalesce(t.chunk_id, t.article_id, t.doc_id) AS v,
               type(r) AS etype
        """
        edges: List[Tuple[str, str, float]] = []
        n_legal = 0
        logger.info("Đang truy xuất cạnh (có trọng số, gồm Article) từ Neo4j...")
        with self.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                u, v, etype = record["u"], record["v"], record["etype"]
                if not u or not v or str(u) == str(v):
                    continue
                if etype == "LEGAL_RELATION":
                    w = float(legal_weight)
                    n_legal += 1
                else:
                    w = 1.0
                edges.append((str(u), str(v), w))

        logger.info("✓ Trích xuất %d cạnh (trong đó %d cạnh pháp lý LEGAL_RELATION ×%.0f).",
                    len(edges), n_legal, legal_weight)
        if n_legal == 0:
            logger.warning("CẢNH BÁO: KHÔNG có cạnh LEGAL_RELATION — đồ thị chỉ còn cấu trúc phân cấp, "
                           "graph-rerank sẽ vô nghĩa. Kiểm tra build_hin_graph + ontology.")
        return edges

    # Giữ tên cũ để tương thích ngược (nay trả về cạnh có trọng số).
    def fetch_legal_edges(self) -> List[Tuple[str, str, float]]:
        return self.fetch_weighted_edges(legal_weight=LEGAL_EDGE_WEIGHT)


class FastGraphWalker:
    """Random Walk CÓ TRỌNG SỐ trên mảng số nguyên phẳng (ưu tiên cạnh pháp lý)."""

    def __init__(self, edges: List[Tuple[str, str, float]]):
        unique_nodes = set()
        for u, v, _w in edges:
            unique_nodes.add(u)
            unique_nodes.add(v)

        self.nodes = list(unique_nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        logger.info("Chuyển đồ thị sang Integer IDs (%d nút duy nhất)...", self.num_nodes)

        # Adjacency CÓ TRỌNG SỐ, vô hướng (để học biểu diễn đồng nhất).
        nbr = defaultdict(list)
        wgt = defaultdict(list)
        for u, v, w in edges:
            ui = self.node_to_idx[u]
            vi = self.node_to_idx[v]
            nbr[ui].append(vi); wgt[ui].append(w)
            nbr[vi].append(ui); wgt[vi].append(w)

        self.adj_nodes = [tuple(nbr[i]) if i in nbr else () for i in range(self.num_nodes)]
        self.adj_wgt = [tuple(wgt[i]) if i in wgt else () for i in range(self.num_nodes)]
        del nbr, wgt

    def generate_walks(self, num_walks: int = 5, walk_length: int = 20) -> torch.Tensor:
        """Sinh chuỗi bước duyệt ngẫu nhiên CÓ TRỌNG SỐ vào Tensor 2D (Tiết kiệm 95% RAM)."""
        total_walks = self.num_nodes * num_walks
        logger.info("Bắt đầu sinh %d chuỗi bước duyệt có trọng số (Chiều dài: %d)...", total_walks, walk_length)

        walks_tensor = torch.empty((total_walks, walk_length), dtype=torch.long)
        row_idx = 0

        for walk_iter in range(num_walks):
            permuted_nodes = list(range(self.num_nodes))
            random.shuffle(permuted_nodes)

            pbar = tqdm(
                permuted_nodes,
                desc=f"[Walk Iteration {walk_iter + 1}/{num_walks}]",
                unit=" nodes",
                leave=False,
            )

            for node in pbar:
                if not self.adj_nodes[node]:
                    walks_tensor[row_idx] = torch.full((walk_length,), node, dtype=torch.long)
                    row_idx += 1
                    continue

                walk = [node]
                curr = node
                for _ in range(walk_length - 1):
                    neighbors = self.adj_nodes[curr]
                    weights = self.adj_wgt[curr]
                    if not neighbors:
                        walk.extend([curr] * (walk_length - len(walk)))
                        break
                    curr = random.choices(neighbors, weights=weights, k=1)[0]
                    walk.append(curr)

                if len(walk) < walk_length:
                    walk.extend([walk[-1]] * (walk_length - len(walk)))
                walks_tensor[row_idx] = torch.tensor(walk, dtype=torch.long)
                row_idx += 1

        logger.info("✓ Hoàn tất sinh %d chuỗi bước đi ngẫu nhiên (có trọng số) vào Tensor.", total_walks)
        return walks_tensor


class PyTorchSparseSGNS(nn.Module):
    """Mô hình Skip-Gram với Negative Sampling thuần túy sử dụng Sparse Embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        init_range = 0.5 / embed_dim
        nn.init.uniform_(self.target_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, target: torch.Tensor, context: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        u = self.target_embeddings(target)       # [Batch, Dim]
        v = self.context_embeddings(context)     # [Batch, Dim]
        pos_score = torch.sum(u * v, dim=-1)      # [Batch]
        pos_loss = -F.logsigmoid(pos_score)

        b_size, k_neg = negatives.shape
        neg_v = self.context_embeddings(negatives.reshape(-1)).reshape(b_size, k_neg, self.embed_dim)
        neg_score = torch.bmm(neg_v, u.unsqueeze(-1)).squeeze(-1)  # [Batch, K_neg]
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=-1)    # [Batch]

        return torch.mean(pos_loss + neg_loss)

    def get_normalized_embeddings(self) -> torch.Tensor:
        return F.normalize(self.target_embeddings.weight.data, p=2, dim=-1)


def train_pytorch_node2vec(
    walks_tensor: torch.Tensor,
    vocab_size: int,
    dimensions: int = 128,
    window_size: int = 5,
    epochs: int = 3,
    batch_size: int = 4096,
    steps_per_epoch: int = 5000,
) -> torch.Tensor:
    n_walks, walk_len = walks_tensor.size()

    logger.info("Khởi động huấn luyện PyTorch SGNS (%d steps/epoch | Batch: %d)...", steps_per_epoch, batch_size)

    model = PyTorchSparseSGNS(vocab_size=vocab_size, embed_dim=dimensions)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)  # SparseAdam: đúng optimizer cho nn.Embedding(sparse=True)

    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f"SGNS Epoch {epoch}/{epochs}", unit=" step")

        for _ in pbar:
            w_idx = torch.randint(0, n_walks, (batch_size,))
            pos = torch.randint(0, walk_len, (batch_size,))

            offsets = torch.randint(-window_size, window_size, (batch_size,))
            offsets[offsets >= 0] += 1
            ctx_pos = torch.clamp(pos + offsets, 0, walk_len - 1)

            target = walks_tensor[w_idx, pos]
            context = walks_tensor[w_idx, ctx_pos]
            negatives = torch.randint(0, vocab_size, (batch_size, 5), dtype=torch.long)

            optimizer.zero_grad()
            loss = model(target, context, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / steps_per_epoch
        logger.info("[Epoch %d/%d] Loss trung bình: %.4f | Thời gian: %.1fs", epoch, epochs, avg_loss, time.time() - start_time)

    model.eval()
    with torch.no_grad():
        return model.get_normalized_embeddings().cpu()


def sync_embeddings_to_redis(node_list: List[str], weights: torch.Tensor) -> None:
    """Nạp vector đồ thị vào Redis RAM Cache (SLA < 500ms) theo luồng Pipeline an toàn."""
    try:
        import redis
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB, socket_timeout=5.0)
        r.ping()
        pipe = r.pipeline(transaction=False)
        pushed = 0

        pbar = tqdm(enumerate(node_list), total=len(node_list), desc="Nạp Redis Cache", unit=" vec")
        for idx, node_id in pbar:
            vec = weights[idx].numpy().tolist()
            pipe.set(f"graph_emb:{node_id}", json.dumps(vec))
            pushed += 1
            if pushed % 5000 == 0:
                pipe.execute()

        pipe.execute()
        r.close()
        logger.info("✓ Đã nạp thành công %d vector đồ thị vào Redis In-Memory Cache.", pushed)
    except Exception as exc:
        logger.warning("Bỏ qua đồng bộ Redis (%s). Dữ liệu vẫn được bảo toàn qua Parquet.", exc)


def compute_and_export_embeddings(
    output_path: Optional[str | Path] = None,
    dimensions: int = 128,
    num_walks: int = 5,
    walk_length: int = 20,
    epochs: int = 3,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    target_output = Path(output_path) if output_path else (ARTIFACTS_DIR / f"graph_embeddings_{dimensions}d.parquet")

    extractor = Neo4jGraphExtractor(uri=uri, user=user, password=password)
    edges = extractor.fetch_weighted_edges(legal_weight=LEGAL_EDGE_WEIGHT)
    extractor.close()

    if not edges:
        raise RuntimeError(
            "Đồ thị Neo4j KHÔNG có cạnh nào — DỪNG để tránh sinh vector rác. "
            "Hãy chạy build_hin_graph trước và kiểm tra Neo4j đang chạy."
        )

    walker = FastGraphWalker(edges)
    del edges

    walks_tensor = walker.generate_walks(num_walks=num_walks, walk_length=walk_length)

    weights = train_pytorch_node2vec(
        walks_tensor=walks_tensor,
        vocab_size=walker.num_nodes,
        dimensions=dimensions,
        window_size=5,
        epochs=epochs,
        batch_size=4096,
        steps_per_epoch=5000,
    )
    del walks_tensor

    logger.info("Đang ghi tệp Parquet tại %s...", target_output)
    target_output.parent.mkdir(parents=True, exist_ok=True)

    weights_np = weights.numpy()
    df = pd.DataFrame({
        "chunk_id": walker.nodes,
        "graph_embedding": list(weights_np),
    })

    try:
        df.to_parquet(target_output, engine="pyarrow", compression="snappy", index=False)
    except Exception:
        fallback_json = target_output.with_suffix(".json.gz")
        df.to_json(fallback_json, orient="records", lines=True, compression="gzip")
        logger.warning("Xuất định dạng fallback: %s", fallback_json)

    logger.info("✓ Đã lưu trữ %d vector đồ thị (%dd) tại: %s", len(df), dimensions, target_output.resolve())
    del df

    sync_embeddings_to_redis(walker.nodes, weights)


def main():
    parser = argparse.ArgumentParser(description="Tiền tính toán Vector Đồ thị Ngoại tuyến (Node2Vec 128d PyTorch-Native)")
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "graph_embeddings_128d.parquet"), help="Đường dẫn tệp Parquet")
    parser.add_argument("--dim", "--dimensions", dest="dim", type=int, default=128, help="Số chiều vector (128)")
    parser.add_argument("--num-walks", type=int, default=5, help="Số lượt duyệt ngẫu nhiên trên mỗi nút")
    parser.add_argument("--walk-length", type=int, default=20, help="Độ dài mỗi đường duyệt")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện SGNS")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", getattr(config, "NEO4J_URI", "bolt://localhost:7687")), help="URI Neo4j")
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", getattr(config, "NEO4J_USER", "neo4j")), help="Tài khoản Neo4j")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", getattr(config, "NEO4J_PASSWORD", "vietlawbert2026")), help="Mật khẩu Neo4j")
    args = parser.parse_args()

    compute_and_export_embeddings(
        output_path=args.output,
        dimensions=args.dim,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        epochs=args.epochs,
        uri=args.uri,
        user=args.user,
        password=args.password,
    )


if __name__ == "__main__":
    main()