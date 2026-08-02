"""
generate_training_data.py - Pipeline sinh Triplet Dataset cho VietLawBERT.

Thuật toán: Graph-Guided Semantic-Lexical Mining (GG-SLM)
  Tầng 1 (Graph Topology):   Truy vấn ứng viên Negative qua Neo4j
                               - Sibling Strategy:    HAS_CHUNK anh em cùng Điều
                               - Neighbor Strategy:   Chunk thuộc Điều anh em cùng doc_id
                               - Referential Strategy: Chunk của văn bản được dẫn chiếu
                                 (CAN_CO_BAN_HANH / VAN_BAN_AP_DUNG)
  Tầng 2 (Lexical Filter):   BM25 để chọn Hard Negative theo công thức:
                               Score_Hardness(h) = γ·BM25(q,h) + (1-γ)·exp(-dist_graph(P,h))
  Query Source:               Synthetic Query từ LLM (OpenAI-compatible via Ollama/Groq)
                               hoặc dùng tiêu đề Điều luật làm Query đơn giản.

Output: triplet_training_data.jsonl
  {"query": "...", "positive": "...", "hard_negative": "...",
   "meta": {"pos_id": "...", "neg_id": "...", "mining_strategy": "sibling/neighbor/reference",
            "hardness_score": 0.0, "pos_doc_id": "...", "neg_doc_id": "..."}}

Usage:
    python -m generate_training_data --input data/json/final_contextual_chunks.jsonl \\
                                     --diagram-dir data/raw/diagram \\
                                     --output data/json/triplet_training_data.jsonl \\
                                     --gamma 0.6 --tau 0.1 --max-triplets 50000 \\
                                     [--use-llm-query] [--ollama-url http://localhost:11434]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# ponytail: rank_bm25 dùng BM25Okapi; upgrade lên BM25+ nếu cần smoothing tốt hơn
from rank_bm25 import BM25Okapi

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GG-SLM")


# ---------------------------------------------------------------------------
# 1. Load dữ liệu
# ---------------------------------------------------------------------------

def load_chunks(jsonl_path: str) -> list[dict]:
    """Đọc final_contextual_chunks.jsonl → list chunk."""
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks):,} chunks từ {jsonl_path}")
    return chunks


def load_diagram_edges(diagram_dir: str) -> dict[str, list[dict]]:
    """
    Đọc thư mục data/raw/diagram/*.json → dict {doc_id: [edge, ...]}
    Mỗi edge có dạng: {target_id, edge_type, direction, graph_layer, ...}
    """
    edges: dict[str, list[dict]] = defaultdict(list)
    diagram_path = Path(diagram_dir)
    files = list(diagram_path.glob("*.json"))
    logger.info(f"Đọc {len(files):,} diagram JSON files từ {diagram_dir}")
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                src = str(rec.get("source_doc_id", fp.stem))
                edges[src].append(rec)
        except Exception as e:
            logger.warning(f"Bỏ qua {fp.name}: {e}")
    return dict(edges)


# ---------------------------------------------------------------------------
# 2. Index cấu trúc cho Tầng 1 (Graph Topology)
# ---------------------------------------------------------------------------

def build_indices(chunks: list[dict]):
    """
    Trả về:
      chunk_by_id   : {chunk_id -> chunk}
      chunks_by_doc : {doc_id  -> [chunk_id, ...]}     (Neighbor Strategy)
      chunks_by_dieu: {(doc_id, dieu) -> [chunk_id, ...]} (Sibling Strategy)
    """
    chunk_by_id: dict[str, dict] = {}
    chunks_by_doc: dict[str, list[str]] = defaultdict(list)
    chunks_by_dieu: dict[tuple, list[str]] = defaultdict(list)

    for ck in chunks:
        cid = ck["chunk_id"]
        chunk_by_id[cid] = ck
        meta = ck.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        dieu = (meta.get("hierarchy_path") or {}).get("điều") or ""
        chunks_by_doc[doc_id].append(cid)
        if dieu:
            chunks_by_dieu[(doc_id, dieu)].append(cid)

    logger.info(
        f"Index: {len(chunk_by_id):,} chunks | "
        f"{len(chunks_by_doc):,} docs | "
        f"{len(chunks_by_dieu):,} (doc,dieu) nhóm"
    )
    return chunk_by_id, dict(chunks_by_doc), dict(chunks_by_dieu)


# ---------------------------------------------------------------------------
# 3. Tầng 1 - Graph Topology: Tập ứng viên C
# ---------------------------------------------------------------------------

def get_candidate_set(
    pos_chunk: dict,
    chunk_by_id: dict,
    chunks_by_doc: dict,
    chunks_by_dieu: dict,
    diagram_edges: dict,
    strategy: str = "all",
) -> list[tuple[str, str, int]]:
    """
    Trả về list (chunk_id, mining_strategy, graph_dist) cho các ứng viên Negative.
    graph_dist là khoảng cách ước lượng trên đồ thị:
      sibling   → 1 (cùng Điều)
      neighbor  → 2 (cùng doc, khác Điều)
      reference → 3 (sang văn bản khác qua edge dẫn chiếu)
    """
    meta = pos_chunk.get("metadata", {})
    pos_id = pos_chunk["chunk_id"]
    doc_id = meta.get("doc_id", "")
    dieu = (meta.get("hierarchy_path") or {}).get("điều") or ""

    candidates: list[tuple[str, str, int]] = []

    # --- Sibling (dist=1): cùng Điều, khác chunk ---
    if strategy in ("all", "sibling") and dieu:
        siblings = chunks_by_dieu.get((doc_id, dieu), [])
        for cid in siblings:
            if cid != pos_id:
                candidates.append((cid, "sibling", 1))

    # --- Neighbor (dist=2): cùng doc_id, khác Điều ---
    if strategy in ("all", "neighbor"):
        doc_chunks = chunks_by_doc.get(doc_id, [])
        for cid in doc_chunks:
            if cid == pos_id:
                continue
            ck_meta = chunk_by_id[cid].get("metadata", {})
            ck_dieu = (ck_meta.get("hierarchy_path") or {}).get("điều") or ""
            if ck_dieu != dieu:
                candidates.append((cid, "neighbor", 2))

    # --- Referential (dist=3): qua CAN_CO_BAN_HANH / VAN_BAN_AP_DUNG ---
    if strategy in ("all", "reference"):
        edges = diagram_edges.get(doc_id, [])
        ref_doc_ids = set()
        for edge in edges:
            if edge.get("edge_type") in ("CAN_CO_BAN_HANH", "VAN_BAN_AP_DUNG"):
                ref_doc_ids.add(str(edge.get("target_id", "")))
        for ref_did in ref_doc_ids:
            for cid in chunks_by_doc.get(ref_did, []):
                if cid != pos_id:
                    candidates.append((cid, "reference", 3))

    return candidates


# ---------------------------------------------------------------------------
# 4. Tầng 2 - Lexical Filter: Hardness Score (công thức của Daniel)
# ---------------------------------------------------------------------------

def score_hardness(
    query: str,
    candidates: list[tuple[str, str, int]],
    chunk_by_id: dict,
    gamma: float = 0.6,
) -> list[tuple[str, str, float]]:
    """
    Score_Hardness(h) = γ·BM25(q,h) + (1-γ)·exp(-dist_graph(P,h))

    Trả về list (chunk_id, mining_strategy, score) đã sắp xếp giảm dần.
    """
    if not candidates:
        return []

    corpus_texts = []
    for cid, _, _ in candidates:
        text = chunk_by_id[cid].get("contextualized_text", "")
        corpus_texts.append(text)

    # BM25 tokenize đơn giản (whitespace); upgrade path: dùng underthesea word_tokenize
    tokenized_corpus = [t.split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # Normalize BM25 về [0,1]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    scored: list[tuple[str, str, float]] = []
    for i, (cid, strategy, dist) in enumerate(candidates):
        bm25_norm = float(bm25_scores[i]) / max_bm25
        graph_component = math.exp(-dist)
        score = gamma * bm25_norm + (1 - gamma) * graph_component
        scored.append((cid, strategy, round(score, 6)))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# 5. Synthetic Query Generation (LLM hoặc fallback tiêu đề Điều)
# ---------------------------------------------------------------------------

def generate_query_from_llm(
    chunk_text: str,
    ollama_url: str,
    model: str = "deepseek-r1:14b",
) -> str | None:
    """
    Gọi LLM qua OpenAI-compatible API để sinh câu hỏi từ nội dung chunk.
    Trả về None nếu lỗi (pipeline sẽ dùng fallback).
    """
    try:
        from openai import OpenAI
        from config import config
        client = OpenAI(base_url=config.LLM_API_BASE, api_key=config.LLM_API_KEY)

        prompt = (
            "Bạn là chuyên gia pháp lý Việt Nam. Đọc đoạn văn bản pháp luật sau "
            "và viết MỘT câu hỏi ngắn (dưới 30 từ) mà đoạn này trả lời được. "
            "Chỉ trả về câu hỏi, không giải thích.\n\n"
            f"VĂN BẢN:\n{chunk_text[:800]}\n\nCÂU HỎI:"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.debug(f"LLM query gen lỗi: {e}")
        return None


def make_query(
    chunk: dict,
    use_llm: bool = False,
    ollama_url: str = "http://localhost:11434",
    model: str = "deepseek-r1:14b",
) -> str:
    """
    Fallback chain:
      1. LLM-generated question (nếu --use-llm-query)
      2. Tiêu đề Điều luật (hierarchy_path.điều)
      3. 100 ký tự đầu của original_text
    """
    if use_llm:
        text = chunk.get("contextualized_text", "")
        q = generate_query_from_llm(text, ollama_url, model)
        if q:
            return q

    meta = chunk.get("metadata", {})
    hp = meta.get("hierarchy_path") or {}
    dieu = hp.get("điều") or ""
    khoan = hp.get("khoản") or ""
    doc_number = meta.get("doc_number", "")

    if dieu:
        label = f"{dieu}" + (f", {khoan}" if khoan else "")
        return f"{label} của {doc_number} quy định gì?"

    # Fallback: 100 ký tự đầu
    text = chunk.get("original_text", "")
    return text[:100].strip() + "?"


# ---------------------------------------------------------------------------
# 6. Pipeline chính
# ---------------------------------------------------------------------------

def run_pipeline(
    chunks_path: str,
    diagram_dir: str,
    output_path: str,
    gamma: float = 0.6,
    tau: float = 0.1,
    max_triplets: int = 50_000,
    strategy: str = "all",
    use_llm_query: bool = False,
    ollama_url: str = "http://localhost:11434",
    llm_model: str = "deepseek-r1:14b",
):
    # Load
    chunks = load_chunks(chunks_path)
    diagram_edges = load_diagram_edges(diagram_dir)

    # Build indices
    chunk_by_id, chunks_by_doc, chunks_by_dieu = build_indices(chunks)

    # Stats
    stats = {"total": 0, "sibling": 0, "neighbor": 0, "reference": 0, "skipped_tau": 0}
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Bắt đầu mining | gamma={gamma} | tau={tau} | max={max_triplets:,}")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, pos_chunk in enumerate(chunks):
            if stats["total"] >= max_triplets:
                break

            pos_id = pos_chunk["chunk_id"]
            pos_text = pos_chunk.get("contextualized_text", "")
            if not pos_text.strip():
                continue

            # Query
            query = make_query(
                pos_chunk, use_llm=use_llm_query,
                ollama_url=ollama_url, model=llm_model
            )

            # Tầng 1: Graph candidates
            candidates = get_candidate_set(
                pos_chunk, chunk_by_id, chunks_by_doc,
                chunks_by_dieu, diagram_edges, strategy
            )

            if not candidates:
                continue

            # Tầng 2: BM25 Hardness Scoring
            scored = score_hardness(query, candidates, chunk_by_id, gamma)

            # Chọn Semi-Hard Negative (tau <= score <= tau_max)
            # tau_max = 0.85 ngăn cản việc bốc các true positive giả (Easy/Positive overlapping)
            tau_max = 0.85
            best = next(
                ((cid, strat, sc) for cid, strat, sc in scored if tau <= sc <= tau_max),
                None
            )
            if best is None:
                stats["skipped_tau"] += 1
                continue

            neg_id, mining_strategy, hardness_score = best
            neg_chunk = chunk_by_id[neg_id]
            neg_text = neg_chunk.get("contextualized_text", "")

            pos_meta = pos_chunk.get("metadata", {})
            neg_meta = neg_chunk.get("metadata", {})

            record = {
                "query": query,
                "positive": pos_text,
                "hard_negative": neg_text,
                "meta": {
                    "pos_id": pos_id,
                    "neg_id": neg_id,
                    "mining_strategy": mining_strategy,
                    "hardness_score": hardness_score,
                    "pos_doc_id": pos_meta.get("doc_id", ""),
                    "neg_doc_id": neg_meta.get("doc_id", ""),
                    "pos_doc_number": pos_meta.get("doc_number", ""),
                    "neg_doc_number": neg_meta.get("doc_number", ""),
                    "pos_dieu": (pos_meta.get("hierarchy_path") or {}).get("điều", ""),
                    "neg_dieu": (neg_meta.get("hierarchy_path") or {}).get("điều", ""),
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["total"] += 1
            stats[mining_strategy] = stats.get(mining_strategy, 0) + 1

            if stats["total"] % 1000 == 0:
                logger.info(
                    f"[{stats['total']:,}/{max_triplets:,}] "
                    f"sibling={stats['sibling']:,} | "
                    f"neighbor={stats['neighbor']:,} | "
                    f"reference={stats['reference']:,} | "
                    f"skipped_tau={stats['skipped_tau']:,}"
                )

    logger.info(
        f"Hoàn thành! Tổng triplet: {stats['total']:,} | "
        f"sibling={stats['sibling']:,} | neighbor={stats['neighbor']:,} | "
        f"reference={stats['reference']:,} | skipped_tau={stats['skipped_tau']:,}\n"
        f"Output: {output_file.resolve()}"
    )
    return stats


# ---------------------------------------------------------------------------
# 7. Kiểm thử nhanh (assert-based self-check)
# ---------------------------------------------------------------------------

def _self_check():
    """Chạy bằng: python generate_training_data.py --test"""
    import math

    # Test hardness score formula
    gamma = 0.6
    bm25_norm = 1.0  # best case
    dist = 1         # sibling
    expected = gamma * bm25_norm + (1 - gamma) * math.exp(-dist)
    assert abs(expected - (0.6 + 0.4 * math.exp(-1))) < 1e-9, "Hardness formula sai!"

    # Test BM25 với corpus giả
    corpus = ["nồng độ cồn vượt quá quy định bị phạt", "đèn đỏ giao thông"]
    bm25 = BM25Okapi([t.split() for t in corpus])
    scores = bm25.get_scores("nồng độ cồn".split())
    assert scores[0] > scores[1], "BM25 phải ưu tiên doc chứa query terms!"

    print("✅ Self-check passed: Hardness formula + BM25 hoạt động đúng.")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GG-SLM Triplet Generator for VietLawBERT")
    parser.add_argument("--input", default="data/json/final_contextual_chunks.jsonl")
    parser.add_argument("--diagram-dir", default="data/raw/diagram")
    parser.add_argument("--output", default="data/json/triplet_training_data.jsonl")
    parser.add_argument("--gamma", type=float, default=0.6,
                        help="Trọng số BM25 vs Graph distance (0-1)")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Ngưỡng Hardness Score tối thiểu để chấp nhận Hard Negative")
    parser.add_argument("--max-triplets", type=int, default=50_000)
    parser.add_argument("--strategy", choices=["all", "sibling", "neighbor", "reference"],
                        default="all")
    parser.add_argument("--use-llm-query", action="store_true",
                        help="Dùng LLM (Ollama) để sinh câu hỏi thay vì tiêu đề Điều")
    parser.add_argument("--ollama-url", default=config.OLLAMA_BASE_URL)
    parser.add_argument("--llm-model", default=config.GENERATOR_MODEL)
    parser.add_argument("--test", action="store_true", help="Chạy self-check")
    args = parser.parse_args()

    if args.test:
        _self_check()
        return

    run_pipeline(
        chunks_path=args.input,
        diagram_dir=args.diagram_dir,
        output_path=args.output,
        gamma=args.gamma,
        tau=args.tau,
        max_triplets=args.max_triplets,
        strategy=args.strategy,
        use_llm_query=args.use_llm_query,
        ollama_url=args.ollama_url,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
