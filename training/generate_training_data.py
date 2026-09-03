"""
generate_training_data.py - Động cơ sinh bộ ba mẫu đối lập pháp lý (Triplet Mining Engine).
Hiện thực hóa thuật toán GG-SLM (Graph-Guided Semantic-Lexical Mining):
Kết hợp khoảng cách topo đồ thị tri thức Neo4j và điểm liên quan từ vựng BM25.
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

from rank_bm25 import BM25Okapi

from configs.paths import ROOT_DIR, DATA_STORAGE_ROOT, ARTIFACTS_DIR, get_log_path
from configs.config import config
from artifacts.canonical import read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(get_log_path("generate_training_data"), encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("VietLawBERT_GGSLM")


def tokenize_vietnamese(text: str) -> List[str]:
    """Tách từ vựng cơ bản chuẩn hóa cho tiếng Việt, loại bỏ ký tự đặc biệt."""
    text_clean = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [w for w in text_clean.split() if w]


def load_chunks_unified(file_path: Path | str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu chunks hỗ trợ cả .jsonl, .jsonl.gz và .parquet."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu chunks: {path}")

    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    chunks = []
    for record in read_jsonl(path):
        chunks.append(record)

    logger.info(f"✓ Đã nạp thành công {len(chunks):,} chunks từ {path.name}")
    return chunks


def load_diagram_edges(diagram_dir: Path | str) -> Dict[str, List[Dict[str, Any]]]:
    """Nạp danh bạ cạnh quan hệ pháp lý từ thư mục diagrams hoặc tệp tổng hợp."""
    edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    d_path = Path(diagram_dir)
    if not d_path.exists():
        logger.warning(f"Thư mục diagram không tồn tại: {d_path}. Tiếp tục chỉ với quan hệ nội bộ văn bản.")
        return dict(edges)

    json_files = list(d_path.glob("*.json"))
    logger.info(f"Đang phân tích {len(json_files):,} tệp diagram quan hệ từ {d_path.name}...")

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                records = json.load(f)
                if isinstance(records, dict):
                    records = [records]
                for rec in records:
                    src = str(rec.get("source_doc_id") or fp.stem)
                    edges[src].append(rec)
        except Exception as exc:
            logger.debug(f"Bỏ qua tệp {fp.name}: {exc}")

    return dict(edges)


def build_indices(chunks: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[Tuple[str, str], List[str]]]:
    """
    Xây dựng chỉ mục tìm kiếm nhanh trên RAM:
    - chunk_by_id: Tra cứu thực thể chunk theo chunk_id.
    - chunks_by_doc: Tra cứu danh sách chunk theo doc_id (Chiến lược Neighbor).
    - chunks_by_dieu: Tra cứu danh sách chunk theo cặp (doc_id, điều) (Chiến lược Sibling).
    """
    chunk_by_id: Dict[str, Dict[str, Any]] = {}
    chunks_by_doc: Dict[str, List[str]] = defaultdict(list)
    chunks_by_dieu: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for ck in chunks:
        cid = str(ck.get("chunk_id", ""))
        if not cid:
            continue
        chunk_by_id[cid] = ck
        meta = ck.get("metadata") or {}
        doc_id = str(meta.get("doc_id") or ck.get("doc_id") or "")
        hierarchy = meta.get("hierarchy_path") or {}
        dieu = str(hierarchy.get("điều") or "").strip()

        chunks_by_doc[doc_id].append(cid)
        if dieu:
            chunks_by_dieu[(doc_id, dieu)].append(cid)

    logger.info(f"Chỉ mục Topo: {len(chunk_by_id):,} chunks | {len(chunks_by_doc):,} văn bản | {len(chunks_by_dieu):,} nhóm (Văn bản, Điều)")
    return chunk_by_id, dict(chunks_by_doc), dict(chunks_by_dieu)


VALID_REFERENTIAL_EDGES = {
    "CAN_CU_BAN_HANH", "CAN_CO_BAN_HANH", "VAN_BAN_AP_DUNG",
    "HUONG_DAN_AP_DUNG", "DAN_CHIEU", "THAY_THE", "SUA_DOI_BO_SUNG",
    "QUY_DINH_CHI_TIET_HUONG_DAN",
}


def get_candidate_set(
    pos_chunk: Dict[str, Any],
    chunk_by_id: Dict[str, Dict[str, Any]],
    chunks_by_doc: Dict[str, List[str]],
    chunks_by_dieu: Dict[Tuple[str, str], List[str]],
    diagram_edges: Dict[str, List[Dict[str, Any]]],
    strategy: str = "all",
) -> List[Tuple[str, str, int]]:
    """
    Tầng 1 (Graph Topology): Khai thác tập ứng viên Negative dựa trên khoảng cách đồ thị tri thức:
    - dist = 1 (Sibling): Thuộc cùng một Điều luật nhưng khác Khoản/Điểm.
    - dist = 2 (Neighbor): Thuộc cùng văn bản pháp lý nhưng quy định ở Điều khác.
    - dist = 3 (Referential): Thuộc văn bản khác nhưng có quan hệ dẫn chiếu/sửa đổi/hướng dẫn.
    """
    meta = pos_chunk.get("metadata") or {}
    pos_id = str(pos_chunk.get("chunk_id", ""))
    doc_id = str(meta.get("doc_id") or pos_chunk.get("doc_id") or "")
    hierarchy = meta.get("hierarchy_path") or {}
    dieu = str(hierarchy.get("điều") or "").strip()

    candidates: List[Tuple[str, str, int]] = []

    # 1. Sibling Strategy (dist = 1)
    if strategy in ("all", "sibling") and dieu:
        for cid in chunks_by_dieu.get((doc_id, dieu), []):
            if cid != pos_id:
                candidates.append((cid, "sibling", 1))

    # 2. Neighbor Strategy (dist = 2)
    if strategy in ("all", "neighbor"):
        for cid in chunks_by_doc.get(doc_id, []):
            if cid == pos_id:
                continue
            ck_meta = chunk_by_id[cid].get("metadata") or {}
            ck_dieu = str((ck_meta.get("hierarchy_path") or {}).get("điều") or "").strip()
            if ck_dieu != dieu:
                candidates.append((cid, "neighbor", 2))

    # 3. Referential Strategy (dist = 3)
    if strategy in ("all", "reference"):
        edges = diagram_edges.get(doc_id, [])
        ref_doc_ids = set()
        for edge in edges:
            if edge.get("edge_type") in VALID_REFERENTIAL_EDGES:
                target = str(edge.get("target_id") or "")
                if target:
                    ref_doc_ids.add(target)

        for ref_id in ref_doc_ids:
            for cid in chunks_by_doc.get(ref_id, []):
                if cid != pos_id:
                    candidates.append((cid, "reference", 3))

    return candidates


def score_hardness(
    query: str,
    candidates: List[Tuple[str, str, int]],
    chunk_by_id: Dict[str, Dict[str, Any]],
    gamma: float = 0.6,
) -> List[Tuple[str, str, float]]:
    """
    Tầng 2 (Lexical Scoring): Tính toán điểm mẫu khó Hardness Score:
    Score_Hardness(h) = gamma * BM25_norm(q, h) + (1 - gamma) * exp(-dist_graph(P, h))
    """
    if not candidates:
        return []

    tokenized_query = tokenize_vietnamese(query)
    corpus_tokens = []

    for cid, _, _ in candidates:
        text = chunk_by_id[cid].get("contextualized_text") or chunk_by_id[cid].get("original_text", "")
        corpus_tokens.append(tokenize_vietnamese(text))

    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(tokenized_query)
    max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0

    scored: List[Tuple[str, str, float]] = []
    for i, (cid, strat, dist) in enumerate(candidates):
        bm25_norm = float(bm25_scores[i]) / max_bm25
        graph_component = math.exp(-float(dist))
        score = (gamma * bm25_norm) + ((1.0 - gamma) * graph_component)
        scored.append((cid, strat, round(score, 6)))

    # Sắp xếp giảm dần theo điểm khó
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def generate_query_from_llm(chunk_text: str, model: str) -> Optional[str]:
    """Sinh câu hỏi pháp lý giả lập qua OpenAI-compatible API."""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=getattr(config, "LLM_API_BASE", "http://localhost:11434/v1"),
            api_key=getattr(config, "LLM_API_KEY", "ollama"),
            timeout=15.0,
        )

        prompt = (
            "Bạn là chuyên gia thẩm định văn bản pháp luật Việt Nam. Hãy đọc đoạn trích sau "
            "và viết MỘT câu hỏi tra cứu tình huống thực tế (dưới 35 từ) mà đoạn trích này giải đáp. "
            "Chỉ trả về câu hỏi duy nhất, không thêm bất kỳ lời dẫn nào.\n\n"
            f"VĂN BẢN QUY ĐỊNH:\n{chunk_text[:1200]}\n\nCÂU HỎI:"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=90,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.debug(f"Bỏ qua sinh câu hỏi LLM: {exc}")
        return None


def make_query(chunk: Dict[str, Any], use_llm: bool = False, model: Optional[str] = None) -> str:
    """Tạo câu hỏi truy vấn: Ưu tiên LLM Synthetic Query, Fallback sang cấu trúc Điều/Khoản."""
    text = chunk.get("contextualized_text") or chunk.get("original_text", "")

    if use_llm:
        target_model = model or getattr(config, "CONTEXTUALIZER_MODEL", "qwen2.5:1.5b")
        q = generate_query_from_llm(text, target_model)
        if q and len(q) >= 15:
            return q

    meta = chunk.get("metadata") or {}
    hp = meta.get("hierarchy_path") or {}
    dieu = str(hp.get("điều") or "").strip()
    khoan = str(hp.get("khoản") or "").strip()
    doc_number = str(meta.get("doc_number") or chunk.get("doc_number") or "")

    if dieu and doc_number:
        scope = f"{dieu}, {khoan}" if khoan else dieu
        return f"Quy định pháp lý tại {scope} của {doc_number} là gì?"

    # Fallback trích xuất tiêu đề ngắn
    raw_preview = chunk.get("original_text", "")[:120].strip()
    return f"Nội dung quy định liên quan đến: {raw_preview}?"


def run_pipeline(
    chunks_path: Path | str,
    diagram_dir: Path | str,
    output_path: Path | str,
    gamma: float = 0.6,
    tau: float = 0.1,
    max_triplets: int = 50_000,
    strategy: str = "all",
    use_llm_query: bool = False,
    llm_model: Optional[str] = None,
) -> Dict[str, int]:
    """Điều phối toàn bộ chu trình khai phá mẫu khó đối lập GG-SLM."""
    chunks = load_chunks_unified(chunks_path)
    diagram_edges = load_diagram_edges(diagram_dir)
    chunk_by_id, chunks_by_doc, chunks_by_dieu = build_indices(chunks)

    stats = {"total": 0, "sibling": 0, "neighbor": 0, "reference": 0, "skipped_tau": 0}
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(f"{out_file.suffix}.tmp_{os.getpid()}")

    logger.info(f"Bắt đầu khai phá GG-SLM | gamma={gamma} | tau={tau} | mục tiêu={max_triplets:,}")

    with open(tmp_file, "w", encoding="utf-8") as out_f:
        for pos_chunk in chunks:
            if stats["total"] >= max_triplets:
                break

            pos_id = str(pos_chunk.get("chunk_id", ""))
            pos_text = pos_chunk.get("contextualized_text") or pos_chunk.get("original_text", "")
            if not pos_text.strip():
                continue

            query = make_query(pos_chunk, use_llm=use_llm_query, model=llm_model)

            candidates = get_candidate_set(
                pos_chunk, chunk_by_id, chunks_by_doc, chunks_by_dieu, diagram_edges, strategy=strategy
            )
            if not candidates:
                continue

            scored = score_hardness(query, candidates, chunk_by_id, gamma=gamma)

            # Lọc Semi-Hard Negatives trong dải [tau, tau_max] (tau_max=0.85 ngăn cản false negatives)
            tau_max = 0.85
            best = next(((cid, strat, sc) for cid, strat, sc in scored if tau <= sc <= tau_max), None)

            if best is None:
                stats["skipped_tau"] += 1
                continue

            neg_id, mining_strat, hardness = best
            neg_chunk = chunk_by_id[neg_id]
            neg_text = neg_chunk.get("contextualized_text") or neg_chunk.get("original_text", "")

            pos_meta = pos_chunk.get("metadata") or {}
            neg_meta = neg_chunk.get("metadata") or {}

            record = {
                "query": query,
                "positive": pos_text,
                "hard_negative": neg_text,
                "meta": {
                    "pos_id": pos_id,
                    "neg_id": neg_id,
                    "mining_strategy": mining_strat,
                    "hardness_score": hardness,
                    "pos_doc_number": str(pos_meta.get("doc_number") or pos_chunk.get("doc_number") or ""),
                    "neg_doc_number": str(neg_meta.get("doc_number") or neg_chunk.get("doc_number") or ""),
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["total"] += 1
            stats[mining_strat] = stats.get(mining_strat, 0) + 1

            if stats["total"] % 5000 == 0:
                logger.info(f"Tiến độ: {stats['total']:,}/{max_triplets:,} triplets (sibling: {stats['sibling']}, neighbor: {stats['neighbor']}, ref: {stats['reference']})")

    tmp_file.replace(out_file)
    logger.info(f"✓ Hoàn tất xuất dữ liệu GG-SLM: {stats['total']:,} mẫu huấn luyện -> {out_file.resolve()}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Chương trình khai phá mẫu đối lập GG-SLM cho VietLawBERT")
    parser.add_argument("--input", default=DATA_STORAGE_ROOT / "processed" / "contextual_chunks.jsonl", help="Đường dẫn file chunks")
    parser.add_argument("--diagram-dir", default=DATA_STORAGE_ROOT / "raw_shards", help="Thư mục chứa sơ đồ quan hệ")
    parser.add_argument("--output", default=ARTIFACTS_DIR / "triplet_training_data.jsonl", help="Tệp xuất dữ liệu đối lập")
    parser.add_argument("--gamma", type=float, default=0.6, help="Trọng số tương quan BM25 vs Đồ thị")
    parser.add_argument("--tau", type=float, default=0.1, help="Ngưỡng Hardness tối thiểu")
    parser.add_argument("--max-triplets", type=int, default=50000, help="Số lượng bộ ba mục tiêu")
    parser.add_argument("--strategy", choices=["all", "sibling", "neighbor", "reference"], default="all")
    parser.add_argument("--use-llm-query", action="store_true", help="Sử dụng LLM sinh câu hỏi tình huống")
    parser.add_argument("--llm-model", default=getattr(config, "CONTEXTUALIZER_MODEL", "qwen2.5:1.5b"))
    args = parser.parse_args()

    run_pipeline(
        chunks_path=args.input,
        diagram_dir=args.diagram_dir,
        output_path=args.output,
        gamma=args.gamma,
        tau=args.tau,
        max_triplets=args.max_triplets,
        strategy=args.strategy,
        use_llm_query=args.use_llm_query,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()