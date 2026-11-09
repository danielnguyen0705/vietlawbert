r"""
benchmark/measure_cpu_latency.py — Đo độ trễ CPU cho query encoding, reranker và toàn luồng truy xuất.
Đồng thời đánh giá chất lượng của reranker nhẹ (mmarco-mMiniLMv2-L12-H384-v1).
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

import numpy as np
import httpx
from elasticsearch import Elasticsearch

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from benchmark.metrics_article import ndcg_at_k, recall_at_k, mrr_at_k, f2_at_k
from benchmark.rerank_topn import rerank_first_n
from benchmark.tune_vlqa_fusion import weighted_rrf
from benchmark.evaluate_vlqa_article import aggregate_pieces
from preprocess.vi_segment import get_segmenter, segment_text

logger = get_subsystem_logger("benchmark", "measure_cpu_latency")


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def summarize(lat_list: List[float]) -> Dict[str, float]:
    if not lat_list:
        return {"P50_ms": 0.0, "P95_ms": 0.0, "share_under_500ms": 0.0}
    p50 = percentile(lat_list, 50)
    p95 = percentile(lat_list, 95)
    under_500 = sum(1 for x in lat_list if x < 500.0) / len(lat_list)
    return {
        "P50_ms": p50,
        "P95_ms": p95,
        "share_under_500ms": under_500,
    }


def rebuild_piece_texts(
    articles: Dict[str, str],
    piece_meta: List[Dict[str, Any]],
    tokenizer: Any,
    window: int = 510,
    stride: int = 384,
) -> List[str]:
    """
    Tái tạo văn bản từng piece từ legal_corpus dựa trên character offsets của tokenizer.
    Kiểm tra n_pieces và n_tokens của từng piece khớp chính xác với piece_meta.
    """
    pm_by_chunk = defaultdict(list)
    for p in piece_meta:
        pm_by_chunk[p["chunk_id"]].append(p)

    rebuilt = [None] * len(piece_meta)

    for cid, p_list in pm_by_chunk.items():
        if cid not in articles:
            raise ValueError(f"Không tìm thấy article cho chunk_id: {cid}")
        txt = articles[cid].strip()

        try:
            enc = tokenizer(txt, add_special_tokens=False, return_offsets_mapping=True)
        except TypeError:
            enc = tokenizer(txt)

        if isinstance(enc, dict):
            offsets = enc.get("offset_mapping")
        else:
            offsets = getattr(enc, "offset_mapping", None)

        if offsets is None:
            raise ValueError("Tokenizer không trả về offset_mapping")

        total_tokens = len(offsets)
        exp_n_pieces = p_list[0].get("n_pieces", len(p_list))
        if len(p_list) != exp_n_pieces:
            raise AssertionError(f"n_pieces mismatch cho {cid}: {len(p_list)} != {exp_n_pieces}")

        for p in p_list:
            pidx = p["piece_index"]
            start_t = pidx * stride
            end_t = min(start_t + window, total_tokens)
            n_tokens = max(0, end_t - start_t)
            exp_tokens = p.get("n_tokens")
            if exp_tokens is not None and n_tokens != exp_tokens:
                raise AssertionError(f"n_tokens mismatch cho piece {p.get('piece_id')}: {n_tokens} != {exp_tokens}")

            start_c = offsets[start_t][0]
            end_c = offsets[end_t - 1][1]
            p_text = txt[start_c:end_c]
            pid = p["piece_id"]
            rebuilt[pid] = p_text

    if any(x is None for x in rebuilt):
        raise ValueError("Có piece_id chưa được khôi phục nội dung")

    return rebuilt


def load_dev_questions(path: Path | str) -> List[Dict[str, Any]]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    raw_vlqa_dir = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    vlqa_dir = raw_vlqa_dir if raw_vlqa_dir else "D:\\VLQA"

    default_corpus = str(Path(vlqa_dir) / "legal_corpus.json")
    default_vectors_dir = str(Path(vlqa_dir) / "vectors" / "bge-m3")
    default_rerank_file = str(Path(vlqa_dir) / "task6" / "kaggle_out" / "rerank_top50.jsonl")
    default_fusion_params = str(BENCHMARK_DIR / "results" / "vlqa_fusion_params.json")
    default_dev_file = str(BENCHMARK_DIR / "vlqa_dev500.jsonl")

    ap = argparse.ArgumentParser(description="Đo độ trễ CPU và chất lượng reranker nhẹ")
    ap.add_argument("--parts", default=None, help="Danh sách phần cần chạy: B1,B2,B3")
    ap.add_argument("--quality-light", action="store_true", help="Chạy đánh giá Part C cho mMiniLM trên tập dev")
    ap.add_argument("--corpus", default=default_corpus)
    ap.add_argument("--vectors-dir", default=default_vectors_dir)
    ap.add_argument("--rerank-file", default=default_rerank_file)
    ap.add_argument("--fusion-params", default=default_fusion_params)
    ap.add_argument("--dev-file", default=default_dev_file)
    ap.add_argument("--out-latency", default=str(BENCHMARK_DIR / "results" / "cpu_latency.csv"))
    ap.add_argument("--out-light", default=str(BENCHMARK_DIR / "results" / "vlqa_light_reranker_dev.csv"))
    args = ap.parse_args(argv)

    if args.parts is not None:
        parts = [p.strip().upper() for p in args.parts.split(",") if p.strip()]
    elif args.quality_light:
        parts = []
    else:
        parts = ["B1", "B2", "B3"]

    # Nạp tập dev questions
    dev_questions = load_dev_questions(args.dev_file)
    logger.info("Nạp %d câu hỏi dev từ %s", len(dev_questions), args.dev_file)

    # Nạp piece_meta và legal_corpus để tái tạo piece texts nếu cần cho B2, B3, Part C
    needs_rebuild = any(p in parts for p in ["B2", "B3"]) or args.quality_light
    rebuilt_piece_texts_list: List[str] = []
    chunk_to_piece_ids: Dict[str, List[int]] = defaultdict(list)
    piece_vectors = None

    if needs_rebuild:
        from transformers import AutoTokenizer

        logger.info("Đang nạp piece_meta từ %s...", Path(args.vectors_dir) / "piece_meta.jsonl")
        piece_meta = []
        with open(Path(args.vectors_dir) / "piece_meta.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    piece_meta.append(item)
                    chunk_to_piece_ids[item["chunk_id"]].append(item["piece_id"])

        logger.info("Đang nạp legal_corpus từ %s...", args.corpus)
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        articles_dict = {}
        for doc in corpus_data:
            for art in doc.get("content", []):
                aid = art.get("aid")
                if aid is not None:
                    c = art.get("content_Article") or art.get("content") or ""
                    if c.strip():
                        articles_dict[f"vlqa_{aid}"] = c.strip()

        logger.info("Đang tái tạo %d pieces bằng tokenizer BAAI/bge-m3...", len(piece_meta))
        tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        rebuilt_piece_texts_list = rebuild_piece_texts(articles_dict, piece_meta, tok, window=510, stride=384)
        logger.info("✓ Tái tạo thành công %d pieces văn bản", len(rebuilt_piece_texts_list))

        if "B3" in parts:
            pvec_path = Path(args.vectors_dir) / "piece_vectors.npy"
            logger.info("Nạp piece vectors từ %s...", pvec_path)
            piece_vectors = np.load(pvec_path)

    # Nạp rerank records nếu cần cho B2 hoặc Part C
    dev_rerank_records = []
    if "B2" in parts or args.quality_light:
        with open(args.rerank_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("split") == "dev":
                        dev_rerank_records.append(item)
        logger.info("Nạp %d bản ghi rerank dev từ %s", len(dev_rerank_records), args.rerank_file)

    latency_rows: List[Dict[str, Any]] = []

    # =========================================================================
    # B1 — Query encoding
    # =========================================================================
    if "B1" in parts:
        logger.info("--- Bắt đầu đo B1: Query encoding với BAAI/bge-m3 trên CPU ---")
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        st_model.max_seq_length = 128

        # 5 câu warm-up
        logger.info("Khởi động (warm-up) 5 câu...")
        for i in range(5):
            st_model.encode(dev_questions[i]["query"], normalize_embeddings=True)

        # Đo thời gian 100 câu
        timed_latencies_b1 = []
        cpu_vectors_100 = []
        logger.info("Đo độ trễ 100 câu dev đầu tiên...")
        for i in range(100):
            q_text = dev_questions[i]["query"]
            t0 = time.perf_counter()
            vec = st_model.encode(q_text, normalize_embeddings=True)
            t1 = time.perf_counter()
            timed_latencies_b1.append((t1 - t0) * 1000.0)
            cpu_vectors_100.append(vec)

        # So sánh cosine với Kaggle query vectors
        qvecs_kaggle = np.load(Path(args.vectors_dir) / "query_vectors.npy")
        cosines = []
        for i in range(100):
            cpu_v = cpu_vectors_100[i]
            kag_v = qvecs_kaggle[i]
            c_val = float(np.dot(cpu_v, kag_v) / (np.linalg.norm(cpu_v) * np.linalg.norm(kag_v)))
            cosines.append(c_val)

        min_cos = min(cosines)
        mean_cos = sum(cosines) / len(cosines)
        logger.info("✓ So sánh Cosine với Kaggle query vectors (100 câu): min=%.6f, mean=%.6f", min_cos, mean_cos)
        if min_cos < 0.99:
            logger.error("DỪNG: min cosine %.6f < 0.99!", min_cos)
            sys.exit(1)

        summ_b1 = summarize(timed_latencies_b1)
        latency_rows.append({
            "part": "B1",
            "pipeline_or_model": "BAAI/bge-m3",
            "n": 1,
            "n_questions": 100,
            "P50_ms": f"{summ_b1['P50_ms']:.2f}",
            "P95_ms": f"{summ_b1['P95_ms']:.2f}",
            "share_under_500ms": f"{summ_b1['share_under_500ms']:.4f}",
        })
        logger.info("B1 Hoàn thành: P50=%.2fms, P95=%.2fms, share<500ms=%.2f%%",
                    summ_b1["P50_ms"], summ_b1["P95_ms"], summ_b1["share_under_500ms"] * 100)

    # =========================================================================
    # B2 — Reranker only
    # =========================================================================
    if "B2" in parts:
        logger.info("--- Bắt đầu đo B2: Reranker only trên CPU ---")
        from sentence_transformers import CrossEncoder

        reranker_configs = [
            ("BAAI/bge-reranker-v2-m3", 5),
            ("BAAI/bge-reranker-v2-m3", 10),
            ("BAAI/bge-reranker-v2-m3", 20),
            ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 10),
            ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 20),
            ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 50),
        ]

        # Khởi tạo cache model
        loaded_models: Dict[str, CrossEncoder] = {}

        for model_id, n_cand in reranker_configs:
            if model_id not in loaded_models:
                logger.info("Đang nạp mô hình reranker %s...", model_id)
                loaded_models[model_id] = CrossEncoder(model_id, max_length=512, device="cpu")

            m = loaded_models[model_id]
            logger.info("Đo mô hình %s với N=%d...", model_id, n_cand)

            # 2 warm-up questions
            for i in range(2):
                r = dev_rerank_records[i]
                q_text = dev_questions[i]["query"]
                pairs = [(q_text, rebuilt_piece_texts_list[pid]) for pid in r["best_piece_ids"][:n_cand]]
                m.predict(pairs, batch_size=n_cand)

            # 30 timed questions
            timed_lats = []
            for i in range(30):
                r = dev_rerank_records[i]
                q_text = dev_questions[i]["query"]
                pairs = [(q_text, rebuilt_piece_texts_list[pid]) for pid in r["best_piece_ids"][:n_cand]]
                t0 = time.perf_counter()
                m.predict(pairs, batch_size=n_cand)
                t1 = time.perf_counter()
                timed_lats.append((t1 - t0) * 1000.0)

            summ_b2 = summarize(timed_lats)
            display_name = "bge-reranker-v2-m3" if "bge" in model_id else "mMiniLM"
            latency_rows.append({
                "part": "B2",
                "pipeline_or_model": display_name,
                "n": n_cand,
                "n_questions": 30,
                "P50_ms": f"{summ_b2['P50_ms']:.2f}",
                "P95_ms": f"{summ_b2['P95_ms']:.2f}",
                "share_under_500ms": f"{summ_b2['share_under_500ms']:.4f}",
            })
            logger.info("  [%s, N=%d] P50=%.2fms, P95=%.2fms, share<500ms=%.2f%%",
                        display_name, n_cand, summ_b2["P50_ms"], summ_b2["P95_ms"], summ_b2["share_under_500ms"] * 100)

    # =========================================================================
    # B3 — End-to-end trên 50 câu dev
    # =========================================================================
    if "B3" in parts:
        logger.info("--- Bắt đầu đo B3: Toàn luồng truy xuất (End-to-End) trên CPU (50 câu dev) ---")
        from sentence_transformers import SentenceTransformer, CrossEncoder

        # Khởi tạo các thành phần
        st_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        st_model.max_seq_length = 128

        reranker_bge = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
        reranker_mini = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512, device="cpu")

        seg_fn = get_segmenter("pyvi")

        # Nạp tham số fusion
        with open(args.fusion_params, "r", encoding="utf-8") as f:
            f_params = json.load(f)
        w_dense = float(f_params.get("w_dense", 0.7))
        w_bm25 = float(f_params.get("w_bm25", 0.3))
        rrf_k = int(f_params.get("rrf_k", 10))

        # Đọc N tối ưu từ bảng vlqa_rerank_topn_quality.csv
        topn_csv = Path(BENCHMARK_DIR / "results" / "vlqa_rerank_topn_quality.csv")
        n_tune = 30
        if topn_csv.exists():
            with open(topn_csv, "r", encoding="utf-8") as qf:
                reader = csv.DictReader(qf)
                for r in reader:
                    if r.get("split") == "tune" and r.get("chosen_on_tune") == "True":
                        n_tune = int(r["n"])
                        break
        chosen_n_b3 = min(n_tune, 20)
        logger.info("N chọn từ tune: %d, N áp dụng cho B3 (cap at 20): %d", n_tune, chosen_n_b3)
        print(f"N chọn từ tune: {n_tune}, N áp dụng cho B3 (cap at 20): {chosen_n_b3}")

        # Khởi tạo ES client và HTTP client Qdrant
        es_client = Elasticsearch([config.ES_HOST])
        qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
        qdrant_http = httpx.Client(base_url=qdrant_url, timeout=120.0)

        def run_dense_pipeline(q_text: str) -> List[str]:
            cpu_vec = st_model.encode(q_text, normalize_embeddings=True)
            res = qdrant_http.post(
                "/collections/vlqa_dense_bge_m3/points/search",
                json={
                    "vector": cpu_vec.tolist(),
                    "limit": 400,
                    "params": {"exact": False},
                    "with_payload": ["chunk_id"],
                },
            )
            data = res.json().get("result", [])
            hits = [(pt.get("payload", {}).get("chunk_id"), pt.get("score", 0.0)) for pt in data]
            return aggregate_pieces(hits)[:10]

        def run_hybrid_wrrf_pipeline(q_text: str, top_k_ret: int = 10) -> Tuple[List[str], np.ndarray]:
            cpu_vec = st_model.encode(q_text, normalize_embeddings=True)
            res = qdrant_http.post(
                "/collections/vlqa_dense_bge_m3/points/search",
                json={
                    "vector": cpu_vec.tolist(),
                    "limit": 400,
                    "params": {"exact": False},
                    "with_payload": ["chunk_id"],
                },
            )
            data = res.json().get("result", [])
            dense_hits = [(pt.get("payload", {}).get("chunk_id"), pt.get("score", 0.0)) for pt in data]
            dense_ranked = aggregate_pieces(dense_hits)

            seg_q = segment_text(q_text, seg_fn)
            es_res = es_client.search(
                index="vlqa_sparse_seg_idx",
                query={"match": {"content_seg": seg_q}},
                size=100,
                _source=["chunk_id"],
            )
            bm25_hits = [h["_source"]["chunk_id"] for h in es_res["hits"]["hits"]]
            fused = weighted_rrf([bm25_hits, dense_ranked], [w_bm25, w_dense], k=rrf_k, depth=top_k_ret)
            return fused, cpu_vec

        def run_rerank_pipeline(q_text: str, model: CrossEncoder, n_cand: int = 20) -> List[str]:
            top_candidates, cpu_vec = run_hybrid_wrrf_pipeline(q_text, top_k_ret=n_cand)
            pairs = []
            for cid in top_candidates:
                pids = chunk_to_piece_ids[cid]
                if len(pids) == 1:
                    best_pid = pids[0]
                else:
                    scores = np.dot(piece_vectors[pids], cpu_vec)
                    best_pid = pids[int(np.argmax(scores))]
                pairs.append((q_text, rebuilt_piece_texts_list[best_pid]))

            scores = model.predict(pairs, batch_size=n_cand)
            sorted_candidates = [c for c, _ in sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)]
            return sorted_candidates[:10]

        b3_configs = [
            ("dense", lambda q: run_dense_pipeline(q), 10),
            ("hybrid_wrrf", lambda q: run_hybrid_wrrf_pipeline(q, 10)[0], 10),
            ("hybrid_wrrf + rerank bge-reranker-v2-m3", lambda q: run_rerank_pipeline(q, reranker_bge, chosen_n_b3), chosen_n_b3),
            ("hybrid_wrrf + rerank mMiniLM", lambda q: run_rerank_pipeline(q, reranker_mini, 20), 20),
        ]

        for pipe_name, pipe_fn, n_cand in b3_configs:
            logger.info("Đo pipeline End-to-End: %s...", pipe_name)
            # 3 câu warm-up
            for i in range(3):
                pipe_fn(dev_questions[i]["query"])

            # 50 câu đo độ trễ
            timed_lats = []
            for i in range(50):
                q_text = dev_questions[i]["query"]
                t0 = time.perf_counter()
                pipe_fn(q_text)
                t1 = time.perf_counter()
                timed_lats.append((t1 - t0) * 1000.0)

            summ_b3 = summarize(timed_lats)
            latency_rows.append({
                "part": "B3",
                "pipeline_or_model": pipe_name,
                "n": n_cand,
                "n_questions": 50,
                "P50_ms": f"{summ_b3['P50_ms']:.2f}",
                "P95_ms": f"{summ_b3['P95_ms']:.2f}",
                "share_under_500ms": f"{summ_b3['share_under_500ms']:.4f}",
            })
            logger.info("  [%s] P50=%.2fms, P95=%.2fms, share<500ms=%.2f%%",
                        pipe_name, summ_b3["P50_ms"], summ_b3["P95_ms"], summ_b3["share_under_500ms"] * 100)

        qdrant_http.close()
        es_client.close()

    # Ghi file kết quả độ trễ nếu có chạy phần B nào
    if latency_rows:
        out_lat = Path(args.out_latency)
        out_lat.parent.mkdir(parents=True, exist_ok=True)
        headers = ["part", "pipeline_or_model", "n", "n_questions", "P50_ms", "P95_ms", "share_under_500ms"]
        with open(out_lat, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(latency_rows)
        logger.info("✓ Đã lưu bảng kết quả độ trễ CPU vào %s", out_lat)
        print("\n" + "=" * 80)
        print("BẢNG KẾT QUẢ ĐỘ TRỄ CPU (cpu_latency.csv):")
        print("=" * 80)
        with open(out_lat, "r", encoding="utf-8") as f:
            print(f.read().strip())
        print("=" * 80 + "\n")

    # =========================================================================
    # Part C — Light reranker quality on dev
    # =========================================================================
    if args.quality_light:
        logger.info("--- Bắt đầu Part C: Đánh giá chất lượng reranker nhẹ mMiniLM trên 500 câu dev ---")
        from sentence_transformers import CrossEncoder

        mini_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512, device="cpu")

        # Nạp gold dev
        gold_map: Dict[int, Set[str]] = {}
        with open(args.dev_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gold_map[int(item["qid"])] = set(item.get("relevant_chunk_ids", []))

        ndcg_l, rec_l, mrr_l, f2_l = [], [], [], []
        logger.info("Đang suy luận reranking 500 câu dev (N=20)...")
        for i, r in enumerate(dev_rerank_records):
            qid = int(r["qid"])
            q_text = dev_questions[i]["query"]
            cands = r["candidates"]
            best_pids = r["best_piece_ids"][:20]
            pairs = [(q_text, rebuilt_piece_texts_list[pid]) for pid in best_pids]
            scores = mini_model.predict(pairs, batch_size=20)
            ranked = rerank_first_n(cands, scores, 20)

            g = gold_map.get(qid, set())
            ndcg_l.append(ndcg_at_k(ranked, g, 10))
            rec_l.append(recall_at_k(ranked, g, 10))
            mrr_l.append(mrr_at_k(ranked, g, 10))
            f2_l.append(f2_at_k(ranked, g, 2))

            if (i + 1) % 100 == 0 or (i + 1) == len(dev_rerank_records):
                logger.info("  Đã xử lý %d/%d câu...", i + 1, len(dev_rerank_records))

        ndcg_mini = sum(ndcg_l) / len(ndcg_l)
        rec_mini = sum(rec_l) / len(rec_l)
        mrr_mini = sum(mrr_l) / len(mrr_l)
        f2_mini = sum(f2_l) / len(f2_l)

        # Đọc kết quả từ bảng vlqa_rerank_topn_quality.csv cho hybrid_rrf (n=0) và bge-reranker-v2-m3 (n=20) trên dev
        topn_csv = Path(BENCHMARK_DIR / "results" / "vlqa_rerank_topn_quality.csv")
        dev_n0 = {}
        dev_n20 = {}
        if topn_csv.exists():
            with open(topn_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("split") == "dev":
                        if int(r.get("n", -1)) == 0:
                            dev_n0 = r
                        elif int(r.get("n", -1)) == 20:
                            dev_n20 = r

        light_rows = [
            {
                "model": "hybrid_rrf (baseline)",
                "n": 0,
                "NDCG@10": f"{float(dev_n0.get('NDCG@10', 0.6083)):.4f}",
                "Recall@10": f"{float(dev_n0.get('Recall@10', 0.7589)):.4f}",
                "MRR@10": f"{float(dev_n0.get('MRR@10', 0.5989)):.4f}",
                "F2@2": f"{float(dev_n0.get('F2@2', 0.4748)):.4f}",
            },
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "n": 20,
                "NDCG@10": f"{float(dev_n20.get('NDCG@10', 0.6829)):.4f}",
                "Recall@10": f"{float(dev_n20.get('Recall@10', 0.8083)):.4f}",
                "MRR@10": f"{float(dev_n20.get('MRR@10', 0.6813)):.4f}",
                "F2@2": f"{float(dev_n20.get('F2@2', 0.5403)):.4f}",
            },
            {
                "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                "n": 20,
                "NDCG@10": f"{ndcg_mini:.4f}",
                "Recall@10": f"{rec_mini:.4f}",
                "MRR@10": f"{mrr_mini:.4f}",
                "F2@2": f"{f2_mini:.4f}",
            },
        ]

        out_light = Path(args.out_light)
        out_light.parent.mkdir(parents=True, exist_ok=True)
        headers = ["model", "n", "NDCG@10", "Recall@10", "MRR@10", "F2@2"]
        with open(out_light, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(light_rows)

        logger.info("✓ Đã lưu bảng so sánh reranker nhẹ vào %s", out_light)
        print("\n" + "=" * 80)
        print("BẢNG SO SÁNH CHẤT LƯỢNG RERANKER NHẸ TRÊN TẬP DEV (vlqa_light_reranker_dev.csv):")
        print("=" * 80)
        with open(out_light, "r", encoding="utf-8") as f:
            print(f.read().strip())
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()