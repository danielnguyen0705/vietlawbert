"""
evaluate_rrf.py - Đánh giá RRF Fusion trên các tập test single-hop và multi-hop.

Sử dụng:
    python evaluate_rrf.py --benchmark_dir ./benchmark
"""

import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("RRF_Eval")


def load_benchmark(benchmark_dir: str):
    """Load các test set từ benchmark directory."""
    datasets = {}
    for filename in ["single_hop.jsonl", "multi_hop.jsonl"]:
        path = os.path.join(benchmark_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Không tìm thấy {path}")
            continue

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        datasets[filename.replace(".jsonl", "")] = samples
        logger.info(f"[LOADED] {filename}: {len(samples)} samples")

    return datasets


def evaluate(retriever, samples: list, top_k: int = 3) -> dict:
    """
    Đánh giá trên một tập test.

    Metrics:
    - Hit@K: Tỷ lệ câu hỏi có đáp án đúng trong top-K
    - MRR: Mean Reciprocal Rank
    - Exact Match Article: Tỷ lệ khớp đúng Điều luật
    """
    hits = 0
    reciprocal_ranks = []
    article_matches = 0

    for sample in samples:
        query = sample["query"]
        gt_article = sample.get("ground_truth_article", "")
        gt_doc_num = sample.get("ground_truth_doc_number", "")
        gt_docs = sample.get("ground_truth_docs", [])

        results = retriever.search_context(query, top_k=top_k)

        # Check hit
        found = False
        for rank, r in enumerate(results, 1):
            if gt_article and gt_article in r.get("article", ""):
                found = True
                reciprocal_ranks.append(1.0 / rank)
                article_matches += 1
                break
            elif gt_docs:
                for gt in gt_docs:
                    if gt["article"] in r.get("article", ""):
                        found = True
                        reciprocal_ranks.append(1.0 / rank)
                        article_matches += 1
                        break
                if found:
                    break

        if found:
            hits += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(samples)
    metrics = {
        "Hit@K": hits / n if n else 0,
        "MRR": sum(reciprocal_ranks) / n if n else 0,
        "Article Match": article_matches / n if n else 0,
        "Total": n,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", default="./benchmark")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # Load Retriever (lazy import để tránh load heavy model khi test)
    try:
        from rag.retriever import LegalRetriever
        retriever = LegalRetriever()
    except Exception as e:
        logger.error(f"Không thể load Retriever: {e}")
        return

    datasets = load_benchmark(args.benchmark_dir)

    results = {}
    for name, samples in datasets.items():
        logger.info(f"\n[EVAL] {name} ({len(samples)} samples)")
        metrics = evaluate(retriever, samples, top_k=args.top_k)
        results[name] = metrics
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Lưu kết quả
    out_path = os.path.join(args.benchmark_dir, "rrf_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n[SAVED] Results -> {out_path}")


if __name__ == "__main__":
    main()