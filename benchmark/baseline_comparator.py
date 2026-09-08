"""
baseline_comparator.py - Hệ thống thực nghiệm đối chứng trực diện (Head-to-Head Baseline Comparison).
Đo đạc hiệu năng IR và kiểm định ý nghĩa thống kê (Paired Student's t-test, p-value) theo chuẩn công bố Q1.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy import stats

from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from benchmark.evaluate_rrf import load_benchmark, evaluate_query

logger = get_subsystem_logger("benchmark", "comparator")


class BaselineComparator:
    """Điều phối đo đạc và kiểm định thống kê giữa VietLawBERT-MRL và các baseline."""

    def __init__(self, benchmark_dir: Path | str = BENCHMARK_DIR):
        self.benchmark_dir = Path(benchmark_dir)
        self.datasets = load_benchmark(self.benchmark_dir)
        if not self.datasets:
            raise FileNotFoundError(f"Thư mục {self.benchmark_dir} chưa có tệp single_hop.jsonl/multi_hop.jsonl!")

    @staticmethod
    def run_eval_for_candidates(
        samples: List[Dict[str, Any]],
        candidates_dict: Dict[str, List[Dict[str, Any]]],
        k_thresholds: List[int] = [1, 5, 10]
    ) -> Tuple[Dict[str, float], List[float]]:
        query_scores = []
        ndcg_vector = []

        for sample in samples:
            q_id = sample.get("benchmark_id") or sample.get("query")
            candidates = candidates_dict.get(q_id, [])
            metrics = evaluate_query(candidates, sample, k_thresholds)
            query_scores.append(metrics)
            ndcg_vector.append(metrics.get("NDCG@10", 0.0))

        total = len(query_scores)
        aggregated = {}
        if total > 0:
            for k in query_scores[0].keys():
                aggregated[k] = round(sum(s[k] for s in query_scores) / total, 4)

        return aggregated, ndcg_vector

    @staticmethod
    def compute_significance(baseline_scores: List[float], proposed_scores: List[float]) -> Tuple[float, str]:
        if len(baseline_scores) != len(proposed_scores) or len(baseline_scores) == 0:
            return 1.0, ""

        t_stat, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
        if p_val < 0.01:
            marker = "^{**}"
        elif p_val < 0.05:
            marker = "^{*}"
        else:
            marker = ""
        return p_val, marker

    def compare_all(self, output_dir: Path | str = ROOT_DIR / "benchmark" / "results") -> str:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        from database.qdrant_client import QdrantClientWrapper
        from rag.es_retriever import LegalElasticsearchRetriever
        from rag.retriever import LegalHybridRetriever
        from sentence_transformers import SentenceTransformer

        qdrant = QdrantClientWrapper(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        es = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=config.ES_INDEX_NAME)
        encoder = SentenceTransformer(config.BASE_MODEL_NAME)
        retriever = LegalHybridRetriever(qdrant_wrapper=qdrant, es_client=es.client, encoder_model=encoder)

        results_by_dataset = {}
        latex_lines = []

        for ds_name, samples in self.datasets.items():
            if ds_name == "vietlawbench_1000" and len(self.datasets) > 1:
                continue

            logger.info(f"\n=== CHẠY ĐỐI CHỨNG THỰC NGHIỆM TRÊN TẬP: {ds_name.upper()} ===")
            
            sparse_candidates = {}
            dense_candidates = {}
            hybrid_candidates = {}

            for idx, s in enumerate(samples, 1):
                q = s["query"]
                q_id = s.get("benchmark_id") or q
                # 1. Sparse BM25 qua Elasticsearch
                sparse_candidates[q_id] = retriever._search_sparse_es(q, top_k=10)
                # 2. Dense MRL 256d qua Qdrant
                dense_candidates[q_id] = retriever._search_dense(q, top_k=10)
                # 3. VietLawBERT Full Hybrid RRF + Graph-injected scoring
                hybrid_candidates[q_id] = retriever.retrieve(q, top_k=10)

                if idx % 50 == 0 or idx == len(samples):
                    logger.info(f"Tiến độ quét: {idx}/{len(samples)}...")

            bm25_metrics, bm25_ndcg = self.run_eval_for_candidates(samples, sparse_candidates)
            dense_metrics, dense_ndcg = self.run_eval_for_candidates(samples, dense_candidates)
            prop_metrics, prop_ndcg = self.run_eval_for_candidates(samples, hybrid_candidates)

            p_val_bm25, mark_bm25 = self.compute_significance(bm25_ndcg, prop_ndcg)
            p_val_dense, mark_dense = self.compute_significance(dense_ndcg, prop_ndcg)

            results_by_dataset[ds_name] = {
                "BM25": bm25_metrics,
                "Dense_Baseline": dense_metrics,
                "VietLawBERT_Proposed": prop_metrics,
                "p_value_vs_BM25": p_val_bm25,
                "p_value_vs_Dense": p_val_dense,
            }

            latex_lines.append(f"% --- Kết quả thực nghiệm cho {ds_name} ---")
            latex_lines.append("\\begin{table*}[t]")
            latex_lines.append("\\centering")
            latex_lines.append("\\small")
            latex_lines.append("\\begin{tabular}{lcccc}")
            latex_lines.append("\\hline")
            latex_lines.append("\\textbf{Model / Architecture} & \\textbf{Hit@1} & \\textbf{Hit@5} & \\textbf{MRR@10} & \\textbf{NDCG@10} \\\\")
            latex_lines.append("\\hline")
            latex_lines.append(f"BM25 (Elasticsearch) & {bm25_metrics.get('Hit@1', 0):.4f} & {bm25_metrics.get('Hit@5', 0):.4f} & {bm25_metrics.get('MRR@10', 0):.4f} & {bm25_metrics.get('NDCG@10', 0):.4f} \\\\")
            latex_lines.append(f"Dense Zero-shot (BGE-M3 256d) & {dense_metrics.get('Hit@1', 0):.4f} & {dense_metrics.get('Hit@5', 0):.4f} & {dense_metrics.get('MRR@10', 0):.4f} & {dense_metrics.get('NDCG@10', 0):.4f} \\\\")
            latex_lines.append(f"\\textbf{{VietLawBERT-MRL (Ours)}} & \\textbf{{{prop_metrics.get('Hit@1', 0):.4f}}} & \\textbf{{{prop_metrics.get('Hit@5', 0):.4f}}} & \\textbf{{{prop_metrics.get('MRR@10', 0):.4f}}} & \\textbf{{{prop_metrics.get('NDCG@10', 0):.4f}}}{mark_dense} \\\\")
            latex_lines.append("\\hline")
            latex_lines.append("\\end{tabular}")
            latex_lines.append(f"\\caption{{Hiệu năng truy xuất đối chứng trên tập VietLawBench ({ds_name}). Dấu $^{{**}}$ biểu thị sự vượt trội có ý nghĩa thống kê so với Baseline ($p < 0.01$).}}")
            latex_lines.append("\\label{tab:" + ds_name + "_results}")
            latex_lines.append("\\end{table*}\n")

        json_report_path = out_path / "baseline_comparison_results.json"
        latex_report_path = out_path / "baseline_table_latex.tex"

        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(results_by_dataset, f, ensure_ascii=False, indent=2)

        latex_content = "\n".join(latex_lines)
        with open(latex_report_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        logger.info(f"Đã lưu bảng kết quả chi tiết: {json_report_path.resolve()}")
        logger.info(f"Đã xuất bảng mã nguồn LaTeX chèn paper: {latex_report_path.resolve()}")
        return latex_content


def main():
    parser = argparse.ArgumentParser(description="Chương trình đối chứng Baseline & Kiểm định t-test cho VietLawBERT")
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "benchmark" / "results"))
    args = parser.parse_args()

    comparator = BaselineComparator(benchmark_dir=args.benchmark_dir)
    comparator.compare_all(output_dir=args.output_dir)


if __name__ == "__main__":
    main()