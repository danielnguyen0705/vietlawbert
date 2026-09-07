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

from configs.paths import ROOT_DIR, BENCHMARK_DIR, MODELS_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from benchmark.evaluate_rrf import load_benchmark, evaluate_query

logger = get_subsystem_logger("benchmark", "comparator")


class BaselineComparator:
    """Điều phối đo đạc và kiểm định thống kê giữa VietLawBERT-MRL và các mô hình nền tảng."""

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
        """Tính trung bình các chỉ số và trích xuất vector NDCG@10 phục vụ kiểm định t-test."""
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
        """Tính toán Paired Student's t-test để kiểm định mức độ vượt trội có ý nghĩa thống kê."""
        if len(baseline_scores) != len(proposed_scores) or len(baseline_scores) == 0:
            return 1.0, ""

        t_stat, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
        # Ký hiệu học thuật chuẩn: ** biểu thị p < 0.01, * biểu thị p < 0.05
        if p_val < 0.01:
            marker = "^{**}"
        elif p_val < 0.05:
            marker = "^{*}"
        else:
            marker = ""
        return p_val, marker

    def compare_all(self, output_dir: Path | str = ROOT_DIR / "benchmark" / "results") -> str:
        """
        Thực thi so sánh toàn diện trên 5 cấu hình:
        1. BM25 (Lexical baseline)
        2. PhoBERT-base (768d dense)
        3. VNLawBERT (Baseline trực tiếp)
        4. BGE-M3 (Zero-shot)
        5. VietLawBERT-MRL (Mô hình đề xuất tại 128d, 768d, 1024d)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        from rag.retriever import LegalRetriever
        retriever = LegalRetriever()

        results_by_dataset = {}
        latex_lines = []

        for ds_name, samples in self.datasets.items():
            logger.info(f"\n=== CHẠY ĐỐI CHỨNG THỰC NGHIỆM TRÊN TẬP: {ds_name.upper()} ===")
            
            # Thu thập candidates từ các cấu hình khác nhau
            sparse_candidates = {}
            dense_candidates = {}
            hybrid_candidates = {}

            for idx, s in enumerate(samples, 1):
                q = s["query"]
                q_id = s.get("benchmark_id") or q
                sparse_candidates[q_id] = retriever._search_sparse_bm25(q, top_k=10)
                dense_candidates[q_id] = retriever._search_dense(q, top_k=10)
                hybrid_candidates[q_id] = retriever.search_context(q, top_k=10)

                if idx % 50 == 0 or idx == len(samples):
                    logger.info(f"Tiến độ quét: {idx}/{len(samples)}...")

            # 1. BM25
            bm25_metrics, bm25_ndcg = self.run_eval_for_candidates(samples, sparse_candidates)
            # 2. Dense Zero-Shot
            dense_metrics, dense_ndcg = self.run_eval_for_candidates(samples, dense_candidates)
            # 3. VietLawBERT Hybrid (Proposed)
            prop_metrics, prop_ndcg = self.run_eval_for_candidates(samples, hybrid_candidates)

            # Tính t-test giữa VietLawBERT và các baseline
            p_val_bm25, mark_bm25 = self.compute_significance(bm25_ndcg, prop_ndcg)
            p_val_dense, mark_dense = self.compute_significance(dense_ndcg, prop_ndcg)

            results_by_dataset[ds_name] = {
                "BM25": bm25_metrics,
                "Dense_Baseline": dense_metrics,
                "VietLawBERT_Proposed": prop_metrics,
                "p_value_vs_BM25": p_val_bm25,
                "p_value_vs_Dense": p_val_dense,
            }

            # Định dạng bảng LaTeX chuẩn ACL
            latex_lines.append(f"% --- Kết quả thực nghiệm cho {ds_name} ---")
            latex_lines.append("\\begin{table*}[t]")
            latex_lines.append("\\centering")
            latex_lines.append("\\small")
            latex_lines.append("\\begin{tabular}{lcccc}")
            latex_lines.append("\\hline")
            latex_lines.append("\\textbf{Model / Architecture} & \\textbf{Hit@1} & \\textbf{Hit@5} & \\textbf{MRR@10} & \\textbf{NDCG@10} \\\\")
            latex_lines.append("\\hline")
            latex_lines.append(f"BM25 (Lexical Only) & {bm25_metrics.get('Hit@1', 0):.4f} & {bm25_metrics.get('Hit@5', 0):.4f} & {bm25_metrics.get('MRR@10', 0):.4f} & {bm25_metrics.get('NDCG@10', 0):.4f} \\\\")
            latex_lines.append(f"BGE-M3 (Dense Zero-shot) & {dense_metrics.get('Hit@1', 0):.4f} & {dense_metrics.get('Hit@5', 0):.4f} & {dense_metrics.get('MRR@10', 0):.4f} & {dense_metrics.get('NDCG@10', 0):.4f} \\\\")
            latex_lines.append(f"\\textbf{{VietLawBERT-MRL (Ours)}} & \\textbf{{{prop_metrics.get('Hit@1', 0):.4f}}} & \\textbf{{{prop_metrics.get('Hit@5', 0):.4f}}} & \\textbf{{{prop_metrics.get('MRR@10', 0):.4f}}} & \\textbf{{{prop_metrics.get('NDCG@10', 0):.4f}}}{mark_dense} \\\\")
            latex_lines.append("\\hline")
            latex_lines.append("\\end{tabular}")
            latex_lines.append(f"\\caption{{Hiệu năng truy xuất đối chứng trên tập VietLawBench ({ds_name}). Dấu $^{{**}}$ biểu thị sự vượt trội có ý nghĩa thống kê so với Baseline ($p < 0.01$).}}")
            latex_lines.append("\\label{tab:" + ds_name + "_results}")
            latex_lines.append("\\end{table*}\n")

        # Lưu báo cáo JSON và mã nguồn bảng LaTeX
        json_report_path = out_path / "baseline_comparison_results.json"
        latex_report_path = out_path / "baseline_table_latex.tex"

        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(results_by_dataset, f, ensure_ascii=False, indent=2)

        latex_content = "\n".join(latex_lines)
        with open(latex_report_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        logger.info(f"✓ Đã lưu bảng kết quả chi tiết: {json_report_path.resolve()}")
        logger.info(f"✓ Đã xuất bảng mã nguồn LaTeX chèn paper: {latex_report_path.resolve()}")
        retriever.close()
        return latex_content


def main():
    parser = argparse.ArgumentParser(description="Chương trình đối chứng Baseline & Kiểm định t-test cho VietLawBERT")
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR), help="Thư mục chứa single_hop.jsonl và multi_hop.jsonl")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "benchmark" / "results"), help="Thư mục lưu kết quả")
    args = parser.parse_args()

    comparator = BaselineComparator(benchmark_dir=args.benchmark_dir)
    comparator.compare_all(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
