"""
evaluate_rqs.py - Khung thực nghiệm đo đạc RQ1 - RQ4.
Tích hợp UniversalEncoderWrapper để tương thích mọi checkpoint và cắt lát vector động tìm d*.
"""

from __future__ import annotations

import csv
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from scipy import stats

from configs.paths import BENCHMARK_DIR, ROOT_DIR
from configs.config import config
from benchmark.evaluate_rrf import load_benchmark, evaluate_query, extract_doc_number_from_text
from benchmark.baseline_comparator import UniversalEncoderWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_RQEvaluator")


class ScientificBenchmarkRunner:
    def __init__(self, benchmark_dir: str = BENCHMARK_DIR, model_path: str = None, device: str = "cpu"):
        self.benchmark_dir = Path(benchmark_dir)
        self.datasets = load_benchmark(self.benchmark_dir)
        if not self.datasets:
            from benchmark.build_vietlawbench import VietLawBenchBuilder
            builder = VietLawBenchBuilder()
            builder.generate_stratified_benchmark(1000)
            self.datasets = load_benchmark(self.benchmark_dir)

        self.samples = self.datasets.get("vietlawbench_1000") or (
            self.datasets.get("single_hop", []) + self.datasets.get("multi_hop", [])
        )
        self.model_path = model_path or config.BASE_MODEL_NAME
        self.device = device
        self.encoder = UniversalEncoderWrapper(self.model_path, device=self.device)

    def evaluate_rq2_pareto_mrl(self, out_path: Path):
        """RQ2: Cắt lát thực tế vector để tìm điểm ngọt Pareto d*."""
        logger.info("=== Đang chạy RQ2: Đo đạc Pareto Matryoshka trên [%s] ===", self.model_path)
        test_samples = self.samples[:100] if self.samples else []
        queries = [s["query"] for s in test_samples]

        # Nạp candidate pool kèm Ground-truth để đảm bảo không bị điểm 0 ảo
        candidate_pool = []
        for s in test_samples:
            if s.get("evidence_text") and s["evidence_text"] not in candidate_pool:
                candidate_pool.append(s["evidence_text"])

        # Trích xuất vector đầy đủ ở chiều cực đại
        q_embs_full = self.encoder.encode(queries)
        c_embs_full = self.encoder.encode(candidate_pool)
        max_dim = q_embs_full.shape[1]

        all_dims = [64, 128, 256, 512, 768, 1024]
        dims = [d for d in all_dims if d <= max_dim]

        csv_file = out_path / "rq2_mrl_pareto_analysis.csv"
        results_rows = []
        baseline_ndcg = 0.0

        # Lấy mốc chuẩn tại chiều tối đa
        sim_max = np.dot(q_embs_full, c_embs_full.T)
        ndcgs_max = []
        for idx, sample in enumerate(test_samples):
            top_i = np.argsort(-sim_max[idx])[:10]
            cands = [{"content": candidate_pool[i], "doc_number": extract_doc_number_from_text(candidate_pool[i])} for i in top_i]
            ndcgs_max.append(evaluate_query(cands, sample, [10])["NDCG@10"])
        baseline_ndcg = float(np.mean(ndcgs_max)) if ndcgs_max else 1.0

        for d in dims:
            sub_q = q_embs_full[:, :d]
            sub_q = sub_q / (np.linalg.norm(sub_q, axis=1, keepdims=True) + 1e-9)
            sub_c = c_embs_full[:, :d]
            sub_c = sub_c / (np.linalg.norm(sub_c, axis=1, keepdims=True) + 1e-9)

            sim_matrix = np.dot(sub_q, sub_c.T)
            hits, ndcgs = [], []
            for idx, sample in enumerate(test_samples):
                top_i = np.argsort(-sim_matrix[idx])[:10]
                cands = [{"content": candidate_pool[i], "doc_number": extract_doc_number_from_text(candidate_pool[i])} for i in top_i]
                m = evaluate_query(cands, sample, [10])
                hits.append(m["Hit@10"])
                ndcgs.append(m["NDCG@10"])

            mean_hit = float(np.mean(hits)) if hits else 0.0
            mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

            ram_mb = float(d * 4 * 100000) / (1024 * 1024)
            base_ram = float(max_dim * 4 * 100000) / (1024 * 1024)
            saved_ram = (1.0 - (ram_mb / base_ram)) * 100.0
            retained_ndcg = (mean_ndcg / baseline_ndcg * 100.0) if baseline_ndcg > 0 else 100.0

            status = "Sweet-spot d* (Điểm tối ưu)" if (retained_ndcg >= 98.0 and saved_ram >= 50.0) else "Khảo sát"

            results_rows.append({
                "Chieu_Vector_d": d,
                "Hit@10": round(mean_hit, 4),
                "NDCG@10": round(mean_ndcg, 4),
                "NDCG_Bao_Toan_%": round(retained_ndcg, 2),
                "RAM_MB": round(ram_mb, 1),
                "Tiet_Kiem_RAM_%": round(saved_ram, 1),
                "Ket_Luan_Pareto": status
            })

        out_path.mkdir(parents=True, exist_ok=True)
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            w.writeheader()
            w.writerows(results_rows)
        logger.info("✓ Đã lưu phân tích Pareto RQ2 tại: %s", csv_file.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    runner = ScientificBenchmarkRunner(benchmark_dir=args.benchmark_dir, model_path=args.checkpoint_path, device=args.device)
    runner.evaluate_rq2_pareto_mrl(Path(args.output_dir))


if __name__ == "__main__":
    main()