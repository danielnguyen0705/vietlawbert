"""
benchmark/baseline_comparator.py - Hệ thống thực nghiệm đối chứng Master Benchmark.
So sánh trực diện công bằng trên cùng kho tài liệu đầy đủ giữa các mô hình gốc và VietLawBERT-MRL.
Hỗ trợ kiểm định thống kê Paired Student's t-test có dấu và xuất bảng tổng hợp CSV.
"""

from __future__ import annotations

import re
import csv
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from scipy import stats
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.config import config
from configs.logging_config import get_subsystem_logger
from benchmark.evaluate_rrf import load_benchmark, evaluate_query

logger = get_subsystem_logger("benchmark", "comparator")


class UniversalEncoderWrapper:
    """Nạp và suy luận chuẩn hóa cho mọi kiến trúc Encoder: SBERT, HuggingFace AutoModel, Seq2Seq."""
    def __init__(self, model_id_or_path: str, device: str = "cpu", slice_dim: Optional[int] = None, is_e5: bool = False):
        self.device = device
        self.slice_dim = slice_dim
        self.is_e5 = is_e5
        path_obj = Path(model_id_or_path)

        if path_obj.exists() and (path_obj / "vietlawbert_mrl.pt").exists():
            logger.info("Nạp Checkpoint Fine-tuned nội bộ: %s", model_id_or_path)
            self.is_custom = True

            model_cfg = AutoConfig.from_pretrained(str(path_obj), trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(str(path_obj), trust_remote_code=True)

            is_seq2seq = getattr(model_cfg, "is_encoder_decoder", False)
            if (path_obj / "model_meta.json").exists():
                try:
                    with open(path_obj / "model_meta.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        is_seq2seq = meta.get("is_seq2seq", is_seq2seq)
                except Exception:
                    pass

            if is_seq2seq:
                try:
                    full_seq = AutoModelForSeq2SeqLM.from_pretrained(str(path_obj), config=model_cfg).to(device)
                    self.core_encoder = full_seq.get_encoder()
                except Exception:
                    self.core_encoder = AutoModel.from_pretrained(str(path_obj), config=model_cfg).to(device)
            else:
                self.core_encoder = AutoModel.from_pretrained(str(path_obj), config=model_cfg).to(device)

            state_dict = torch.load(path_obj / "vietlawbert_mrl.pt", map_location=device, weights_only=False)
            cleaned_state = {}
            for k, v in state_dict.items():
                new_k = re.sub(r"^(encoder\.|model\.)", "", k)
                cleaned_state[new_k] = v

            load_res = self.core_encoder.load_state_dict(cleaned_state, strict=False)
            if len(load_res.missing_keys) > 0:
                logger.warning("Trọng số nạp thiếu (missing keys): %s", load_res.missing_keys[:5])
            self.core_encoder.eval()
        else:
            try:
                self.sbert = SentenceTransformer(model_id_or_path, device=device)
                self.is_custom = False
            except Exception:
                logger.info("Nạp trực tiếp qua HuggingFace AutoModel: %s", model_id_or_path)
                self.is_custom = True
                model_cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
                if getattr(model_cfg, "is_encoder_decoder", False):
                    full_seq = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path, config=model_cfg).to(device)
                    self.core_encoder = full_seq.get_encoder()
                else:
                    self.core_encoder = AutoModel.from_pretrained(model_id_or_path, config=model_cfg).to(device)
                self.core_encoder.eval()

    def encode(self, texts: List[str], batch_size: int = 16, is_query: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.slice_dim or 768))

        # Tiền tố E5 phân định rạch ròi truy vấn và tài liệu
        if self.is_e5:
            prefix = "query: " if is_query else "passage: "
            texts = [f"{prefix}{t}" for t in texts]

        if not self.is_custom:
            embs = self.sbert.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=False)
        else:
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                tok = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                encoder_inputs = {k: v for k, v in tok.items() if k in ["input_ids", "attention_mask"]}
                
                with torch.no_grad():
                    outputs = self.core_encoder(**encoder_inputs)
                    last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                    mask = encoder_inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_emb = torch.sum(last_hidden * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    batch_vecs = (sum_emb / sum_mask).cpu().numpy()
                    all_embs.append(batch_vecs)
            embs = np.vstack(all_embs)

        if self.slice_dim is not None and self.slice_dim < embs.shape[1]:
            embs = embs[:, :self.slice_dim]

        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        return embs / norms


class MultiModelBaselineComparator:
    def __init__(self, benchmark_dir: Path | str = BENCHMARK_DIR, checkpoints_dir: str = "checkpoints", device: str = "cpu"):
        self.benchmark_dir = Path(benchmark_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device
        self.datasets = load_benchmark(self.benchmark_dir)
        if not self.datasets:
            from benchmark.build_vietlawbench import VietLawBenchBuilder
            builder = VietLawBenchBuilder()
            builder.build_master_benchmark(total_target=1000)
            self.datasets = load_benchmark(self.benchmark_dir)

    @staticmethod
    def compute_significance(baseline_scores: List[float], proposed_scores: List[float]) -> Tuple[float, str]:
        """Kiểm định Paired t-test có xét dấu chênh lệch hiệu năng."""
        if len(baseline_scores) != len(proposed_scores) or len(baseline_scores) < 2:
            return 1.0, "N/A"
        diff = np.array(proposed_scores) - np.array(baseline_scores)
        mean_diff = np.mean(diff)
        if np.all(diff == 0):
            return 1.0, "Đồng nhất"

        _, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
        p_val = float(p_val) if not np.isnan(p_val) else 1.0

        if p_val < 0.01:
            sig = "Tăng vượt trội (p < 0.01)" if mean_diff > 0 else "Giảm sút (p < 0.01)"
        elif p_val < 0.05:
            sig = "Tăng có ý nghĩa (p < 0.05)" if mean_diff > 0 else "Giảm sút (p < 0.05)"
        else:
            sig = "Chênh lệch không đáng kể"
        return p_val, sig

    def compare_all(self, output_dir: Path | str = ROOT_DIR / "benchmark" / "results"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        from rag.es_retriever import LegalElasticsearchRetriever
        es = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=config.ES_INDEX_NAME)

        # Danh sách so sánh đồng số chiều đầy đủ (full dim) trước khi nén MRL
        raw_backbones = [
            ("VNLawBERT (Chau et al., 2020)", "Chau/VNLawBERT", False, 768),
            ("vinai/phobert-base-v2", "vinai/phobert-base-v2", False, 768),
            ("vinai/phobert-large", "vinai/phobert-large", False, 1024),
            ("intfloat/multilingual-e5-base", "intfloat/multilingual-e5-base", True, 768),
            ("BAAI/bge-m3", "BAAI/bge-m3", False, 1024),
        ]

        summary_rows = []

        for ds_name, samples in self.datasets.items():
            if ds_name == "vietlawbench_1000" and len(self.datasets) > 1:
                continue

            test_samples = samples[:200]
            logger.info("=== ĐỐI CHUẨN MASTER BENCHMARK: %s (%d MẪU) ===", ds_name.upper(), len(test_samples))

            # Xây dựng kho ứng viên toàn diện đồng nhất cho tất cả các mô hình
            candidate_pool_dicts = []
            seen_chunks = set()
            for s in test_samples:
                c_id = s.get("ground_truth_chunk") or str(len(seen_chunks))
                if c_id not in seen_chunks:
                    seen_chunks.add(c_id)
                    candidate_pool_dicts.append({
                        "chunk_id": c_id,
                        "doc_number": s.get("ground_truth_doc_number", ""),
                        "article": s.get("ground_truth_article", ""),
                        "content": s.get("raw_content") or s.get("evidence_text", "N/A")
                    })

            candidate_texts = [c["content"] for c in candidate_pool_dicts]
            eval_results = {}
            ndcg_vectors = {}

            # 1. Đánh giá BM25 trên Elasticsearch
            bm25_scores, bm25_ndcg = [], []
            for s in test_samples:
                cands = es.search(s["query"], top_k=10)
                m = evaluate_query(cands, s, [1, 5, 10])
                bm25_scores.append(m)
                bm25_ndcg.append(m.get("NDCG@10", 0.0))
            t = len(bm25_scores) if bm25_scores else 1
            eval_results["BM25 (Elasticsearch)"] = {k: round(sum(x[k] for x in bm25_scores) / t, 4) for k in bm25_scores[0].keys() if k != "is_valid_query"}
            ndcg_vectors["BM25 (Elasticsearch)"] = bm25_ndcg

            # 2. Đánh giá các Dense Backbones ở CÙNG SỐ CHIỀU GỐC
            for display_name, model_id, is_e5, full_dim in raw_backbones:
                label_zero = f"{display_name} (Zero-shot @ d={full_dim})"
                try:
                    wrap_zero = UniversalEncoderWrapper(model_id, device=self.device, is_e5=is_e5)
                    q_z = wrap_zero.encode([s["query"] for s in test_samples], is_query=True)
                    c_z = wrap_zero.encode(candidate_texts, is_query=False)
                    sim_z = np.dot(q_z, c_z.T)

                    scores_z, ndcg_z = [], []
                    for idx, s in enumerate(test_samples):
                        top_idx = np.argsort(-sim_z[idx])[:10]
                        cands = [candidate_pool_dicts[i] for i in top_idx]
                        m = evaluate_query(cands, s, [1, 5, 10])
                        scores_z.append(m)
                        ndcg_z.append(m.get("NDCG@10", 0.0))

                    t_z = len(scores_z) if scores_z else 1
                    eval_results[label_zero] = {k: round(sum(x[k] for x in scores_z) / t_z, 4) for k in scores_z[0].keys() if k != "is_valid_query"}
                    ndcg_vectors[label_zero] = ndcg_z
                except Exception as ex:
                    logger.warning("Bỏ qua bản gốc [%s]: %s", label_zero, ex)

                # Đánh giá bản Fine-tuned (nếu có checkpoint) ở CÙNG SỐ CHIỀU GỐC trước
                safe_name = model_id.replace("/", "_")
                ckpt_path = self.checkpoints_dir / safe_name
                if ckpt_path.exists():
                    label_ft_full = f"{display_name} (Ours-MRL Full @ d={full_dim})"
                    label_ft_256 = f"{display_name} (Ours-MRL Slice @ d=256)"
                    for lbl, s_dim in [(label_ft_full, full_dim), (label_ft_256, 256)]:
                        try:
                            wrap_ft = UniversalEncoderWrapper(str(ckpt_path), device=self.device, slice_dim=s_dim, is_e5=is_e5)
                            q_ft = wrap_ft.encode([s["query"] for s in test_samples], is_query=True)
                            c_ft = wrap_ft.encode(candidate_texts, is_query=False)
                            sim_ft = np.dot(q_ft, c_ft.T)

                            scores_ft, ndcg_ft = [], []
                            for idx, s in enumerate(test_samples):
                                top_idx = np.argsort(-sim_ft[idx])[:10]
                                cands = [candidate_pool_dicts[i] for i in top_idx]
                                m = evaluate_query(cands, s, [1, 5, 10])
                                scores_ft.append(m)
                                ndcg_ft.append(m.get("NDCG@10", 0.0))

                            t_ft = len(scores_ft) if scores_ft else 1
                            eval_results[lbl] = {k: round(sum(x[k] for x in scores_ft) / t_ft, 4) for k in scores_ft[0].keys() if k != "is_valid_query"}
                            ndcg_vectors[lbl] = ndcg_ft
                        except Exception as ex:
                            logger.warning("Lỗi đánh giá bản fine-tune [%s]: %s", lbl, ex)

            # Cố định Proposed Model để đối chứng: BGE-M3 (Ours-MRL Full)
            proposed_key = next((k for k in eval_results if "Ours-MRL Full" in k), None) or list(eval_results.keys())[-1]
            proposed_vec = ndcg_vectors[proposed_key]

            for k in eval_results.keys():
                m = eval_results[k]
                p_val, sig = self.compute_significance(ndcg_vectors[k], proposed_vec) if k != proposed_key else (0.0, "Mô hình đề xuất (Ours)")
                summary_rows.append({
                    "Dataset": ds_name,
                    "Mo_Hinh_Embedding": k,
                    "Hit@1": m.get("Hit@1", 0.0),
                    "Hit@5": m.get("Hit@5", 0.0),
                    "Hit@10": m.get("Hit@10", 0.0),
                    "MRR@10": m.get("MRR@10", 0.0),
                    "NDCG@10": m.get("NDCG@10", 0.0),
                    "P_Value_vs_Proposed": f"{p_val:.2e}" if p_val > 0 else "-",
                    "Kiem_Dinh_Y_Nghia": sig
                })

        sum_file = out_path / "master_baseline_comparison.csv"
        with open(sum_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "Dataset", "Mo_Hinh_Embedding", "Hit@1", "Hit@5", "Hit@10",
                "MRR@10", "NDCG@10", "P_Value_vs_Proposed", "Kiem_Dinh_Y_Nghia"
            ])
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("✓ Đã lưu bảng tổng hợp Master Benchmark tại: %s", sum_file.resolve())


# Tạo alias để tương thích benchmark/__init__.py
BaselineComparator = MultiModelBaselineComparator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=str(BENCHMARK_DIR))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "benchmark" / "results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    comparator = MultiModelBaselineComparator(benchmark_dir=args.benchmark_dir, checkpoints_dir=args.checkpoints_dir, device=args.device)
    comparator.compare_all(output_dir=args.output_dir)


if __name__ == "__main__":
    main()