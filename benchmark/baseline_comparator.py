"""
baseline_comparator.py - Hệ thống thực nghiệm đối chứng Master Benchmark.
Tự động đối đầu trực diện giữa [Mô hình Gốc (Zero-shot)] và [Mô hình Đã Tinh Chỉnh (Ours-MRL)]
trên toàn bộ 13 kiến trúc. Đo đạc Paired Student's t-test và xuất bảng tổng hợp CSV.
"""

from __future__ import annotations

import csv
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
    """Nạp vạn năng mọi loại mô hình: SentenceTransformer, HuggingFace AutoModel, và Seq2Seq Encoder."""
    def __init__(self, model_id_or_path: str, device: str = "cpu", slice_dim: int = None, is_e5: bool = False):
        self.device = device
        self.slice_dim = slice_dim
        self.is_e5 = is_e5
        path_obj = Path(model_id_or_path)

        if path_obj.exists() and (path_obj / "vietlawbert_mrl.pt").exists():
            logger.info("Nạp Checkpoint Fine-tuned nội bộ: %s", model_id_or_path)
            self.is_custom = True
            with open(path_obj / "model_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.tokenizer = AutoTokenizer.from_pretrained(str(path_obj), trust_remote_code=True)
            model_cfg = AutoConfig.from_dict(meta["base_config"])

            if meta.get("is_seq2seq", False):
                full_seq = AutoModelForSeq2SeqLM.from_pretrained(str(path_obj), config=model_cfg).to(device)
                self.core_encoder = full_seq.get_encoder()
            else:
                self.core_encoder = AutoModel.from_pretrained(str(path_obj), config=model_cfg).to(device)

            try:
                state_dict = torch.load(path_obj / "vietlawbert_mrl.pt", map_location=device, weights_only=False)
            except TypeError:
                state_dict = torch.load(path_obj / "vietlawbert_mrl.pt", map_location=device)

            cleaned_state = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            self.core_encoder.load_state_dict(cleaned_state, strict=False)
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

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.is_e5:
            texts = [f"query: {t}" for t in texts]

        if not self.is_custom:
            embs = self.sbert.encode(texts, show_progress_bar=False, normalize_embeddings=False)
        else:
            tok = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.core_encoder(**tok)
                last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                mask = tok["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_emb = torch.sum(last_hidden * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                embs = (sum_emb / sum_mask).cpu().numpy()

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
            builder.generate_stratified_benchmark(1000)
            self.datasets = load_benchmark(self.benchmark_dir)

    @staticmethod
    def compute_significance(baseline_scores: List[float], proposed_scores: List[float]) -> Tuple[float, str]:
        if len(baseline_scores) != len(proposed_scores) or len(baseline_scores) < 2:
            return 1.0, "N/A"
        diff = np.array(proposed_scores) - np.array(baseline_scores)
        if np.all(diff == 0):
            return 1.0, "Đồng nhất"
        _, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
        p_val = float(p_val) if not np.isnan(p_val) else 1.0
        sig = "p < 0.01 (Vượt trội có ý nghĩa)" if p_val < 0.01 else ("p < 0.05" if p_val < 0.05 else "Chênh lệch không đáng kể")
        return p_val, sig

    def compare_all(self, output_dir: Path | str = ROOT_DIR / "benchmark" / "results"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        from rag.es_retriever import LegalElasticsearchRetriever
        es = LegalElasticsearchRetriever(hosts=[config.ES_HOST], index_name=config.ES_INDEX_NAME)

        raw_backbones = [
            ("VNLawBERT (Chau et al., 2020)", "Chau/VNLawBERT", False, 256),
            ("bert-base-multilingual-cased", "bert-base-multilingual-cased", False, 256),
            ("vinai/phobert-base-v2", "vinai/phobert-base-v2", False, 256),
            ("vinai/phobert-large", "vinai/phobert-large", False, 256),
            ("xlm-roberta-base", "xlm-roberta-base", False, 256),
            ("Fsoft-AIC/vi-electra-base-generator", "Fsoft-AIC/vi-electra-base-generator", False, 256),
            ("bkai-foundation-models/videberta-base", "bkai-foundation-models/videberta-base", False, 256),
            ("bkai-foundation-models/vietnamese-bi-encoder", "bkai-foundation-models/vietnamese-bi-encoder", False, 256),
            ("intfloat/multilingual-e5-base", "intfloat/multilingual-e5-base", True, 256),
            ("intfloat/multilingual-e5-large", "intfloat/multilingual-e5-large", True, 256),
            ("BAAI/bge-m3", "BAAI/bge-m3", False, 256),
            ("vinai/bartpho-syllable", "vinai/bartpho-syllable", False, 256),
            ("VietAI/vit5-base", "VietAI/vit5-base", False, 256),
        ]

        summary_rows = []

        for ds_name, samples in self.datasets.items():
            if ds_name == "vietlawbench_1000" and len(self.datasets) > 1:
                continue

            test_samples = samples[:100]
            logger.info("=== ĐỐI CHUẨN MA TRẬN MASTER BENCHMARK: %s (%d MẪU) ===", ds_name.upper(), len(test_samples))

            # Bảo toàn metadata đầy đủ cho candidate pool để hàm evaluate_query không bị điểm 0 ảo
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
                        "content": s.get("raw_content") or s.get("evidence_text", "")
                    })

            candidate_texts = [c["content"] for c in candidate_pool_dicts]
            eval_results = {}
            ndcg_vectors = {}

            # 1. Đánh giá BM25 (Elasticsearch Lexical)
            bm25_scores, bm25_ndcg = [], []
            for s in test_samples:
                cands = es.search(s["query"], top_k=10)
                m = evaluate_query(cands, s, [1, 5, 10])
                bm25_scores.append(m)
                bm25_ndcg.append(m.get("NDCG@10", 0.0))
            t = len(bm25_scores) if bm25_scores else 1
            eval_results["BM25 (Elasticsearch Lexical)"] = {k: round(sum(x[k] for x in bm25_scores) / t, 4) for k in bm25_scores[0].keys()} if bm25_scores else {}
            ndcg_vectors["BM25 (Elasticsearch Lexical)"] = bm25_ndcg

            # 2. Đánh giá 13 Backbone (Zero-shot & Fine-tuned MRL)
            for display_name, model_id, is_e5, d_star in raw_backbones:
                label_zero = f"{display_name} (Zero-shot)"
                try:
                    wrap_zero = UniversalEncoderWrapper(model_id, device=self.device, is_e5=is_e5)
                    q_z = wrap_zero.encode([s["query"] for s in test_samples])
                    c_z = wrap_zero.encode(candidate_texts)
                    sim_z = np.dot(q_z, c_z.T)

                    scores_z, ndcg_z = [], []
                    for idx, s in enumerate(test_samples):
                        top_idx = np.argsort(-sim_z[idx])[:10]
                        cands = [candidate_pool_dicts[i] for i in top_idx]
                        m = evaluate_query(cands, s, [1, 5, 10])
                        scores_z.append(m)
                        ndcg_z.append(m.get("NDCG@10", 0.0))

                    t_z = len(scores_z) if scores_z else 1
                    eval_results[label_zero] = {k: round(sum(x[k] for x in scores_z) / t_z, 4) for k in scores_z[0].keys()}
                    ndcg_vectors[label_zero] = ndcg_z
                except Exception as ex:
                    logger.warning("Bỏ qua bản gốc [%s]: %s", label_zero, ex)

                safe_name = model_id.replace("/", "_")
                ckpt_path = self.checkpoints_dir / safe_name
                if ckpt_path.exists():
                    label_ft = f"{display_name} (Ours-MRL @ d={d_star})"
                    try:
                        wrap_ft = UniversalEncoderWrapper(str(ckpt_path), device=self.device, slice_dim=d_star, is_e5=is_e5)
                        q_ft = wrap_ft.encode([s["query"] for s in test_samples])
                        c_ft = wrap_ft.encode(candidate_texts)
                        sim_ft = np.dot(q_ft, c_ft.T)

                        scores_ft, ndcg_ft = [], []
                        for idx, s in enumerate(test_samples):
                            top_idx = np.argsort(-sim_ft[idx])[:10]
                            cands = [candidate_pool_dicts[i] for i in top_idx]
                            m = evaluate_query(cands, s, [1, 5, 10])
                            scores_ft.append(m)
                            ndcg_ft.append(m.get("NDCG@10", 0.0))

                        t_ft = len(scores_ft) if scores_ft else 1
                        eval_results[label_ft] = {k: round(sum(x[k] for x in scores_ft) / t_ft, 4) for k in scores_ft[0].keys()}
                        ndcg_vectors[label_ft] = ndcg_ft
                    except Exception as ex:
                        logger.warning("Lỗi đánh giá bản fine-tune [%s]: %s", label_ft, ex)

            if not eval_results:
                continue

            best_model_key = max(eval_results.keys(), key=lambda k: eval_results[k].get("NDCG@10", 0.0))
            proposed_vec = ndcg_vectors[best_model_key]
            logger.info("Mô hình đạt SOTA cao nhất: %s (NDCG@10: %.4f)", best_model_key, eval_results[best_model_key].get("NDCG@10", 0.0))

            for k in eval_results.keys():
                m = eval_results[k]
                p_val, sig = self.compute_significance(ndcg_vectors[k], proposed_vec) if k != best_model_key else (0.0, "Proposed SOTA")
                summary_rows.append({
                    "Dataset": ds_name,
                    "Mo_Hinh_Embedding": k,
                    "Hit@1": m.get("Hit@1", 0.0),
                    "Hit@5": m.get("Hit@5", 0.0),
                    "Hit@10": m.get("Hit@10", 0.0),
                    "MRR@10": m.get("MRR@10", 0.0),
                    "NDCG@10": m.get("NDCG@10", 0.0),
                    "P_Value_vs_SOTA": f"{p_val:.2e}" if p_val > 0 else "-",
                    "Kiem_Dinh_Y_Nghia": sig
                })

        sum_file = out_path / "master_baseline_comparison.csv"
        with open(sum_file, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "Dataset", "Mo_Hinh_Embedding", "Hit@1", "Hit@5", "Hit@10",
                "MRR@10", "NDCG@10", "P_Value_vs_SOTA", "Kiem_Dinh_Y_Nghia"
            ])
            w.writeheader()
            w.writerows(summary_rows)
        logger.info("✓ Đã lưu bảng tổng hợp Master Benchmark tại: %s", sum_file.resolve())


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