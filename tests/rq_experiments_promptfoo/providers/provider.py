"""
provider.py - Custom Python Provider cho Promptfoo đánh giá Encoder Models.
Hỗ trợ nạp động nhiều Backbone, Checkpoint .pt (Ours-MRL), mini-batch encoding và cắt lát số chiều Matryoshka.
"""

from __future__ import annotations

import os
import re
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------------------------
# 0. KHÓA CỨNG MÔI TRƯỜNG CHẠY CPU & OFFLINE TRÊN MÁY TRẠM
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH GỐC DỰ ÁN & BỘ PHÂN GIẢI ĐƯỜNG DẪN (PATH RESOLVER)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("VIETLAWBERT_ROOT") or next((str(p) for p in Path(__file__).resolve().parents if (p / "rag").is_dir()), str(Path(__file__).resolve().parent)))
EVAL_DIR = Path(__file__).resolve().parent
EMB_CACHE_DIR = EVAL_DIR / ".cache_promptfoo_embeddings"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def resolve_existing_path(path_str: str | None) -> str | None:
    """Tự động phân giải đường dẫn tương đối về đúng thư mục gốc dự án."""
    if not path_str:
        return None
    raw_path = Path(path_str)
    
    # 1. Đường dẫn tuyệt đối hợp lệ
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)
    
    # 2. Duyệt qua các vị trí neo khả dĩ
    candidates = [
        PROJECT_ROOT / raw_path,
        Path.cwd() / raw_path,
        Path(__file__).resolve().parent / raw_path,
        Path(__file__).resolve().parent.parent / "vietlawbert" / raw_path,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
            
    # 3. Trả về chuỗi gốc nếu là định danh tải từ Hugging Face Hub
    return path_str


CORPUS_FILE_CANDIDATES = [
    PROJECT_ROOT / "benchmark" / "vietlawbench_1000.jsonl",
    Path("/home/dell/Desktop/vietlawbert/benchmark/vietlawbench_1000.jsonl"),
    Path(__file__).resolve().parent / "vietlawbench_1000.jsonl",
    Path.cwd() / "vietlawbench_1000.jsonl",
]

# -----------------------------------------------------------------------------
# 2. KHO CHUNK ỨNG VIÊN ĐỂ SO KHỚP (CANDIDATE CORPUS POOL)
# -----------------------------------------------------------------------------
class CorpusManager:
    _chunks: List[Dict[str, Any]] = []

    @classmethod
    def get_corpus(cls) -> List[Dict[str, Any]]:
        if not cls._chunks:
            target_file = None
            for candidate in CORPUS_FILE_CANDIDATES:
                if candidate.exists():
                    target_file = candidate
                    break
            
            if not target_file:
                raise FileNotFoundError(
                    f"Không tìm thấy file corpus benchmark. Đã duyệt các đường dẫn: "
                    f"{[str(p) for p in CORPUS_FILE_CANDIDATES]}"
                )

            seen = set()
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    item = json.loads(line_str)
                    cid = item.get("ground_truth_chunk")
                    if cid and cid not in seen:
                        seen.add(cid)
                        cls._chunks.append({
                            "chunk_id": cid,
                            "doc_number": item.get("ground_truth_doc_number", ""),
                            "article": item.get("ground_truth_article", ""),
                            "text": item.get("evidence_text", "")
                        })
        return cls._chunks

# -----------------------------------------------------------------------------
# 3. BỘ QUẢN LÝ VÀ CACHE MÔ HÌNH (MODEL REGISTRY & ENCODER)
# -----------------------------------------------------------------------------
class EncoderRegistry:
    _loaded_models = {}

    @staticmethod
    def _encode_texts(
        model,
        tokenizer,
        texts: List[str],
        device: str,
        batch_size: int = 32,
        is_query: bool = False,
        model_identifier: str = ""
    ) -> torch.Tensor:
        """Mã hóa theo mini-batch chống tràn bộ nhớ, tự động chèn tiền tố E5."""
        if not texts:
            return torch.empty(0, model.config.hidden_size)

        is_e5 = "e5" in str(model_identifier).lower()
        processed_texts = []
        for t in texts:
            clean_text = t if t and t.strip() else "[EMPTY]"
            if is_e5:
                prefix = "query: " if is_query else "passage: "
                clean_text = prefix + clean_text
            processed_texts.append(clean_text)

        all_embeddings = []
        with torch.inference_mode():
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                # Mean pooling có xét attention mask
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    @classmethod
    def get_model(
        cls,
        model_path: str,
        checkpoint_pt: str | None = None,
        device: str | None = "cpu"
    ):
        # Ép chạy hoàn toàn trên CPU để bảo vệ tiến trình khỏi lỗi CUDA OOM
        device = "cpu"

        resolved_model_path = resolve_existing_path(model_path)
        resolved_checkpoint_pt = resolve_existing_path(checkpoint_pt)

        key = f"{resolved_model_path}::{resolved_checkpoint_pt}::{device}"
        if key in cls._loaded_models:
            return cls._loaded_models[key]

        tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(resolved_model_path, trust_remote_code=True)

        # Nạp trọng số MRL .pt đã fine-tune
        if resolved_checkpoint_pt and os.path.exists(resolved_checkpoint_pt):
            state = torch.load(resolved_checkpoint_pt, map_location="cpu")
            for sub_key in ["state_dict", "model_state_dict", "model"]:
                if isinstance(state, dict) and sub_key in state and isinstance(state[sub_key], dict):
                    state = state[sub_key]
                    break

            cleaned_state = {}
            for k, v in state.items():
                clean_k = re.sub(r"^(module\.|encoder\.|model\.)", "", k)
                cleaned_state[clean_k] = v

            model.load_state_dict(cleaned_state, strict=False)

        model.to(device)
        model.eval()

        corpus = CorpusManager.get_corpus()
        corpus_texts = [c["text"] if c["text"] else c["chunk_id"] for c in corpus]

        # Kiểm tra cache đĩa cứng để tránh việc mã hóa lại 1000 chunks gây timeout 300s
        safe_model_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", str(resolved_model_path).split("/")[-1])
        pt_tag = "mrl" if (resolved_checkpoint_pt and os.path.exists(resolved_checkpoint_pt)) else "vanilla"
        cache_file = EMB_CACHE_DIR / f"corpus_emb_{safe_model_tag}_{pt_tag}.pt"

        if cache_file.exists():
            raw_corpus_embeddings = torch.load(cache_file, map_location="cpu")
        else:
            # Mã hóa toàn bộ corpus lần đầu và lưu đệm xuống đĩa
            raw_corpus_embeddings = cls._encode_texts(
                model=model,
                tokenizer=tokenizer,
                texts=corpus_texts,
                device=device,
                batch_size=32,
                is_query=False,
                model_identifier=model_path
            )
            torch.save(raw_corpus_embeddings.cpu(), cache_file)

        cls._loaded_models[key] = (tokenizer, model, corpus, raw_corpus_embeddings, device)
        return cls._loaded_models[key]

# -----------------------------------------------------------------------------
# 4. ĐIỂM VÀO THỰC THI CHO PROMPTFOO (PROMPTFOO ENTRYPOINT)
# -----------------------------------------------------------------------------
def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Hàm giao tiếp chính với Promptfoo, bọc try-except chống sập tiến trình worker."""
    try:
        cfg = options.get("config", {})
        model_path = cfg.get("model_path", "xlm-roberta-base")
        checkpoint_pt = cfg.get("checkpoint_pt", None)
        target_dim = cfg.get("dimension", None)
        top_k = cfg.get("top_k", 5)
        device = "cpu"

        query = context.get("vars", {}).get("query", prompt)

        tokenizer, model, corpus, raw_corpus_embeddings, runtime_device = EncoderRegistry.get_model(
            model_path=model_path,
            checkpoint_pt=checkpoint_pt,
            device=device
        )

        # 1. Mã hóa câu truy vấn đơn lẻ
        q_emb = EncoderRegistry._encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=[query],
            device=runtime_device,
            batch_size=1,
            is_query=True,
            model_identifier=model_path
        )  # Tensor dạng [1, hidden_dim] trên CPU

        c_emb = raw_corpus_embeddings  # Tensor dạng [N, hidden_dim] trên CPU

        # 2. Cắt lát số chiều Matryoshka (MRL Slicing) TRƯỚC KHI chuẩn hóa L2
        if target_dim is not None and target_dim < q_emb.shape[-1]:
            q_emb = q_emb[:, :target_dim]
            c_emb = c_emb[:, :target_dim]

        # 3. Chuẩn hóa L2 vector sau khi cắt lát
        q_emb = F.normalize(q_emb, p=2, dim=1)
        c_emb = F.normalize(c_emb, p=2, dim=1)

        # 4. Tính toán phân phối điểm Cosine Similarity
        scores = torch.mm(q_emb, c_emb.t())[0]
        actual_k = min(top_k, len(corpus))
        top_indices = torch.topk(scores, k=actual_k).indices.tolist()

        retrieved_chunks = [corpus[idx]["chunk_id"] for idx in top_indices]
        retrieved_docs = [corpus[idx]["doc_number"] for idx in top_indices]
        top_1_text = corpus[top_indices[0]]["text"] if top_indices else ""
        top_contexts = "\n---\n".join([
            f"[{corpus[i]['chunk_id']} | Số hiệu: {corpus[i]['doc_number']}]: {corpus[i]['text']}"
            for i in top_indices[:3]
        ])

        return {
            "output": json.dumps({
                "retrieved_chunks": retrieved_chunks,
                "retrieved_docs": retrieved_docs,
                "top_chunk_id": retrieved_chunks[0] if retrieved_chunks else "",
                "top_evidence_text": top_1_text,
                "top_contexts": top_contexts
            }, ensure_ascii=False)
        }

    except Exception as e:
        # Bắt ngoại lệ và trả về định dạng chuẩn của Promptfoo để tránh crash worker
        return {
            "error": f"EncoderProviderError: {type(e).__name__} - {str(e)}"
        }