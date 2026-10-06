"""
train_mrl.py - Động cơ huấn luyện VietLawBERT với Hierarchy-Aware Matryoshka InfoNCE Loss.
Hỗ trợ đầy đủ 13 backbones:
- Pure Encoders (768d & 1024d): BERT, PhoBERT (base/large), XLM-RoBERTa, viELECTRA,
  viDeBERTa, BGE-M3, Multilingual-E5 (base/large), BKAI Bi-Encoder, VNLawBERT.
- Seq2Seq Encoders: BARTpho, ViT5 (tự động bóc tách tầng Encoder).
Tối ưu hóa GPU tương thích PyTorch 2.x SDPA tự nhiên, loại bỏ xung đột Flash Attention 2.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

try:
    torch.set_num_threads(2)
except Exception:
    pass

from configs.config import config
from configs.paths import MODELS_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s")
logger = logging.getLogger("VietLawBERT_TrainMRL")


class HierarchyAwareMatryoshkaLoss(nn.Module):
    def __init__(
        self,
        matryoshka_dims: List[int],
        temperature: float = 0.05,
        hierarchy_weight: float = 0.15,
    ):
        super().__init__()
        self.matryoshka_dims = sorted(matryoshka_dims)
        self.tau = temperature
        self.gamma = hierarchy_weight

        raw_weights = [1.0 / math.log2(float(d) + 2.0) for d in self.matryoshka_dims]
        total_w = sum(raw_weights)
        self.weights = [w / total_w for w in raw_weights]
        logger.info("Cấu hình Matryoshka dims: %s | Trọng số: %s", self.matryoshka_dims, [round(w, 4) for w in self.weights])

    def forward(
        self,
        anchor_rep: torch.Tensor,
        pos_rep: torch.Tensor,
        neg_rep: torch.Tensor,
        hierarchy_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = anchor_rep.size(0)
        labels = torch.arange(B, device=anchor_rep.device)
        candidates = torch.cat([pos_rep, neg_rep], dim=0)
        total_loss = 0.0

        for dim, weight in zip(self.matryoshka_dims, self.weights):
            if dim > anchor_rep.size(-1):
                continue
            sub_anchor = F.normalize(anchor_rep[:, :dim], p=2, dim=-1)
            sub_candidates = F.normalize(candidates[:, :dim], p=2, dim=-1)
            logits = torch.matmul(sub_anchor, sub_candidates.T) / self.tau
            total_loss = total_loss + weight * F.cross_entropy(logits, labels)

        if hierarchy_labels is not None and self.gamma > 0 and anchor_rep.size(-1) >= 64:
            sub_macro = F.normalize(anchor_rep[:, :64], p=2, dim=-1)
            sim_macro = torch.matmul(sub_macro, sub_macro.T) / self.tau

            label_mask = torch.eq(hierarchy_labels.unsqueeze(1), hierarchy_labels.unsqueeze(0)).float()
            diag_mask = torch.eye(B, device=anchor_rep.device)
            pos_mask = label_mask * (1.0 - diag_mask)

            max_sim, _ = torch.max(sim_macro, dim=1, keepdim=True)
            exp_sim = torch.exp(sim_macro - max_sim.detach()) * (1.0 - diag_mask)
            denom = exp_sim.sum(dim=1, keepdim=True) + 1e-9
            log_prob = (sim_macro - max_sim.detach()) - torch.log(denom)

            num_pos = pos_mask.sum(dim=1)
            valid = num_pos > 0
            if valid.any():
                sup_con = -(pos_mask * log_prob).sum(dim=1)[valid] / num_pos[valid]
                total_loss = total_loss + self.gamma * sup_con.mean()

        return total_loss


class VietLawBERTMRL(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        freeze_bottom_layers: int = 0,
        enable_checkpointing: bool = True,
        precision: str = "bf16",
        is_cuda: bool = False,
    ):
        super().__init__()
        logger.info("Khởi tạo backbone: %s (Device CUDA: %s, Precision: %s)", base_model_name, is_cuda, precision)
        
        model_kwargs = {}
        if is_cuda:
            model_kwargs["torch_dtype"] = torch.bfloat16 if (precision == "bf16" and torch.cuda.is_bf16_supported()) else (torch.float16 if precision == "fp16" else torch.float32)

        self.model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        self.is_seq2seq = getattr(self.model_config, "is_encoder_decoder", False)

        # Nạp mô hình có cơ chế fallback an toàn: thử SDPA trước, nếu không hỗ trợ thì lùi về mặc định
        if self.is_seq2seq:
            logger.info("-> Bóc tách khối Encoder từ Seq2Seq LM (%s)...", base_model_name)
            try:
                full_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, config=self.model_config, attn_implementation="sdpa", **model_kwargs)
            except Exception:
                full_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, config=self.model_config, **model_kwargs)
            self.encoder = full_model.get_encoder()
            self.hidden_size = getattr(self.encoder.config, "d_model", getattr(self.encoder.config, "hidden_size", 768))
        else:
            try:
                self.encoder = AutoModel.from_pretrained(base_model_name, config=self.model_config, attn_implementation="sdpa", **model_kwargs)
            except Exception:
                self.encoder = AutoModel.from_pretrained(base_model_name, config=self.model_config, **model_kwargs)
            self.hidden_size = getattr(self.encoder.config, "hidden_size", 768)

        if enable_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        if freeze_bottom_layers > 0 and hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            if hasattr(self.encoder, "embeddings"):
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_bottom_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def _mean_pooling(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.is_seq2seq:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = out.last_hidden_state
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self._mean_pooling(last_hidden, attention_mask)


class TripletParquetDataset(Dataset):
    def __init__(self, parquet_path: str, max_length: int = 256, max_samples: Optional[int] = None):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"File Parquet rỗng: {path}")

        col_a = "anchor" if "anchor" in df.columns else ("query" if "query" in df.columns else None)
        col_p = "positive" if "positive" in df.columns else None
        col_n = "negative" if "negative" in df.columns else ("hard_negative" if "hard_negative" in df.columns else None)

        if not col_a or not col_p or not col_n:
            raise KeyError(f"Cần các cột anchor, positive, negative. Tìm thấy: {list(df.columns)}")

        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.anchors = df[col_a].astype(str).tolist()
        self.positives = df[col_p].astype(str).tolist()
        self.negatives = df[col_n].astype(str).tolist()
        self.labels = df["hierarchy_label"].astype("category").cat.codes.tolist() if "hierarchy_label" in df.columns else [0] * len(self.anchors)
        self.max_length = max_length

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return {
            "anchor": self.anchors[idx],
            "positive": self.positives[idx],
            "negative": self.negatives[idx],
            "label": self.labels[idx],
        }


def collate_fn_triplets(batch, tokenizer, max_len=256):
    anchors = [x["anchor"] for x in batch]
    positives = [x["positive"] for x in batch]
    negatives = [x["negative"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    a_tok = tokenizer(anchors, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    p_tok = tokenizer(positives, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    n_tok = tokenizer(negatives, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return a_tok, p_tok, n_tok, labels


def save_checkpoint(model: VietLawBERTMRL, tokenizer: AutoTokenizer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not model.is_seq2seq:
        model.encoder.save_pretrained(out_dir)
    else:
        torch.save(model.encoder.state_dict(), out_dir / "encoder_model.bin")
        model.model_config.save_pretrained(out_dir)

    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), out_dir / "vietlawbert_mrl.pt")

    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "hidden_size": model.hidden_size,
            "is_seq2seq": model.is_seq2seq,
            "base_config": model.model_config.to_dict()
        }, f, indent=2)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    is_cpu = device.type == "cpu"
    logger.info("Bắt đầu huấn luyện mô hình [%s] trên: %s", args.model_name, device)

    max_len = min(args.max_seq_length, 256) if is_cpu else args.max_seq_length
    micro_batch = min(args.batch_size, 2) if is_cpu else args.batch_size
    accum_steps = max(1, args.gradient_accumulation_steps)
    freeze_layers = args.freeze_layers if args.freeze_layers is not None else (18 if is_cpu else 0)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = VietLawBERTMRL(
        base_model_name=args.model_name,
        freeze_bottom_layers=freeze_layers,
        enable_checkpointing=True if is_cpu else False,
        precision=args.precision,
        is_cuda=not is_cpu,
    ).to(device)

    if args.matryoshka_dims:
        dims = [d for d in args.matryoshka_dims if d <= model.hidden_size]
    else:
        dims = [64, 128, 256, 512, 768] if model.hidden_size == 768 else [64, 128, 256, 512, 768, 1024]

    dataset = TripletParquetDataset(args.train_parquet, max_length=max_len, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers if not is_cpu else 0,
        pin_memory=True if not is_cpu else False,
        collate_fn=lambda b: collate_fn_triplets(b, tokenizer, max_len),
    )

    criterion = HierarchyAwareMatryoshkaLoss(matryoshka_dims=dims, temperature=args.tau, hierarchy_weight=args.hierarchy_weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(dataloader) // accum_steps
    total_steps = args.max_steps if args.max_steps is not None else (steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(10, int(total_steps * 0.1)), num_training_steps=total_steps)

    amp_dtype = torch.bfloat16 if (args.precision == "bf16" and torch.cuda.is_bf16_supported()) else (torch.float16 if args.precision == "fp16" else torch.float32)
    use_amp = (not is_cpu) and (args.precision in ["bf16", "fp16"])

    model.train()
    optimizer.zero_grad()
    global_step = 0
    out_dir = Path(args.output_dir)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        step_in_epoch = 0

        for step, (a_tok, p_tok, n_tok, labels) in enumerate(dataloader):
            a_tok = {k: v.to(device) for k, v in a_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = criterion(model(**a_tok), model(**p_tok), model(**n_tok), hierarchy_labels=labels)
            else:
                loss = criterion(model(**a_tok), model(**p_tok), model(**n_tok), hierarchy_labels=labels)

            loss = loss / accum_steps
            loss.backward()
            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step_in_epoch += 1
                global_step += 1

                if global_step % args.logging_steps == 0:
                    logger.info("Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f", epoch + 1, args.epochs, global_step, total_steps, loss.item() * accum_steps)

                if args.max_steps is not None and global_step >= args.max_steps:
                    break

        logger.info("=== Epoch %d Hoàn thành | Avg Loss: %.4f ===", epoch + 1, epoch_loss / max(step_in_epoch * accum_steps, 1))
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    save_checkpoint(model, tokenizer, out_dir)
    logger.info("✓ Hoàn tất huấn luyện. Trọng số đã lưu tại: %s", out_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description="VietLawBERT Multi-Backbone MRL Trainer")
    parser.add_argument("--model-name", default=config.BASE_MODEL_NAME, help="Tên hoặc đường dẫn backbone HuggingFace")
    parser.add_argument("--train-parquet", default=str(ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"))
    parser.add_argument("--output-dir", default=str(MODELS_DIR / "vietlawbert_mrl_final"), help="Thư mục lưu checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--freeze-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--matryoshka-dims", nargs="+", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--hierarchy-weight", type=float, default=0.15)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
