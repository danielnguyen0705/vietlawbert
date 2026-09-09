"""
train_mrl.py - Động cơ huấn luyện VietLawBERT với Hierarchy-Aware Matryoshka InfoNCE Loss.
Tối ưu hóa đa tầng biểu diễn lồng nhau và ràng buộc hình học vĩ mô ở chiều d=64.
Hỗ trợ Mixed Precision (FP16) và lưu trữ tương thích hoàn toàn với SentenceTransformer.
"""

from __future__ import annotations

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
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from configs.config import config
from configs.paths import MODELS_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s")
logger = logging.getLogger("VietLawBERT_HierarchyMRL")


class HierarchyAwareMatryoshkaLoss(nn.Module):
    def __init__(
        self,
        matryoshka_dims: List[int] = [64, 128, 256, 512, 768, 1024],
        temperature: float = 0.05,
        hierarchy_weight: float = 0.15,
    ):
        super().__init__()
        self.matryoshka_dims = matryoshka_dims
        self.tau = temperature
        self.gamma = hierarchy_weight

        raw_weights = [1.0 / math.log2(float(d) + 2.0) for d in self.matryoshka_dims]
        total_w = sum(raw_weights)
        self.weights = [w / total_w for w in raw_weights]

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

        # 1. Multi-tier Matryoshka InfoNCE Loss
        for dim, weight in zip(self.matryoshka_dims, self.weights):
            sub_anchor = F.normalize(anchor_rep[:, :dim], p=2, dim=-1)
            sub_candidates = F.normalize(candidates[:, :dim], p=2, dim=-1)

            logits = torch.matmul(sub_anchor, sub_candidates.T) / self.tau
            loss_d = F.cross_entropy(logits, labels)
            total_loss = total_loss + weight * loss_d

        # 2. Hierarchy Supervised Contrastive Loss tại chiều vĩ mô d=64
        if hierarchy_labels is not None and self.gamma > 0:
            sub_macro = F.normalize(anchor_rep[:, :64], p=2, dim=-1)
            sim_macro = torch.matmul(sub_macro, sub_macro.T) / self.tau

            label_mask = torch.eq(hierarchy_labels.unsqueeze(1), hierarchy_labels.unsqueeze(0)).float()
            diag_mask = torch.eye(B, device=anchor_rep.device)
            pos_mask = label_mask * (1.0 - diag_mask)

            max_sim, _ = torch.max(sim_macro, dim=1, keepdim=True)
            exp_sim = torch.exp(sim_macro - max_sim.detach()) * (1.0 - diag_mask)

            denom = exp_sim.sum(dim=1, keepdim=True) + 1e-9
            log_prob = (sim_macro - max_sim.detach()) - torch.log(denom)

            num_positives = pos_mask.sum(dim=1)
            valid_rows = num_positives > 0

            if valid_rows.any():
                sup_con = -(pos_mask * log_prob).sum(dim=1)[valid_rows] / num_positives[valid_rows]
                total_loss = total_loss + self.gamma * sup_con.mean()

        return total_loss


class VietLawBERTMRL(nn.Module):
    def __init__(self, base_model_name: str = "BAAI/bge-m3", output_dim: int = 1024):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, output_dim, bias=False) if hidden_size != output_dim else nn.Identity()

    def _mean_pooling(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask, dim=1)
        sum_mask = torch.clamp(input_mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.projection(pooled)


class TripletParquetDataset(Dataset):
    def __init__(self, parquet_path: str, max_length: int = 512):
        df = pd.read_parquet(parquet_path)
        self.anchors = df["anchor"].tolist()
        self.positives = df["positive"].tolist()
        self.negatives = df["negative"].tolist()

        if "hierarchy_label" in df.columns:
            self.labels = df["hierarchy_label"].astype("category").cat.codes.tolist()
        else:
            self.labels = [0] * len(self.anchors)

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


def collate_fn_triplets(batch, tokenizer, max_len=512):
    anchors = [x["anchor"] for x in batch]
    positives = [x["positive"] for x in batch]
    negatives = [x["negative"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    a_tok = tokenizer(anchors, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    p_tok = tokenizer(positives, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    n_tok = tokenizer(negatives, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    return a_tok, p_tok, n_tok, labels


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info("Khởi chạy huấn luyện mô hình trên thiết bị: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = VietLawBERTMRL(args.model_name, output_dim=args.output_dim).to(device)

    dataset = TripletParquetDataset(args.train_parquet, max_length=args.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_fn_triplets(b, tokenizer, args.max_seq_length),
    )

    criterion = HierarchyAwareMatryoshkaLoss(
        matryoshka_dims=list(config.MATRYOSHKA_DIMS),
        temperature=args.tau,
        hierarchy_weight=args.hierarchy_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, (a_tok, p_tok, n_tok, labels) in enumerate(dataloader):
            a_tok = {k: v.to(device) for k, v in a_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                a_rep = model(**a_tok)
                p_rep = model(**p_tok)
                n_rep = model(**n_tok)
                loss = criterion(a_rep, p_rep, n_rep, hierarchy_labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
                logger.info(
                    "Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f",
                    epoch + 1,
                    args.epochs,
                    step + 1,
                    len(dataloader),
                    loss.item(),
                )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info("=== Epoch %d Hoàn thành | Average Loss: %.4f ===", epoch + 1, avg_loss)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Lưu định dạng chuẩn HuggingFace để SentenceTransformer có thể nạp trực tiếp
    model.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), out_dir / "vietlawbert_mrl.pt")
    logger.info("✓ Đã lưu thành công trọng số VietLawBERT-MRL tương thích SentenceTransformer tại: %s", out_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện VietLawBERT với Hierarchy-Aware MRL Loss")
    parser.add_argument("--train-parquet", default=str(ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"))
    parser.add_argument("--model-name", default=config.BASE_MODEL_NAME)
    parser.add_argument("--output-dir", default=str(MODELS_DIR / "vietlawbert_mrl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=config.MAX_SEQ_LENGTH)
    parser.add_argument("--output-dim", type=int, default=config.EMBEDDING_DIM)
    parser.add_argument("--tau", type=float, default=config.TEMPERATURE)
    parser.add_argument("--hierarchy-weight", type=float, default=config.HIERARCHY_WEIGHT)
    parser.add_argument("--device", default=config.EMBED_DEVICE)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()