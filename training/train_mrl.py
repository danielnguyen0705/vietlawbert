"""
train_mrl.py - Động cơ huấn luyện VietLawBERT với Hierarchy-Aware Matryoshka InfoNCE Loss.
Tối ưu hóa đa tầng biểu diễn lồng nhau D = {64, 128, 256, 512, 768, 1024} và ràng buộc hình học vĩ mô ở chiều d=64.
Tương thích kép: Khống chế OOM trên CPU máy trạm (Dell G7) và khai phóng hiệu năng cực đại trên Cloud GPU (A100/RTX 4090).
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
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

# Khóa cứng 2 luồng CPU phòng vệ khi chạy cục bộ
try:
    torch.set_num_threads(2)
except Exception:
    pass

from configs.config import config
from configs.paths import MODELS_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(name)s - %(message)s")
logger = logging.getLogger("VietLawBERT_HierarchyMRL")


class HierarchyAwareMatryoshkaLoss(nn.Module):
    """
    Hàm mất mát hỗn hợp định hướng phân cấp:
    1. Multi-tier InfoNCE trên 6 lát cắt không gian vector D = {64, 128, 256, 512, 768, 1024}
    2. Supervised Contrastive Loss tại chiều vĩ mô d=64 ép các điều khoản cùng nhóm quan hệ tiệm cận nhau.
    """
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

        # Trọng số phạt phân cấp theo lý thuyết thông tin: w_d = 1 / log2(d + 2)
        raw_weights = [1.0 / math.log2(float(d) + 2.0) for d in self.matryoshka_dims]
        total_w = sum(raw_weights)
        self.weights = [w / total_w for w in raw_weights]
        logger.info("Trọng số phân bổ Matryoshka (%s): %s", self.matryoshka_dims, [round(w, 4) for w in self.weights])

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

        # 2. Hierarchy Supervised Contrastive Loss tại lát cắt vĩ mô d=64
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
    def __init__(
        self,
        base_model_name: str = "BAAI/bge-m3",
        output_dim: int = 1024,
        freeze_bottom_layers: int = 0,
        enable_checkpointing: bool = True,
        precision: str = "bf16",
        is_cuda: bool = False,
    ):
        super().__init__()
        logger.info("Khởi tạo Backbone: %s...", base_model_name)

        model_kwargs: Dict[str, Any] = {}
        if is_cuda:
            if precision == "bf16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif precision == "fp16":
                model_kwargs["torch_dtype"] = torch.float16

            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("✓ Kích hoạt FlashAttention-2 tối ưu hóa phần cứng.")
            except Exception:
                model_kwargs["attn_implementation"] = "sdpa"

        self.encoder = AutoModel.from_pretrained(base_model_name, **model_kwargs)

        # 1. Gradient Checkpointing
        if enable_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("✓ Kích hoạt Gradient Checkpointing.")

        # 2. Đóng băng tầng dưới
        if freeze_bottom_layers > 0 and hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            if hasattr(self.encoder, "embeddings"):
                for param in self.encoder.embeddings.parameters():
                    param.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_bottom_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("✓ Đã đóng băng Embeddings và %d/%d tầng đầu của mô hình.", freeze_bottom_layers, len(self.encoder.encoder.layer))
        elif freeze_bottom_layers == 0:
            logger.info("✓ Mở khóa toàn bộ backbone phục vụ huấn luyện sâu.")

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
    def __init__(self, parquet_path: str, max_length: int = 256, max_samples: Optional[int] = None):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp Triplet Parquet tại: {path}")

        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Tệp Triplet Parquet tại {path} hoàn toàn rỗng!")

        col_anchor = "anchor" if "anchor" in df.columns else ("query" if "query" in df.columns else None)
        col_pos = "positive" if "positive" in df.columns else None
        col_neg = "negative" if "negative" in df.columns else ("hard_negative" if "hard_negative" in df.columns else None)

        if not col_anchor or not col_pos or not col_neg:
            raise KeyError(f"Schema Parquet không hợp lệ. Cần chứa anchor/positive/negative. Tìm thấy: {list(df.columns)}")

        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.anchors = df[col_anchor].astype(str).tolist()
        self.positives = df[col_pos].astype(str).tolist()
        self.negatives = df[col_neg].astype(str).tolist()

        if "hierarchy_label" in df.columns:
            self.labels = df["hierarchy_label"].astype("category").cat.codes.tolist()
        else:
            self.labels = [0] * len(self.anchors)

        self.max_length = max_length
        logger.info("✓ Khởi tạo Dataset thành công với %d mẫu bộ ba đối lập.", len(self.anchors))

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


def save_checkpoint_artifacts(model: VietLawBERTMRL, tokenizer: AutoTokenizer, out_dir: Path, output_dim: int):
    """Xuất checkpoint định dạng kép: HuggingFace AutoModel và SentenceTransformer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), out_dir / "vietlawbert_mrl.pt")

    pooling_dir = out_dir / "1_Pooling"
    pooling_dir.mkdir(parents=True, exist_ok=True)
    with open(pooling_dir / "config.json", "w", encoding="utf-8") as pf:
        json.dump({
            "word_embedding_dimension": output_dim,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        }, pf, indent=2)

    modules_config = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]
    with open(out_dir / "modules.json", "w", encoding="utf-8") as mf:
        json.dump(modules_config, mf, indent=2)


def sync_to_huggingface_hub(output_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = True) -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or os.getenv("HF_TOKEN"))
        logger.info("Đang tải bộ trọng số lên Hugging Face Hub: %s (Private=%s)...", repo_id, private)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("✓ Đồng bộ thành công 100%% trọng số lên Hugging Face Hub: %s", repo_id)
    except Exception as exc:
        logger.error("Lỗi khi đồng bộ lên Hugging Face Hub: %s. Trọng số vẫn được lưu tại ổ đĩa.", exc)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    is_cpu = device.type == "cpu"
    logger.info("Khởi chạy huấn luyện mô hình trên thiết bị: %s", device)

    # Điều phối tham số phòng vệ chống OOM trên CPU và bung hiệu năng trên GPU
    max_len = min(args.max_seq_length, 256) if is_cpu else args.max_seq_length
    micro_batch = min(args.batch_size, 2) if is_cpu else args.batch_size
    accum_steps = max(1, args.gradient_accumulation_steps if is_cpu else args.gradient_accumulation_steps)
    freeze_layers = args.freeze_layers if args.freeze_layers is not None else (18 if is_cpu else 0)

    logger.info(
        "Chiến lược huấn luyện: Micro-Batch=%d | Accumulation Steps=%d (Batch hiệu dụng=%d) | Max Length=%d | Frozen Layers=%d",
        micro_batch, accum_steps, micro_batch * accum_steps, max_len, freeze_layers
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = VietLawBERTMRL(
        base_model_name=args.model_name,
        output_dim=args.output_dim,
        freeze_bottom_layers=freeze_layers,
        enable_checkpointing=True if is_cpu else False,
        precision=args.precision,
        is_cuda=not is_cpu,
    ).to(device)

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

    matryoshka_dims = list(getattr(config, "MATRYOSHKA_DIMS", [64, 128, 256, 512, 768, 1024]))
    criterion = HierarchyAwareMatryoshkaLoss(
        matryoshka_dims=matryoshka_dims,
        temperature=args.tau,
        hierarchy_weight=args.hierarchy_weight,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(dataloader) // accum_steps
    total_expected_steps = steps_per_epoch * args.epochs
    total_training_steps = args.max_steps if args.max_steps is not None else total_expected_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, int(total_training_steps * args.warmup_ratio)),
        num_training_steps=total_training_steps,
    )

    # Cấu hình Precision Context
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)
    use_cuda_amp = (not is_cpu) and (args.precision in ["bf16", "fp16"])

    model.train()
    optimizer.zero_grad()

    global_step = 0
    stop_training = False
    out_dir = Path(args.output_dir)

    for epoch in range(args.epochs):
        if stop_training:
            break

        epoch_loss = 0.0
        step_in_epoch = 0

        for step, (a_tok, p_tok, n_tok, labels) in enumerate(dataloader):
            a_tok = {k: v.to(device) for k, v in a_tok.items()}
            p_tok = {k: v.to(device) for k, v in p_tok.items()}
            n_tok = {k: v.to(device) for k, v in n_tok.items()}
            labels = labels.to(device)

            if use_cuda_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    a_rep = model(**a_tok)
                    p_rep = model(**p_tok)
                    n_rep = model(**n_tok)
                    loss = criterion(a_rep, p_rep, n_rep, hierarchy_labels=labels)
            else:
                a_rep = model(**a_tok)
                p_rep = model(**p_tok)
                n_rep = model(**n_tok)
                loss = criterion(a_rep, p_rep, n_rep, hierarchy_labels=labels)

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

                if global_step % args.logging_steps == 0 or (args.max_steps and global_step == args.max_steps):
                    current_loss = loss.item() * accum_steps
                    logger.info(
                        "Epoch [%d/%d] | Step [%d/%d] | Global Step [%d/%d] | Mini-Batch Loss: %.4f",
                        epoch + 1,
                        args.epochs,
                        step_in_epoch,
                        steps_per_epoch,
                        global_step,
                        total_training_steps,
                        current_loss,
                    )
                    if is_cpu:
                        gc.collect()

                # Lưu checkpoint định kỳ
                if args.save_steps and global_step % args.save_steps == 0:
                    ckpt_dir = out_dir / f"checkpoint-{global_step}"
                    logger.info("Đang lưu checkpoint định kỳ tại: %s", ckpt_dir)
                    save_checkpoint_artifacts(model, tokenizer, ckpt_dir, args.output_dim)

                # Dừng kiểm thử sớm nếu đạt trần max_steps
                if args.max_steps is not None and global_step >= args.max_steps:
                    logger.info("✓ Đạt ngưỡng kiểm thử max_steps=%d. Hoàn tất sớm quá trình huấn luyện!", args.max_steps)
                    stop_training = True
                    break

        avg_loss = epoch_loss / max(step_in_epoch * accum_steps, 1)
        logger.info("=== Epoch %d Hoàn thành | Average Loss: %.4f ===", epoch + 1, avg_loss)

    # Lưu trọng số mô hình cuối cùng
    logger.info("Đang ghi lưu trọng số mô hình hoàn chỉnh tại: %s...", out_dir.resolve())
    save_checkpoint_artifacts(model, tokenizer, out_dir, args.output_dim)
    logger.info("✓ Đã lưu thành công trọng số VietLawBERT-MRL tại: %s", out_dir.resolve())

    # Tự động đồng bộ lên Hugging Face Hub
    if args.push_to_hub and args.hub_model_id:
        sync_to_huggingface_hub(
            output_dir=out_dir,
            repo_id=args.hub_model_id,
            token=args.hf_token,
            private=not args.hub_public,
        )


def main():
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện VietLawBERT với Hierarchy-Aware MRL Loss")
    parser.add_argument("--train-parquet", default=str(ARTIFACTS_DIR / "triplets" / "hin_triplets.parquet"), help="Đường dẫn tệp Triplet Dataset")
    parser.add_argument("--model-name", default=getattr(config, "BASE_MODEL_NAME", "BAAI/bge-m3"), help="Mô hình Backbone gốc")
    parser.add_argument("--output-dir", default=str(MODELS_DIR / "vietlawbert_mrl"), help="Thư mục xuất trọng số")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=8, help="Kích thước micro-batch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Số bước tích lũy gradient")
    parser.add_argument("--freeze-layers", type=int, default=None, help="Số tầng đóng băng (Mặc định: 18 trên CPU, 0 trên GPU)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Tốc độ học (Learning Rate)")
    parser.add_argument("--max-seq-length", type=int, default=getattr(config, "MAX_SEQ_LENGTH", 256), help="Độ dài ngữ cảnh tối đa")
    parser.add_argument("--output-dim", type=int, default=getattr(config, "EMBEDDING_DIM", 1024), help="Số chiều vector biểu diễn tối đa")
    parser.add_argument("--tau", type=float, default=getattr(config, "TEMPERATURE", 0.05), help="Nhiệt độ InfoNCE")
    parser.add_argument("--hierarchy-weight", type=float, default=getattr(config, "HIERARCHY_WEIGHT", 0.15), help="Trọng số gamma cho SupCon tại d=64")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16", help="Độ chính xác tính toán (Mặc định: bf16)")
    parser.add_argument("--device", default=getattr(config, "EMBED_DEVICE", "cpu"), help="Thiết bị thực thi (cpu/cuda)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Tỷ lệ bước khởi động Learning Rate")

    # Các cờ điều khiển thực nghiệm và kiểm thử nhanh (Dry-run)
    parser.add_argument("--max-steps", type=int, default=None, help="Giới hạn số bước cập nhật để test nhanh (Dry-run)")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu triplet nạp vào RAM")
    parser.add_argument("--logging-steps", type=int, default=25, help="Tần suất in log tiến trình bước")
    parser.add_argument("--save-steps", type=int, default=None, help="Tần suất lưu checkpoint trung gian")
    parser.add_argument("--num-workers", type=int, default=2, help="Số luồng nạp dữ liệu cho DataLoader")

    # Cấu hình tự động đồng bộ Hugging Face Hub
    parser.add_argument("--push-to-hub", action="store_true", help="Tự động tải lên Hugging Face Hub sau huấn luyện")
    parser.add_argument("--hub-model-id", type=str, default=None, help="Tên repo trên HF (VD: username/vietlawbert-mrl-v1)")
    parser.add_argument("--hub-public", action="store_true", help="Mở công khai repo (Mặc định: Private)")
    parser.add_argument("--hf-token", type=str, default=None, help="User Access Token của Hugging Face")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
