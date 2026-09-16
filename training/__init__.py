"""
vietlawbert.training
~~~~~~~~~~~~~~~~~~~~
Phân hệ khai phá mẫu khó trên đồ thị dị thể và huấn luyện biểu diễn nhúng VietLawBERT-MRL (v3):
- HINTripletMiner: Khai phá Hard Negatives theo giải thuật GG-SLM (Node2Vec 128d + BM25).
- HierarchyAwareMatryoshkaLoss: Hàm mất mát MRL InfoNCE kết hợp phạt phân cụm hình học vĩ mô d=64.
- VietLawBERTMRL: Kiến trúc Bi-Encoder Transformer thích ứng phần cứng (CPU Laptop / Cloud GPU A100).
- compute_graph_embeddings: Tiền tính toán vector tô-pô 128 chiều bằng PyTorch Sparse SGNS.
"""

from .generate_hin_triplets import (
    HINTripletMiner,
    mine_and_export_hin_triplets,
)
from .train_mrl import (
    HierarchyAwareMatryoshkaLoss,
    VietLawBERTMRL,
    train as train_mrl,
    main as train_mrl_main,
)
from .compute_graph_embeddings import (
    compute_and_export_embeddings,
    FastGraphWalker,
    PyTorchSparseSGNS,
)

__all__ = [
    "HINTripletMiner",
    "mine_and_export_hin_triplets",
    "HierarchyAwareMatryoshkaLoss",
    "VietLawBERTMRL",
    "train_mrl",
    "train_mrl_main",
    "compute_and_export_embeddings",
    "FastGraphWalker",
    "PyTorchSparseSGNS",
]