"""
vietlawbert.training
~~~~~~~~~~~~~~~~~~~~
Phân hệ khai phá mẫu khó trên đồ thị dị thể và huấn luyện biểu diễn nhúng VietLawBERT-MRL (v3):
- HINTripletMiner: Khai phá Hard Negatives qua Random Walk with Restart (RWR) trên ma trận kề thưa.
- HierarchyAwareMatryoshkaLoss: Hàm mất mát MRL InfoNCE kết hợp phạt phân cụm vĩ mô d=64.
- VietLawBERTMRL: Kiến trúc Bi-Encoder Transformer kết hợp Mean Pooling và Linear Projection.
- compute_graph_embeddings: Tiền tính toán véc-tơ đồ thị 128 chiều phục vụ Reranking Compile-time.
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
)

__all__ = [
    "HINTripletMiner",
    "mine_and_export_hin_triplets",
    "HierarchyAwareMatryoshkaLoss",
    "VietLawBERTMRL",
    "train_mrl",
    "train_mrl_main",
    "compute_and_export_embeddings",
]