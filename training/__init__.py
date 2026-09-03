"""
vietlawbert.training
~~~~~~~~~~~~~~~~~~~~
Phân hệ khai phá mẫu khó và huấn luyện biểu diễn nhúng pháp lý VietLawBERT-MRL:
- generate_training_data: Thuật toán Graph-Guided Semantic-Lexical Mining (GG-SLM).
- train_mrl: Động cơ huấn luyện Matryoshka Representation Learning tích hợp InfoNCE Loss.
- fine_tune: Điểm vào tương thích ngược (Backward-Compatible Fine-tuning Wrapper).
"""

from .generate_training_data import (
    score_hardness,
    get_candidate_set,
    run_pipeline as run_triplet_mining,
)
from .train_mrl import (
    load_triplets,
    compute_matryoshka_weights,
    train as train_mrl_model,
)
from .fine_tune import main as fine_tune_main

__all__ = [
    "score_hardness",
    "get_candidate_set",
    "run_triplet_mining",
    "load_triplets",
    "compute_matryoshka_weights",
    "train_mrl_model",
    "fine_tune_main",
]