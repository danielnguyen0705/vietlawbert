"""Unit test hàm mất mát Hierarchy-Aware Matryoshka InfoNCE Loss."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from training.train_mrl import HierarchyAwareMatryoshkaLoss


def test_mrl_loss_forward_and_backward():
    batch_size = 4
    hidden_dim = 1024
    dims = [64, 128, 256, 512, 768, 1024]

    criterion = HierarchyAwareMatryoshkaLoss(
        matryoshka_dims=dims,
        temperature=0.05,
        hierarchy_weight=0.15
    )

    anchor = torch.randn(batch_size, hidden_dim, requires_grad=True)
    pos = torch.randn(batch_size, hidden_dim, requires_grad=True)
    neg = torch.randn(batch_size, hidden_dim, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss = criterion(anchor, pos, neg, hierarchy_labels=labels)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Kiểm tra lan truyền ngược (Backpropagation)
    loss.backward()
    assert anchor.grad is not None
    assert pos.grad is not None
    assert neg.grad is not None