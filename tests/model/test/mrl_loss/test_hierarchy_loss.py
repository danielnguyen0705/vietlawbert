"""Tier 3 - Tách kiểm định THÀNH PHẦN phạt nhận thức phân cấp (gamma * L_Hierarchy @ d=64).

test_tier3 và test_mrl_loss đã phủ InfoNCE + ổn định số học; file này cô lập đóng góp của
nhãn phân cấp: khi truyền hierarchy_labels, tổng mất mát phải KHÁC so với khi bỏ nhãn,
chứng minh gamma*L_Hierarchy thực sự tham gia đồ thị tính toán.
Đặc tả: Giai đoạn 2 > Hàm mất mát Hierarchy-Aware (SupCon d=64).
"""
import pytest
import torch
from training.train_mrl import HierarchyAwareMatryoshkaLoss

pytestmark = pytest.mark.tier3

DIMS = [64, 128, 256, 512, 768, 1024]


def _mk(batch=8, dim=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, dim, generator=g, requires_grad=True)
    p = torch.randn(batch, dim, generator=g, requires_grad=True)
    n = torch.randn(batch, dim, generator=g, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3][:batch], dtype=torch.long)
    return a, p, n, labels


def test_hierarchy_term_changes_total_loss():
    crit = HierarchyAwareMatryoshkaLoss(matryoshka_dims=DIMS, temperature=0.05, hierarchy_weight=0.2)
    a, p, n, labels = _mk()
    loss_no = crit(a, p, n, hierarchy_labels=None)
    loss_hi = crit(a, p, n, hierarchy_labels=labels)
    assert torch.isfinite(loss_no) and torch.isfinite(loss_hi)
    assert not torch.isclose(loss_no, loss_hi), (
        "Tổng mất mát không đổi khi thêm hierarchy_labels - thành phần L_Hierarchy có thể chưa được cộng."
    )


def test_hierarchy_loss_backward_ok():
    crit = HierarchyAwareMatryoshkaLoss(matryoshka_dims=DIMS, temperature=0.05, hierarchy_weight=0.2)
    a, p, n, labels = _mk()
    loss = crit(a, p, n, hierarchy_labels=labels)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all(), "Gradient NaN/Inf hoặc không chảy về anchor."
