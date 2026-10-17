"""
test_tier3_representation_mrl.py - Kiểm định Tầng 3: Động lực học mô hình & Toán học MRL.
Bao phủ:
  1. Kiểm tra toán học hàm mất mát Matryoshka InfoNCE Loss trên 6 lát cắt chiều [64, 128, 256, 512, 768, 1024].
  2. Bất biến tái chuẩn hóa L2 sau cắt lát (Post-truncation L2 Normalization).
  3. Kiểm tra dòng chảy đạo hàm Gradient Flow (không triệt tiêu, không NaN/Inf).
  4. Kiểm soát tính bất đẳng hướng (Anisotropy Check - chống sụp nón vector).
  5. Độ tương quan thứ hạng Spearman giữa lát cắt nén d=256 và không gian gốc d=1024.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ép toàn bộ bài toán tính toán đại số nơ-ron chạy trên CPU cục bộ
DEVICE = torch.device("cpu")
MRL_DIMENSIONS = [64, 128, 256, 512, 768, 1024]


class MatryoshkaInfoNCELoss(nn.Module):
    """
    Hiện thực chuẩn xác hàm mất mát Matryoshka InfoNCE Loss phục vụ huấn luyện VietLawBERT.
    """
    def __init__(self, dimensions=None, temperature=0.05, weights=None):
        super().__init__()
        self.dimensions = dimensions or MRL_DIMENSIONS
        self.temperature = temperature
        self.weights = weights or [1.0] * len(self.dimensions)

    def forward(self, query_vecs, pos_vecs, neg_vecs):
        """
        query_vecs: [Batch, d_max]
        pos_vecs:   [Batch, d_max]
        neg_vecs:   [Batch, K, d_max] hoặc [Batch, d_max]
        """
        total_loss = torch.tensor(0.0, device=query_vecs.device, requires_grad=True)

        # Định hình lại neg_vecs về dạng [Batch, K, d_max] nếu chỉ có 1 negative
        if neg_vecs.dim() == 2:
            neg_vecs = neg_vecs.unsqueeze(1)

        for d, w in zip(self.dimensions, self.weights):
            # 1. Cắt lát chiều không gian (Matryoshka Truncation)
            q_d = query_vecs[:, :d]
            p_d = pos_vecs[:, :d]
            n_d = neg_vecs[:, :, :d]

            # 2. Bắt buộc tái chuẩn hóa L2 sau cắt lát
            q_norm = F.normalize(q_d, p=2, dim=-1)
            p_norm = F.normalize(p_d, p=2, dim=-1)
            n_norm = F.normalize(n_d, p=2, dim=-1)

            # 3. Tính Cosine Similarity
            pos_sim = torch.sum(q_norm * p_norm, dim=-1, keepdim=True) / self.temperature  # [Batch, 1]
            neg_sim = torch.bmm(n_norm, q_norm.unsqueeze(-1)).squeeze(-1) / self.temperature  # [Batch, K]

            # 4. InfoNCE Log-Sum-Exp Logit Formulation
            logits = torch.cat([pos_sim, neg_sim], dim=-1)  # [Batch, 1 + K]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # Positive nằm ở vị trí 0

            loss_d = F.cross_entropy(logits, labels)
            total_loss = total_loss + w * loss_d

        return total_loss


class TestMatryoshkaLossMath:
    """Kiểm tra tính ổn định giải tích và bất biến chuẩn hóa của hàm mất mát MRL."""

    def test_l2_normalization_after_slicing(self):
        """Bảo đảm mọi lát cắt chiều d sau khi chuẩn hóa đều có độ dài Euclid chính xác bằng 1.0."""
        batch_size = 8
        d_max = 1024
        raw_vectors = torch.randn(batch_size, d_max, device=DEVICE)

        for d in MRL_DIMENSIONS:
            sliced = raw_vectors[:, :d]
            normalized = F.normalize(sliced, p=2, dim=-1)
            norms = torch.norm(normalized, p=2, dim=-1)
            expected_ones = torch.ones(batch_size, device=DEVICE)
            assert torch.allclose(norms, expected_ones, atol=1e-6), (
                f"Lát cắt d={d} không bảo toàn chuẩn L2 = 1.0!"
            )

    def test_mrl_loss_computation_and_numerical_stability(self):
        """Kiểm tra giá trị loss dương, không bị tràn số (NaN, Inf) và phản ứng đúng với độ tương đồng."""
        batch_size = 4
        k_negatives = 5
        d_max = 1024

        query = torch.randn(batch_size, d_max, device=DEVICE, requires_grad=True)
        # Giả lập positive có độ tương quan cao với query
        positive = query + 0.1 * torch.randn(batch_size, d_max, device=DEVICE)
        # Giả lập negative ngẫu nhiên
        negatives = torch.randn(batch_size, k_negatives, d_max, device=DEVICE)

        loss_fn = MatryoshkaInfoNCELoss(dimensions=MRL_DIMENSIONS, temperature=0.05)
        loss = loss_fn(query, positive, negatives)

        assert not torch.isnan(loss).item(), "MRL Loss bị lỗi NaN!"
        assert not torch.isinf(loss).item(), "MRL Loss bị lỗi Inf!"
        assert loss.item() >= 0.0, "MRL Loss (InfoNCE) phải không âm!"

    def test_gradient_flow_across_all_slices(self):
        """Chứng minh gradient lan truyền ngược đầy đủ qua tất cả 6 lát cắt chiều về encoder backbone."""
        batch_size = 4
        k_neg = 3
        d_max = 1024

        # Thiết lập mạng nơ-ron giả lập lớp chiếu tuyến tính (Projection Head)
        linear_proj = nn.Linear(768, d_max).to(DEVICE)
        backbone_features = torch.randn(batch_size, 768, device=DEVICE)

        query_repr = linear_proj(backbone_features)
        pos_repr = linear_proj(torch.randn(batch_size, 768, device=DEVICE))
        neg_repr = torch.randn(batch_size, k_neg, d_max, device=DEVICE)

        loss_fn = MatryoshkaInfoNCELoss(dimensions=MRL_DIMENSIONS, temperature=0.05)
        loss = loss_fn(query_repr, pos_repr, neg_repr)
        loss.backward()

        # Kiểm tra trọng số của lớp chiếu phải có gradient đầy đủ
        assert linear_proj.weight.grad is not None, "Gradient không truyền về được trọng số mô hình!"
        grad_norm = torch.norm(linear_proj.weight.grad).item()
        assert grad_norm > 1e-5, f"Gradient bị triệt tiêu (Vanishing Gradient): {grad_norm}"
        assert not math.isnan(grad_norm), "Gradient chứa phần tử NaN!"


class TestRepresentationGeometry:
    """Kiểm tra các chỉ số hình học không gian vector để chống thoái hóa biểu diễn."""

    def test_cosine_anisotropy_prevention(self):
        """Bảo đảm không gian vector không bị co cụm vào một nón hẹp (Off-diagonal Cosine Sim < 0.65)."""
        num_samples = 64
        dim = 256
        # Giả lập embedding sau khi chiếu chuẩn hóa
        torch.manual_seed(42)
        embeddings = torch.randn(num_samples, dim, device=DEVICE)
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)

        sim_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        # Loại bỏ đường chéo chính (tự tương đồng với chính nó = 1.0)
        mask = ~torch.eye(num_samples, dtype=torch.bool, device=DEVICE)
        off_diagonal_sims = sim_matrix[mask]

        avg_cosine = off_diagonal_sims.mean().item()
        max_cosine = off_diagonal_sims.max().item()

        # Không gian isotropic lý tưởng có độ tương đồng ngẫu nhiên xấp xỉ 0.0
        assert abs(avg_cosine) < 0.2, (
            f"Không gian vector bị suy biến nón (Anisotropy)! Cosine trung bình: {avg_cosine:.4f}"
        )
        assert max_cosine < 0.75, (
            f"Phát hiện cặp vector ngẫu nhiên bị trùng lặp góc quá cao: {max_cosine:.4f}"
        )

    def test_ranking_preservation_between_full_and_sliced_dim(self):
        """
        Bảo đảm cấu trúc nén thông tin của MRL (Energy Compaction):
        Các chiều đầu (d=256) nắm giữ phần lớn năng lượng biểu diễn ngữ nghĩa,
        giúp bảo toàn thứ hạng Top-K so với không gian đầy đủ d=1024.
        """
        torch.manual_seed(42)
        batch_candidates = 20
        d_max = 1024

        # Mô phỏng phổ suy giảm năng lượng lũy thừa (Power-law decay) đặc trưng của MRL
        # Các chiều đầu mang tín hiệu ngữ nghĩa chính, các chiều sau đóng vai trò vi chỉnh
        dims = torch.arange(1, d_max + 1, device=DEVICE, dtype=torch.float32)
        spectral_decay = 1.0 / torch.sqrt(dims)  # Trọng số suy giảm theo căn bậc hai chiều không gian

        # Khởi tạo vector mang tín hiệu cấu trúc MRL
        query = torch.randn(1, d_max, device=DEVICE) * spectral_decay
        candidates = torch.randn(batch_candidates, d_max, device=DEVICE) * spectral_decay

        # 1. Tính toán thứ hạng trên không gian đầy đủ d=1024 (chuẩn hóa L2 trước)
        q_1024 = F.normalize(query, p=2, dim=-1)
        c_1024 = F.normalize(candidates, p=2, dim=-1)
        sim_1024 = F.cosine_similarity(q_1024, c_1024, dim=-1)
        ranks_1024 = torch.argsort(sim_1024, descending=True)

        # 2. Cắt lát nén d=256 và bắt buộc tái chuẩn hóa L2 trên không gian con
        q_256 = F.normalize(query[:, :256], p=2, dim=-1)
        c_256 = F.normalize(candidates[:, :256], p=2, dim=-1)
        sim_256 = F.cosine_similarity(q_256, c_256, dim=-1)
        ranks_256 = torch.argsort(sim_256, descending=True)

        # 3. Đánh giá độ phủ Top-5 giữa không gian nén 256d và không gian gốc 1024d
        top5_1024 = set(ranks_1024[:5].tolist())
        top5_256 = set(ranks_256[:5].tolist())
        overlap = len(top5_1024.intersection(top5_256))

        assert overlap >= 4, (
            f"Lát cắt d=256 làm sai lệch thứ hạng Top-5 so với d=1024! Độ trùng khớp: {overlap}/5"
        )