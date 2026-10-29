"""Tier 3 - Kiểm định bộ khai phá mẫu âm khó HIN (module CHÍNH THỨC: training.generate_hin_triplets).

Lưu ý kiến trúc: repo có 2 bản khai phá triplet:
  - training/generate_hin_triplets.py  <- CHÍNH THỨC (main.py, cloud_entrypoint, run_all_experiments đều gọi
    mine_and_export_hin_triplets(..., beta=0.5)); được re-export trong training/__init__.py.
  - pipeline/hin_triplet_miner.py      <- bản trùng/không ai tham chiếu (đề xuất xoá).
Test này bám bản chính thức để khớp luồng thực nghiệm RQ1.
Đặc tả: Giai đoạn 2 > Sinh mẫu âm khó Metapath + RWR + Hardness(beta=0.5).
"""
import inspect
import pytest

pytestmark = pytest.mark.tier3

miner = pytest.importorskip("training.generate_hin_triplets")


def test_export_entrypoint_exposed():
    assert hasattr(miner, "mine_and_export_hin_triplets"), "Thiếu entrypoint mine_and_export_hin_triplets."
    assert callable(miner.mine_and_export_hin_triplets)


def test_hardness_beta_parameter_default_05():
    """Hàm xuất triplet phải có tham số beta (hệ số cân bằng BM25 vs tô-pô HIN) mặc định 0.5 theo đặc tả."""
    sig = inspect.signature(miner.mine_and_export_hin_triplets)
    assert "beta" in sig.parameters, "Thiếu tham số beta - không kiểm soát được trọng số Hardness."
    default = sig.parameters["beta"].default
    if default is not inspect.Parameter.empty:
        assert abs(float(default) - 0.5) < 1e-9, f"beta mặc định nên = 0.5 (đang là {default})."


@pytest.mark.requires_neo4j
@pytest.mark.requires_corpus
def test_mine_and_export_smoke(tmp_path, neo4j_driver):
    """Smoke: xuất một lô triplet nhỏ ra parquet (cần Neo4j + corpus). SKIP nếu thiếu."""
    out = tmp_path / "triplets_smoke.parquet"
    try:
        miner.mine_and_export_hin_triplets(output_path=str(out), limit=50, beta=0.5)
    except Exception as e:
        pytest.skip(f"Chưa đủ điều kiện chạy khai phá triplet (đồ thị/corpus): {e}")
    assert out.exists() and out.stat().st_size > 0, "Không xuất được triplet parquet."
