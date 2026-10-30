"""
test_tier4_ir_benchmark.py - Kiểm định Tầng 4: Động cơ truy xuất lai (IR Benchmarking & RQ1-RQ3).
Đọc trực tiếp dữ liệu từ file vietlawbench_promptfoo_tests.json chính thức.
Bao phủ:
  1. Kiểm tra độc lập nhánh Sparse BM25 (Elasticsearch).
  2. Kiểm tra độc lập nhánh Dense Vector (Qdrant) qua _search_dense.
  3. Kiểm tra tính toàn vẹn và khử trùng lặp của thuật toán Reciprocal Rank Fusion (_rrf_fusion).
  4. Đánh giá kiểm thử toàn vẹn chu trình (End-to-End) trên tập mẫu VietLawBench và đo đạc độ trễ phản hồi.
"""

import json
import random
from pathlib import Path
import pytest
from providers.retrieval_provider import call_api

from _common.paths import DATASETS_DIR

BENCHMARK_FILE = DATASETS_DIR / "demo.json"

# Đăng ký marker Tầng 4 để `pytest -m tier4` chọn được bộ test này.
# Các test gọi retriever thật → cần Qdrant + Elasticsearch đang chạy.
pytestmark = [pytest.mark.tier4, pytest.mark.requires_qdrant, pytest.mark.requires_es]


@pytest.fixture(scope="module")
def vietlawbench_samples():
    """
    Nạp dữ liệu từ vietlawbench_promptfoo_tests.json.
    Lấy mẫu 10 testcases ngẫu nhiên để phục vụ CI/CD kiểm thử nhanh trên máy trạm cục bộ (CPU).
    """
    if not BENCHMARK_FILE.exists():
        pytest.fail(f"Không tìm thấy file benchmark tại: {BENCHMARK_FILE}")

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0, "File vietlawbench_promptfoo_tests.json rỗng!"
    random.seed(42)
    sample_size = min(10, len(data))
    return random.sample(data, sample_size)


class TestIRPipelineComponents:
    """Kiểm tra hoạt động phản hồi của từng nhánh truy xuất độc lập."""

    def test_sparse_bm25_search_response(self, vietlawbench_samples):
        """Nhánh Sparse BM25 (Elasticsearch) phản hồi hợp lệ."""
        sample_query = vietlawbench_samples[0]["vars"]["query"]
        res_raw = call_api(sample_query, options={"config": {"mode": "sparse_only", "top_k": 3}})
        assert "error" not in res_raw, f"Lỗi nhánh Sparse BM25: {res_raw.get('error')}"

        data = json.loads(res_raw["output"])
        assert len(data["retrieved_chunks"]) > 0, "BM25 không trả về chunk nào!"
        # Ngưỡng trần 5000ms tính cả độ trễ I/O ban đầu trên CPU máy trạm
        assert data["latency_ms"] < 5000.0, f"BM25 latency quá cao: {data['latency_ms']}ms"

    def test_dense_vector_search_response(self, vietlawbench_samples):
        """Nhánh Dense Vector 256d (Qdrant) phản hồi hợp lệ trên CPU qua _search_dense."""
        sample_query = vietlawbench_samples[1]["vars"]["query"]
        res_raw = call_api(sample_query, options={"config": {"mode": "dense_only", "top_k": 3}})
        assert "error" not in res_raw, f"Lỗi nhánh Dense Vector: {res_raw.get('error')}"

        data = json.loads(res_raw["output"])
        assert len(data["retrieved_chunks"]) > 0, "Qdrant không trả về chunk nào!"
        # Ngưỡng suy luận trên CPU máy trạm (Forward pass BGE-M3 568M params dao động 2.5s - 3.5s)
        assert data["latency_ms"] < 4000.0, f"Qdrant CPU latency quá cao: {data['latency_ms']}ms"


class TestHybridRRFAndSLA:
    """Kiểm định thuật toán RRF và SLA thời gian phản hồi toàn trình (RQ3)."""

    def test_rrf_rank_fusion_deduplication(self, vietlawbench_samples):
        """Thuật toán RRF dung hợp kết quả và không tạo ra chunk trùng lặp."""
        sample_query = vietlawbench_samples[2]["vars"]["query"]
        res_raw = call_api(sample_query, options={"config": {"mode": "hybrid_rrf", "top_k": 5}})
        assert "error" not in res_raw, f"Lỗi RRF: {res_raw.get('error')}"

        data = json.loads(res_raw["output"])
        chunks = data["retrieved_chunks"]
        assert len(chunks) == len(set(chunks)), "Thuật toán RRF trả về chunk bị trùng lặp!"

    def test_full_architecture_benchmark_sample(self, vietlawbench_samples):
        """
        Chạy kiểm thử trên tập mẫu VietLawBench:
        - Đảm bảo toàn bộ quy trình lai (Dense + Sparse + Graph) thực thi hoàn chỉnh không lỗi ngoại lệ.
        - Kiểm tra mức độ nhận diện văn bản (Doc Hit@5) của mô hình nền tảng Zero-shot (kỳ vọng >= 20%).
        - Đo đạc thời gian đáp ứng trung bình trên môi trường máy trạm CPU.
        """
        doc_hits = 0
        latencies = []

        for item in vietlawbench_samples:
            vars_data = item["vars"]
            query = vars_data["query"]
            target_doc = str(vars_data.get("ground_truth_doc_number", "")).strip().lower()

            res_raw = call_api(query, options={"config": {"mode": "full_architecture", "top_k": 5}})
            assert "error" not in res_raw, f"Lỗi Full Architecture: {res_raw.get('error')}"

            data = json.loads(res_raw["output"])
            latencies.append(data["latency_ms"])

            retrieved_docs = [str(d).strip().lower() for d in data.get("retrieved_docs", [])]
            if any(target_doc in d or d in target_doc for d in retrieved_docs if d):
                doc_hits += 1

        hit_rate = doc_hits / len(vietlawbench_samples)
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)

        print(f"\n[Kết quả Tầng 4] Doc Hit@5 (Zero-shot Baseline): {hit_rate * 100:.1f}% ({doc_hits}/{len(vietlawbench_samples)})")
        print(f"[Kết quả Tầng 4] Độ trễ trung bình trên CPU máy trạm: {avg_latency:.1f}ms")

        # Ngưỡng sàn xác nhận pipeline hoạt động chính xác với mô hình Zero-shot trên tập dữ liệu luật chuyên ngành
        assert hit_rate >= 0.20, f"Doc Hit@5 trên tập mẫu chỉ đạt {hit_rate * 100:.1f}% (kỳ vọng >= 20%)"