"""
run_audit.py - Master Runner kiểm toán toàn diện 5 Tầng cho đề tài VietLawBERT.
Bao phủ:
  - Tầng 1: Data Engineering, AST Parsing & Corpus Integrity.
  - Tầng 2: Multi-Store & Knowledge Graph Invariants (Neo4j & Qdrant).
  - Tầng 3: Representation Mechanics & Matryoshka InfoNCE Loss Math.
  - Tầng 4: Hybrid Information Retrieval Benchmarking (RQ1 - RQ3).
  - Tầng 5: Grounded Generation, Faithfulness & Safety Guardrails (RQ4).
"""

import sys
import time
from pathlib import Path
import pytest

EVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_test_suite():
    print("=" * 80)
    print("BẮT ĐẦU KIỂM TOÁN TOÀN DIỆN 5 TẦNG KỸ THUẬT (DATA -> GRAPH -> MRL -> IR -> GENERATION)")
    print("=" * 80)

    tests_dir = EVAL_DIR / "tests"
    target_path = tests_dir if tests_dir.exists() else EVAL_DIR

    pytest_args = [
        str(target_path),
        "-v",
        "-s",
        "-p", "no:deepeval",
        "--durations=5",
    ]

    start_time = time.perf_counter()
    exit_code = pytest.main(pytest_args)
    elapsed = time.perf_counter() - start_time

    print("-" * 80)
    if exit_code == 0:
        print(f"✓ TOÀN BỘ CÁC BÀI KIỂM THỬ TRÊN CẢ 5 TẦNG ĐÃ VƯỢT QUA ({elapsed:.2f}s)")
        print("Hệ thống đạt chuẩn toàn vẹn khoa học từ tầng dữ liệu thô đến mô hình sinh.")
    else:
        print(f"✗ PHÁT HIỆN LỖI KIỂM ĐỊNH (Exit Code: {int(exit_code)}) TRONG {elapsed:.2f}s")
        print("Vui lòng đối soát chi tiết nguyên nhân hiển thị bên trên.")
    print("=" * 80)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(run_test_suite())