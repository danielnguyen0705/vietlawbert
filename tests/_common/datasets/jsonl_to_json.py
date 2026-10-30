"""
jsonl_to_json.py — Sinh bộ test promptfoo (vietlawbench_promptfoo_tests.json) từ JSONL benchmark.

FIX 2026-10-30:
  - Nguồn đổi từ benchmark/vietlawbench_1000.jsonl (FULL, còn 271 case dơ) sang
    vietlawbench_clean.jsonl (729, ĐÃ lọc preamble/boilerplate) → bộ test sạch.
  - Đường dẫn robust theo vị trí script (chạy được cả Ubuntu lẫn Windows, không hardcode).
  - Mang ĐỦ field metric mới cần (doc_core, article_id, clause_id, global_id, retrieval_level...)
    thay vì chỉ 4 field cũ.
  - Có thể trỏ nguồn khác qua biến môi trường VLB_BENCHMARK_SRC.
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("VLB_BENCHMARK_SRC", HERE / "vietlawbench_clean.jsonl"))
OUT = HERE / "vietlawbench_promptfoo_tests.json"

FIELDS = [
    "query", "ground_truth_chunk", "ground_truth_doc_number", "ground_truth_article",
    "doc_norm", "doc_core", "chapter_id", "article_id", "clause_id", "global_id",
    "retrieval_level", "hierarchy_label",
]

tests = []
with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        tests.append({
            "description": f"Test case {data.get('benchmark_id')}",
            "vars": {k: data.get(k) for k in FIELDS},
        })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(tests, f, ensure_ascii=False, indent=2)

print(f"✓ Đã chuyển đổi {len(tests)} testcases từ {SRC.name} sang {OUT.name}")
