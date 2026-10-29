"""
tests/system/test/chatbox/test_interaction_log.py
=====================================================
Test hồi quy (regression) cho tính năng ghi nhật ký hỏi-đáp
(demo/server/interaction_service.py → flagging/interactions.jsonl).

Mục đích:
  1. Bảo đảm schema log KHÔNG vỡ khi refactor (mọi record đủ trường bắt buộc).
  2. Kiểm tra bất biến (invariant) chất lượng: attribution ∈ [0,1], n_docs khớp
     số evidence, rank liên tục 1..n, latency ≥ 0, total = retr + gen.
  3. Hook golden-case: nếu có tests/system/test/chatbox/known_cases.json,
     assert mỗi câu hỏi mẫu truy xuất đúng doc_number kỳ vọng trong top-k.

Triết lý (theo nhóm): KHÔNG để latency gate pass/fail; chỉ kiểm cấu trúc & căn cứ.
Test tự SKIP khi log chưa có (chưa chạy demo lần nào) thay vì làm đỏ cả suite.

Chạy:
    pytest tests/system/test/chatbox/test_interaction_log.py -v
    pytest -m tier5 -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# conftest.py đã ghim REPO_ROOT vào sys.path
from demo.server import interaction_service
from demo.config import INTERACTIONS_FILE

pytestmark = pytest.mark.tier5

REQUIRED_TOP_KEYS = {
    "timestamp", "run_id", "turn_index", "query", "answer",
    "mode", "top_k", "model_used", "is_mock", "metrics", "evidence",
}
REQUIRED_METRIC_KEYS = {
    "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
    "attribution_score", "n_docs",
}
REQUIRED_EVIDENCE_KEYS = {
    "rank", "doc_number", "hierarchy_path", "effective_date",
    "content", "rerank_score", "graph_boost", "rrf_score",
}

KNOWN_CASES_FILE = Path(__file__).resolve().parent / "known_cases.json"


@pytest.fixture(scope="module")
def records():
    recs = interaction_service.load_interactions()
    if not recs:
        pytest.skip(
            f"Chưa có interaction log tại {INTERACTIONS_FILE}. "
            "Chạy demo và đặt vài câu hỏi trước khi test."
        )
    return recs


# ─── Schema ─────────────────────────────────────────────────────────────────

def test_top_level_schema(records):
    """Mọi record phải đủ các trường bắt buộc cấp cao nhất."""
    for i, r in enumerate(records):
        missing = REQUIRED_TOP_KEYS - set(r.keys())
        assert not missing, f"Record #{i} thiếu trường: {missing}"


def test_metrics_schema(records):
    """Khối metrics phải đủ 5 chỉ số."""
    for i, r in enumerate(records):
        m = r.get("metrics", {})
        missing = REQUIRED_METRIC_KEYS - set(m.keys())
        assert not missing, f"Record #{i}.metrics thiếu: {missing}"


def test_evidence_schema(records):
    """Mỗi phần tử evidence phải đủ trường bằng chứng + điểm số."""
    for i, r in enumerate(records):
        for j, ev in enumerate(r.get("evidence", [])):
            missing = REQUIRED_EVIDENCE_KEYS - set(ev.keys())
            assert not missing, f"Record #{i}.evidence[{j}] thiếu: {missing}"


# ─── Bất biến chất lượng ────────────────────────────────────────────────────

def test_attribution_in_range(records):
    """Attribution score (RQ4) luôn thuộc [0, 1]."""
    for i, r in enumerate(records):
        attr = r["metrics"]["attribution_score"]
        assert 0.0 <= attr <= 1.0, f"Record #{i}: attribution={attr} ngoài [0,1]"


def test_ndocs_matches_evidence(records):
    """n_docs trong metrics phải khớp số phần tử evidence."""
    for i, r in enumerate(records):
        assert r["metrics"]["n_docs"] == len(r["evidence"]), \
            f"Record #{i}: n_docs={r['metrics']['n_docs']} != len(evidence)={len(r['evidence'])}"


def test_evidence_rank_contiguous(records):
    """Rank của evidence phải là 1..n liên tục, không trùng/nhảy cóc."""
    for i, r in enumerate(records):
        ranks = [ev["rank"] for ev in r["evidence"]]
        assert ranks == list(range(1, len(ranks) + 1)), \
            f"Record #{i}: rank không liên tục 1..n → {ranks}"


def test_latency_nonnegative_and_total(records):
    """Latency không âm và total ≈ retrieval + generation (sai số làm tròn)."""
    for i, r in enumerate(records):
        m = r["metrics"]
        assert m["retrieval_latency_ms"] >= 0, f"Record #{i}: retrieval latency âm"
        assert m["generation_latency_ms"] >= 0, f"Record #{i}: generation latency âm"
        expected = m["retrieval_latency_ms"] + m["generation_latency_ms"]
        assert math.isclose(m["total_latency_ms"], expected, abs_tol=1.0), \
            f"Record #{i}: total={m['total_latency_ms']} != retr+gen={expected}"


def test_answer_nonempty(records):
    """Câu trả lời không được rỗng (kể cả fallback 'không đủ căn cứ')."""
    for i, r in enumerate(records):
        assert isinstance(r["answer"], str) and r["answer"].strip(), \
            f"Record #{i}: answer rỗng"


def test_topk_respected(records):
    """Số evidence không vượt quá top_k đã yêu cầu."""
    for i, r in enumerate(records):
        assert len(r["evidence"]) <= r["top_k"], \
            f"Record #{i}: {len(r['evidence'])} docs > top_k={r['top_k']}"


# ─── Golden cases (tùy chọn — user tự thêm known_cases.json) ────────────────
# Định dạng known_cases.json:
# [
#   {"query_contains": "kết hôn với người dưới 18", "expect_doc": "109/2026/NĐ-CP", "max_rank": 3},
#   {"query_contains": "sở hữu trí tuệ",            "expect_doc": "134/2026/NĐ-CP", "max_rank": 5}
# ]
# Mỗi case: tìm record mới nhất có query chứa `query_contains`, assert `expect_doc`
# xuất hiện trong evidence ở rank ≤ max_rank.

def _load_known_cases():
    if not KNOWN_CASES_FILE.exists():
        return []
    try:
        return json.loads(KNOWN_CASES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


@pytest.mark.parametrize("case", _load_known_cases())
def test_known_retrieval_cases(records, case):
    """Golden case: câu hỏi mẫu phải truy xuất đúng doc_number trong top-k."""
    q_sub = case["query_contains"].lower()
    expect = str(case["expect_doc"]).lower().replace(" ", "")
    max_rank = int(case.get("max_rank", 5))

    # record mới nhất khớp query
    matched = [r for r in records if q_sub in r.get("query", "").lower()]
    if not matched:
        pytest.skip(f"Chưa có log cho câu hỏi chứa '{case['query_contains']}'")
    rec = matched[-1]

    hits = [
        ev for ev in rec["evidence"]
        if expect in str(ev.get("doc_number", "")).lower().replace(" ", "")
        and ev["rank"] <= max_rank
    ]
    assert hits, (
        f"Câu '{case['query_contains']}' KHÔNG truy xuất '{case['expect_doc']}' "
        f"ở rank ≤ {max_rank}. Evidence: "
        f"{[(ev['rank'], ev['doc_number']) for ev in rec['evidence']]}"
    )
