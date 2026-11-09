"""
tests/system/test/retrieval_ir/test_demo_latency.py — Unit tests for demo latency measurement helpers.
"""
from __future__ import annotations

import os
import sys
import pytest

_removed = [p for p in sys.path if os.path.basename(os.path.normpath(p)) == "_common"]
for p in _removed:
    sys.path.remove(p)

from benchmark.measure_demo_latency import StageTimer, rotated_order, topk_overlap

for p in _removed:
    sys.path.append(p)

pytestmark = pytest.mark.tier4


def test_stage_timer_records_and_totals():
    timer = StageTimer()
    timer.start("stage_a")
    timer.stop("stage_a")
    timer.start("stage_b")
    timer.stop("stage_b")

    d = timer.as_dict()
    assert "stage_a" in d
    assert "stage_b" in d
    assert timer.total() == pytest.approx(d["stage_a"] + d["stage_b"])

    with pytest.raises(RuntimeError):
        timer.stop("unstarted_stage")


def test_rotated_order():
    x = ["a", "b", "c", "d"]
    assert rotated_order(x, 1) == ["b", "c", "d", "a"]
    assert rotated_order(x, 4) == x
    assert rotated_order(x, 0) == x
    assert rotated_order(x, 5) == ["b", "c", "d", "a"]
    assert rotated_order([], 2) == []


def test_topk_overlap():
    assert topk_overlap(["a", "b", "c"], ["c", "x", "a"], k=3) == pytest.approx(2 / 3)
    assert topk_overlap(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)
    assert topk_overlap(["a", "b", "c"], ["x", "y", "z"], k=3) == pytest.approx(0.0)
    assert topk_overlap(["a", "b"], ["a", "b"], k=0) == pytest.approx(0.0)