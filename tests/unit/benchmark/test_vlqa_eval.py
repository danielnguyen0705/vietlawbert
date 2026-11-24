# -*- coding: utf-8 -*-
"""Unit-test vlqa_eval: metric tất định + grounding aid→doc_number + so khớp điều + parse judge.

Phần thuần chạy luôn. Smoke trên data VLQA thật chỉ chạy khi có /home/dell/Desktop/VLQA.
"""
import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from benchmark import vlqa_eval as V  # noqa: E402


# ───── Metric tất định ─────

def test_retrieval_metrics_co_ban():
    m = V.retrieval_metrics(["A", "B", "C"], {"B"}, ks=(1, 3))
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["mrr"] == 0.5                      # B ở hạng 2
    assert round(m["ndcg@3"], 4) == 0.6309      # 1/log2(3)
    assert m["map@3"] == 0.5


def test_metrics_dedup_giu_hang_tot_nhat():
    # nhiều chunk cùng doc_number → tính 1 lần, giữ hạng tốt nhất
    m = V.retrieval_metrics(["A", "A", "B"], {"B"}, ks=(1,))
    assert m["mrr"] == 0.5  # sau dedup B ở hạng 2


def test_metrics_khong_trung():
    m = V.retrieval_metrics(["X", "Y"], {"Z"}, ks=(1, 3))
    assert m["hit@3"] == 0.0 and m["mrr"] == 0.0 and m["map@3"] == 0.0


# ───── Grounding trên corpus tổng hợp ─────

_MINI_CORPUS = json.dumps([
    {"id": 0, "law_id": "14/2022/TT-NHNN", "content": [
        {"aid": 0, "content_Article": "Thông tư này quy định mã số ngạch công chức."},
        {"aid": 1, "content_Article": "Áp dụng với công chức Ngân hàng Nhà nước."}]},
    {"id": 1, "law_id": "100/2019/NĐ-CP", "content": [
        {"aid": 2, "content_Article": "Xử phạt vi phạm giao thông đường bộ."}]},
], ensure_ascii=False)


def test_build_aid_index_va_question_gold(tmp_path):
    p = tmp_path / "corpus.json"
    p.write_text(_MINI_CORPUS, encoding="utf-8")
    idx = V.build_aid_index(p)
    assert set(idx.keys()) == {0, 1, 2}
    assert idx[0]["doc_number"] == V._norm("14/2022/TT-NHNN")
    assert "mã số" in idx[0]["article_text"]
    assert idx[2]["doc_number"] == V._norm("100/2019/NĐ-CP")

    g = V.question_gold({"relevant_laws": [0, 2, 999]}, idx)   # 999 không có → bỏ
    assert g["gold_docnums"] == {V._norm("14/2022/TT-NHNN"), V._norm("100/2019/NĐ-CP")}
    assert len(g["gold_articles"]) == 2 and g["has_gold"]


def test_coverage_loai_cau_khong_phu(tmp_path):
    p = tmp_path / "corpus.json"
    p.write_text(_MINI_CORPUS, encoding="utf-8")
    idx = V.build_aid_index(p)
    vlqa = [{"relevant_laws": [0]}, {"relevant_laws": [2]}, {"relevant_laws": []}]
    ours = {V._norm("14/2022/TT-NHNN")}   # chỉ phủ luật đầu
    cov = V.coverage(vlqa, idx, ours)
    assert cov["questions_with_gold"] == 2
    assert cov["questions_covered"] == 1           # chỉ câu [0] phủ
    assert cov["docnum_coverage"] == 0.5


# ───── So khớp điều-level ─────

def test_match_article():
    gold = "Người điều khiển xe mô tô vượt đèn đỏ bị phạt tiền từ 800 nghìn đến 1 triệu."
    chunks = [
        {"content": "Nội dung không liên quan về thuế thu nhập doanh nghiệp."},
        {"content": "Người điều khiển xe mô tô vượt đèn đỏ bị phạt tiền từ 800.000 đến 1.000.000 đồng."},
    ]
    r = V.match_article(gold, chunks)
    assert r is not None and r["chunk"] is chunks[1] and r["sim"] >= 0.35
    assert V.match_article(gold, chunks[:1]) is None   # chỉ còn chunk không liên quan


# ───── Parse judge ─────

def test_parse_json_lenient_boc_fence_va_regex():
    assert V._parse_json_lenient('```json\n{"correctness": 0.8}\n```')["correctness"] == 0.8
    # hỏng JSON → regex fallback
    out = V._parse_json_lenient('loằng ngoằng "cites_effective": 1.0, "avoids_superseded": 0.5 ...')
    assert out["cites_effective"] == 1.0 and out["avoids_superseded"] == 0.5


class _FakeClient:
    """client giả theo API OpenAI: trả JSON cố định."""
    def __init__(self, payload):
        self._p = payload
        self.chat = self
        self.completions = self
    def create(self, **kw):
        class _R:
            def __init__(s, c): s.choices = [type("x", (), {"message": type("m", (), {"content": c})})]
        return _R(self._p)


def test_judges_voi_client_gia():
    c = _FakeClient('{"cites_effective":1,"avoids_superseded":1,"relation_correct":0.5,"reasoning":"ok"}')
    r1 = V.judge_effectivity_hop(c, "m", "q", "a", [{"doc_number": "100/2019/NĐ-CP", "eff_status": "Còn hiệu lực"}])
    assert r1["cites_effective"] == 1 and r1["relation_correct"] == 0.5
    c2 = _FakeClient('{"correctness":0.9,"completeness":0.7,"reasoning":"ok"}')
    r2 = V.judge_answer_correctness(c2, "m", "q", "a", "ref")
    assert r2["correctness"] == 0.9


# ───── Multi-source: set-recall + completeness ─────

def test_set_recall_va_completeness():
    m = V.retrieval_metrics(["A", "B", "C"], {"A", "C"}, ks=(1, 3))
    assert m["set_recall@3"] == 1.0 and m["set_recall@1"] == 0.5     # top1 chỉ có A trong 2 gold
    assert m["completeness@3"] == 1.0 and m["completeness@1"] == 0.0  # đủ cả {A,C} chỉ ở top3


# ───── Drift labeling (status_map tổng hợp) ─────

def test_label_drift():
    smap = {
        "A": {"in_corpus": True, "superseded": True,  "expired": False, "status": ""},
        "B": {"in_corpus": True, "superseded": False, "expired": True,  "status": "Hết hiệu lực"},
        "C": {"in_corpus": True, "superseded": False, "expired": False, "status": "Còn hiệu lực"},
        # D không có trong corpus
    }
    assert V.label_drift({"A", "C"}, smap)["is_drifted"] is True     # A đã thay thế
    assert V.label_drift({"B"}, smap)["is_drifted"] is True          # B hết hiệu lực
    stable = V.label_drift({"C"}, smap)
    assert stable["is_stable"] is True and stable["is_drifted"] is False
    assert V.label_drift({"D"}, smap)["is_stable"] is False          # không gold trong corpus


# ───── Calibration judge↔người ─────

def test_spearman_va_kappa():
    assert V.spearman([0.1, 0.5, 0.9], [0.2, 0.6, 0.95]) == 1.0       # đồng biến hoàn toàn
    assert V.spearman([0.1, 0.5, 0.9], [0.9, 0.5, 0.1]) == -1.0       # nghịch biến
    assert V.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0           # khớp hoàn toàn
    cal = V.judge_calibration([0.1, 0.5, 0.9, 0.8], [0.2, 0.6, 0.95, 0.75])
    assert cal["spearman_rho"] == 1.0 and cal["trustworthy"] is True


# ───── Smoke trên VLQA thật (skip nếu thiếu) ─────

_VLQA = "/home/dell/Desktop/VLQA"

@pytest.mark.skipif(not os.path.isdir(_VLQA), reason="không có data VLQA")
def test_coverage_vlqa_that_khop_94_phan_tram():
    idx = V.build_aid_index(os.path.join(_VLQA, "legal_corpus.json"))
    train = V.load_vlqa_split(os.path.join(_VLQA, "train.json"))
    # tập doc_number toàn corpus VLQA (xấp xỉ "corpus mình" để kiểm logic coverage chạy)
    all_dn = {a["doc_number"] for a in idx.values()}
    cov = V.coverage(train, idx, all_dn)
    assert cov["questions_with_gold"] >= 2000
    assert cov["question_coverage"] >= 0.99    # gold của VLQA tất nhiên nằm trong corpus VLQA
