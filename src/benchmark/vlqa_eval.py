"""
vlqa_eval.py — Đánh giá hệ thống trên bộ VLQA (external, forum pháp luật).

Ba tầng (xem docs/TongKet-Tuan-2026-11-21.md):
  1. TRUY XUẤT — TẤT ĐỊNH trên `relevant_laws` (= aid toàn cục → law_id/doc_number):
     Hit/MRR/NDCG/MAP@k. Doc-level mặc định; điều-level qua so khớp text (match_article).
  2. JUDGE-1 (LLM) — giá trị ĐỒ THỊ: đáp án trích đúng VB CÒN hiệu lực, không trích bản bị thay thế.
  3. JUDGE-2 (LLM) — chất lượng SINH: correctness vs đáp án VLQA (tương đối).

VLQA schema: train.json = [{qid, question, relevant_laws:[aid...], answer}].
legal_corpus.json = [{id, law_id, content:[{aid, content_Article}]}] — aid TOÀN CỤC 0..N.

Phần thuần (grounding/metric/match) không cần service → unit-test được.
Phần evaluate_vlqa cần retriever + (tùy chọn) LLM client.
"""

from __future__ import annotations

import re
import json
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable

from configs.global_id import normalize_doc_number as _norm
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("benchmark", "vlqa_eval")


# ───────────────────────── Grounding ─────────────────────────

def build_aid_index(corpus_path: str | Path) -> Dict[int, Dict[str, str]]:
    """aid (toàn cục) → {law_id, doc_number (chuẩn hóa), article_text}.
    Đọc theo luồng regex: mỗi `law_id` mở 1 luật; các `aid`/`content_Article` kế tiếp thuộc luật đó."""
    txt = Path(corpus_path).read_text(encoding="utf-8")
    index: Dict[int, Dict[str, str]] = {}
    cur_law: Optional[str] = None
    # Quét tuần tự 3 loại token theo đúng thứ tự xuất hiện trong file.
    pat = re.compile(
        r'"law_id":\s*"([^"]+)"'
        r'|"aid":\s*(\d+)'
        r'|"content_Article":\s*"((?:[^"\\]|\\.)*)"'
    )
    pending_aid: Optional[int] = None
    for m in pat.finditer(txt):
        if m.group(1) is not None:
            cur_law = m.group(1)
        elif m.group(2) is not None:
            pending_aid = int(m.group(2))
            if cur_law is not None:
                index[pending_aid] = {"law_id": cur_law, "doc_number": _norm(cur_law), "article_text": ""}
        elif m.group(3) is not None and pending_aid is not None and pending_aid in index:
            index[pending_aid]["article_text"] = _unescape(m.group(3))
            pending_aid = None
    return index


def _unescape(s: str) -> str:
    return s.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t').replace('\\\\', '\\')


def load_vlqa_split(path: str | Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def question_gold(q: Dict[str, Any], aid_index: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
    """Gold cho 1 câu: tập doc_number (cấp luật) + danh sách điều gold (aid, doc_number, text)."""
    articles = [
        {"aid": a, "doc_number": aid_index[a]["doc_number"], "text": aid_index[a]["article_text"]}
        for a in (q.get("relevant_laws") or []) if a in aid_index
    ]
    return {
        "gold_docnums": {a["doc_number"] for a in articles},
        "gold_articles": articles,
        "has_gold": bool(articles),
    }


def coverage(vlqa: List[Dict[str, Any]], aid_index: Dict[int, Dict[str, str]],
             corpus_docnums: Set[str]) -> Dict[str, float]:
    """Độ phủ gold của VLQA trong corpus mình (câu có >=1 gold-luật nằm trong corpus)."""
    gold_dn: Set[str] = set()
    q_cov = q_gold = 0
    for q in vlqa:
        g = question_gold(q, aid_index)
        if not g["has_gold"]:
            continue
        q_gold += 1
        gold_dn |= g["gold_docnums"]
        if g["gold_docnums"] & corpus_docnums:
            q_cov += 1
    inter = gold_dn & corpus_docnums
    return {
        "gold_docnums": len(gold_dn),
        "gold_docnums_in_corpus": len(inter),
        "docnum_coverage": len(inter) / max(len(gold_dn), 1),
        "questions_with_gold": q_gold,
        "questions_covered": q_cov,
        "question_coverage": q_cov / max(q_gold, 1),
    }


# ───────────────────────── So khớp điều-level ─────────────────────────

_WORD = re.compile(r"\w+", re.UNICODE)

def _toks(s: str) -> List[str]:
    return _WORD.findall((s or "").lower())


def article_similarity(a: str, b: str) -> float:
    """Độ giống 2 đoạn điều luật: difflib trên chuỗi token (ổn với khác chunk/thừa boilerplate)."""
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb).ratio()


def match_article(gold_text: str, candidate_chunks: List[Dict[str, Any]],
                  min_sim: float = 0.35) -> Optional[Dict[str, Any]]:
    """Tìm chunk mình khớp nhất với điều gold VLQA (trong cùng doc_number). None nếu dưới ngưỡng."""
    best, best_s = None, 0.0
    for ch in candidate_chunks:
        s = article_similarity(gold_text, ch.get("content") or ch.get("text") or "")
        if s > best_s:
            best, best_s = ch, s
    if best is not None and best_s >= min_sim:
        return {"chunk": best, "sim": round(best_s, 4)}
    return None


# ───────────────────────── Metric truy xuất (tất định) ─────────────────────────

def _dedup_rank(ranked_keys: Iterable[str]) -> List[str]:
    """Giữ thứ hạng TỐT NHẤT cho mỗi khóa (vd nhiều chunk cùng doc_number → tính 1 lần)."""
    seen: Set[str] = set()
    out: List[str] = []
    for k in ranked_keys:
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out


def hit_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    return 1.0 if gold & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: List[str], gold: Set[str]) -> float:
    for i, key in enumerate(ranked, start=1):
        if key in gold:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = 0
    score = 0.0
    for i, key in enumerate(ranked[:k], start=1):
        if key in gold:
            hits += 1
            score += hits / i
    return score / min(len(gold), k)


def ndcg_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    import math
    dcg = sum(1.0 / math.log2(i + 1) for i, key in enumerate(ranked[:k], start=1) if key in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def set_recall_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """Multi-source: lấy được BAO NHIÊU trong tập nguồn gold ở top-k (0..1). Với câu nhiều VB đáp án."""
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def completeness_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    """Lấy ĐỦ toàn bộ nguồn gold ở top-k chưa (tòa căn cứ nhiều VB → cần đủ, không chỉ 1)."""
    return 1.0 if gold and gold <= set(ranked[:k]) else 0.0


def retrieval_metrics(ranked_keys: List[str], gold: Set[str], ks: Tuple[int, ...] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Toàn bộ metric cho 1 câu. ranked_keys đã theo thứ hạng; dedup giữ hạng tốt nhất.
    Gồm metric đơn-nguồn (hit/mrr/ndcg/map) và đa-nguồn (set_recall/completeness)."""
    ranked = _dedup_rank(ranked_keys)
    out: Dict[str, float] = {"mrr": reciprocal_rank(ranked, gold)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        out[f"map@{k}"] = average_precision_at_k(ranked, gold, k)
        out[f"set_recall@{k}"] = set_recall_at_k(ranked, gold, k)
        out[f"completeness@{k}"] = completeness_at_k(ranked, gold, k)
    return out


# ───────────────────────── Temporal drift (tách stable/drifted) ─────────────────────────

_DRIFT_REL = ("THAY_THE", "BAI_BO", "DINH_CHI_THI_HANH")


def fetch_law_status(neo4j_client, gold_docnums: Set[str]) -> Dict[str, Dict[str, Any]]:
    """I/O: {doc_number(norm) -> {status, superseded, expired, in_corpus}} cho gold.
    CHUẨN HÓA 2 PHÍA (Neo4j lưu doc_number dạng thô, gold đã chuẩn hóa → exact-match sẽ hụt)."""
    norm2raw, norm2status = {}, {}
    for r in neo4j_client.execute_query(
        "MATCH (d:LawDocument) WHERE d.doc_number IS NOT NULL AND d.doc_number<>'N/A' "
        "RETURN d.doc_number AS dn, d.status AS st"
    ):
        n = _norm(r["dn"]); norm2raw[n] = r["dn"]; norm2status[n] = r["st"] or ""
    in_corpus = gold_docnums & set(norm2raw)
    raw = [norm2raw[n] for n in in_corpus]
    superseded: Set[str] = set()
    if raw:
        for r in neo4j_client.execute_query(
            "UNWIND $d AS dn MATCH (x:LawDocument {doc_number: dn}) "
            "MATCH (:LawDocument)-[rel:LEGAL_RELATION]->(x) WHERE rel.type IN $t "
            "RETURN DISTINCT dn AS dn", {"d": raw, "t": list(_DRIFT_REL)}
        ):
            superseded.add(_norm(r["dn"]))
    out: Dict[str, Dict[str, Any]] = {}
    for n in in_corpus:
        st = norm2status.get(n, "")
        out[n] = {"status": st, "superseded": n in superseded,
                  "expired": "hết hiệu lực" in st.lower(), "in_corpus": True}
    return out


def label_drift(gold_docnums: Set[str], status_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """THUẦN: gắn cờ 1 câu là stable/drifted. drifted = có gold-luật (trong corpus) đã
    bị thay thế/bãi bỏ HOẶC hết hiệu lực. status_map từ fetch_law_status (inject được → test)."""
    in_corp = {d for d in gold_docnums if status_map.get(d, {}).get("in_corpus")}
    drifted = {d for d in in_corp if status_map[d]["superseded"] or status_map[d]["expired"]}
    return {
        "in_corpus_golds": in_corp,
        "drifted_golds": drifted,
        "is_drifted": bool(drifted),
        "is_stable": bool(in_corp) and not drifted,
    }


# ───────────────────────── LLM Judges ─────────────────────────

JUDGE1_HOP_PROMPT = """Bạn là thẩm phán pháp lý đánh giá năng lực TRUY XUẤT THEO ĐỒ THỊ HIỆU LỰC.
Cho câu hỏi, câu trả lời của hệ thống, và danh sách văn bản được trích (kèm tình trạng hiệu lực).

[Câu hỏi] {question}
[Câu trả lời hệ thống] {answer}
[Văn bản được trích] {contexts}

Chấm (JSON, không lời dẫn):
{{
  "cites_effective": <0..1, trích đúng VB CÒN hiệu lực liên quan>,
  "avoids_superseded": <0..1, KHÔNG trích VB đã bị thay thế/bãi bỏ như thể còn hiệu lực>,
  "relation_correct": <0..1, nếu cần đi theo quan hệ (dẫn chiếu/thay thế) thì có đi đúng>,
  "reasoning": "<ngắn gọn>"
}}"""

JUDGE2_CORRECTNESS_PROMPT = """Bạn là thẩm phán đánh giá ĐỘ ĐÚNG câu trả lời pháp luật so với đáp án tham chiếu (của chuyên gia/forum).
Đáp án tham chiếu có thể diễn đạt khác; chấm theo Ý ĐÚNG, không bắt khớp từng chữ.

[Câu hỏi] {question}
[Câu trả lời hệ thống] {answer}
[Đáp án tham chiếu] {reference}

Chấm (JSON, không lời dẫn):
{{
  "correctness": <0..1, cùng kết luận/ý chính với đáp án tham chiếu>,
  "completeness": <0..1, đủ ý quan trọng>,
  "reasoning": "<ngắn gọn>"
}}"""


def _parse_json_lenient(raw: str) -> Dict[str, Any]:
    """Bóc JSON kể cả khi LLM bọc ```json fences hoặc chèn lời dẫn."""
    raw = (raw or "").strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        out: Dict[str, Any] = {}
        for key in ("cites_effective", "avoids_superseded", "relation_correct", "correctness", "completeness"):
            mm = re.search(rf'"{key}"\s*:\s*([0-9.]+)', raw)
            if mm:
                out[key] = float(mm.group(1))
        out.setdefault("reasoning", "parsed_by_regex")
        return out


def judge_effectivity_hop(client, model: str, question: str, answer: str,
                          contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """JUDGE-1: trích đúng VB còn hiệu lực + đi đúng quan hệ (giá trị đồ thị)."""
    ctx = "\n".join(
        f"- {c.get('doc_number','?')} | hiệu lực: {c.get('eff_status') or c.get('status') or '?'} "
        f"| {c.get('hierarchy_path','')}"
        for c in (contexts or [])[:8]
    ) or "Không có."
    prompt = JUDGE1_HOP_PROMPT.format(question=question, answer=answer, contexts=ctx)
    return _call_judge(client, model, prompt)


def judge_answer_correctness(client, model: str, question: str, answer: str, reference: str) -> Dict[str, Any]:
    """JUDGE-2: độ đúng so với đáp án VLQA (tương đối)."""
    prompt = JUDGE2_CORRECTNESS_PROMPT.format(question=question, answer=answer, reference=reference or "(trống)")
    return _call_judge(client, model, prompt)


def _call_judge(client, model: str, prompt: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1,
        )
        return _parse_json_lenient(resp.choices[0].message.content)
    except Exception as exc:
        logger.warning("Judge lỗi: %s", str(exc)[:120])
        return {"error": str(exc)[:160]}


# ───────────────────────── Calibration judge↔người ─────────────────────────

def _rank(xs: List[float]) -> List[float]:
    """Hạng trung bình (xử lý đồng hạng) — cho Spearman không cần scipy."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def spearman(human: List[float], judge: List[float]) -> float:
    """Tương quan hạng Spearman ρ giữa điểm người và judge."""
    if len(human) != len(judge) or len(human) < 2:
        return 0.0
    return round(_pearson(_rank(human), _rank(judge)), 4)


def cohen_kappa(human: List[float], judge: List[float], thr: float = 0.5) -> float:
    """Cohen's κ sau khi nhị phân hóa ở ngưỡng (đạt/không) — mức đồng thuận hơn ngẫu nhiên."""
    n = len(human)
    if n == 0 or n != len(judge):
        return 0.0
    ha = [x >= thr for x in human]
    hb = [x >= thr for x in judge]
    po = sum(1 for i in range(n) if ha[i] == hb[i]) / n
    pa, pb = sum(ha) / n, sum(hb) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def judge_calibration(human: List[float], judge: List[float], thr: float = 0.5,
                      min_rho: float = 0.8) -> Dict[str, Any]:
    """Báo cáo mức tin cậy của judge so với nhãn người: ρ, κ, và có ĐỦ tin để dùng tự động không."""
    rho = spearman(human, judge)
    return {
        "n": len(human),
        "spearman_rho": rho,
        "cohen_kappa": cohen_kappa(human, judge, thr),
        "trustworthy": rho >= min_rho,   # lit: chỉ dùng judge tự động khi ρ đủ cao
    }


# ───────────────────────── Orchestrator ─────────────────────────

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _agg_metrics(per_q: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_q:
        return {}
    return {key: round(_mean([p[key] for p in per_q]), 4) for key in per_q[0]}


def evaluate_vlqa(
    retriever,
    vlqa_path: str | Path,
    corpus_path: str | Path,
    corpus_docnums: Optional[Set[str]] = None,
    neo4j_client=None,
    top_k: int = 10,
    limit: Optional[int] = None,
    generator=None,
    judge_client=None,
    judge_model: str = "",
    article_level: bool = True,
) -> Dict[str, Any]:
    """Chạy eval VLQA, TÁCH stable/drifted (~76% câu có gold đã thay thế/hết hiệu lực).
    retriever.retrieve(q, top_k)->list[dict] (có doc_number[, content,...]).
    neo4j_client (tùy chọn) → gắn cờ drift; generator+judge_client (tùy chọn) → chấm LLM.
    Metric tất định báo theo 3 nhóm: all / stable / drifted. Judge-1 (hiệu lực) nhắm nhóm drifted."""
    aid_index = build_aid_index(corpus_path)
    data = load_vlqa_split(vlqa_path)
    if limit:
        data = data[:limit]

    # Gắn cờ drift (nếu có Neo4j): lấy status cho toàn bộ gold doc_number 1 lần
    status_map: Dict[str, Dict[str, Any]] = {}
    if neo4j_client is not None:
        all_gold: Set[str] = set()
        for q in data:
            all_gold |= question_gold(q, aid_index)["gold_docnums"]
        status_map = fetch_law_status(neo4j_client, all_gold)

    buckets: Dict[str, List[Dict[str, float]]] = {"all": [], "stable": [], "drifted": []}
    art_hits: List[float] = []
    j1_drift: List[float] = []
    j2_all: List[float] = []
    n_eval = n_drift = 0

    for q in data:
        g = question_gold(q, aid_index)
        if not g["has_gold"]:
            continue
        if corpus_docnums is not None and not (g["gold_docnums"] & corpus_docnums):
            continue   # gold không nằm trong corpus → bỏ, kẻo Hit=0 oan
        n_eval += 1

        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        ranked_dn = [_norm(c.get("doc_number") or "") for c in retrieved]
        m = retrieval_metrics(ranked_dn, g["gold_docnums"])
        buckets["all"].append(m)

        drift = label_drift(g["gold_docnums"], status_map) if status_map else None
        is_drift = bool(drift and drift["is_drifted"])
        if status_map:
            buckets["drifted" if is_drift else "stable"].append(m)
            if is_drift:
                n_drift += 1

        if article_level and g["gold_articles"]:
            matched = any(
                match_article(ga["text"], [c for c in retrieved if _norm(c.get("doc_number") or "") == ga["doc_number"]])
                for ga in g["gold_articles"]
            )
            art_hits.append(1.0 if matched else 0.0)

        if generator is not None and judge_client is not None:
            ans = generator.ask(q["question"], contexts=retrieved).get("answer", "")
            r2 = judge_answer_correctness(judge_client, judge_model, q["question"], ans, q.get("answer", ""))
            if "correctness" in r2:
                j2_all.append(r2.get("correctness", 0.0))
            if is_drift:   # Judge-1 (hiệu lực/hop) chỉ ý nghĩa ở nhóm drifted
                r1 = judge_effectivity_hop(judge_client, judge_model, q["question"], ans, retrieved)
                if "cites_effective" in r1:
                    j1_drift.append(_mean([r1.get("cites_effective", 0), r1.get("avoids_superseded", 0), r1.get("relation_correct", 0)]))

    agg: Dict[str, Any] = {"n_evaluated": n_eval, "n_drifted": n_drift,
                           "drift_rate": round(n_drift / n_eval, 4) if n_eval else 0.0}
    agg["retrieval_all"] = _agg_metrics(buckets["all"])
    if status_map:
        agg["retrieval_stable"] = _agg_metrics(buckets["stable"])
        agg["retrieval_drifted"] = _agg_metrics(buckets["drifted"])
    if art_hits:
        agg["article_match_rate"] = round(_mean(art_hits), 4)
    if j1_drift:
        agg["judge1_effectivity_drifted"] = round(_mean(j1_drift), 4)
    if j2_all:
        agg["judge2_correctness"] = round(_mean(j2_all), 4)
    return agg


def _corpus_docnums_from_neo4j(client) -> Set[str]:
    return {
        _norm(r["dn"]) for r in client.execute_query(
            "MATCH (d:LawDocument) WHERE d.doc_number IS NOT NULL AND d.doc_number<>'N/A' "
            "RETURN DISTINCT d.doc_number AS dn"
        )
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá VLQA (drift-aware, 3 tầng)")
    parser.add_argument("--vlqa", default="/home/dell/Desktop/VLQA/train.json")
    parser.add_argument("--corpus", default="/home/dell/Desktop/VLQA/legal_corpus.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-drift", action="store_true", help="Bỏ gắn cờ drift (không cần Neo4j)")
    parser.add_argument("--judge", action="store_true", help="Bật Judge-1/Judge-2 (cần EVAL_API_KEY)")
    parser.add_argument("--out", default="outputs/benchmark/results/vlqa_eval.json")
    args = parser.parse_args()

    # Import nặng chỉ khi chạy thật (giữ module nhẹ cho unit-test).
    from rag.retriever import LegalHybridRetriever
    from database.neo4j_client import Neo4jClient

    retriever = LegalHybridRetriever()
    neo = None if args.no_drift else Neo4jClient()
    corpus_dns = _corpus_docnums_from_neo4j(neo) if neo is not None else None

    gen = jclient = None
    jmodel = ""
    if args.judge:
        from rag.generator import LegalGenerator
        from benchmark.llm_judge import LegalLLMJudge
        gen = LegalGenerator(retriever=retriever)
        _j = LegalLLMJudge()
        jclient, jmodel = _j.client, _j.model

    res = evaluate_vlqa(
        retriever=retriever, vlqa_path=args.vlqa, corpus_path=args.corpus,
        corpus_docnums=corpus_dns, neo4j_client=neo, top_k=args.top_k, limit=args.limit,
        generator=gen, judge_client=jclient, judge_model=jmodel,
    )
    if neo is not None:
        neo.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã ghi kết quả VLQA eval → %s", out)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
