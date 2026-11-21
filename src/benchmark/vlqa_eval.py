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


def retrieval_metrics(ranked_keys: List[str], gold: Set[str], ks: Tuple[int, ...] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Toàn bộ metric cho 1 câu. ranked_keys đã theo thứ hạng; dedup giữ hạng tốt nhất."""
    ranked = _dedup_rank(ranked_keys)
    out: Dict[str, float] = {"mrr": reciprocal_rank(ranked, gold)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        out[f"map@{k}"] = average_precision_at_k(ranked, gold, k)
    return out


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


# ───────────────────────── Orchestrator ─────────────────────────

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_vlqa(
    retriever,
    vlqa_path: str | Path,
    corpus_path: str | Path,
    corpus_docnums: Optional[Set[str]] = None,
    top_k: int = 10,
    limit: Optional[int] = None,
    generator=None,
    judge_client=None,
    judge_model: str = "",
    article_level: bool = True,
) -> Dict[str, Any]:
    """Chạy eval VLQA. retriever.retrieve(q, top_k)->list[dict] (có doc_number[, content,...]).
    generator (tùy chọn) cho câu trả lời để judge; judge_client (tùy chọn) để chấm LLM.
    Trả về metric trung bình + coverage. Chỉ chấm câu có gold phủ trong corpus."""
    aid_index = build_aid_index(corpus_path)
    data = load_vlqa_split(vlqa_path)
    if limit:
        data = data[:limit]

    per_q: List[Dict[str, float]] = []
    art_hits: List[float] = []
    j1: List[float] = []
    j2: List[float] = []
    n_eval = 0

    for q in data:
        g = question_gold(q, aid_index)
        if not g["has_gold"]:
            continue
        if corpus_docnums is not None and not (g["gold_docnums"] & corpus_docnums):
            continue   # gold không nằm trong corpus → bỏ, kẻo Hit=0 oan
        n_eval += 1

        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        ranked_dn = [_norm(c.get("doc_number") or "") for c in retrieved]
        per_q.append(retrieval_metrics(ranked_dn, g["gold_docnums"]))

        # Điều-level: có khớp được chunk truy xuất với điều gold không
        if article_level and g["gold_articles"]:
            matched = False
            for ga in g["gold_articles"]:
                cands = [c for c in retrieved if _norm(c.get("doc_number") or "") == ga["doc_number"]]
                if match_article(ga["text"], cands):
                    matched = True
                    break
            art_hits.append(1.0 if matched else 0.0)

        # Judges (tùy chọn)
        if generator is not None and judge_client is not None:
            ans = generator.ask(q["question"], contexts=retrieved).get("answer", "")
            r1 = judge_effectivity_hop(judge_client, judge_model, q["question"], ans, retrieved)
            r2 = judge_answer_correctness(judge_client, judge_model, q["question"], ans, q.get("answer", ""))
            if "cites_effective" in r1:
                j1.append(_mean([r1.get("cites_effective", 0), r1.get("avoids_superseded", 0), r1.get("relation_correct", 0)]))
            if "correctness" in r2:
                j2.append(r2.get("correctness", 0.0))

    agg: Dict[str, Any] = {"n_evaluated": n_eval}
    if per_q:
        for key in per_q[0]:
            agg[key] = round(_mean([p[key] for p in per_q]), 4)
    if art_hits:
        agg["article_match_rate"] = round(_mean(art_hits), 4)
    if j1:
        agg["judge1_graph_score"] = round(_mean(j1), 4)
    if j2:
        agg["judge2_correctness"] = round(_mean(j2), 4)
    return agg
