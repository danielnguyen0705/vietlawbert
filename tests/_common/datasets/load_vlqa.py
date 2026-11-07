"""
tests/_common/datasets/load_vlqa.py — Nạp VLQA làm BENCHMARK ĐÁNH GIÁ (không train).

Ánh xạ (xác nhận từ dữ liệu thật 06/11):
  train.json       : [{qid, question, relevant_laws:"[aid,...]", answer}]  (relevant_laws là CHUỖI json các aid)
  legal_corpus.json: [{id, law_id(=doc_number), content:[{aid, content_Article}]}]  (aid TOÀN CỤC)
  → aid → law_id(doc_number) → canonical_doc_id.

Xuất: vietlawbench_vlqa.jsonl (schema tương thích load_tests.js). KHÔNG commit (đã .gitignore) — Điều 4.3.
Tuân thủ biên bản VLQA: chỉ dùng đánh giá; trích dẫn VLSP DRiLL 2025 + công trình VLQA trong mọi công bố.

Chạy:  python -m tests._common.datasets.load_vlqa            # sinh jsonl + báo độ phủ (nếu có ES)
"""
from __future__ import annotations
import os, re, json, argparse, logging
from pathlib import Path
from configs.config import config
from configs.global_id import canonical_doc_id, normalize_doc_number
from configs.paths import BENCHMARK_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_VLQA")


def _parse_relevant(v):
    """relevant_laws có thể là chuỗi '[53877, ...]' hoặc list sẵn → trả list[int]."""
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str) and v.strip():
        try:
            return [int(x) for x in json.loads(v)]
        except Exception:
            return [int(x) for x in re.findall(r"\d+", v)]
    return []


def build_aid_to_lawid(legal_corpus) -> dict:
    """aid (toàn cục) → law_id (doc_number)."""
    m = {}
    for doc in legal_corpus:
        law_id = str(doc.get("law_id") or "").strip()
        for art in doc.get("content") or []:
            aid = art.get("aid")
            if aid is not None and law_id:
                m[int(aid)] = law_id
    return m


def load_vlqa(vlqa_dir: str):
    tr = json.load(open(os.path.join(vlqa_dir, "train.json"), encoding="utf-8"))
    lc = json.load(open(os.path.join(vlqa_dir, "legal_corpus.json"), encoding="utf-8"))
    aid2law = build_aid_to_lawid(lc)
    cases, unresolved = [], 0
    for rec in tr:
        aids = _parse_relevant(rec.get("relevant_laws"))
        law_ids = []
        for a in aids:
            lw = aid2law.get(a)
            if lw and lw not in law_ids:
                law_ids.append(lw)
        if not law_ids:
            unresolved += 1
            continue
        cases.append({
            "benchmark_id": f"vlqa_{rec.get('qid')}",
            "query_type": "vlqa",
            "retrieval_level": "document",
            "query": str(rec.get("question") or "").strip(),
            "ground_truth_doc_number": law_ids[0],
            "ground_truth_doc_numbers": law_ids,
            "ground_truth_docs": [{"doc_number": x, "article": ""} for x in law_ids],
            "ground_truth_doc_id": canonical_doc_id(law_ids[0]),
            "relevant_aids": aids,
            "answer_ref": str(rec.get("answer") or "")[:500],
        })
    return cases, aid2law, unresolved


def main():
    ap = argparse.ArgumentParser(description="Nạp VLQA làm benchmark đánh giá (map sang doc_number của mình)")
    ap.add_argument("--out", default=str(BENCHMARK_DIR / "vietlawbench_vlqa.jsonl"))
    args = ap.parse_args()
    if not config.VLQA_DIR or not os.path.isdir(config.VLQA_DIR):
        raise SystemExit(f"VLQA_DIR không hợp lệ: {config.VLQA_DIR!r} — set trong .env trước.")
    cases, aid2law, unresolved = load_vlqa(config.VLQA_DIR)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_doc = len({c["ground_truth_doc_number"] for c in cases})
    logger.info("✓ %d câu VLQA (bỏ %d câu không map được aid) | %d doc_number đích phân biệt → %s",
                len(cases), unresolved, n_doc, args.out)

    # Độ phủ: bao nhiêu % câu có ÍT NHẤT một doc đích nằm trong index ES
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(config.ES_HOST)
        if es.ping():
            idx = {normalize_doc_number(b["key"]) for b in es.search(
                index=config.ES_INDEX_NAME, size=0,
                aggs={"d": {"terms": {"field": "doc_number.keyword", "size": 10000}}}
            )["aggregations"]["d"]["buckets"]}
            covered = sum(1 for c in cases
                          if any(normalize_doc_number(x) in idx for x in c["ground_truth_doc_numbers"]))
            logger.info("ĐỘ PHỦ ES: %d/%d câu (%.1f%%) có ≥1 doc đích trong index (phần đo được end-to-end hiện tại).",
                        covered, len(cases), 100 * covered / max(len(cases), 1))
    except Exception as exc:
        logger.info("Bỏ qua đo độ phủ ES (%s).", exc)


if __name__ == "__main__":
    main()
