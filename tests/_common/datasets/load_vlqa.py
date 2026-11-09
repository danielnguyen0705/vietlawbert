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
import os, re, json, argparse, logging, random
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
            "qid": rec.get("qid"),
            "benchmark_id": f"vlqa_{rec.get('qid')}",
            "query_type": "vlqa",
            "retrieval_level": "document",
            "query": str(rec.get("question") or "").strip(),
            "ground_truth_doc_number": law_ids[0],
            "ground_truth_doc_numbers": law_ids,
            "ground_truth_docs": [{"doc_number": x, "article": ""} for x in law_ids],
            "ground_truth_doc_id": canonical_doc_id(law_ids[0]),
            "relevant_aids": aids,
            "relevant_chunk_ids": [f"vlqa_{aid}" for aid in aids],
            "answer_ref": str(rec.get("answer") or "")[:500],
        })
    return cases, aid2law, unresolved


def select_dev(cases: list, n: int, seed: int = 42) -> list:
    """
    Chọn tập dev cố định từ danh sách câu hỏi:
    - Chỉ lấy câu hỏi có ≥ 1 aid hợp lệ (relevant_aids hoặc relevant_chunk_ids không rỗng).
    - Sắp xếp ổn định theo qid (int hoặc trích từ benchmark_id).
    - Xáo trộn bằng random.Random(seed).
    - Lấy N phần tử đầu tiên.
    """
    valid = [
        c for c in cases
        if c.get("relevant_aids") or c.get("relevant_chunk_ids")
    ]

    def _qid_key(c):
        q = c.get("qid")
        if q is not None:
            try:
                return (0, int(q))
            except (ValueError, TypeError):
                return (1, str(q))
        bid = str(c.get("benchmark_id", ""))
        m = re.search(r"\d+", bid)
        if m:
            return (0, int(m.group()))
        return (1, bid)

    valid.sort(key=_qid_key)
    shuffled = list(valid)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:n]


def split_tune(cases: list, dev_qids: set | list) -> list:
    """
    Tách tập tune gồm toàn bộ câu hỏi có ≥ 1 aid hợp lệ và qid KHÔNG nằm trong dev_qids,
    sắp xếp tăng dần theo qid.
    """
    dev_qids_all = set()
    for q in dev_qids:
        dev_qids_all.add(q)
        try:
            dev_qids_all.add(int(q))
        except (ValueError, TypeError):
            pass
        dev_qids_all.add(str(q))

    tune = [
        c for c in cases
        if (c.get("relevant_aids") or c.get("relevant_chunk_ids")) and c.get("qid") not in dev_qids_all
    ]

    def _qid_key(c):
        q = c.get("qid")
        if q is not None:
            try:
                return (0, int(q))
            except (ValueError, TypeError):
                return (1, str(q))
        bid = str(c.get("benchmark_id", ""))
        m = re.search(r"\d+", bid)
        if m:
            return (0, int(m.group()))
        return (1, bid)

    tune.sort(key=_qid_key)
    return tune


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nạp VLQA làm benchmark đánh giá (map sang doc_number của mình)")
    ap.add_argument("--out", default=str(BENCHMARK_DIR / "vietlawbench_vlqa.jsonl"))
    ap.add_argument("--dev-size", type=int, default=0, help="Số lượng câu cho tập dev cố định (0 = giữ hành vi cũ)")
    ap.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên chọn dev")
    ap.add_argument("--dev-out", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--qids-out", default=str(BENCHMARK_DIR / "splits" / "vlqa_dev500_qids.txt"))
    ap.add_argument("--tune-out", default=str(BENCHMARK_DIR / "vlqa_tune1690.jsonl"))
    ap.add_argument("--tune-qids-out", default=str(BENCHMARK_DIR / "splits" / "vlqa_tune1690_qids.txt"))
    args = ap.parse_args(argv)

    vlqa_dir = os.environ.get("VLQA_DIR") or getattr(config, "VLQA_DIR", "")
    if not vlqa_dir or not os.path.isdir(vlqa_dir):
        raise SystemExit(f"VLQA_DIR không hợp lệ: {vlqa_dir!r} — set trong .env trước.")

    cases, aid2law, unresolved = load_vlqa(vlqa_dir)

    if args.dev_size > 0:
        dev_cases = select_dev(cases, args.dev_size, seed=args.seed)
        dev_out_path = Path(args.dev_out)
        dev_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dev_out_path, "w", encoding="utf-8") as f:
            for c in dev_cases:
                f.write(json.dumps(c, ensure_ascii=False) + chr(10))

        qids = []
        for c in dev_cases:
            q = c.get("qid")
            if q is None:
                bid = str(c.get("benchmark_id", ""))
                m = re.search(r"\d+", bid)
                q = int(m.group()) if m else bid
            else:
                try:
                    q = int(q)
                except (ValueError, TypeError):
                    pass
            qids.append(q)

        try:
            qids.sort(key=int)
        except (ValueError, TypeError):
            qids.sort(key=str)

        qids_out_path = Path(args.qids_out)
        qids_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qids_out_path, "w", encoding="utf-8") as f:
            for q in qids:
                f.write(f"{q}" + chr(10))

        logger.info(
            "✓ Đã tạo dev set: %d câu (seed=%d) → %s | %d qids → %s",
            len(dev_cases), args.seed, dev_out_path, len(qids), qids_out_path
        )

        dev_qids_set = {c.get("qid") for c in dev_cases}
        tune_cases = split_tune(cases, dev_qids_set)
        tune_out_path = Path(args.tune_out)
        tune_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tune_out_path, "w", encoding="utf-8") as f:
            for c in tune_cases:
                f.write(json.dumps(c, ensure_ascii=False) + chr(10))

        tune_qids = []
        for c in tune_cases:
            q = c.get("qid")
            if q is None:
                bid = str(c.get("benchmark_id", ""))
                m = re.search(r"\d+", bid)
                q = int(m.group()) if m else bid
            else:
                try:
                    q = int(q)
                except (ValueError, TypeError):
                    pass
            tune_qids.append(q)

        try:
            tune_qids.sort(key=int)
        except (ValueError, TypeError):
            tune_qids.sort(key=str)

        tune_qids_out_path = Path(args.tune_qids_out)
        tune_qids_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tune_qids_out_path, "w", encoding="utf-8") as f:
            for q in tune_qids:
                f.write(f"{q}" + chr(10))

        logger.info(
            "✓ Đã tạo tune set: %d câu → %s | %d qids → %s",
            len(tune_cases), tune_out_path, len(tune_qids), tune_qids_out_path
        )
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + chr(10))
        n_doc = len({c["ground_truth_doc_number"] for c in cases})
        logger.info(
            "✓ %d câu VLQA (bỏ %d câu không map được aid) | %d doc_number đích phân biệt → %s",
            len(cases), unresolved, n_doc, out_path
        )

    eval_target_cases = dev_cases if args.dev_size > 0 else cases
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(config.ES_HOST)
        if es.ping():
            idx = {normalize_doc_number(b["key"]) for b in es.search(
                index=config.ES_INDEX_NAME, size=0,
                aggs={"d": {"terms": {"field": "doc_number.keyword", "size": 10000}}}
            )["aggregations"]["d"]["buckets"]}
            covered = sum(1 for c in eval_target_cases
                          if any(normalize_doc_number(x) in idx for x in c["ground_truth_doc_numbers"]))
            logger.info("ĐỘ PHỦ ES: %d/%d câu (%.1f%%) có ≥1 doc đích trong index.",
                        covered, len(eval_target_cases), 100 * covered / max(len(eval_target_cases), 1))
    except Exception as exc:
        logger.info("Bỏ qua đo độ phủ ES (%s).", exc)


if __name__ == "__main__":
    main()