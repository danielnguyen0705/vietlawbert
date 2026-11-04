"""
tests/_common/datasets/generate_multihop.py — Sinh bộ test MULTI-HOP THẬT từ đồ thị HIN (Track 3.1, v2).

Mục đích: tạo câu hỏi mà ĐÁP ÁN nằm ở văn bản B được A dẫn chiếu/bổ trợ tới.
  - Hybrid tìm được A (vì query lấy từ TIÊU ĐỀ ĐIỀU của A → mạch lạc, truy xuất được),
    nhưng ground-truth lại ở B → chỉ graph-expansion mới kéo B vào pool.

v2 sửa 3 lỗi suy biến của v1 (phát hiện qua eval-mh 01/11: 500 ca chỉ 1 đáp án B):
  1. CYPHER v1 nổ tích Descartes chunk(A)×chunk(B) + LIMIT không ORDER BY → dồn về 1 cặp.
     → v2 lấy DISTINCT cặp (A,B), ORDER BY rand(), mỗi cặp 1 chunk đại diện.
  2. Query v1 là bag-of-words (salient terms) → không truy được cả A.
     → v2 dùng tiêu đề điều của A (dòng "Điều N. ...") làm neo truy xuất.
  3. Phân bố đáp án lệch → v2 giới hạn mỗi văn bản B tối đa MAX_PER_B ca.

Xuất: vietlawbench_multihop.jsonl.  Sau khi sinh, chạy ablation rồi đối chiếu tỷ lệ
"A truy được" và "B truy được (qua expansion)" từ output để phán xử graph.

Chạy:  python -m tests._common.datasets.generate_multihop --limit 500
"""

from __future__ import annotations

import re
import json
import argparse
import logging
import collections
from pathlib import Path

from neo4j import GraphDatabase

from configs.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_MultiHopGen")

# Quan hệ BỔ TRỢ: A (nguồn) → B (được dẫn chiếu/chi tiết hóa) — nơi đáp án thực sự nằm.
BO_TRO = ["QUY_DINH_CHI_TIET_HUONG_DAN", "HUONG_DAN_AP_DUNG", "DAN_CHIEU", "CAN_CU_BAN_HANH", "GIAI_THICH"]
MAX_PER_B = 15  # trần số ca có cùng một đáp án B, tránh phân bố suy biến

# v2: DISTINCT cặp (A,B) + ngẫu nhiên hóa + mỗi cặp 1 chunk đại diện (hết nổ tích Descartes).
CYPHER = """
MATCH (a:LawDocument)-[r:LEGAL_RELATION]->(b:LawDocument)
WHERE toUpper(r.type) IN $bo_tro AND a <> b
WITH DISTINCT a, b, toUpper(r.type) AS rel
WITH a, b, rel, rand() AS rnd ORDER BY rnd LIMIT $cand
MATCH (a)-[:HAS_CHUNK]->(ca:Chunk) WHERE size(coalesce(ca.content,'')) > 150
MATCH (b)-[:HAS_CHUNK]->(cb:Chunk) WHERE size(coalesce(cb.content,'')) > 150
WITH a, b, rel, collect(DISTINCT ca)[0] AS ca, collect(DISTINCT cb)[0] AS cb
RETURN a.doc_number AS a_doc, ca.content AS a_text,
       b.doc_number AS b_doc, cb.chunk_id AS b_chunk, cb.content AS b_text, rel
"""


def _clean(t: str) -> str:
    """Bỏ marker [META]/[HIERARCHY]/[CONTENT] khỏi nội dung chunk."""
    return re.sub(r"\[(META|HIERARCHY|CONTENT)\][^\n]*\n?", "", t or "").strip()


def _heading(text: str) -> str:
    """Trích tiêu đề điều (dòng 'Điều N. ...') làm neo truy xuất cho A."""
    for line in _clean(text).split("\n"):
        line = line.strip()
        if len(line) >= 8:
            return line[:200]
    return ""


def _normdoc(s: str) -> str:
    """Chuẩn hóa doc_number để so khớp (bỏ dấu, đ->d, thường, bỏ khoảng trắng)."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = re.sub(r"[\u0300-\u036f]", "", s)
    return s.lower().replace("\u0111", "d").replace(" ", "")


def _load_indexed(path):
    """Nạp tập doc_number đang có trong index ES (1 doc_number/dòng). None nếu không dùng."""
    if not path:
        return None
    idx = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                idx.add(_normdoc(t))
    logger.info("Nạp %d doc_number có trong index để lọc cặp (A,B).", len(idx))
    return idx


def generate(limit: int, out: str, indexed_path: str = None) -> int:
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, getattr(config, "NEO4J_PASSWORD", "")),
    )
    # lấy dư ứng viên để sau khi lọc trần-mỗi-B và câu rỗng vẫn đủ `limit`
    cand = max(int(limit) * 8, 4000) if indexed_path else max(int(limit) * 4, 2000)
    indexed = _load_indexed(indexed_path)
    n_skip_idx = 0
    cases = []
    per_b = collections.Counter()
    try:
        with driver.session() as sess:
            for rec in sess.run(CYPHER, bo_tro=BO_TRO, cand=cand):
                if len(cases) >= limit:
                    break
                a_text = _clean(str(rec["a_text"] or ""))
                b_doc = str(rec["b_doc"] or "")
                b_chunk = str(rec["b_chunk"] or "")
                heading = _heading(a_text)
                if not heading or not b_doc or not b_chunk:
                    continue
                if indexed is not None:
                    a_doc = str(rec["a_doc"] or "")
                    if _normdoc(a_doc) not in indexed or _normdoc(b_doc) not in indexed:
                        n_skip_idx += 1
                        continue
                if per_b[b_doc] >= MAX_PER_B:       # chặn một B chiếm quá nhiều ca
                    continue
                per_b[b_doc] += 1
                # Query neo vào TIÊU ĐỀ điều của A (truy được A), đáp án ở B → buộc multi-hop.
                query = f"{heading} Quy định chi tiết và văn bản được dẫn chiếu liên quan nằm ở đâu?"
                cases.append({
                    "benchmark_id": f"mh_{len(cases):05d}",
                    "query_type": "multi_hop",
                    "query": query,
                    "ground_truth_doc_number": b_doc,
                    "ground_truth_chunk": b_chunk,
                    "anchor_doc_number": str(rec["a_doc"] or ""),
                    "relation": str(rec["rel"] or ""),
                    "note": "Đáp án ở B (bổ trợ từ A) — cần graph-expansion. Query neo theo tiêu đề điều A.",
                })
    finally:
        driver.close()

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if indexed is not None:
        logger.info("Đã loại %d cặp vì A hoặc B không nằm trong index ES.", n_skip_idx)
    n_b = len(set(c["ground_truth_doc_number"] for c in cases))
    n_rel = collections.Counter(c["relation"] for c in cases)
    logger.info("✓ Sinh %d ca | %d đáp án B phân biệt | quan hệ=%s → %s",
                len(cases), n_b, dict(n_rel), outp.resolve())
    if n_b < max(10, len(cases) // 20):
        logger.warning("⚠ Phân bố B vẫn hẹp (%d) — kiểm lại đồ thị có đủ cặp bổ trợ đa dạng không.", n_b)
    return len(cases)


def main():
    ap = argparse.ArgumentParser(description="Sinh bộ test multi-hop thật từ đồ thị HIN (Track 3.1 v2)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--indexed-docs", default=None,
                    help="File doc_number (1/dòng) có trong index ES; chỉ giữ cặp (A,B) đều thuộc tập này (Hướng A).")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "vietlawbench_multihop.jsonl"))
    args = ap.parse_args()
    generate(args.limit, args.out, args.indexed_docs)


if __name__ == "__main__":
    main()
