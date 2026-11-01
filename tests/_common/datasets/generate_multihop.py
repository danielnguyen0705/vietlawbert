"""
tests/_common/datasets/generate_multihop.py — Sinh bộ test MULTI-HOP THẬT từ đồ thị HIN (Track 3.1).

Mục đích: tạo câu hỏi mà ĐÁP ÁN nằm ở văn bản B liên-kết-với A qua quan hệ BỔ TRỢ.
  - Text retrieval (Hybrid) tìm ra A theo ngữ nghĩa, nhưng ground-truth nằm ở B
    → CHỈ graph-expansion mới kéo B vào pool và trả đúng.
  - Đây là bộ DUY NHẤT đo được delta của graph (Hybrid vs Hybrid+Expansion).

Xuất: vietlawbench_multihop.jsonl (schema tương thích promptfoo tests + cờ multi_hop + anchor).

⚠️ Query hiện ở dạng TEMPLATE (salient terms của A + stem). Nên dùng LLM cascade (rag.generator)
   để diễn đạt tự nhiên hơn trước khi coi là gold chính thức — xem --naturalize (để ngỏ, chưa bật mặc định).

Chạy:  python -m tests._common.datasets.generate_multihop --limit 500
"""

from __future__ import annotations

import re
import json
import argparse
import logging
from pathlib import Path

from neo4j import GraphDatabase

from configs.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("VietLawBERT_MultiHopGen")

# Quan hệ BỔ TRỢ: A (cha/nguồn) → B (chi tiết/hướng dẫn/dẫn chiếu) — nơi đáp án thực sự nằm.
BO_TRO = ["QUY_DINH_CHI_TIET_HUONG_DAN", "HUONG_DAN_AP_DUNG", "DAN_CHIEU", "CAN_CU_BAN_HANH", "GIAI_THICH"]

STOP = {
    "căn", "cứ", "luật", "số", "ngày", "tháng", "năm", "của", "về", "việc", "quy", "định",
    "và", "các", "có", "được", "tại", "theo", "khoản", "điều", "này", "meta", "hierarchy", "content",
}

CYPHER = """
MATCH (a:LawDocument)-[r:LEGAL_RELATION]->(b:LawDocument)
WHERE toUpper(r.type) IN $bo_tro
MATCH (a)-[:HAS_CHUNK]->(ca:Chunk) WHERE size(coalesce(ca.content,'')) > 120
MATCH (b)-[:HAS_CHUNK]->(cb:Chunk) WHERE size(coalesce(cb.content,'')) > 120
WITH a, b, r, ca, cb LIMIT $limit
RETURN a.doc_number AS a_doc, ca.content AS a_text,
       b.doc_number AS b_doc, cb.chunk_id AS b_chunk, cb.content AS b_text,
       toUpper(r.type) AS rel
"""


def _clean(t: str) -> str:
    """Bỏ marker [META]/[HIERARCHY]/[CONTENT] khỏi nội dung chunk."""
    return re.sub(r"\[(META|HIERARCHY|CONTENT)\][^\n]*\n?", "", t or "").strip()


def _salient(text: str, n: int = 10) -> str:
    toks = re.findall(r"\w+", text.lower())
    sal = [t for t in toks if t not in STOP and len(t) > 2]
    return " ".join(sal[:n]) if len(sal) >= 3 else " ".join(toks[:n])


def generate(limit: int, out: str) -> int:
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, getattr(config, "NEO4J_PASSWORD", "")),
    )
    cases = []
    try:
        with driver.session() as sess:
            for rec in sess.run(CYPHER, bo_tro=BO_TRO, limit=int(limit)):
                a_text = _clean(str(rec["a_text"] or ""))
                b_chunk = str(rec["b_chunk"] or "")
                b_doc = str(rec["b_doc"] or "")
                if not a_text or not b_chunk or not b_doc:
                    continue
                kw = _salient(a_text)
                # Query NEO vào ngữ cảnh A; đáp án ở B → bắt buộc multi-hop.
                query = f"Về {kw}, văn bản quy định chi tiết/hướng dẫn thi hành nêu cụ thể như thế nào?"
                cases.append({
                    "benchmark_id": f"mh_{len(cases):05d}",
                    "query_type": "multi_hop",
                    "query": query,
                    "ground_truth_doc_number": b_doc,
                    "ground_truth_chunk": b_chunk,
                    "anchor_doc_number": str(rec["a_doc"] or ""),
                    "relation": str(rec["rel"] or ""),
                    "note": "Đáp án ở B (bổ trợ từ A) — cần graph-expansion; template, nên LLM-naturalize.",
                })
    finally:
        driver.close()

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("✓ Sinh %d ca multi-hop → %s", len(cases), outp.resolve())
    return len(cases)


def main():
    ap = argparse.ArgumentParser(description="Sinh bộ test multi-hop thật từ đồ thị HIN (Track 3.1)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "vietlawbench_multihop.jsonl"))
    args = ap.parse_args()
    generate(args.limit, args.out)


if __name__ == "__main__":
    main()
