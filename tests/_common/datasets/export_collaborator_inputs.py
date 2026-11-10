"""
tests/_common/datasets/export_collaborator_inputs.py
Xuất MỘT LẦN toàn bộ Input cho gói cộng sự (task-briefs mục A):
  - 50 triplet dạng ĐỌC ĐƯỢC (3.3)           ← từ artifacts/triplets/*.parquet
  - Mẫu Cochran ~382 chunk phân tầng (1.6)    ← Neo4j (nếu nối được)
  - Danh sách doc_number phân biệt (DATA-1)   ← Neo4j (nếu nối được)
Chạy:  python -m tests._common.datasets.export_collaborator_inputs --cochran
KHÔNG đụng VLQA (chỉ corpus của mình). File ra nằm trong collaborator_inputs/ (đã .gitignore nếu cần).
"""
from __future__ import annotations
import os, json, math, glob, argparse, random
from pathlib import Path
from configs.config import config
from configs.paths import TRIPLETS_DIR, ROOT_DIR
from configs.global_id import classify_doc_number
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("tests", "export_collab")

_ANCHOR = ["anchor", "anchor_text", "query"]
_POS = ["positive", "positive_text", "pos"]
_NEG = ["negative", "negative_text", "hard_negative", "neg"]


def _pick(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None


def detect_cols(df):
    """Dò tên cột text anchor/pos/neg bất kể schema miner xuất kiểu nào."""
    cols = set(df.columns)
    return _pick(cols, _ANCHOR), _pick(cols, _POS), _pick(cols, _NEG)


def sample_triplets(df, n=50, seed=42):
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def render_triplet_md(row, a, p, g, idx):
    """Một triplet thành khối markdown có ô chấm trống cho cộng sự."""
    def _t(col):
        return str(row[col]).strip().replace("\n", " ")[:800] if col else "(thiếu cột)"
    return (
        f"## Triplet {idx+1}\n"
        f"**ANCHOR:** {_t(a)}\n\n"
        f"**POSITIVE (phải GẦN/bổ trợ anchor):** {_t(p)}\n\n"
        f"**NEGATIVE (phải ĐỐI KHÁNG — bị anchor thay thế/bãi bỏ):** {_t(g)}\n\n"
        f"- [ ] positive ĐÚNG cực tính?  (đúng/sai + lý do): \n"
        f"- [ ] negative ĐÚNG cực tính?  (đúng/sai + lý do): \n\n---\n"
    )


def export_triplets(out_dir: Path, n: int):
    import pandas as pd
    cand = TRIPLETS_DIR / "hin_triplets.parquet"
    path = cand if cand.exists() else None
    if path is None:
        hits = sorted(glob.glob(str(TRIPLETS_DIR / "*.parquet")))
        if not hits:
            logger.warning("Không thấy parquet triplet trong %s — bỏ qua 3.3.", TRIPLETS_DIR)
            return None
        path = Path(hits[0])
    df = pd.read_parquet(path)
    a, p, g = detect_cols(df)
    if not (a and p):
        logger.warning("Parquet %s thiếu cột anchor/positive (%s) — bỏ qua.", path.name, list(df.columns))
        return None
    sub = sample_triplets(df, n)
    md = out_dir / "triplets_50_review.md"
    jl = out_dir / "triplets_50_review.jsonl"
    with open(md, "w", encoding="utf-8") as fm, open(jl, "w", encoding="utf-8") as fj:
        fm.write(f"# 50 triplet kiểm cực tính (task 3.3) — nguồn: {path.name}\n\n")
        fm.write("POSITIVE = gần/bổ trợ anchor · NEGATIVE = bị anchor thay thế/bãi bỏ (gần từ vựng nhưng SAI hiệu lực).\n\n---\n")
        for i, row in sub.iterrows():
            fm.write(render_triplet_md(row, a, p, g, i))
            fj.write(json.dumps({
                "idx": int(i),
                "anchor": str(row[a]), "positive": str(row[p]),
                "negative": (str(row[g]) if g else None),
            }, ensure_ascii=False) + "\n")
    logger.info("✓ 3.3: %d triplet → %s | %s", len(sub), md, jl)
    return md


def export_from_neo4j(out_dir: Path, cochran_n: int):
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        drv.verify_connectivity()
    except Exception as exc:
        logger.warning("Không nối được Neo4j (%s) — bỏ qua 1.6 + DATA-1. Chạy lại trên máy có Neo4j.", str(exc)[:80])
        return
    with drv.session() as s:
        # DATA-1: doc_number phân biệt
        dns = [r["dn"] for r in s.run(
            "MATCH (d:LawDocument) WHERE d.doc_number IS NOT NULL RETURN DISTINCT d.doc_number AS dn")]
        import csv as _csv, collections as _col
        cat = _col.Counter()
        with open(out_dir / "doc_numbers_categorized.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f); w.writerow(["doc_number", "status"])
            for dn in sorted(map(str, dns)):
                st = classify_doc_number(dn)[0]
                cat[st] += 1
                w.writerow([dn, st])
        (out_dir / "doc_numbers_neo4j.txt").write_text("\n".join(sorted(map(str, dns))), encoding="utf-8")
        logger.info("✓ DATA-1: %d doc_number → phân loại VALID=%d · NO_NUMBER=%d · CONTAMINATED=%d → doc_numbers_categorized.csv",
                    len(dns), cat.get("VALID",0), cat.get("NO_NUMBER",0), cat.get("CONTAMINATED",0))
        # 1.6: mẫu Cochran phân tầng theo doc_type
        rows = s.run(
            "MATCH (d:LawDocument)-[:HAS_CHUNK]->(c:Chunk) "
            "WHERE c.content IS NOT NULL AND size(c.content) > 50 "
            "RETURN c.chunk_id AS cid, d.doc_number AS dn, "
            "coalesce(d.doc_type,'KHAC') AS dtype, c.content AS content"
        ).data()
    drv.close()
    if not rows:
        logger.warning("Không có chunk để lấy mẫu Cochran."); return
    # phân tầng theo doc_type, tỉ lệ theo quần thể
    random.seed(42)
    by = {}
    for r in rows:
        by.setdefault(r["dtype"], []).append(r)
    N = len(rows)
    take_total = min(cochran_n, N)
    sample = []
    for dtype, grp in by.items():
        k = max(1, round(take_total * len(grp) / N))
        random.shuffle(grp)
        sample.extend(grp[:k])
    random.shuffle(sample)
    sample = sample[:take_total]
    import csv
    with open(out_dir / "cochran_sample_382.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "doc_number", "doc_type", "content",
                    "nhan_SACH_BAN", "co_OCR", "co_lechnganh", "co_trunglap", "co_catcut", "ly_do"])
        for r in sample:
            w.writerow([r["cid"], r["dn"], r["dtype"], str(r["content"])[:1200], "", "", "", "", "", ""])
    logger.info("✓ 1.6: mẫu Cochran %d chunk (phân tầng %d loại) → cochran_sample_382.csv",
                len(sample), len(by))


def main():
    ap = argparse.ArgumentParser(description="Xuất Input gói cộng sự (3.3 + 1.6 + DATA-1)")
    ap.add_argument("--out-dir", default=str(ROOT_DIR / "collaborator_inputs"))
    ap.add_argument("--n-triplets", type=int, default=50)
    ap.add_argument("--cochran", action="store_true", help="Thêm mẫu Cochran + doc_number (cần Neo4j)")
    ap.add_argument("--cochran-n", type=int, default=382)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    export_triplets(out, args.n_triplets)
    if args.cochran:
        export_from_neo4j(out, args.cochran_n)
    logger.info("Hoàn tất. Giao thư mục %s cho cộng sự.", out)


if __name__ == "__main__":
    main()
