"""
benchmark/inventory_team_corpus.py — Kiểm kê kho văn bản parquet của nhóm tự cào (read-only)
và đo khả năng liên kết với bộ dữ liệu đánh giá VLQA.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.global_id import normalize_doc_number
from configs.logging_config import get_subsystem_logger

logger = get_subsystem_logger("benchmark", "inventory_team_corpus")

def char_ngrams(text: str, n: int = 5) -> Set[str]:
    """
    Tách tập n-gram ký tự sau khi viết thường và gộp khoảng trắng liên tiếp.
    Trả về tập rỗng nếu text rỗng hoặc độ dài nhỏ hơn n.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if len(s) < n or n <= 0:
        return set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}

def containment(a: Set[str] | str, b: Set[str] | str, n: int = 5) -> float:
    """
    Tính độ bao hàm len(a & b) / len(a).
    Hỗ trợ đầu vào là Set[str] n-gram hoặc str văn bản thô.
    Nếu a rỗng, trả về 0.0 (không chia cho 0).
    """
    set_a = char_ngrams(a, n=n) if isinstance(a, str) else a
    set_b = char_ngrams(b, n=n) if isinstance(b, str) else b
    if not set_a:
        return 0.0
    return len(set_a & set_b) / float(len(set_a))

def scan_parquet_corpus(root: Path | str) -> Dict[str, Any]:
    """
    Quét kiểm kê toàn bộ file parquet dưới thư mục root.
    """
    root_path = Path(root)
    files = sorted(list(root_path.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file parquet nào dưới: {root_path}")

    logger.info("Tìm thấy %d file parquet dưới %s", len(files), root_path)

    # 1. Schema từ file đầu tiên
    schema0 = pq.read_schema(files[0])
    schema_dict = {name: str(typ) for name, typ in zip(schema0.names, schema0.types)}

    schema_diffs = []
    for f in files[1:]:
        s = pq.read_schema(f)
        if s != schema0:
            schema_diffs.append(f.name)

    logger.info("Schema đồng nhất trên mọi file: %s (khác biệt: %d)", len(schema_diffs) == 0, len(schema_diffs))

    # 2. Thu thập dữ liệu
    total_rows = 0
    doc_ids = set()
    doc_numbers = set()
    norm_doc_numbers = set()
    level_counts = Counter()
    is_admin_counts = Counter()
    has_eff_date_counts = Counter()
    doc_seen = set()
    doc_types = Counter()
    issue_years = Counter()
    all_lengths: List[int] = []
    hierarchy_samples: List[str] = []

    columns_to_read = [
        "doc_id", "doc_number", "doc_title", "hierarchy_path",
        "level", "content", "issue_date", "effective_date", "is_administrative"
    ]

    for idx, f in enumerate(files):
        t = pq.read_table(f, columns=columns_to_read)
        num_rows = t.num_rows
        total_rows += num_rows

        d_ids = t["doc_id"].to_pylist()
        d_nums = t["doc_number"].to_pylist()
        d_titles = t["doc_title"].to_pylist()
        d_hiers = t["hierarchy_path"].to_pylist()
        d_levels = t["level"].to_pylist()
        d_contents = t["content"].to_pylist()
        d_issues = t["issue_date"].to_pylist()
        d_effs = t["effective_date"].to_pylist()
        d_admins = t["is_administrative"].to_pylist()

        doc_ids.update(d_ids)
        doc_numbers.update(d_nums)
        for num in d_nums:
            norm_num = normalize_doc_number(num)
            if norm_num:
                norm_doc_numbers.add(norm_num)

        level_counts.update(d_levels)
        is_admin_counts.update(d_admins)

        for eff in d_effs:
            has_eff_date_counts[bool(eff and str(eff).strip())] += 1

        for c in d_contents:
            all_lengths.append(len(c) if c else 0)

        for h in d_hiers:
            if h and len(hierarchy_samples) < 3:
                h_str = str(h).strip()[:120]
                if h_str not in hierarchy_samples:
                    hierarchy_samples.append(h_str)

        for d_id, title, dt in zip(d_ids, d_titles, d_issues):
            if d_id not in doc_seen:
                doc_seen.add(d_id)
                m = re.match(r"^(Luật|Nghị định|Nghị quyết|Quyết định|Thông tư|Chỉ thị|Pháp lệnh|Hiến pháp|Thông tư liên tịch)", str(title).strip())
                dtype = m.group(1) if m else "Khác"
                doc_types[dtype] += 1
                yr = str(dt)[:4] if dt and str(dt)[:4].isdigit() else "Unknown"
                issue_years[yr] += 1

        if (idx + 1) % 40 == 0 or (idx + 1) == len(files):
            logger.info("  Đã quét %d/%d files (%d chunks)...", idx + 1, len(files), total_rows)

    # Thống kê độ dài text
    arr_len = np.array(all_lengths, dtype=np.int32)
    mean_len = float(np.mean(arr_len)) if len(arr_len) > 0 else 0.0
    p50_len = float(np.percentile(arr_len, 50)) if len(arr_len) > 0 else 0.0
    p95_len = float(np.percentile(arr_len, 95)) if len(arr_len) > 0 else 0.0
    max_len = int(np.max(arr_len)) if len(arr_len) > 0 else 0

    # Ước tính RAM chỉ mục dense
    ram_1024_f32_bytes = total_rows * 1024 * 4
    ram_256_f32_bytes = total_rows * 256 * 4
    ram_256_i8_bytes = total_rows * 256 * 1

    inventory_data = {
        "schema": schema_dict,
        "schema_uniform_across_all_files": (len(schema_diffs) == 0),
        "schema_differing_files": schema_diffs,
        "counts": {
            "n_files": len(files),
            "total_rows_chunks": total_rows,
            "n_distinct_doc_ids": len(doc_ids),
            "n_distinct_doc_numbers": len(doc_numbers),
            "n_distinct_normalized_doc_numbers": len(norm_doc_numbers),
            "distinct_documents_count_column": "doc_number",
        },
        "chunk_level": {
            "distribution": dict(level_counts),
            "hierarchy_samples": hierarchy_samples,
        },
        "content_length_chars": {
            "mean": round(mean_len, 2),
            "P50": round(p50_len, 2),
            "P95": round(p95_len, 2),
            "max": max_len,
        },
        "distributions": {
            "is_administrative": {str(k): v for k, v in is_admin_counts.items()},
            "has_effective_date": {str(k): v for k, v in has_eff_date_counts.items()},
            "doc_types_by_document": dict(doc_types.most_common()),
            "issue_years_top15": dict(issue_years.most_common(15)),
        },
        "dense_index_ram_estimates": {
            "float32_1024d_bytes": ram_1024_f32_bytes,
            "float32_1024d_gb": round(ram_1024_f32_bytes / (1024**3), 3),
            "float32_256d_bytes": ram_256_f32_bytes,
            "float32_256d_gb": round(ram_256_f32_bytes / (1024**3), 3),
            "int8_256d_bytes": ram_256_i8_bytes,
            "int8_256d_gb": round(ram_256_i8_bytes / (1024**3), 3),
        },
    }

    return inventory_data

def link_vlqa(
    root: Path | str,
    vlqa_dir: Path | str,
    dev_path: Path | str,
    tune_path: Path | str,
    seed: int = 42,
    n_sample: int = 200,
) -> Dict[str, Any]:
    """
    Đo khả năng liên kết dữ liệu giữa kho văn bản nhóm tự cào và VLQA.
    """
    root_path = Path(root)
    files = sorted(list(root_path.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file parquet nào dưới: {root_path}")

    logger.info("=== LIÊN KẾT VỚI DỮ LIỆU VLQA ===")

    # 1. Thu thập normalized doc numbers của kho nhóm
    team_docs = set()
    for f in files:
        t = pq.read_table(f, columns=["doc_number"])
        for d in t["doc_number"].to_pylist():
            norm = normalize_doc_number(d)
            if norm:
                team_docs.add(norm)

    logger.info("Tổng số doc_number chuẩn hóa trong kho nhóm: %d", len(team_docs))

    # 2. Đọc legal_corpus.json của VLQA
    corpus_file = Path(vlqa_dir) / "legal_corpus.json"
    if not corpus_file.exists():
        raise FileNotFoundError(f"Không tìm thấy legal_corpus.json tại: {corpus_file}")

    with open(corpus_file, "r", encoding="utf-8") as f:
        vlqa_corpus = json.load(f)

    vlqa_docs = set()
    aid_to_doc = {}
    aid_to_text = {}

    for doc in vlqa_corpus:
        law_id = str(doc.get("law_id") or doc.get("doc_number") or "").strip()
        norm_law = normalize_doc_number(law_id)
        if norm_law:
            vlqa_docs.add(norm_law)
        for art in doc.get("content", []):
            aid = art.get("aid")
            txt = art.get("content_Article") or art.get("content") or ""
            if aid is not None:
                aid_int = int(aid)
                aid_to_doc[aid_int] = norm_law
                aid_to_text[aid_int] = txt

    overlap_docs = vlqa_docs & team_docs
    share_vlqa_docs = len(overlap_docs) / len(vlqa_docs) if vlqa_docs else 0.0
    logger.info("Văn bản VLQA có trong kho nhóm: %d/%d (%.2f%%)",
                len(overlap_docs), len(vlqa_docs), share_vlqa_docs * 100.0)

    # 3. Đánh giá dev set và tune set
    def evaluate_split(split_file: Path | str) -> Dict[str, Any]:
        with open(split_file, "r", encoding="utf-8") as sf:
            questions = [json.loads(line) for line in sf if line.strip()]

        all_exist_count = 0
        any_exist_count = 0

        for q in questions:
            doc_nums = q.get("ground_truth_doc_numbers") or [q.get("ground_truth_doc_number")]
            norms = [normalize_doc_number(dn) for dn in doc_nums if dn]
            if norms and all(dn in team_docs for dn in norms):
                all_exist_count += 1
            if any(dn in team_docs for dn in norms):
                any_exist_count += 1

        n_q = len(questions)
        return {
            "n_questions": n_q,
            "all_gold_docs_exist_count": all_exist_count,
            "share_all_gold_docs_exist": round(all_exist_count / n_q, 4) if n_q > 0 else 0.0,
            "at_least_one_gold_doc_exists_count": any_exist_count,
            "share_at_least_one_gold_doc_exists": round(any_exist_count / n_q, 4) if n_q > 0 else 0.0,
        }

    dev_stats = evaluate_split(dev_path)
    logger.info("Dev set (%d câu): ALL gold exist = %.2f%%, ANY gold exist = %.2f%%",
                dev_stats["n_questions"],
                dev_stats["share_all_gold_docs_exist"] * 100.0,
                dev_stats["share_at_least_one_gold_doc_exists"] * 100.0)

    tune_stats = evaluate_split(tune_path)
    logger.info("Tune set (%d câu): ALL gold exist = %.2f%%, ANY gold exist = %.2f%%",
                tune_stats["n_questions"],
                tune_stats["share_all_gold_docs_exist"] * 100.0,
                tune_stats["share_at_least_one_gold_doc_exists"] * 100.0)

    # 4. Article level: 200 random gold aids từ dev
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_questions = [json.loads(line) for line in f if line.strip()]

    distinct_dev_aids = sorted(list(set(aid for q in dev_questions for aid in q.get("relevant_aids", []))))
    eligible_aids = [aid for aid in distinct_dev_aids if aid_to_doc.get(aid) in team_docs]

    logger.info("Tổng số gold aid duy nhất trong dev: %d; đủ điều kiện (doc tồn tại trong kho nhóm): %d",
                len(distinct_dev_aids), len(eligible_aids))

    rng = random.Random(seed)
    actual_sample_size = min(n_sample, len(eligible_aids))
    sample_aids = sorted(rng.sample(eligible_aids, actual_sample_size))

    target_docs = set(aid_to_doc[aid] for aid in sample_aids)
    logger.info("Lấy mẫu %d aid (seed=%d) thuộc %d văn bản mục tiêu",
                actual_sample_size, seed, len(target_docs))

    # Thu thập chunk n-grams cho target_docs từ kho parquet
    doc_chunk_ngrams: Dict[str, List[Set[str]]] = {d: [] for d in target_docs}
    for f in files:
        t = pq.read_table(f, columns=["doc_number", "content"])
        for d, c in zip(t["doc_number"].to_pylist(), t["content"].to_pylist()):
            norm = normalize_doc_number(d)
            if norm in doc_chunk_ngrams and c and str(c).strip():
                doc_chunk_ngrams[norm].append(char_ngrams(c, 5))

    total_chunks_collected = sum(len(v) for v in doc_chunk_ngrams.values())
    logger.info("Đã thu thập %d chunk cho %d văn bản mục tiêu", total_chunks_collected, len(target_docs))

    best_A_ge_80 = 0
    best_C_ge_80 = 0
    neither_ge_50 = 0

    for aid in sample_aids:
        norm_doc = aid_to_doc[aid]
        art_text = aid_to_text[aid]
        g_art = char_ngrams(art_text, 5)
        chunks_g = doc_chunk_ngrams.get(norm_doc, [])

        if not chunks_g or not g_art:
            neither_ge_50 += 1
            continue

        best_A = max(containment(g_art, g_c) for g_c in chunks_g)
        best_C = max(containment(g_c, g_art) for g_c in chunks_g)

        if best_A >= 0.8:
            best_A_ge_80 += 1
        if best_C >= 0.8:
            best_C_ge_80 += 1
        if best_A < 0.5 and best_C < 0.5:
            neither_ge_50 += 1

    share_A = best_A_ge_80 / actual_sample_size if actual_sample_size > 0 else 0.0
    share_C = best_C_ge_80 / actual_sample_size if actual_sample_size > 0 else 0.0
    share_neither = neither_ge_50 / actual_sample_size if actual_sample_size > 0 else 0.0

    logger.info("Mẫu %d aid: best containment_A >= 0.8: %d (%.2f%%)",
                actual_sample_size, best_A_ge_80, share_A * 100.0)
    logger.info("Mẫu %d aid: best containment_C >= 0.8: %d (%.2f%%)",
                actual_sample_size, best_C_ge_80, share_C * 100.0)
    logger.info("Mẫu %d aid: neither >= 0.5: %d (%.2f%%)",
                actual_sample_size, neither_ge_50, share_neither * 100.0)

    vlqa_link_data = {
        "document_level": {
            "vlqa_total_docs": len(vlqa_corpus),
            "vlqa_normalized_unique_docs": len(vlqa_docs),
            "team_normalized_unique_docs": len(team_docs),
            "vlqa_docs_in_team_corpus_count": len(overlap_docs),
            "vlqa_docs_in_team_corpus_share": round(share_vlqa_docs, 4),
            "dev_split": dev_stats,
            "tune_split": tune_stats,
        },
        "article_level_sample": {
            "sample_size": actual_sample_size,
            "seed": seed,
            "target_docs_count": len(target_docs),
            "total_target_chunks_evaluated": total_chunks_collected,
            "best_containment_A_ge_0_8_count": best_A_ge_80,
            "best_containment_A_ge_0_8_share": round(share_A, 4),
            "best_containment_C_ge_0_8_count": best_C_ge_80,
            "best_containment_C_ge_0_8_share": round(share_C, 4),
            "neither_ge_0_5_count": neither_ge_50,
            "neither_ge_0_5_share": round(share_neither, 4),
        },
    }

    return vlqa_link_data

def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t0 = time.perf_counter()

    ap = argparse.ArgumentParser(description="Kiểm kê kho văn bản parquet của nhóm và liên kết VLQA")
    ap.add_argument("--root", default=r"D:\vietlawbert_data\processed", help="Thư mục gốc chứa các file parquet")
    ap.add_argument("--vlqa", action="store_true", help="Thực hiện phân tích liên kết với dữ liệu VLQA")
    ap.add_argument("--vlqa-dir", default=os.environ.get("VLQA_DIR", r"D:\VLQA"), help="Thư mục chứa VLQA dataset")
    ap.add_argument("--dev-file", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--tune-file", default=str(BENCHMARK_DIR / "vlqa_tune1690.jsonl"))
    ap.add_argument("--out", default=str(BENCHMARK_DIR / "results" / "team_corpus_inventory.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-size", type=int, default=200)

    args = ap.parse_args(argv)

    logger.info("=== BẮT ĐẦU KIỂM KÊ KHO VĂN BẢN NHÓM ===")
    data = scan_parquet_corpus(args.root)

    if args.vlqa:
        vlqa_link_res = link_vlqa(
            root=args.root,
            vlqa_dir=args.vlqa_dir,
            dev_path=args.dev_file,
            tune_path=args.tune_file,
            seed=args.seed,
            n_sample=args.sample_size,
        )
        data["vlqa_link"] = vlqa_link_res

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("✓ Đã lưu kết quả kiểm kê vào: %s", out_path)

    # In kết quả tóm tắt console
    print("\n" + "=" * 80)
    print("TỔNG KẾT KIỂM KÊ KHO VĂN BẢN PARQUET NHÓM TỰ CÀO:")
    print("=" * 80)
    cnts = data["counts"]
    print(f"  Số file parquet:                 {cnts['n_files']}")
    print(f"  Tổng số đoạn (chunks):           {cnts['total_rows_chunks']:,}")
    print(f"  Số văn bản duy nhất (doc_id):    {cnts['n_distinct_doc_ids']:,}")
    print(f"  Số số hiệu duy nhất (doc_number):{cnts['n_distinct_doc_numbers']:,}")
    print(f"  Số số hiệu đã chuẩn hóa:         {cnts['n_distinct_normalized_doc_numbers']:,}")
    print("\nPhân bố cấp độ chunk (level):")
    for lvl, cnt in data["chunk_level"]["distribution"].items():
        print(f"  - {lvl}: {cnt:,} ({cnt/cnts['total_rows_chunks']*100:.2f}%)")
    print("\nMẫu đường dẫn phân cấp (hierarchy samples):")
    for s in data["chunk_level"]["hierarchy_samples"]:
        print(f"  * {s}")
    clen = data["content_length_chars"]
    print(f"\nĐộ dài nội dung (ký tự): Trung bình={clen['mean']}, P50={clen['P50']}, P95={clen['P95']}, Max={clen['max']:,}")
    rams = data["dense_index_ram_estimates"]
    print(f"\nƯớc tính RAM chỉ mục Dense ({cnts['total_rows_chunks']:,} chunks):")
    print(f"  - 1024 chiều (float32): {rams['float32_1024d_gb']:.2f} GB ({rams['float32_1024d_bytes']:,} bytes)")
    print(f"  - 256 chiều (float32):  {rams['float32_256d_gb']:.2f} GB ({rams['float32_256d_bytes']:,} bytes)")
    print(f"  - 256 chiều (int8):     {rams['int8_256d_gb']:.2f} GB ({rams['int8_256d_bytes']:,} bytes)")

    if "vlqa_link" in data:
        vlink = data["vlqa_link"]
        dlevel = vlink["document_level"]
        print("\n" + "-" * 80)
        print("KẾT QUẢ LIÊN KẾT VỚI BỘ DỮ LIỆU VLQA:")
        print("-" * 80)
        print(f"  Văn bản VLQA tồn tại trong kho nhóm: {dlevel['vlqa_docs_in_team_corpus_count']:,}/{dlevel['vlqa_normalized_unique_docs']:,} ({dlevel['vlqa_docs_in_team_corpus_share']*100:.2f}%)")
        dev_s = dlevel["dev_split"]
        tune_s = dlevel["tune_split"]
        print(f"  Dev (500 câu):   ALL gold exist = {dev_s['share_all_gold_docs_exist']*100:.2f}% | ANY gold exist = {dev_s['share_at_least_one_gold_doc_exists']*100:.2f}%")
        print(f"  Tune (1690 câu): ALL gold exist = {tune_s['share_all_gold_docs_exist']*100:.2f}% | ANY gold exist = {tune_s['share_at_least_one_gold_doc_exists']*100:.2f}%")
        asamp = vlink["article_level_sample"]
        print(f"  Mẫu {asamp['sample_size']} gold aids dev (seed={asamp['seed']}):")
        print(f"    - Chứa trong chunk nhóm (containment_A >= 0.8): {asamp['best_containment_A_ge_0_8_share']*100:.2f}%")
        print(f"    - Chunk nhóm nằm trong điều (containment_C >= 0.8): {asamp['best_containment_C_ge_0_8_share']*100:.2f}%")
        print(f"    - Không khớp (neither >= 0.5):                  {asamp['neither_ge_0_5_share']*100:.2f}%")

    t_end = time.perf_counter()
    print("=" * 80)
    print(f"Hoàn thành trong: {t_end - t0:.2f} giây")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()