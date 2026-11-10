"""
benchmark/prepare_team_dense.py — Chuẩn bị dữ liệu đầu vào cho bước dense retrieval trên Kaggle.

Bao gồm 3 chức năng chính (sub-commands):
1. dupcheck:   Kiểm tra 410 chunk_id bị lặp, xác nhận nội dung và quy tắc lưu trữ của ES.
2. export-bm25: Xuất danh sách ứng viên BM25-seg top-100 cho VLQA test và VietLawBench.
3. slim:       Tạo kho văn bản rút gọn (chunk_id, level, content) đã loại bỏ trùng lặp để tải lên Kaggle.
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import hashlib
import heapq
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter

import pyarrow as pa
import pyarrow.parquet as pq
from elasticsearch import Elasticsearch

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger
from preprocess.vi_segment import get_segmenter, segment_text
from benchmark.evaluate_team_corpus import to_group_ranking
from benchmark.metrics_article import ndcg_at_k

logger = get_subsystem_logger("benchmark", "prepare_team_dense")


def compute_dedup_drops(files_with_cids: List[Tuple[str, List[str]]]) -> Set[Tuple[int, int]]:
    """
    Xác định các hàng trùng lặp cần loại bỏ theo quy tắc: giữ lại lần xuất hiện cuối cùng
    theo thứ tự tên file tăng dần, rồi đến thứ tự hàng tăng dần.
    
    Quét ngược từ file cuối cùng về file đầu tiên, và từ hàng cuối cùng về hàng đầu tiên.
    Bản ghi nào đã thấy (seen) ở file/hàng phía sau thì bản ghi ở phía trước sẽ bị loại bỏ (drop).
    
    Trả về tập hợp các cặp (file_index, row_index) cần loại bỏ.
    """
    seen: Set[str] = set()
    drops: Set[Tuple[int, int]] = set()

    n_files = len(files_with_cids)
    for file_idx in range(n_files - 1, -1, -1):
        _, cids = files_with_cids[file_idx]
        n_rows = len(cids)
        for row_idx in range(n_rows - 1, -1, -1):
            cid = cids[row_idx]
            if cid in seen:
                drops.add((file_idx, row_idx))
            else:
                seen.add(cid)

    return drops


def apply_dedup_to_table(table: pa.Table, drop_row_indices: Set[int]) -> pa.Table:
    """
    Lọc bỏ các hàng trong drop_row_indices và chỉ giữ lại các cột [chunk_id, level, content].
    """
    cols = ["chunk_id", "level", "content"]
    if not drop_row_indices:
        return table.select(cols)

    keep_mask = [i not in drop_row_indices for i in range(table.num_rows)]
    filtered = table.filter(pa.array(keep_mask, type=pa.bool_()))
    return filtered.select(cols)


def check_duplicates(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phân tích danh sách các bản ghi để đếm và phân loại trùng lặp chunk_id.
    """
    cid_occs = defaultdict(list)
    for r in records:
        cid_occs[r["chunk_id"]].append(r)

    repeated_ids = {cid: occs for cid, occs in cid_occs.items() if len(occs) > 1}

    identical_content = 0
    different_content = 0
    diff_same_doc = 0
    diff_diff_doc = 0
    diff_chunk_ids = []

    for cid, occs in repeated_ids.items():
        hashes = set()
        for o in occs:
            cnt = o.get("content", "")
            h = hashlib.sha256(cnt.encode("utf-8")).hexdigest() if cnt else ""
            hashes.add(h)

        if len(hashes) == 1:
            identical_content += 1
        else:
            different_content += 1
            diff_chunk_ids.append(cid)
            dnums = set(o.get("doc_number", "") for o in occs)
            if len(dnums) == 1:
                diff_same_doc += 1
            else:
                diff_diff_doc += 1

    return {
        "repeated_ids_count": len(repeated_ids),
        "identical_content_count": identical_content,
        "different_content_count": different_content,
        "different_content_same_doc_count": diff_same_doc,
        "different_content_diff_doc_count": diff_diff_doc,
        "diff_chunk_ids": diff_chunk_ids,
        "repeated_ids": repeated_ids,
    }


def run_dupcheck(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=== BẮT ĐẦU KIỂM TRA CHUNK_ID TRÙNG LẶP (DUPCHECK) ===")

    root_path = Path(args.root)
    files = sorted(list(root_path.glob("*.parquet")))
    logger.info("Tìm thấy %d file parquet trong %s", len(files), root_path)

    cid_occs = defaultdict(list)
    total_rows = 0

    for pf in files:
        t = pq.read_table(pf, columns=["chunk_id", "doc_number", "content"])
        cids = t["chunk_id"].to_pylist()
        dnums = t["doc_number"].to_pylist()
        contents = t["content"].to_pylist()
        total_rows += len(cids)

        for row_idx, (cid, dnum, cnt) in enumerate(zip(cids, dnums, contents)):
            h = hashlib.sha256(cnt.encode("utf-8")).hexdigest() if cnt else ""
            cid_occs[cid].append({
                "file": pf.name,
                "row": row_idx,
                "doc_number": dnum,
                "content_hash": h,
            })

    unique_cids_count = len(cid_occs)
    repeated_ids = {cid: occs for cid, occs in cid_occs.items() if len(occs) > 1}
    logger.info("Tổng số hàng: %d | Chunk ID duy nhất: %d | Chunk ID lặp lại: %d",
                total_rows, unique_cids_count, len(repeated_ids))

    identical_content = 0
    different_content = 0
    diff_same_doc = 0
    diff_diff_doc = 0
    diff_chunk_ids = []
    file_distribution = Counter()

    for cid, occs in repeated_ids.items():
        hashes = set(o["content_hash"] for o in occs)
        file_tuple = tuple(o["file"] for o in occs)
        file_distribution[file_tuple] += 1

        if len(hashes) == 1:
            identical_content += 1
        else:
            different_content += 1
            diff_chunk_ids.append(cid)
            dnums = set(o["doc_number"] for o in occs)
            if len(dnums) == 1:
                diff_same_doc += 1
            else:
                diff_diff_doc += 1

    logger.info("  * Trùng lặp nhưng nội dung GIỐNG NHAU hoàn toàn: %d", identical_content)
    logger.info("  * Trùng lặp có nội dung KHÁC NHAU:               %d", different_content)
    logger.info("    - Cùng doc_number: %d", diff_same_doc)
    logger.info("    - Khác doc_number: %d", diff_diff_doc)

    # Phân bố theo cặp file
    file_dist_str = {}
    for ft, cnt in file_distribution.items():
        key_str = ", ".join(ft)
        file_dist_str[key_str] = cnt
        logger.info("    - Cặp file [%s]: %d ids", key_str, cnt)

    # Xác nhận bản sao Elasticsearch đã lưu
    es = Elasticsearch([config.ES_HOST])
    sample_ids = list(repeated_ids.keys())[:args.es_check_ids]
    es_hash_matches_last = True

    for cid in sample_ids:
        doc = es.get(index=args.index, id=cid)
        c_es = doc["_source"]["content_plain"]
        h_es = hashlib.sha256(c_es.encode("utf-8")).hexdigest()
        # Lấy hash của bản ghi xuất hiện cuối cùng
        last_hash = repeated_ids[cid][-1]["content_hash"]
        if h_es != last_hash:
            es_hash_matches_last = False
            logger.error("Hash trong ES không khớp bản ghi cuối cùng cho cid: %s", cid)

    logger.info("Xác nhận Elasticsearch: khớp bản sao cuối cùng = %s (trên %d mẫu)",
                es_hash_matches_last, len(sample_ids))

    # Đối chiếu với gold của VLQA và VietLawBench
    diff_ids_set = set(diff_chunk_ids)
    vlqa_affected_qids = []
    vlqa_gold_file = Path(args.vlqa_gold)
    if vlqa_gold_file.exists():
        with open(vlqa_gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                qid = d["qid"]
                for aid, ainfo in d.get("per_aid", {}).items():
                    for gcid in ainfo.get("gold_chunks", []):
                        if gcid in diff_ids_set:
                            vlqa_affected_qids.append(qid)
                            break

    vlb_affected_ids = []
    vlb_gold_file = Path(args.vlb_gold)
    if vlb_gold_file.exists():
        with open(vlb_gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                bid = d["benchmark_id"]
                for gcid in d.get("relevant_chunk_ids", []):
                    if gcid in diff_ids_set:
                        vlb_affected_ids.append(bid)
                        break

    logger.info("Đối chiếu với tập gold:")
    logger.info("  * VLQA test:        %d câu hỏi bị ảnh hưởng", len(vlqa_affected_qids))
    logger.info("  * VietLawBench tune: %d câu hỏi bị ảnh hưởng", len(vlb_affected_ids))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "total_unique_chunk_ids": unique_cids_count,
        "repeated_chunk_ids_count": len(repeated_ids),
        "identical_content_count": identical_content,
        "different_content_count": different_content,
        "different_content_same_doc_number": diff_same_doc,
        "different_content_different_doc_number": diff_diff_doc,
        "repeated_files_distribution": file_dist_str,
        "es_kept_rule": "last_occurrence_by_filename_and_row_order",
        "es_content_hash_matches_last_copy": es_hash_matches_last,
        "sample_checked_chunk_ids": sample_ids,
        "gold_cross_check": {
            "vlqa_affected_gold_chunk_ids_count": len(vlqa_affected_qids),
            "vlqa_affected_qids": sorted(list(set(vlqa_affected_qids))),
            "vlb_affected_gold_chunk_ids_count": len(vlb_affected_ids),
            "vlb_affected_benchmark_ids": sorted(list(set(vlb_affected_ids))),
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    logger.info("✓ Đã lưu kết quả kiểm tra vào: %s", out_path)

    if len(vlqa_affected_qids) > 20:
        logger.error("DỪNG: Có hơn 20 câu hỏi VLQA bị ảnh hưởng (%d câu)!", len(vlqa_affected_qids))
        sys.exit(1)

    logger.info("✓ Hoàn tất dupcheck trong %.2f giây", time.perf_counter() - t0)


def run_export_bm25(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=== BẮT ĐẦU XUẤT ỨNG VIÊN BM25-SEG (EXPORT-BM25) ===")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sao chép byte-identical file VLQA và VLB
    vlqa_src = Path(args.vlqa_file)
    vlb_src = Path(args.vlb_file)

    vlqa_dst = out_dir / vlqa_src.name
    vlb_dst = out_dir / vlb_src.name

    shutil.copyfile(vlqa_src, vlqa_dst)
    shutil.copyfile(vlb_src, vlb_dst)

    assert hashlib.sha256(vlqa_src.read_bytes()).hexdigest() == hashlib.sha256(vlqa_dst.read_bytes()).hexdigest()
    assert hashlib.sha256(vlb_src.read_bytes()).hexdigest() == hashlib.sha256(vlb_dst.read_bytes()).hexdigest()
    logger.info("✓ Đã sao chép byte-identical %s và %s vào %s", vlqa_src.name, vlb_src.name, out_dir)

    # 2. Chuẩn bị tìm kiếm ES
    es = Elasticsearch([config.ES_HOST], timeout=60)
    seg_fn = get_segmenter("pyvi")

    bm25_out_path = out_dir / "bm25_seg_top100_team.jsonl"
    bm25_file = open(bm25_out_path, "w", encoding="utf-8")

    # 3. VLQA test set (2.190 câu)
    logger.info("Đang truy xuất và xuất top-100 cho 2.190 câu VLQA test...")
    vlqa_items = []
    with open(vlqa_src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vlqa_items.append(json.loads(line))

    vlqa_ndcg_list = []
    for idx, item in enumerate(vlqa_items):
        qid = item["qid"]
        query = item["query"]
        per_aid = item.get("per_aid", {})

        chunk_to_group = {}
        gold_groups = set()
        for aid_str, ainfo in per_aid.items():
            gid = f"aid:{aid_str}"
            gold_groups.add(gid)
            for cid in ainfo.get("gold_chunks", []):
                chunk_to_group[cid] = gid

        seg_query = segment_text(query, seg_fn)
        resp = es.search(
            index=args.index,
            query={"match": {"content_seg": seg_query}},
            size=args.depth,
            _source=["chunk_id"],
        )
        hits = resp["hits"]["hits"]
        ranked = [h["_source"]["chunk_id"] for h in hits]
        scores = [float(h["_score"]) for h in hits]

        bm25_file.write(json.dumps({
            "set": "vlqa",
            "qid": qid,
            "ranked": ranked,
            "scores": scores,
        }, ensure_ascii=False) + "\n")

        ranked_groups = to_group_ranking(ranked, chunk_to_group)
        score_ndcg = ndcg_at_k(ranked_groups, gold_groups, 10)
        vlqa_ndcg_list.append(score_ndcg)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(vlqa_items):
            logger.info("  VLQA: Đã xử lý %d/%d câu hỏi...", idx + 1, len(vlqa_items))

    # 4. VietLawBench tune (710 câu mapped)
    logger.info("Đang truy xuất và xuất top-100 cho 710 câu VietLawBench mapped...")
    vlb_items = []
    with open(vlb_src, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "mapped":
                    vlb_items.append(row)

    vlb_ndcg_list = []
    for idx, item in enumerate(vlb_items):
        bid = item["benchmark_id"]
        query = item["query"]
        relevant_cids = item.get("relevant_chunk_ids", [])

        gid = f"vlb:{bid}"
        chunk_to_group = {cid: gid for cid in relevant_cids}
        gold_groups = {gid}

        seg_query = segment_text(query, seg_fn)
        resp = es.search(
            index=args.index,
            query={"match": {"content_seg": seg_query}},
            size=args.depth,
            _source=["chunk_id"],
        )
        hits = resp["hits"]["hits"]
        ranked = [h["_source"]["chunk_id"] for h in hits]
        scores = [float(h["_score"]) for h in hits]

        bm25_file.write(json.dumps({
            "set": "vlb",
            "qid": bid,
            "ranked": ranked,
            "scores": scores,
        }, ensure_ascii=False) + "\n")

        ranked_groups = to_group_ranking(ranked, chunk_to_group)
        score_ndcg = ndcg_at_k(ranked_groups, gold_groups, 10)
        vlb_ndcg_list.append(score_ndcg)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(vlb_items):
            logger.info("  VLB: Đã xử lý %d/%d câu hỏi...", idx + 1, len(vlb_items))

    bm25_file.close()
    logger.info("✓ Đã lưu danh sách ứng viên vào: %s", bm25_out_path)

    # 5. Cross-check số liệu NDCG@10
    mean_vlqa_ndcg = float(sum(vlqa_ndcg_list) / len(vlqa_ndcg_list))
    mean_vlb_ndcg = float(sum(vlb_ndcg_list) / len(vlb_ndcg_list))

    expected_vlqa = 0.3751
    expected_vlb = 0.4698

    logger.info("=== KIỂM TRA ĐỐI CHIẾU (CROSS-CHECK) NDCG@10 ===")
    logger.info("  VLQA all_2190:  thực tế = %.4f | kỳ vọng = %.4f | độ lệch = %+.4f",
                mean_vlqa_ndcg, expected_vlqa, mean_vlqa_ndcg - expected_vlqa)
    logger.info("  VLB mapped_710: thực tế = %.4f | kỳ vọng = %.4f | độ lệch = %+.4f",
                mean_vlb_ndcg, expected_vlb, mean_vlb_ndcg - expected_vlb)

    if abs(mean_vlqa_ndcg - expected_vlqa) > 0.001:
        logger.error("DỪNG: NDCG@10 của VLQA lệch vượt quá 0.001!")
        sys.exit(1)

    if abs(mean_vlb_ndcg - expected_vlb) > 0.001:
        logger.error("DỪNG: NDCG@10 của VietLawBench lệch vượt quá 0.001!")
        sys.exit(1)

    logger.info("✓ Đối chiếu khớp chính xác hoàn toàn!")
    logger.info("✓ Hoàn tất export-bm25 trong %.2f giây", time.perf_counter() - t0)


def run_slim(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=== BẮT ĐẦU TẠO KHO VĂN BẢN RÚT GỌN (SLIM) ===")

    root_path = Path(args.root)
    files = sorted(list(root_path.glob("*.parquet")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bước 1: Quét ID để xác định các hàng cần loại bỏ
    logger.info("Bước 1: Quét chunk_id của %d file để xác định bản ghi lặp cần drop...", len(files))
    files_with_cids: List[Tuple[str, List[str]]] = []
    for pf in files:
        t = pq.read_table(pf, columns=["chunk_id"])
        files_with_cids.append((pf.name, t["chunk_id"].to_pylist()))

    drops = compute_dedup_drops(files_with_cids)
    logger.info("Tổng số hàng cần drop trên toàn bộ kho: %d", len(drops))

    file_drop_map = defaultdict(set)
    for f_idx, r_idx in drops:
        file_drop_map[f_idx].add(r_idx)

    # Bước 2: Tạo các file slim và tính toán thống kê
    logger.info("Bước 2: Xuất %d file parquet slim vào %s...", len(files), out_dir)
    total_rows = 0
    total_bytes = 0
    sha256_dict = {}

    total_chars = 0
    chunks_gt_2k = 0
    chunks_gt_20k = 0
    chunks_gt_200k = 0
    top_5_lengths: List[int] = []

    for f_idx, pf in enumerate(files):
        t_raw = pq.read_table(pf, columns=["chunk_id", "level", "content"])
        drop_indices = file_drop_map.get(f_idx, set())
        t_slim = apply_dedup_to_table(t_raw, drop_indices)

        out_file = out_dir / pf.name
        pq.write_table(t_slim, out_file)

        # Tính hash sha256 và kích thước
        file_bytes = out_file.read_bytes()
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        sha256_dict[pf.name] = file_sha256

        file_size = len(file_bytes)
        total_bytes += file_size
        total_rows += t_slim.num_rows

        # Thống kê độ dài văn bản
        for cnt in t_slim["content"].to_pylist():
            c_len = len(cnt) if cnt else 0
            total_chars += c_len
            if c_len > 2000:
                chunks_gt_2k += 1
            if c_len > 20000:
                chunks_gt_20k += 1
            if c_len > 200000:
                chunks_gt_200k += 1

            if len(top_5_lengths) < 5:
                heapq.heappush(top_5_lengths, c_len)
            else:
                heapq.heappushpop(top_5_lengths, c_len)

        if (f_idx + 1) % 40 == 0 or (f_idx + 1) == len(files):
            logger.info("  Đã xuất %d/%d file slim...", f_idx + 1, len(files))

    expected_rows = 1130317
    if total_rows != expected_rows:
        logger.error("DỪNG: Tổng số hàng slim (%d) khác 1.130.317!", total_rows)
        sys.exit(1)

    top_5_sorted = sorted(top_5_lengths, reverse=True)

    manifest = {
        "n_files": len(files),
        "n_rows": total_rows,
        "total_bytes": total_bytes,
        "sha256": sha256_dict,
        "dedup_rule": "keep_last_occurrence_by_filename_and_row_order",
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("✓ Đã lưu manifest vào: %s", manifest_path)
    logger.info("=== THỐNG KÊ KHO VĂN BẢN RÚT GỌN ===")
    logger.info("  * Tổng số file:              %d", len(files))
    logger.info("  * Tổng số đoạn (chunks):     %d", total_rows)
    logger.info("  * Dung lượng lưu trữ:        %.2f MB", total_bytes / (1024 * 1024))
    logger.info("  * Tổng số ký tự nội dung:    %d", total_chars)
    logger.info("  * Số chunk > 2.000 ký tự:    %d", chunks_gt_2k)
    logger.info("  * Số chunk > 20.000 ký tự:   %d", chunks_gt_20k)
    logger.info("  * Số chunk > 200.000 ký tự:  %d", chunks_gt_200k)
    logger.info("  * 5 độ dài lớn nhất:         %s", top_5_sorted)
    logger.info("✓ Hoàn tất slim trong %.2f giây", time.perf_counter() - t0)


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu cho dense retrieval trên kho nhóm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: dupcheck
    p_dup = subparsers.add_parser("dupcheck", help="Kiểm tra chunk_id trùng lặp")
    p_dup.add_argument("--root", default=r"D:\vietlawbert_data\processed\processed_parquet")
    p_dup.add_argument("--vlqa-gold", default=str(BENCHMARK_DIR / "vlqa_test2190_team.jsonl"))
    p_dup.add_argument("--vlb-gold", default=str(BENCHMARK_DIR / "vietlawbench_tune_team.jsonl"))
    p_dup.add_argument("--out", default=str(BENCHMARK_DIR / "results" / "team_duplicate_chunk_ids.json"))
    p_dup.add_argument("--index", default="team_sparse_seg_idx")
    p_dup.add_argument("--es-check-ids", type=int, default=5)

    # Sub-command: export-bm25
    p_bm25 = subparsers.add_parser("export-bm25", help="Xuất ứng viên BM25-seg top-100")
    p_bm25.add_argument("--index", default="team_sparse_seg_idx")
    p_bm25.add_argument("--vlqa-file", default=str(BENCHMARK_DIR / "vlqa_test2190_team.jsonl"))
    p_bm25.add_argument("--vlb-file", default=str(BENCHMARK_DIR / "vietlawbench_tune_team.jsonl"))
    p_bm25.add_argument("--out-dir", default=r"D:\VLQA\task9d")
    p_bm25.add_argument("--depth", type=int, default=100)

    # Sub-command: slim
    p_slim = subparsers.add_parser("slim", help="Tạo kho văn bản rút gọn")
    p_slim.add_argument("--root", default=r"D:\vietlawbert_data\processed\processed_parquet")
    p_slim.add_argument("--out-dir", default=r"D:\VLQA\task9d\team_corpus_slim")

    args = parser.parse_args(argv)

    if args.command == "dupcheck":
        run_dupcheck(args)
    elif args.command == "export-bm25":
        run_export_bm25(args)
    elif args.command == "slim":
        run_slim(args)


if __name__ == "__main__":
    main()
