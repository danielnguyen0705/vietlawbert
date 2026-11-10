"""
benchmark/map_gold_to_team.py — Ánh xạ nhãn đúng (gold answers) của VLQA và VietLawBench
sang chunk của kho văn bản nhóm (processed parquet corpus).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import random
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.global_id import normalize_doc_number
from configs.logging_config import get_subsystem_logger
from benchmark.inventory_team_corpus import char_ngrams, containment

logger = get_subsystem_logger("benchmark", "map_gold_to_team")

def match_article(article_text: str, candidate_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    So khớp văn bản điều luật với danh sách các chunk ứng viên của cùng văn bản.
    Quy tắc quyết định:
      score_A(chunk) = containment(G(article), G(chunk))  (article nằm trong chunk)
      score_C(chunk) = containment(G(chunk), G(article))   (chunk nằm trong article)
      1) "single": chunk tốt nhất theo max(score_A, score_C) có max >= 0.80 -> gold = chunk đó.
         (Nếu score_A >= 0.80 thì article chứa trong 1 chunk duy nhất).
      2) "split": nếu quy tắc 1 không thỏa (không có chunk đơn nào chứa trọn >= 80% điều),
         lấy tất cả các chunk có score_C >= 0.80; nếu tổng n-gram của chúng bao phủ >= 80%
         n-gram của điều -> gold = các chunk đó.
      3) "duplicate copies": sau 1) hoặc 2), bổ sung mọi chunk từ doc_id KHÁC có cùng số hiệu
         chuẩn hóa mà có điểm so với điều >= 0.95 (bản sao trùng lặp).
      4) "weak": điểm max tốt nhất trong [0.50, 0.80) -> không có gold.
      5) "missing_doc": không có chunk ứng viên (văn bản không tồn tại trong kho nhóm) -> không có gold.
         "no_match": điểm max tốt nhất < 0.50 -> không có gold.
    """
    if not candidate_chunks:
        return {
            "decision": "missing_doc",
            "gold_chunk_ids": [],
            "best_chunk_id": None,
            "score_A": 0.0,
            "score_C": 0.0,
            "best_score": 0.0,
        }

    g_art = char_ngrams(article_text, 5)
    if not g_art:
        return {
            "decision": "no_match",
            "gold_chunk_ids": [],
            "best_chunk_id": candidate_chunks[0].get("chunk_id"),
            "score_A": 0.0,
            "score_C": 0.0,
            "best_score": 0.0,
        }

    scored_chunks = []
    for c in candidate_chunks:
        g_c = c.get("ngrams")
        if g_c is None:
            g_c = char_ngrams(c.get("content", ""), 5)
        s_A = containment(g_art, g_c)
        s_C = containment(g_c, g_art)
        max_s = max(s_A, s_C)
        scored_chunks.append({
            "chunk": c,
            "score_A": s_A,
            "score_C": s_C,
            "max_score": max_s,
            "ngrams": g_c,
        })

    # Sắp xếp điểm giảm dần
    scored_chunks.sort(
        key=lambda x: (x["max_score"], x["score_A"], x["chunk"].get("chunk_id", "")),
        reverse=True
    )
    best_item = scored_chunks[0]
    best_score = best_item["max_score"]
    best_cid = best_item["chunk"]["chunk_id"]

    base_decision: Optional[str] = None
    initial_gold: List[Dict[str, Any]] = []

    # Kiểm tra split: nếu điều bị phân mảnh trên nhiều chunk (mỗi chunk nằm trong điều, score_C >= 0.8)
    split_candidates = [item for item in scored_chunks if item["score_C"] >= 0.80]
    is_split = False
    if len(split_candidates) > 1 and best_item["score_A"] < 0.80:
        g_combined = set().union(*(item["ngrams"] for item in split_candidates))
        coverage = len(g_art & g_combined) / float(len(g_art))
        if coverage >= 0.80:
            is_split = True
            base_decision = "split"
            initial_gold = split_candidates

    if not is_split:
        if best_score >= 0.80:
            base_decision = "single"
            initial_gold = [best_item]

    if not base_decision:
        if 0.50 <= best_score < 0.80:
            decision = "weak"
        else:
            decision = "no_match"
        return {
            "decision": decision,
            "gold_chunk_ids": [],
            "best_chunk_id": best_cid,
            "score_A": round(best_item["score_A"], 4),
            "score_C": round(best_item["score_C"], 4),
            "best_score": round(best_score, 4),
        }

    # Quy tắc 3: Duplicate copies từ doc_id khác
    initial_doc_ids = {item["chunk"]["doc_id"] for item in initial_gold}
    final_gold = list(initial_gold)
    has_duplicate = False

    for item in scored_chunks:
        if item["chunk"]["doc_id"] not in initial_doc_ids:
            if item["max_score"] >= 0.95:
                final_gold.append(item)
                has_duplicate = True

    decision = "duplicate copies" if has_duplicate else base_decision

    seen_cids = set()
    gold_cids = []
    for item in final_gold:
        cid = item["chunk"]["chunk_id"]
        if cid not in seen_cids:
            seen_cids.add(cid)
            gold_cids.append(cid)

    return {
        "decision": decision,
        "gold_chunk_ids": gold_cids,
        "best_chunk_id": best_cid,
        "score_A": round(best_item["score_A"], 4),
        "score_C": round(best_item["score_C"], 4),
        "best_score": round(best_score, 4),
    }

def map_vlb_item(item: Dict[str, Any], candidate_chunks: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Ánh xạ 1 câu hỏi VietLawBench sang chunk ARTICLE của kho nhóm.
    Trạng thái: "mapped" | "doc_not_found" | "article_not_found" | "text_mismatch".
    """
    gt_chunk = item.get("ground_truth_chunk", "")
    gt_doc_num = item.get("ground_truth_doc_number", "")
    gt_art = item.get("ground_truth_article", "")
    ev_text = item.get("evidence_text", "")

    if not candidate_chunks:
        return {
            "benchmark_id": item.get("benchmark_id"),
            "query": item.get("query"),
            "query_type": item.get("query_type"),
            "retrieval_level": "article",
            "relevant_chunk_ids": [],
            "status": "doc_not_found",
            "evidence_containment": 0.0,
        }

    # Trích xuất số điều
    art_num_match = re.search(r"\d+", gt_art)
    art_num = art_num_match.group(0) if art_num_match else ""
    if not art_num and "_art_" in gt_chunk:
        art_num = gt_chunk.split("_art_")[1].split("_")[0]

    expected_cid_suffix = f"_art_{art_num}" if art_num else ""
    art_pattern = re.compile(rf"Điều\s+{art_num}\b", re.IGNORECASE) if art_num else None

    # Lọc các chunk cấp ARTICLE khớp điều
    matched_article_chunks = []
    for ch in candidate_chunks:
        if ch.get("level") == "ARTICLE":
            is_match = False
            cid = ch.get("chunk_id", "")
            hp = ch.get("hierarchy_path", "")
            content_start = ch.get("content", "")[:150]

            if expected_cid_suffix and cid.endswith(expected_cid_suffix):
                is_match = True
            elif art_pattern and art_pattern.search(hp):
                is_match = True
            elif art_pattern and art_pattern.search(content_start):
                is_match = True

            if is_match:
                matched_article_chunks.append(ch)

    if not matched_article_chunks:
        return {
            "benchmark_id": item.get("benchmark_id"),
            "query": item.get("query"),
            "query_type": item.get("query_type"),
            "retrieval_level": "article",
            "relevant_chunk_ids": [],
            "status": "article_not_found",
            "evidence_containment": 0.0,
        }

    # Xác minh nội dung đoạn bằng chứng (evidence_text)
    g_ev = char_ngrams(ev_text, 5)
    best_c = 0.0
    passed_chunk_ids = []

    for ch in matched_article_chunks:
        g_c = ch.get("ngrams")
        if g_c is None:
            g_c = char_ngrams(ch.get("content", ""), 5)
        c_score = containment(g_ev, g_c)
        if c_score > best_c:
            best_c = c_score
        if c_score >= 0.80:
            passed_chunk_ids.append(ch["chunk_id"])

    # Loại bỏ trùng lặp id
    seen_cids = set()
    unique_passed = []
    for cid in passed_chunk_ids:
        if cid not in seen_cids:
            seen_cids.add(cid)
            unique_passed.append(cid)

    if unique_passed:
        status = "mapped"
    else:
        status = "text_mismatch"

    return {
        "benchmark_id": item.get("benchmark_id"),
        "query": item.get("query"),
        "query_type": item.get("query_type"),
        "retrieval_level": "article",
        "relevant_chunk_ids": unique_passed,
        "status": status,
        "evidence_containment": round(best_c, 4),
    }

def update_summary_file(summary_path: Path, section_key: str, section_data: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing[section_key] = section_data
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("✓ Đã cập nhật tóm tắt vào: %s (khóa: %s)", summary_path, section_key)

def run_vlqa(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=== BẮT ĐẦU ÁNH XẠ VLQA SANG CHUNK NHÓM (2.190 CÂU HỎI) ===")

    # 1. Đọc các câu hỏi dev và tune
    dev_path = Path(args.dev_file)
    tune_path = Path(args.tune_file)
    if not dev_path.exists() or not tune_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dev ({dev_path}) hoặc tune ({tune_path})")

    questions = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["orig_split"] = "dev"
                questions.append(item)

    with open(tune_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["orig_split"] = "tune"
                questions.append(item)

    logger.info("Tổng số câu hỏi VLQA: %d (dev: 500, tune: 1690)", len(questions))

    # 2. Đọc legal_corpus.json
    corpus_path = Path(args.vlqa_dir) / "legal_corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy legal_corpus.json tại: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    aid_to_info = {}
    for doc in corpus:
        law_id = str(doc.get("law_id") or doc.get("doc_number") or "").strip()
        norm_law = normalize_doc_number(law_id)
        for art in doc.get("content", []):
            aid = art.get("aid")
            if aid is not None:
                aid_int = int(aid)
                txt = art.get("content_Article") or art.get("content") or ""
                aid_to_info[aid_int] = {
                    "law_id": law_id,
                    "norm_law": norm_law,
                    "text": txt,
                }

    # Thu thập tất cả các aid xuất hiện trong bộ câu hỏi
    target_aids = set()
    for q in questions:
        for a in q.get("relevant_aids", []):
            target_aids.add(int(a))

    target_norm_laws = set()
    for aid in target_aids:
        if aid in aid_to_info:
            target_norm_laws.add(aid_to_info[aid]["norm_law"])

    logger.info("Tổng số gold aid duy nhất cần ánh xạ: %d (thuộc %d văn bản mục tiêu)",
                len(target_aids), len(target_norm_laws))

    # 3. Quét kho parquet lấy candidate chunks cho target_norm_laws
    root_path = Path(args.root)
    parquet_files = sorted(list(root_path.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy file parquet nào dưới: {root_path}")

    logger.info("Quét %d file parquet để thu thập chunk ứng viên...", len(parquet_files))
    norm_law_to_chunks: Dict[str, List[Dict[str, Any]]] = {nl: [] for nl in target_norm_laws}

    for idx, pf in enumerate(parquet_files):
        t = pq.read_table(pf, columns=["chunk_id", "doc_id", "doc_number", "content", "level"])
        for cid, did, dn, content, lvl in zip(
            t["chunk_id"].to_pylist(),
            t["doc_id"].to_pylist(),
            t["doc_number"].to_pylist(),
            t["content"].to_pylist(),
            t["level"].to_pylist()
        ):
            norm = normalize_doc_number(dn)
            if norm in norm_law_to_chunks and content and str(content).strip():
                norm_law_to_chunks[norm].append({
                    "chunk_id": cid,
                    "doc_id": did,
                    "doc_number": dn,
                    "content": content,
                    "level": lvl,
                    "ngrams": char_ngrams(content, 5),
                })

        if (idx + 1) % 40 == 0 or (idx + 1) == len(parquet_files):
            logger.info("  Đã quét %d/%d files...", idx + 1, len(parquet_files))

    total_cands = sum(len(v) for v in norm_law_to_chunks.values())
    logger.info("Đã nạp %d chunk ứng viên cho %d văn bản mục tiêu", total_cands, len(target_norm_laws))

    # 4. Ánh xạ từng aid duy nhất
    logger.info("Tiến hành so khớp %d aid với candidate chunks...", len(target_aids))
    aid_mapping_results: Dict[int, Dict[str, Any]] = {}
    decision_counts = Counter()
    accepted_scores = []

    for aid in sorted(list(target_aids)):
        info = aid_to_info.get(aid)
        if not info:
            res = {
                "decision": "missing_doc",
                "gold_chunk_ids": [],
                "best_chunk_id": None,
                "score_A": 0.0,
                "score_C": 0.0,
                "best_score": 0.0,
            }
        else:
            candidates = norm_law_to_chunks.get(info["norm_law"], [])
            res = match_article(info["text"], candidates)

        aid_mapping_results[aid] = res
        dec = res["decision"]
        decision_counts[dec] += 1
        if dec in {"single", "split", "duplicate copies"}:
            accepted_scores.append(res["best_score"])

    logger.info("Phân bố quyết định của các aid:")
    for dec, count in decision_counts.items():
        logger.info("  - %s: %d (%.2f%%)", dec, count, count / len(target_aids) * 100.0)

    # 5. Ghép nhãn cho 2.190 câu hỏi
    out_q_path = Path(args.out_questions)
    out_q_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts_overall = Counter()
    status_counts_dev = Counter()
    status_counts_tune = Counter()
    questions_with_dups = 0

    with open(out_q_path, "w", encoding="utf-8") as qf:
        for q in questions:
            qid = q["qid"]
            query = q["query"]
            orig_split = q["orig_split"]
            rel_aids = [int(a) for a in q.get("relevant_aids", [])]

            team_cids_union = []
            per_aid_decisions = {}
            has_dup_for_q = False

            for a in rel_aids:
                m_res = aid_mapping_results.get(a, {
                    "decision": "missing_doc",
                    "gold_chunk_ids": [],
                    "best_score": 0.0,
                })
                per_aid_decisions[str(a)] = {
                    "decision": m_res["decision"],
                    "gold_chunks": m_res["gold_chunk_ids"],
                    "best_score": m_res.get("best_score", 0.0),
                }
                for cid in m_res["gold_chunk_ids"]:
                    if cid not in team_cids_union:
                        team_cids_union.append(cid)
                if m_res["decision"] == "duplicate copies":
                    has_dup_for_q = True

            if has_dup_for_q:
                questions_with_dups += 1

            # Xác định trạng thái câu hỏi
            if not rel_aids:
                q_status = "unreachable"
            else:
                all_mapped = all(
                    per_aid_decisions[str(a)]["decision"] in {"single", "split", "duplicate copies"}
                    for a in rel_aids
                )
                any_mapped = any(
                    per_aid_decisions[str(a)]["decision"] in {"single", "split", "duplicate copies"}
                    for a in rel_aids
                )
                if all_mapped:
                    q_status = "full"
                elif any_mapped:
                    q_status = "partial"
                else:
                    q_status = "unreachable"

            status_counts_overall[q_status] += 1
            if orig_split == "dev":
                status_counts_dev[q_status] += 1
            else:
                status_counts_tune[q_status] += 1

            record = {
                "qid": qid,
                "query": query,
                "orig_split": orig_split,
                "vlqa_relevant_aids": rel_aids,
                "relevant_chunk_ids": team_cids_union,
                "per_aid": per_aid_decisions,
                "status": q_status,
            }
            qf.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("✓ Đã lưu %d câu hỏi VLQA kèm chunk nhóm vào: %s", len(questions), out_q_path)

    # 6. Ghi D:\VLQA\task9\vlqa_aid_to_team_chunk.jsonl
    out_aids_path = Path(args.out_aids)
    out_aids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_aids_path, "w", encoding="utf-8") as af:
        for aid in sorted(list(target_aids)):
            m_res = aid_mapping_results[aid]
            info = aid_to_info.get(aid, {})
            af.write(json.dumps({
                "aid": aid,
                "law_id": info.get("law_id", ""),
                "decision": m_res["decision"],
                "gold_chunk_ids": m_res["gold_chunk_ids"],
                "best_score_A": m_res.get("score_A", 0.0),
                "best_score_C": m_res.get("score_C", 0.0),
                "best_max_score": m_res.get("best_score", 0.0),
            }, ensure_ascii=False) + "\n")
    logger.info("✓ Đã lưu danh sách ánh xạ từng aid vào: %s", out_aids_path)

    # 7. Ghi D:\VLQA\task9\mapping_review_sample.csv (50 dòng review: 30 thấp nhất + 20 ngẫu nhiên)
    out_rev_path = Path(args.out_review)
    out_rev_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_aids = [
        aid for aid, res in aid_mapping_results.items()
        if res["decision"] in {"single", "split", "duplicate copies"}
    ]

    # Sắp xếp theo best_score tăng dần
    accepted_aids.sort(key=lambda a: (aid_mapping_results[a]["best_score"], a))
    sample_lowest_30 = accepted_aids[:30]
    remaining_accepted = accepted_aids[30:]

    rng = random.Random(42)
    sample_random_20 = sorted(rng.sample(remaining_accepted, min(20, len(remaining_accepted))))
    final_review_aids = sample_lowest_30 + sample_random_20

    # Lập mapping chunk_id -> content từ candidates
    cid_to_content: Dict[str, str] = {}
    for c_list in norm_law_to_chunks.values():
        for ch in c_list:
            cid_to_content[ch["chunk_id"]] = ch["content"]

    fieldnames = [
        "aid", "law_id", "team_chunk_id", "decision",
        "score_A", "score_C", "vlqa_text_first300", "team_text_first300",
        "reviewer_ok", "reviewer_note"
    ]

    with open(out_rev_path, "w", encoding="utf-8", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=fieldnames)
        writer.writeheader()
        for aid in final_review_aids:
            res = aid_mapping_results[aid]
            info = aid_to_info.get(aid, {})
            t_cid = res["best_chunk_id"] or (res["gold_chunk_ids"][0] if res["gold_chunk_ids"] else "")
            vlqa_txt = re.sub(r"\s+", " ", info.get("text", "")).strip()[:300]
            team_txt = re.sub(r"\s+", " ", cid_to_content.get(t_cid, "")).strip()[:300]

            writer.writerow({
                "aid": aid,
                "law_id": info.get("law_id", ""),
                "team_chunk_id": t_cid,
                "decision": res["decision"],
                "score_A": res.get("score_A", 0.0),
                "score_C": res.get("score_C", 0.0),
                "vlqa_text_first300": vlqa_txt,
                "team_text_first300": team_txt,
                "reviewer_ok": "",
                "reviewer_note": "",
            })

    logger.info("✓ Đã lưu 50 dòng mẫu kiểm tra (review) vào: %s", out_rev_path)

    # 8. Cập nhật summary
    p5_score = float(np.percentile(accepted_scores, 5)) if accepted_scores else 0.0
    p50_score = float(np.percentile(accepted_scores, 50)) if accepted_scores else 0.0
    elapsed = round(time.perf_counter() - t0, 2)

    vlqa_summary = {
        "total_questions": len(questions),
        "distinct_aids": len(target_aids),
        "aids_by_decision": dict(decision_counts),
        "questions_by_status": {
            "overall": dict(status_counts_overall),
            "dev": dict(status_counts_dev),
            "tune": dict(status_counts_tune),
        },
        "questions_with_duplicate_copies": questions_with_dups,
        "accepted_mappings_score_distribution": {
            "P5": round(p5_score, 4),
            "P50": round(p50_score, 4),
        },
        "run_time_seconds": elapsed,
    }

    update_summary_file(Path(args.summary_file), "vlqa", vlqa_summary)
    logger.info("✓ Hoàn thành ánh xạ VLQA trong %.2f giây", elapsed)

def run_vlb(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=== BẮT ĐẦU ÁNH XẠ VIETLAWBENCH SANG CHUNK NHÓM (729 CÂU HỎI) ===")

    vlb_path = Path(args.vlb_file)
    if not vlb_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file VietLawBench tại: {vlb_path}")

    with open(vlb_path, "r", encoding="utf-8") as f:
        vlb_items = [json.loads(line) for line in f if line.strip()]

    logger.info("Đã nạp %d câu hỏi VietLawBench", len(vlb_items))

    # Tìm các target doc_id và target normalized doc_number
    target_doc_ids = set()
    target_norm_nums = set()
    for item in vlb_items:
        gt_chunk = item.get("ground_truth_chunk", "")
        if "_art_" in gt_chunk:
            target_doc_ids.add(gt_chunk.split("_art_")[0])
        dn = item.get("ground_truth_doc_number", "")
        if dn:
            target_norm_nums.add(normalize_doc_number(dn))

    root_path = Path(args.root)
    parquet_files = sorted(list(root_path.rglob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy file parquet nào dưới: {root_path}")

    logger.info("Quét %d file parquet để thu thập chunk của %d doc_id và %d số hiệu...",
                len(parquet_files), len(target_doc_ids), len(target_norm_nums))

    doc_chunks: Dict[str, List[Dict[str, Any]]] = {}
    norm_to_doc_ids: Dict[str, Set[str]] = {}

    for idx, pf in enumerate(parquet_files):
        t = pq.read_table(pf, columns=["chunk_id", "doc_id", "doc_number", "hierarchy_path", "content", "level"])
        for cid, did, dn, hp, content, lvl in zip(
            t["chunk_id"].to_pylist(),
            t["doc_id"].to_pylist(),
            t["doc_number"].to_pylist(),
            t["hierarchy_path"].to_pylist(),
            t["content"].to_pylist(),
            t["level"].to_pylist()
        ):
            norm = normalize_doc_number(dn)
            if did in target_doc_ids or norm in target_norm_nums:
                if did not in doc_chunks:
                    doc_chunks[did] = []
                doc_chunks[did].append({
                    "chunk_id": cid,
                    "doc_id": did,
                    "doc_number": dn,
                    "norm_doc_number": norm,
                    "hierarchy_path": hp or "",
                    "content": content or "",
                    "level": lvl,
                    "ngrams": char_ngrams(content or "", 5),
                })
                if norm:
                    if norm not in norm_to_doc_ids:
                        norm_to_doc_ids[norm] = set()
                    norm_to_doc_ids[norm].add(did)

        if (idx + 1) % 40 == 0 or (idx + 1) == len(parquet_files):
            logger.info("  Đã quét %d/%d files...", idx + 1, len(parquet_files))

    logger.info("Đã nạp chunk cho %d doc_id", len(doc_chunks))

    out_vlb_path = Path(args.out_questions)
    out_vlb_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts = Counter()
    by_query_type: Dict[str, Counter] = {}

    with open(out_vlb_path, "w", encoding="utf-8") as out_f:
        for item in vlb_items:
            gt_chunk = item.get("ground_truth_chunk", "")
            gt_doc_num = item.get("ground_truth_doc_number", "")
            q_type = item.get("query_type", "unknown")

            did = gt_chunk.split("_art_")[0] if "_art_" in gt_chunk else ""
            norm_dn = normalize_doc_number(gt_doc_num)

            candidate_doc_ids = set()
            if did in doc_chunks:
                candidate_doc_ids.add(did)
            if norm_dn and norm_dn in norm_to_doc_ids:
                candidate_doc_ids.update(norm_to_doc_ids[norm_dn])

            all_cands_for_q = []
            for cdid in candidate_doc_ids:
                all_cands_for_q.extend(doc_chunks.get(cdid, []))

            res = map_vlb_item(item, all_cands_for_q if all_cands_for_q else None)
            st = res["status"]
            status_counts[st] += 1

            if q_type not in by_query_type:
                by_query_type[q_type] = Counter()
            by_query_type[q_type][st] += 1

            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")

    logger.info("✓ Đã lưu %d câu hỏi VietLawBench kèm chunk nhóm vào: %s", len(vlb_items), out_vlb_path)
    logger.info("Phân bố trạng thái VietLawBench:")
    for st, cnt in status_counts.items():
        logger.info("  - %s: %d (%.2f%%)", st, cnt, cnt / len(vlb_items) * 100.0)

    elapsed = round(time.perf_counter() - t0, 2)
    vlb_summary = {
        "total_questions": len(vlb_items),
        "questions_by_status": dict(status_counts),
        "questions_by_query_type": {qt: dict(c) for qt, c in by_query_type.items()},
        "run_time_seconds": elapsed,
    }

    update_summary_file(Path(args.summary_file), "vietlawbench", vlb_summary)
    logger.info("✓ Hoàn thành ánh xạ VietLawBench trong %.2f giây", elapsed)

def main(argv: Optional[List[str]] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Ánh xạ gold answers của VLQA và VietLawBench sang chunk nhóm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand vlqa
    p_vlqa = subparsers.add_parser("vlqa", help="Ánh xạ 2.190 câu hỏi VLQA sang chunk nhóm")
    p_vlqa.add_argument("--root", default=r"D:\vietlawbert_data\processed")
    p_vlqa.add_argument("--vlqa-dir", default=os.environ.get("VLQA_DIR", r"D:\VLQA"))
    p_vlqa.add_argument("--dev-file", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    p_vlqa.add_argument("--tune-file", default=str(BENCHMARK_DIR / "vlqa_tune1690.jsonl"))
    p_vlqa.add_argument("--out-questions", default=str(BENCHMARK_DIR / "vlqa_test2190_team.jsonl"))
    p_vlqa.add_argument("--out-aids", default=r"D:\VLQA\task9\vlqa_aid_to_team_chunk.jsonl")
    p_vlqa.add_argument("--out-review", default=r"D:\VLQA\task9\mapping_review_sample.csv")
    p_vlqa.add_argument("--summary-file", default=str(BENCHMARK_DIR / "results" / "team_gold_mapping_summary.json"))

    # Subcommand vlb
    p_vlb = subparsers.add_parser("vlb", help="Ánh xạ 729 câu hỏi VietLawBench sang chunk ARTICLE nhóm")
    p_vlb.add_argument("--root", default=r"D:\vietlawbert_data\processed")
    p_vlb.add_argument("--vlb-file", default=str(ROOT_DIR / "tests" / "_common" / "datasets" / "vietlawbench_clean.jsonl"))
    p_vlb.add_argument("--out-questions", default=str(BENCHMARK_DIR / "vietlawbench_tune_team.jsonl"))
    p_vlb.add_argument("--summary-file", default=str(BENCHMARK_DIR / "results" / "team_gold_mapping_summary.json"))

    args = parser.parse_args(argv)

    if args.command == "vlqa":
        run_vlqa(args)
    elif args.command == "vlb":
        run_vlb(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()