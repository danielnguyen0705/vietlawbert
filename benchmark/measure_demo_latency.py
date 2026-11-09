r"""
benchmark/measure_demo_latency.py — Đo độ trễ chi tiết từng công đoạn trên CPU cho dense và hybrid có trọng số,
so sánh giữa Qdrant exact và HNSW phục vụ Gradio demo.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import requests
import psutil
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from configs.config import config
from configs.paths import ROOT_DIR, BENCHMARK_DIR
from configs.logging_config import get_subsystem_logger

from benchmark.evaluate_vlqa_article import aggregate_pieces
from benchmark.tune_vlqa_fusion import weighted_rrf
from preprocess.vi_segment import get_segmenter, segment_text
from benchmark.measure_cpu_latency import summarize, percentile

logger = get_subsystem_logger("benchmark", "measure_demo_latency")


class StageTimer:
    """
    Bộ đo thời gian từng công đoạn (stage) bằng milliseconds.
    """
    def __init__(self) -> None:
        self._starts: Dict[str, float] = {}
        self.stages: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage not in self._starts:
            raise RuntimeError(f"Giai đoạn '{stage}' chưa được khởi động (chưa gọi start).")
        elapsed_ms = (time.perf_counter() - self._starts.pop(stage)) * 1000.0
        self.stages[stage] = elapsed_ms
        return elapsed_ms

    def total(self) -> float:
        # Tổng thời gian các công đoạn thành phần (bỏ qua 'total' nếu được ghi độc lập)
        sub_stages = [v for k, v in self.stages.items() if k != "total"]
        if sub_stages:
            return sum(sub_stages)
        return sum(self.stages.values())

    def as_dict(self) -> Dict[str, float]:
        return dict(self.stages)


def rotated_order(pipelines: List[Any], i: int) -> List[Any]:
    """
    Xoay vòng danh sách pipeline theo chỉ số i (câu i bắt đầu với pipeline i % len).
    """
    if not pipelines:
        return []
    idx = i % len(pipelines)
    return pipelines[idx:] + pipelines[:idx]


def topk_overlap(a: List[str], b: List[str], k: int = 10) -> float:
    """
    Tính tỷ lệ trùng lặp top-k giữa 2 danh sách kết quả xếp hạng.
    """
    if k <= 0:
        return 0.0
    set_a = set(a[:k])
    set_b = set(b[:k])
    return len(set_a & set_b) / float(k)


def load_dev_questions(path: Path | str) -> List[Dict[str, Any]]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def main(argv: List[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t_script_start = time.perf_counter()

    ap = argparse.ArgumentParser(description="Đo độ trễ chi tiết từng công đoạn cho demo trên CPU")
    ap.add_argument("--dev-file", default=str(BENCHMARK_DIR / "vlqa_dev500.jsonl"))
    ap.add_argument("--perquery-file", default=str(BENCHMARK_DIR / "results" / "vlqa_dev500_perquery.jsonl"))
    ap.add_argument("--fusion-params", default=str(BENCHMARK_DIR / "results" / "vlqa_fusion_params.json"))
    ap.add_argument("--out-csv", default=str(BENCHMARK_DIR / "results" / "demo_latency.csv"))
    ap.add_argument("--out-env", default=str(BENCHMARK_DIR / "results" / "demo_env.json"))
    ap.add_argument("--n-questions", type=int, default=100)
    args = ap.parse_args(argv)

    # -------------------------------------------------------------------------
    # Bước 1: Kiểm tra phần cứng & môi trường (Step 1)
    # -------------------------------------------------------------------------
    logger.info("=== KIỂM TRA MÔI TRƯỜNG VÀ PHẦN CỨNG ===")

    # Kiểm tra nguồn điện AC
    try:
        res = subprocess.run(
            ["powershell", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
            capture_output=True,
            text=True,
            check=True,
        )
        bat_status_int = int(res.stdout.strip())
    except Exception as e:
        logger.warning("Không thể đọc BatteryStatus: %s; giả định đang cắm sạc", e)
        bat_status_int = 2

    logger.info("Trạng thái pin (BatteryStatus): %d (2 = on AC)", bat_status_int)
    if bat_status_int != 2:
        logger.error("DỪNG: Máy đang chạy bằng PIN (BatteryStatus = %d). Vui lòng cắm sạc AC!", bat_status_int)
        sys.exit(1)

    # Power plan
    try:
        res = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, check=True)
        active_plan_raw = res.stdout.strip()
    except Exception:
        active_plan_raw = "unknown"
    logger.info("Chế độ nguồn điện hoạt động: %s", active_plan_raw)

    # RAM trước khi nạp model
    ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    ram_avail_before_gb = round(psutil.virtual_memory().available / (1024**3), 2)
    logger.info("RAM hệ thống: Tổng=%.2f GB, Khả dụng trước khi nạp=%.2f GB", ram_total_gb, ram_avail_before_gb)

    # Docker stats
    try:
        res = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}} {{.CPUPerc}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        docker_stats_raw = res.stdout.strip()
    except Exception as e:
        docker_stats_raw = f"error: {e}"
    logger.info("Docker stats:\n%s", docker_stats_raw)

    # wslconfig
    user_profile = os.environ.get("USERPROFILE", "")
    wsl_path = Path(user_profile) / ".wslconfig" if user_profile else None
    if wsl_path and wsl_path.exists():
        wsl_content = wsl_path.read_text(encoding="utf-8").strip()
    else:
        wsl_content = "absent"
    logger.info("Nội dung .wslconfig: %s", wsl_content)

    # Qdrant collection info
    qdrant_url = f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}"
    try:
        r = requests.get(f"{qdrant_url}/collections/vlqa_dense_bge_m3", timeout=10.0)
        q_data = r.json().get("result", {})
        q_config = q_data.get("config", {}).get("params", {}).get("vectors", {})
        q_info = {
            "name": "vlqa_dense_bge_m3",
            "vectors_on_disk": q_config.get("on_disk"),
            "points_count": q_data.get("points_count"),
            "indexed_vectors_count": q_data.get("indexed_vectors_count"),
            "status": q_data.get("status"),
        }
    except Exception as e:
        logger.error("Không thể kết nối Qdrant: %s", e)
        sys.exit(1)
    logger.info("Thông tin collection Qdrant: %s", q_info)

    torch_threads = torch.get_num_threads()
    logger.info("Số luồng CPU của PyTorch: %d", torch_threads)

    # -------------------------------------------------------------------------
    # Bước 2: Nạp mô hình & chuẩn bị dịch vụ
    # -------------------------------------------------------------------------
    logger.info("Đang nạp mô hình BAAI/bge-m3 lên CPU (max_seq_length=128)...")
    st_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    st_model.max_seq_length = 128

    ram_avail_after_gb = round(psutil.virtual_memory().available / (1024**3), 2)
    logger.info("RAM khả dụng sau khi nạp bge-m3: %.2f GB", ram_avail_after_gb)

    # Nạp pyvi segmenter
    seg_fn = get_segmenter("pyvi")

    # Nạp tham số fusion
    params_path = Path(args.fusion_params)
    if not params_path.exists():
        logger.error("Không tìm thấy file tham số fusion: %s", params_path)
        sys.exit(1)
    with open(params_path, "r", encoding="utf-8") as f:
        fusion_params = json.load(f)
    w_dense = float(fusion_params.get("w_dense", 0.7))
    w_bm25 = float(fusion_params.get("w_bm25", 0.3))
    rrf_k = int(fusion_params.get("rrf_k", 10))
    logger.info("Tham số Weighted RRF: w_dense=%.2f, w_bm25=%.2f, k=%d", w_dense, w_bm25, rrf_k)

    # Nạp tập dev questions
    dev_questions = load_dev_questions(args.dev_file)
    logger.info("Đã nạp %d câu hỏi dev từ %s", len(dev_questions), args.dev_file)

    # Khởi tạo HTTP session cho Qdrant và client cho ES
    session = requests.Session()
    es_client = Elasticsearch([config.ES_HOST])

    # -------------------------------------------------------------------------
    # Định nghĩa các hàm thực thi pipeline
    # -------------------------------------------------------------------------
    def run_dense(q_text: str, exact: bool) -> Tuple[List[str], StageTimer]:
        timer = StageTimer()
        t0 = time.perf_counter()

        timer.start("encode")
        vec = st_model.encode(q_text, normalize_embeddings=True, show_progress_bar=False)
        timer.stop("encode")

        timer.start("qdrant")
        res = session.post(
            f"{qdrant_url}/collections/vlqa_dense_bge_m3/points/search",
            json={
                "vector": vec.tolist(),
                "limit": 400,
                "params": {"exact": exact},
                "with_payload": ["chunk_id"],
            },
        )
        hits = [(pt.get("payload", {}).get("chunk_id"), pt.get("score", 0.0)) for pt in res.json().get("result", [])]
        timer.stop("qdrant")

        timer.start("aggregate")
        ranked = aggregate_pieces(hits)[:10]
        timer.stop("aggregate")

        wall_total_ms = (time.perf_counter() - t0) * 1000.0
        timer.stages["total"] = wall_total_ms
        return ranked, timer

    def run_wrrf(q_text: str, exact: bool) -> Tuple[List[str], StageTimer]:
        timer = StageTimer()
        t0 = time.perf_counter()

        timer.start("encode")
        vec = st_model.encode(q_text, normalize_embeddings=True, show_progress_bar=False)
        timer.stop("encode")

        timer.start("qdrant")
        res = session.post(
            f"{qdrant_url}/collections/vlqa_dense_bge_m3/points/search",
            json={
                "vector": vec.tolist(),
                "limit": 400,
                "params": {"exact": exact},
                "with_payload": ["chunk_id"],
            },
        )
        hits = [(pt.get("payload", {}).get("chunk_id"), pt.get("score", 0.0)) for pt in res.json().get("result", [])]
        timer.stop("qdrant")

        timer.start("aggregate")
        dense_ranked = aggregate_pieces(hits)
        timer.stop("aggregate")

        timer.start("segment")
        seg_query = segment_text(q_text, seg_fn)
        timer.stop("segment")

        timer.start("es")
        es_res = es_client.search(
            index="vlqa_sparse_seg_idx",
            query={"match": {"content_seg": seg_query}},
            size=100,
            _source=["chunk_id"],
        )
        bm25_hits = [h["_source"]["chunk_id"] for h in es_res["hits"]["hits"]]
        timer.stop("es")

        timer.start("fuse")
        fused = weighted_rrf([bm25_hits, dense_ranked[:100]], [w_bm25, w_dense], k=rrf_k, depth=100)
        ranked = fused[:10]
        timer.stop("fuse")

        wall_total_ms = (time.perf_counter() - t0) * 1000.0
        timer.stages["total"] = wall_total_ms
        return ranked, timer

    def execute_pipeline_by_name(p_name: str, q_text: str) -> Tuple[List[str], StageTimer]:
        if p_name == "dense_exact":
            return run_dense(q_text, exact=True)
        elif p_name == "dense_hnsw":
            return run_dense(q_text, exact=False)
        elif p_name == "wrrf_exact":
            return run_wrrf(q_text, exact=True)
        elif p_name == "wrrf_hnsw":
            return run_wrrf(q_text, exact=False)
        else:
            raise ValueError(f"Tên pipeline không hợp lệ: {p_name}")

    # -------------------------------------------------------------------------
    # Bước 3: Cold start & Warm-up
    # -------------------------------------------------------------------------
    logger.info("--- Đo Cold Start trên câu đầu tiên với wrrf_exact ---")
    _, cold_timer = execute_pipeline_by_name("wrrf_exact", dev_questions[0]["query"])
    cold_first_query_ms = round(cold_timer.stages["total"], 2)
    logger.info("✓ Cold start đầu tiên (wrrf_exact): %.2f ms", cold_first_query_ms)

    logger.info("--- Khởi động (Warm-up) 5 câu qua toàn bộ 4 pipelines ---")
    for w_i in range(5):
        w_text = dev_questions[w_i]["query"]
        for p_name in ["dense_exact", "dense_hnsw", "wrrf_exact", "wrrf_hnsw"]:
            execute_pipeline_by_name(p_name, w_text)
    logger.info("✓ Hoàn thành warm-up")

    # -------------------------------------------------------------------------
    # Bước 4: Đo đạc 100 câu dev đầu tiên
    # -------------------------------------------------------------------------
    n_measure = min(args.n_questions, len(dev_questions))
    logger.info("--- Bắt đầu đo %d câu dev đầu tiên (xoay vòng pipeline mỗi câu) ---", n_measure)

    all_pipelines = ["dense_exact", "dense_hnsw", "wrrf_exact", "wrrf_hnsw"]
    stage_timings: Dict[str, Dict[str, List[float]]] = {
        p: {s: [] for s in (["encode", "qdrant", "aggregate", "total"] if "dense" in p else ["encode", "qdrant", "aggregate", "segment", "es", "fuse", "total"])}
        for p in all_pipelines
    }
    results_top10: Dict[str, List[List[str]]] = {p: [] for p in all_pipelines}

    for idx in range(n_measure):
        q_text = dev_questions[idx]["query"]
        current_pipes = rotated_order(all_pipelines, idx)
        for p_name in current_pipes:
            top10, timer = execute_pipeline_by_name(p_name, q_text)
            results_top10[p_name].append(top10)
            for stage_name, ms_val in timer.as_dict().items():
                if stage_name in stage_timings[p_name]:
                    stage_timings[p_name][stage_name].append(ms_val)

        if (idx + 1) % 20 == 0 or (idx + 1) == n_measure:
            logger.info("  Đã xử lý %d/%d câu hỏi...", idx + 1, n_measure)

    # -------------------------------------------------------------------------
    # Bước 5: Kiểm tra độ chính xác (Correctness checks)
    # -------------------------------------------------------------------------
    logger.info("=== KIỂM TRA ĐỘ CHÍNH XÁC (CORRECTNESS CHECKS) ===")

    # 1. Overlap checks
    dense_overlaps = [
        topk_overlap(results_top10["dense_hnsw"][i], results_top10["dense_exact"][i], k=10)
        for i in range(n_measure)
    ]
    overlap_dense_mean = round(sum(dense_overlaps) / len(dense_overlaps), 4)
    overlap_dense_min = round(min(dense_overlaps), 4)

    wrrf_overlaps = [
        topk_overlap(results_top10["wrrf_hnsw"][i], results_top10["wrrf_exact"][i], k=10)
        for i in range(n_measure)
    ]
    overlap_wrrf_mean = round(sum(wrrf_overlaps) / len(wrrf_overlaps), 4)
    overlap_wrrf_min = round(min(wrrf_overlaps), 4)

    logger.info("Top-10 Overlap dense_hnsw vs dense_exact: mean=%.4f, min=%.4f", overlap_dense_mean, overlap_dense_min)
    logger.info("Top-10 Overlap wrrf_hnsw vs wrrf_exact: mean=%.4f, min=%.4f", overlap_wrrf_mean, overlap_wrrf_min)

    # 2. Recall@10 checks vs perquery.jsonl
    first_100_qids = set(dev_questions[i]["qid"] for i in range(n_measure))
    perquery_dense_recs = []
    perquery_wrrf_recs = []

    perquery_path = Path(args.perquery_file)
    if perquery_path.exists():
        with open(perquery_path, "r", encoding="utf-8") as pf:
            for line in pf:
                if line.strip():
                    item = json.loads(line)
                    if item.get("qid") in first_100_qids:
                        g = set(item.get("gold_ids", []))
                        rec = len(set(item.get("top_10_ids", [])) & g) / len(g) if g else 0.0
                        if item.get("mode") == "dense":
                            perquery_dense_recs.append(rec)
                        elif item.get("mode") == "hybrid_wrrf":
                            perquery_wrrf_recs.append(rec)

    perquery_dense_rec10 = round(sum(perquery_dense_recs) / len(perquery_dense_recs), 4) if perquery_dense_recs else 0.0
    perquery_wrrf_rec10 = round(sum(perquery_wrrf_recs) / len(perquery_wrrf_recs), 4) if perquery_wrrf_recs else 0.0

    # Recall@10 đo đạc thực tế
    actual_dense_recs = []
    actual_wrrf_recs = []
    for i in range(n_measure):
        g = set(dev_questions[i].get("relevant_chunk_ids", []))
        r_dense = len(set(results_top10["dense_exact"][i][:10]) & g) / len(g) if g else 0.0
        r_wrrf = len(set(results_top10["wrrf_exact"][i][:10]) & g) / len(g) if g else 0.0
        actual_dense_recs.append(r_dense)
        actual_wrrf_recs.append(r_wrrf)

    dense_exact_recall10 = round(sum(actual_dense_recs) / len(actual_dense_recs), 4)
    wrrf_exact_recall10 = round(sum(actual_wrrf_recs) / len(actual_wrrf_recs), 4)

    diff_dense = round(abs(dense_exact_recall10 - perquery_dense_rec10), 4)
    diff_wrrf = round(abs(wrrf_exact_recall10 - perquery_wrrf_rec10), 4)

    logger.info("Recall@10 dense_exact: thực tế=%.4f, perquery=%.4f (lệch=%.4f)",
                dense_exact_recall10, perquery_dense_rec10, diff_dense)
    logger.info("Recall@10 wrrf_exact:  thực tế=%.4f, perquery=%.4f (lệch=%.4f)",
                wrrf_exact_recall10, perquery_wrrf_rec10, diff_wrrf)

    if diff_dense > 0.02:
        logger.error("DỪNG: Chênh lệch Recall@10 dense_exact (%.4f) vượt ngưỡng 0.02!", diff_dense)
        sys.exit(1)
    if diff_wrrf > 0.02:
        logger.error("DỪNG: Chênh lệch Recall@10 wrrf_exact (%.4f) vượt ngưỡng 0.02!", diff_wrrf)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Bước 6: Lưu kết quả CSV và JSON
    # -------------------------------------------------------------------------
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    stage_orders = {
        "dense_exact": ["encode", "qdrant", "aggregate", "total"],
        "dense_hnsw": ["encode", "qdrant", "aggregate", "total"],
        "wrrf_exact": ["encode", "qdrant", "aggregate", "segment", "es", "fuse", "total"],
        "wrrf_hnsw": ["encode", "qdrant", "aggregate", "segment", "es", "fuse", "total"],
    }

    for p_name in all_pipelines:
        for s_name in stage_orders[p_name]:
            vals = stage_timings[p_name][s_name]
            mean_v = sum(vals) / len(vals)
            p50_v = percentile(vals, 50)
            p95_v = percentile(vals, 95)
            share_val = (sum(1 for v in vals if v < 500.0) / len(vals)) if s_name == "total" else None

            csv_rows.append({
                "pipeline": p_name,
                "stage": s_name,
                "n_questions": n_measure,
                "mean_ms": f"{mean_v:.2f}",
                "P50_ms": f"{p50_v:.2f}",
                "P95_ms": f"{p95_v:.2f}",
                "share_under_500ms": f"{share_val:.4f}" if share_val is not None else "",
            })

    headers = ["pipeline", "stage", "n_questions", "mean_ms", "P50_ms", "P95_ms", "share_under_500ms"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("✓ Đã lưu bảng kết quả chi tiết vào %s", out_csv)

    # Lưu demo_env.json
    out_env = Path(args.out_env)
    out_env.parent.mkdir(parents=True, exist_ok=True)

    env_payload = {
        "power": {
            "battery_status": bat_status_int,
            "on_ac": (bat_status_int == 2),
        },
        "active_power_plan": active_plan_raw,
        "ram": {
            "total_gb": ram_total_gb,
            "available_before_load_gb": ram_avail_before_gb,
            "available_after_load_gb": ram_avail_after_gb,
        },
        "docker_stats": {
            "raw": docker_stats_raw,
        },
        "wslconfig": wsl_content,
        "qdrant_collection": q_info,
        "torch_num_threads": torch_threads,
        "checks": {
            "cold_first_query_ms": cold_first_query_ms,
            "overlap_dense_hnsw_vs_exact_mean": overlap_dense_mean,
            "overlap_dense_hnsw_vs_exact_min": overlap_dense_min,
            "overlap_wrrf_hnsw_vs_exact_mean": overlap_wrrf_mean,
            "overlap_wrrf_hnsw_vs_exact_min": overlap_wrrf_min,
            "dense_exact_recall10": dense_exact_recall10,
            "perquery_dense_recall10": perquery_dense_rec10,
            "dense_recall10_diff": diff_dense,
            "wrrf_exact_recall10": wrrf_exact_recall10,
            "perquery_wrrf_recall10": perquery_wrrf_rec10,
            "wrrf_recall10_diff": diff_wrrf,
        },
    }

    with open(out_env, "w", encoding="utf-8") as ef:
        json.dump(env_payload, ef, indent=2, ensure_ascii=False)
        ef.write("\n")
    logger.info("✓ Đã lưu thông tin môi trường và kiểm tra vào %s", out_env)

    # Đóng kết nối
    session.close()
    es_client.close()

    t_script_end = time.perf_counter()
    logger.info("✓ Toàn bộ quá trình hoàn thành trong %.2f giây", t_script_end - t_script_start)

    # In kết quả console
    print("\n" + "=" * 90)
    print("BẢNG KẾT QUẢ ĐỘ TRỄ CHI TIẾT TỪNG CÔNG ĐOẠN (demo_latency.csv):")
    print("=" * 90)
    with open(out_csv, "r", encoding="utf-8") as f:
        print(f.read().strip())
    print("=" * 90)

    print("\nTỔNG KẾT ĐỘ TRỄ TOÀN BỘ (TOTAL) TỪNG PIPELINE TRÊN CPU:")
    for p_name in all_pipelines:
        total_vals = stage_timings[p_name]["total"]
        p50 = percentile(total_vals, 50)
        p95 = percentile(total_vals, 95)
        share = (sum(1 for v in total_vals if v < 500.0) / len(total_vals)) * 100.0
        print(f"  {p_name:12s}: Total P50 = {p50:6.2f} ms | P95 = {p95:6.2f} ms | Share < 500ms = {share:5.1f}%")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()