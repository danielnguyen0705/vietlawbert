#!/usr/bin/env bash
# =============================================================================
# run_p0_to_p4.sh — PRE-FLIGHT CHỈ-CPU (chạy trên laptop TRƯỚC khi thuê cloud).
# Kiểm chứng logic nền dữ liệu mà KHÔNG embed toàn kho (578s/32 chunk ở CPU).
# Pha cloud (crawl→ingest→graph→train→eval) nay nằm TRỌN trong run_all_experiments.sh
# (gọi tự động qua auto_cloud_runner.sh). KHÔNG còn hai script chồng vai trò.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
log(){ printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }

if [ "${1:-cpu}" = "cloud" ]; then
  echo "Pha cloud đã hợp nhất vào run_all_experiments.sh."
  echo "Chạy:  bash auto_cloud_runner.sh [IP] [PORT] [USER] [PASS]   (tự SSH + chạy run_all_experiments)"
  echo "Hoặc trực tiếp trên máy cloud:  BUILD_DATA=1 ./run_all_experiments.sh cuda"
  exit 0
fi

log "P0 — GATE ĐƠN VỊ: global_id + AST per-doctype (không cần service)"
python -m pytest tests/system/test/global_id/test_global_id.py \
                 tests/system/test/global_id/test_doc_number_sanitize.py \
                 tests/system/test/parser/test_ast_per_doctype.py \
                 tests/system/test/parser/test_flat_resplit.py -q

log "P1 — DỰNG LẠI GRAPH SẠCH bằng khóa canonical + parser/doc_type mới (cần Neo4j)"
python -m database.build_hin_graph --rebuild

log "P2 — GATE TOÀN VẸN KHÓA: no-dup, canonical, ES⊆Neo4j (cross-store ĐỎ tới khi embed cloud)"
python -m pytest tests/system/test/global_id/test_id_consistency.py -q \
  || echo "  [!] cross-store cần embed ở cloud — bình thường ở CPU."

log "P3 — NẠP VLQA + ĐỘ PHỦ (gate khoa học trước khi tin số)"
python -m tests._common.datasets.load_vlqa

log "P4 — SMOKE eval VLQA trên subset đã embed (lộ bug plumbing; số thấp OK)"
python -m benchmark.evaluate_rqs --rq vlqa --device cpu || echo "  [!] cần Qdrant/ES có dữ liệu để ra số thật."

log "PRE-FLIGHT CPU XONG. Thuê A100 rồi chạy: bash auto_cloud_runner.sh [IP] [PORT] [USER] [PASS]"
