#!/usr/bin/env bash
# =============================================================================
# run_p0_to_p4.sh — Chạy MỘT LƯỢT P0→P4 cho VietLawBERT (lộ lỗi end-to-end).
#
# Hai tầng tài nguyên theo Đề cương v3.1 §IV:
#   STAGE A (cpu)   — validate LOGIC trên laptop Ubuntu, KHÔNG embed toàn kho.
#   STAGE B (cloud) — chạy DỮ LIỆU→SỐ một mạch trên A100 (crawl→ingest→graph→eval).
#
# Căn cứ tách tầng: đã đo thực nghiệm embedding bge-m3 ~578s/32 chunk ở max_seq=8192
# trên CPU => nhúng toàn bộ 3,4 triệu chunk là việc của GPU, không ép lên laptop.
#
# Dùng:  bash run_p0_to_p4.sh cpu            # chạy ngay, kiểm chứng logic
#        bash run_p0_to_p4.sh cloud cuda     # khi đã thuê A100
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
log(){ printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }

STAGE="${1:-cpu}"

if [ "$STAGE" = "cpu" ]; then
  log "P0 — GATE ĐƠN VỊ: global_id (khóa canonical) + AST per-doctype  [không cần service]"
  python -m pytest tests/system/test/global_id/test_global_id.py \
                   tests/system/test/parser/test_ast_per_doctype.py -q

  log "P1 — DỰNG LẠI GRAPH SẠCH bằng khóa canonical + parser/doc_type MỚI  [cần Neo4j]"
  echo "     (WIPE Neo4j + xóa checkpoint; mật khẩu lấy từ .env — KHÔNG hardcode)"
  python -m database.build_hin_graph --rebuild

  log "P2 — GATE TOÀN VẸN KHÓA: không trùng node (ratio<=1.05), canonical 100%, ES⊆Neo4j"
  echo "     (nhánh cross-store ES⊆Neo4j sẽ ĐỎ tới khi STAGE B tái-embed — bình thường ở CPU)"
  python -m pytest tests/system/test/global_id/test_id_consistency.py -q \
    || echo "     [!] tier2 chưa xanh hết — cross-store cần embed ở STAGE B; xem log phần Neo4j."

  log "P3 — NẠP VLQA + ĐO ĐỘ PHỦ  [gate khoa học TRƯỚC khi tin số P4]"
  python -m tests._common.datasets.load_vlqa
  echo "     -> Độ phủ ES thấp = law_id VLQA chưa khớp doc_number của mình => sửa map TRƯỚC khi đo P4."

  log "STAGE A (CPU) XONG — logic đã kiểm chứng. Thuê A100 rồi chạy: bash run_p0_to_p4.sh cloud cuda"
  exit 0
fi

if [ "$STAGE" = "cloud" ]; then
  DEV="${2:-cuda}"
  log "P0→P1 — DỰNG DỮ LIỆU MỘT MẠCH: crawl→ingest(fresh)→graph(rebuild)  [$DEV]"
  python -m main --phase build-data --device "$DEV"

  log "P2 — GATE TOÀN VẸN KHÓA (giờ cross-store PHẢI xanh sau khi đã embed)"
  python -m pytest tests/system/test/global_id/test_id_consistency.py -q

  log "P3 — NẠP VLQA + ĐỘ PHỦ (sau khi index đã đầy)"
  python -m tests._common.datasets.load_vlqa

  log "P4 — HỆ SỐ RA SỐ: VLQA ngoại (Hit@1/5/10, MRR@10, NDCG@10, MAP@10, P95 + t-test) & RQ1–RQ4"
  python -m benchmark.evaluate_rqs --rq vlqa --device "$DEV"
  python -m benchmark.evaluate_rqs --rq all  --device "$DEV"

  if ls evaluation/configs/*.yaml >/dev/null 2>&1; then
    log "TẦNG 4/5 — PROMPTFOO GATE (pass/fail + SLA + trích dẫn)"
    for c in evaluation/configs/*.yaml; do echo "     npx promptfoo eval -c $c"; done
  fi

  log "STAGE B (CLOUD) XONG — bảng số: benchmark/results/ (p4_vlqa_external_numbers.csv, ...)"
  exit 0
fi

echo "Dùng: bash run_p0_to_p4.sh [cpu|cloud] [cuda|cpu]"; exit 1
