#!/usr/bin/env bash
# ==============================================================================
# VIETLAWBERT: MASTER EXPERIMENT AUTOMATION HARNESS (7 BACKBONES SẠCH)
# Vị trí: scripts/run_all_experiments.sh (tự chuyển về gốc repo khi chạy)
# Tự động hóa: Kiểm tra dữ liệu -> Sinh Triplet/Benchmark -> Huấn luyện MRL
#              -> Đo RQ2 Pareto -> Đối chuẩn Master -> Đo RQ1, RQ3, RQ4
# ==============================================================================

set -e
cd "$(dirname "$0")/.."

REQUESTED_DEVICE="${1:-cuda}"

if [ "$REQUESTED_DEVICE" = "cuda" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[CẢNH BÁO] Không tìm thấy card NVIDIA. Tự động chuyển sang thiết bị: cpu"
        DEVICE="cpu"
    else
        DEVICE="cuda"
        # BỔ SUNG: Chỉ định duy nhất GPU 0 cho VietLawBERT, để GPU 1 cho tác vụ khác
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    fi
else
    DEVICE="cpu"
fi

# Batch train theo VRAM: GPU <20GB (vd RTX 5080 16GB) -> 16 tránh OOM bge-m3; else 32. Override: TRAIN_BATCH=...
if [ -z "${TRAIN_BATCH:-}" ]; then
    _VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
    if [ -n "$_VRAM" ] && [ "$_VRAM" -lt 20000 ]; then TRAIN_BATCH=16; else TRAIN_BATCH=32; fi
fi
echo "    [cfg] VRAM=${_VRAM:-?}MB -> TRAIN_BATCH=$TRAIN_BATCH"

# BỔ SUNG: Giới hạn CPU luồng để không chiếm dụng hết 64 nhân EPYC
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# 1. Cấu hình biến môi trường và thư viện
export DATA_STORAGE_ROOT="${DATA_STORAGE_ROOT:-${HOME}/vietlawbert/data}"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"

# Nạp thư viện động cuDNN và NCCL cho PyTorch
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, nvidia.nccl, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib:' + os.path.dirname(nvidia.nccl.__file__) + '/lib')" 2>/dev/null):$LD_LIBRARY_PATH

PARQUET_FILE="outputs/artifacts/triplets/hin_triplets.parquet"
BENCHMARK_DIR="outputs/artifacts/benchmark"
RESULTS_DIR="outputs/benchmark/results"
CHECKPOINTS_DIR="outputs/checkpoints"
LOG_DIR="outputs/logs/experiments"
FINAL_MODEL_DIR="outputs/models/vietlawbert_mrl_final"

mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR" "outputs/models" "${DATA_STORAGE_ROOT}/logs" "outputs/artifacts/triplets" "$BENCHMARK_DIR"

echo "=================================================================="
echo "BẮT ĐẦU CHU KỲ THỰC NGHIỆM VIETLAWBERT ĐẠT CHUẨN BÀI BÁO Q1"
echo "Thiết bị thực thi : $DEVICE"
echo "Thời gian bắt đầu : $(date)"
echo "=================================================================="

# ==============================================================================
# PHA DỮ LIỆU (hợp nhất từ run_p0_to_p4.sh) — mặc định BẬT cho máy MỚI.
#   crawl → ingest(embed Qdrant/ES) → graph(canonical) → gate khóa → re-mine triplet SẠCH.
#   BUILD_DATA=0 để BỎ QUA (resume khi graph/embed đã có).
#   YÊU CẦU: Neo4j/Qdrant/ES đang chạy (vd: docker compose up -d).
# ==============================================================================
BUILD_DATA="${BUILD_DATA:-1}"
if [ "$BUILD_DATA" = "1" ]; then
    echo ">>> [PHA DỮ LIỆU 0/4] Bật hạ tầng lưu trữ (Neo4j/Qdrant/ES) qua docker compose..."
    if command -v docker &>/dev/null; then
        (docker compose up -d 2>/dev/null || docker-compose up -d 2>/dev/null) \
            || echo "  [!] docker compose không chạy được — đảm bảo Neo4j/Qdrant/ES đã bật thủ công."
        echo "    (chờ 15s cho services sẵn sàng...)"; sleep 15
    else
        echo "  [!] Không có docker — đảm bảo Neo4j/Qdrant/ES đã chạy sẵn trước khi tiếp."
    fi

    echo ">>> [PHA DỮ LIỆU 1/4] crawl → ingest → graph (canonical, từ đầu)..."
    python3 -u -m main --phase build-data --device "$DEVICE"

    echo ">>> [PHA DỮ LIỆU 2/4] Gate toàn vẹn khóa (no-dup + canonical + cross-store)..."
    if ! python3 -m pytest tests/system/test/global_id/test_id_consistency.py -q; then
        echo "✗ Gate khóa ĐỎ — DỪNG trước khi train để tránh phí GPU."
        exit 1
    fi

    echo ">>> [PHA DỮ LIỆU 2b/4] Cổng ngữ liệu & Bất biến Lớp 1 (p1_gate_corpus, GATE_SOFT=${GATE_SOFT:-1})..."
    if ! GATE_SOFT="${GATE_SOFT:-1}" python3 -u -m quality.corpus_audit.corpus_gate; then
        echo "✗ Cổng p1_gate_corpus ĐỎ (GATE_SOFT=0) — DỪNG trước khi train để tránh phí GPU."
        exit 1
    fi

    echo ">>> [PHA DỮ LIỆU 3/4] Nạp VLQA + đo độ phủ..."
    python3 -u -m tests._common.datasets.load_vlqa || true

    echo ">>> [PHA DỮ LIỆU 4/4] Re-mine triplet SẠCH trên graph mới (xoá bản cũ stale)..."
    rm -f "$PARQUET_FILE"
else
    echo ">>> [PHA DỮ LIỆU] BỎ QUA (BUILD_DATA=0) — dùng graph/triplet sẵn có."
fi

# ==============================================================================
# BƯỚC 0: KIỂM TRA VÀ SINH DỮ LIỆU ĐẦU VÀO NẾU CHƯA CÓ
# ==============================================================================
if [ ! -f "$PARQUET_FILE" ]; then
    echo ">>> [BƯỚC 0.1] Chưa phát hiện file Triplet. Tiến hành khai phá HIN Triplets..."
    python3 -u -m training.generate_hin_triplets \
        --output "$PARQUET_FILE" \
        --limit 50000 \
        --beta 0.5
    echo "✓ Đã sinh thành công tập huấn luyện: $PARQUET_FILE"
else
    echo "✓ Đã có sẵn tập huấn luyện Triplet: $PARQUET_FILE"
fi

if [ ! -f "${BENCHMARK_DIR}/vietlawbench_vlqa.jsonl" ]; then
    echo ">>> [BƯỚC 0.2] Chưa phát hiện VLQA. Nạp VLQA làm benchmark đánh giá..."
    python3 -u -m tests._common.datasets.load_vlqa
    echo "✓ Đã nạp VLQA tại: $BENCHMARK_DIR"
else
    echo "✓ Đã có sẵn tập kiểm chuẩn VLQA tại: $BENCHMARK_DIR"
fi

# BƯỚC 0.3: Sinh subset MULTI-HOP (TEMPORAL + REFERENCE) qua API — đo multi-hop/completeness mà VLQA không gánh.
# Cần EVAL_API_KEY (hhtechapi). Cổng 0 đối chiếu triplet đã re-mine → chống rò rỉ với tập train.
if [ -z "$EVAL_API_KEY" ]; then
    echo "[!] [BƯỚC 0.3] Bỏ qua sinh multi-hop — chưa đặt EVAL_API_KEY."
else
    for B in TEMPORAL REFERENCE; do
        MH_OUT="${BENCHMARK_DIR}/multihop_$(echo "$B" | tr 'A-Z' 'a-z').jsonl"
        if [ -f "$MH_OUT" ]; then
            echo "✓ Đã có multi-hop $B tại: $MH_OUT"
        else
            echo ">>> [BƯỚC 0.3] Sinh multi-hop $B (target 200)..."
            python3 -u -m benchmark.build_multihop_subset --bucket "$B" --target 200 \
                --out "$MH_OUT" --triplets "$PARQUET_FILE" || echo "  [!] Sinh multi-hop $B lỗi (xem log)."
        fi
    done
fi

# Danh mục 7 Encoder có công bố khoa học chuẩn mực (findings A3): đã loại các kiến trúc Seq2Seq
# không tương thích (ViT5, BARTpho), model lỗi tokenizer (vi-electra, videberta) và định danh đã bị
# gỡ khỏi HuggingFace (Chau/VNLawBERT), bert-base-multilingual-cased (không phải bi-encoder).
MODELS=(
  "BAAI/bge-m3"
  "intfloat/multilingual-e5-large"
  "intfloat/multilingual-e5-base"
  "vinai/phobert-large"
  "vinai/phobert-base-v2"
  "bkai-foundation-models/vietnamese-bi-encoder"
  "xlm-roberta-base"
)

TOTAL=${#MODELS[@]}
CURRENT=0
SUCCESS_COUNT=0

for MODEL_ID in "${MODELS[@]}"; do
  CURRENT=$((CURRENT + 1))
  SAFE_NAME=$(echo "$MODEL_ID" | tr '/' '_')
  CKPT_PATH="${CHECKPOINTS_DIR}/${SAFE_NAME}"
  MODEL_LOG="${LOG_DIR}/${SAFE_NAME}.log"
  RQ2_RESULT="${RESULTS_DIR}/${SAFE_NAME}/rq2_mrl_pareto_analysis.csv"

  echo "------------------------------------------------------------------"
  echo "[$CURRENT/$TOTAL] TIẾN HÀNH CHO BACKBONE: $MODEL_ID"
  echo "------------------------------------------------------------------"

  # BƯỚC 1: KIỂM TRA CHECKPOINT ĐÃ HUẤN LUYỆN
  if [ -f "${CKPT_PATH}/vietlawbert_mrl.pt" ] || [ -f "${CKPT_PATH}/best_model/vietlawbert_mrl.pt" ]; then
      echo "    -> [RESUME] Đã tìm thấy checkpoint hoàn chỉnh tại: $CKPT_PATH"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
      echo ">>> [1/2] Đang fine-tune MRL (có Validation Checkpoint)..."
      if python3 -u -m training.train_mrl \
          --model-name "$MODEL_ID" \
          --train-parquet "$PARQUET_FILE" \
          --output-dir "$CKPT_PATH" \
          --epochs 3 \
          --batch-size "$TRAIN_BATCH" \
          --gradient-accumulation-steps 1 \
          --freeze-layers 0 \
          --lr 2e-5 \
          --precision bf16 \
          --num-workers 4 > "$MODEL_LOG" 2>&1; then
          echo "    ✓ Huấn luyện thành công: $CKPT_PATH"
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
          echo "    ✗ Lỗi khi huấn luyện $MODEL_ID. Xem chi tiết tại: $MODEL_LOG"
          echo "$MODEL_ID" >> "${LOG_DIR}/failed_training.log"
          continue
      fi
  fi

  # Đảm bảo trọng số tối ưu từ best_model nằm ngay tại gốc CKPT_PATH để bộ nạp nhận diện
  if [ -d "${CKPT_PATH}/best_model" ] && [ -f "${CKPT_PATH}/best_model/vietlawbert_mrl.pt" ]; then
      cp -n "${CKPT_PATH}/best_model/"* "${CKPT_PATH}/" 2>/dev/null || true
  fi

  # Tự động đồng bộ mô hình đề xuất (BAAI/bge-m3) sang thư mục phục vụ RAG
  if [ "$MODEL_ID" == "BAAI/bge-m3" ]; then
      echo "    -> Đồng bộ checkpoint BGE-M3 sang $FINAL_MODEL_DIR..."
      rm -rf "$FINAL_MODEL_DIR"
      mkdir -p "$FINAL_MODEL_DIR"
      cp -r "${CKPT_PATH}/"* "$FINAL_MODEL_DIR/" 2>/dev/null || true
      echo "    ✓ Đã tạo mô hình Production tại: $FINAL_MODEL_DIR"
  fi

  # BƯỚC 2: ĐO PARETO RQ2 CHO TỪNG MÔ HÌNH
  mkdir -p "${RESULTS_DIR}/${SAFE_NAME}"
  if [ -f "$RQ2_RESULT" ]; then
      echo "    -> [RESUME] Đã có kết quả RQ2 cho $MODEL_ID. Bỏ qua đo đạc!"
  else
      echo ">>> [2/2] Đang quét lát cắt Pareto (RQ2)..."
      if python3 -u -m benchmark.evaluate_rqs \
          --benchmark-dir "$BENCHMARK_DIR" \
          --checkpoint-path "$CKPT_PATH" \
          --output-dir "${RESULTS_DIR}/${SAFE_NAME}" \
          --device "$DEVICE" \
          --rq "2" >> "$MODEL_LOG" 2>&1; then
          echo "    ✓ Hoàn tất đo đạc RQ2 cho: $MODEL_ID"
      else
          echo "    ✗ Lỗi đo đạc RQ2 cho $MODEL_ID. Xem log: $MODEL_LOG"
      fi
  fi
done

if [ "$SUCCESS_COUNT" -eq 0 ]; then
    echo "=================================================================="
    echo "✗ TẤT CẢ CÁC MÔ HÌNH ĐỀU THẤT BẠI! DỪNG TIẾN TRÌNH."
    echo "=================================================================="
    exit 1
fi

# ==============================================================================
# BƯỚC 2.5: 3 BIẾN THỂ ĐỐI CHỨNG bge-m3 — RQ1 (random/bm25-neg) & RQ2 (FlatMRL)
#   Checkpoint khớp tên trong evaluate_rqs.raw_backbones → tự lên bảng Master.
# ==============================================================================
TRIP_RANDOM="outputs/artifacts/triplets/triplets_random.parquet"
TRIP_BM25="outputs/artifacts/triplets/triplets_bm25.parquet"
[ -f "$TRIP_RANDOM" ] || python3 -u -m training.generate_hin_triplets --neg-strategy random --output "$TRIP_RANDOM" --limit 50000
[ -f "$TRIP_BM25" ]   || python3 -u -m training.generate_hin_triplets --neg-strategy bm25   --output "$TRIP_BM25"   --limit 50000

train_variant() {  # $1=output-dir  $2=train-parquet  $3=extra-flags
    if [ ! -f "$1/best_model/vietlawbert_mrl.pt" ] && [ ! -f "$1/vietlawbert_mrl.pt" ]; then
        python3 -u -m training.train_mrl --model-name "BAAI/bge-m3" --train-parquet "$2" \
            --output-dir "$1" --epochs 3 --batch-size "$TRAIN_BATCH" --gradient-accumulation-steps 1 \
            --freeze-layers 0 --lr 2e-5 --precision bf16 --num-workers 4 $3 \
            > "${LOG_DIR}/$(basename "$1").log" 2>&1 \
            && echo "    ✓ xong biến thể: $1" || { echo "    ✗ lỗi biến thể $1 (xem log)."; return 0; }
    else
        echo "    -> [RESUME] biến thể đã có: $1"
    fi
    # train_mrl lưu best vào best_model/ — đưa lên gốc để UniversalEncoderWrapper nhận diện (giống vòng chính).
    if [ -f "$1/best_model/vietlawbert_mrl.pt" ] && [ ! -f "$1/vietlawbert_mrl.pt" ]; then
        cp -n "$1/best_model/"* "$1/" 2>/dev/null || true
    fi
}
echo ">>> [BƯỚC 2.5] Train 3 biến thể bge-m3 (RQ1 random, RQ1 bm25, RQ2 FlatMRL)..."
train_variant "${CHECKPOINTS_DIR}/bge-m3_rq1_random"  "$TRIP_RANDOM"  ""
train_variant "${CHECKPOINTS_DIR}/bge-m3_rq1_bm25"    "$TRIP_BM25"    ""
train_variant "${CHECKPOINTS_DIR}/bge-m3_rq2_flatmrl" "$PARQUET_FILE" "--hierarchy-weight 0"

echo "=================================================================="
echo "BƯỚC 3: CHẠY BẢNG ĐỐI ĐẦU MASTER BENCHMARK TRÊN TOÀN BỘ MA TRẬN"
echo "=================================================================="
python3 -u -m benchmark.evaluate_rqs --rq baselines \
  --benchmark-dir "$BENCHMARK_DIR" \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --output-dir "$RESULTS_DIR" \
  --device "$DEVICE"

echo "=================================================================="
echo "BƯỚC 4: THỰC THI TOÀN DIỆN CÁC CÂU HỎI NGHIÊN CỨU (RQ1, RQ3, RQ4)"
echo "=================================================================="
TARGET_EVAL_MODEL="${FINAL_MODEL_DIR}"
if [ ! -f "${TARGET_EVAL_MODEL}/vietlawbert_mrl.pt" ]; then
    if [ -f "${CHECKPOINTS_DIR}/BAAI_bge-m3/best_model/vietlawbert_mrl.pt" ]; then
        TARGET_EVAL_MODEL="${CHECKPOINTS_DIR}/BAAI_bge-m3/best_model"
    else
        TARGET_EVAL_MODEL="${CHECKPOINTS_DIR}/BAAI_bge-m3"
    fi
fi

python3 -u -m benchmark.evaluate_rqs \
  --benchmark-dir "$BENCHMARK_DIR" \
  --checkpoint-path "$TARGET_EVAL_MODEL" \
  --output-dir "$RESULTS_DIR" \
  --device "$DEVICE" \
  --rq "all"

echo ">>> [BƯỚC 4b] Eval NGOẠI VLQA (Hit/MRR/NDCG/MAP@K + P95 + t-test ablation)..."
python3 -u -m benchmark.evaluate_rqs \
  --benchmark-dir "$BENCHMARK_DIR" \
  --checkpoint-path "$TARGET_EVAL_MODEL" \
  --output-dir "$RESULTS_DIR" \
  --device "$DEVICE" \
  --rq vlqa || true

echo "=================================================================="
echo "✓ HOÀN TẤT 100% QUY TRÌNH THỰC NGHIỆM! CÁC BẢNG ĐIỂM ĐÃ SẴN SÀNG TẠI:"
echo "  1. Master Benchmark : ${RESULTS_DIR}/master_baseline_comparison.csv"
echo "  2. RQ1 (Mining)     : ${RESULTS_DIR}/rq1_negative_mining_comparison.csv"
echo "  3. RQ2 (Pareto)     : ${RESULTS_DIR}/rq2_mrl_pareto_analysis.csv"
echo "  4. RQ3 (Ablation)   : ${RESULTS_DIR}/rq3_architecture_ablation_latency.csv"
echo "  5. RQ4 (Judge RAG)  : ${RESULTS_DIR}/rq4_llm_judge_summary.csv"
echo "Thời gian kết thúc   : $(date)"
echo "=================================================================="