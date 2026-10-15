#!/usr/bin/env bash
# ==============================================================================
# VIETLAWBERT: MASTER EXPERIMENT AUTOMATION HARNESS (13 BACKBONES)
# Vị trí: /home/dell/Desktop/vietlawbert/run_all_experiments.sh
# Tự động hóa: Kiểm tra dữ liệu -> Sinh Triplet/Benchmark -> Huấn luyện MRL
#              -> Đo RQ2 Pareto -> Đối chuẩn Master -> Đo RQ1, RQ3, RQ4
# ==============================================================================

set -e

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

# BỔ SUNG: Giới hạn CPU luồng để không chiếm dụng hết 64 nhân EPYC
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# 1. Cấu hình biến môi trường và thư viện
export DATA_STORAGE_ROOT="${HOME}/vietlawbert/data"
export PYTHONPATH="${HOME}/vietlawbert:${PYTHONPATH}"

# Nạp thư viện động cuDNN và NCCL cho PyTorch
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, nvidia.nccl, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib:' + os.path.dirname(nvidia.nccl.__file__) + '/lib')" 2>/dev/null):$LD_LIBRARY_PATH

PARQUET_FILE="artifacts/triplets/hin_triplets.parquet"
BENCHMARK_DIR="artifacts/benchmark"
RESULTS_DIR="benchmark/results"
CHECKPOINTS_DIR="checkpoints"
LOG_DIR="logs/experiments"
FINAL_MODEL_DIR="models/vietlawbert_mrl_final"

mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR" "models" "${DATA_STORAGE_ROOT}/logs" "artifacts/triplets" "$BENCHMARK_DIR"

echo "=================================================================="
echo "BẮT ĐẦU CHU KỲ THỰC NGHIỆM VIETLAWBERT ĐẠT CHUẨN BÀI BÁO Q1"
echo "Thiết bị thực thi : $DEVICE"
echo "Thời gian bắt đầu : $(date)"
echo "=================================================================="

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

if [ ! -f "${BENCHMARK_DIR}/vietlawbench_1000.jsonl" ]; then
    echo ">>> [BƯỚC 0.2] Chưa phát hiện VietLawBench. Tiến hành khởi tạo tập kiểm chuẩn..."
    python3 -u -m benchmark.build_vietlawbench \
        --total-size 1000
    echo "✓ Đã sinh thành công tập kiểm chuẩn tại: $BENCHMARK_DIR"
else
    echo "✓ Đã có sẵn tập kiểm chuẩn tại: $BENCHMARK_DIR"
fi

# Danh mục đầy đủ 13 mô hình backbone nghiên cứu
MODELS=(
  "BAAI/bge-m3"
  "bert-base-multilingual-cased"
  "xlm-roberta-base"
  "bkai-foundation-models/vietnamese-bi-encoder"
  "intfloat/multilingual-e5-base"
  "intfloat/multilingual-e5-large"
  "vinai/phobert-base-v2"
  "vinai/phobert-large"
  "Fsoft-AIC/vi-electra-base-generator"
  "bkai-foundation-models/videberta-base"
  "vinai/bartpho-syllable"
  "VietAI/vit5-base"
  "Chau/VNLawBERT"
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
          --batch-size 32 \
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

echo "=================================================================="
echo "BƯỚC 3: CHẠY BẢNG ĐỐI ĐẦU MASTER BENCHMARK TRÊN TOÀN BỘ MA TRẬN"
echo "=================================================================="
python3 -u -m benchmark.baseline_comparator \
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

echo "=================================================================="
echo "✓ HOÀN TẤT 100% QUY TRÌNH THỰC NGHIỆM! CÁC BẢNG ĐIỂM ĐÃ SẴN SÀNG TẠI:"
echo "  1. Master Benchmark : ${RESULTS_DIR}/master_baseline_comparison.csv"
echo "  2. RQ1 (Mining)     : ${RESULTS_DIR}/rq1_negative_mining_comparison.csv"
echo "  3. RQ2 (Pareto)     : ${RESULTS_DIR}/rq2_mrl_pareto_analysis.csv"
echo "  4. RQ3 (Ablation)   : ${RESULTS_DIR}/rq3_architecture_ablation_latency.csv"
echo "  5. RQ4 (Judge RAG)  : ${RESULTS_DIR}/rq4_llm_judge_summary.csv"
echo "Thời gian kết thúc   : $(date)"
echo "=================================================================="