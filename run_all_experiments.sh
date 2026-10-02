#!/usr/bin/env bash
# ==============================================================================
# VIETLAWBERT: MASTER EXPERIMENT AUTOMATION HARNESS (13 BACKBONES)
# Tự động hóa kiểm thử đa mô hình và đo đạc Pareto d* phục vụ Paper Q1
# ==============================================================================

# Cho phép truyền thiết bị qua tham số: ./run_all_experiments.sh cpu hoặc cuda
REQUESTED_DEVICE="${1:-cuda}"

# Kiểm tra tương thích CUDA
if [ "$REQUESTED_DEVICE" = "cuda" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[CẢNH BÁO] Không tìm thấy card NVIDIA. Tự động chuyển sang thiết bị: cpu"
        DEVICE="cpu"
    else
        DEVICE="cuda"
    fi
else
    DEVICE="cpu"
fi

PARQUET_FILE="artifacts/triplets/hin_triplets.parquet"
RESULTS_DIR="benchmark/results"
CHECKPOINTS_DIR="checkpoints"
LOG_DIR="logs/experiments"

mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR"

# Danh mục đầy đủ 13 mô hình học thuật
MODELS=(
  "Chau/VNLawBERT"
  "bert-base-multilingual-cased"
  "vinai/phobert-base-v2"
  "vinai/phobert-large"
  "xlm-roberta-base"
  "Fsoft-AIC/vi-electra-base-generator"
  "bkai-foundation-models/videberta-base"
  "bkai-foundation-models/vietnamese-bi-encoder"
  "intfloat/multilingual-e5-base"
  "intfloat/multilingual-e5-large"
  "BAAI/bge-m3"
  "vinai/bartpho-syllable"
  "VietAI/vit5-base"
)

echo "=================================================================="
echo "BẮT ĐẦU CHU KỲ HUẤN LUYỆN & TÌM DIMENSION TỐI ƯU CHO 13 MÔ HÌNH"
echo "Thiết bị thực thi: $DEVICE"
echo "Thời gian bắt đầu: $(date)"
echo "=================================================================="

TOTAL=${#MODELS[@]}
CURRENT=0

for MODEL_ID in "${MODELS[@]}"; do
  CURRENT=$((CURRENT + 1))
  SAFE_NAME=$(echo "$MODEL_ID" | tr '/' '_')
  CKPT_PATH="${CHECKPOINTS_DIR}/${SAFE_NAME}"
  MODEL_LOG="${LOG_DIR}/${SAFE_NAME}.log"

  echo "------------------------------------------------------------------"
  echo "[$CURRENT/$TOTAL] TIẾN HÀNH CHO BACKBONE: $MODEL_ID"
  echo "------------------------------------------------------------------"

  # Bước 1: Huấn luyện MRL
  echo ">>> [1/2] Đang fine-tune MRL..."
  if python3 -u -m training.train_mrl \
      --model-name "$MODEL_ID" \
      --train-parquet "$PARQUET_FILE" \
      --output-dir "$CKPT_PATH" \
      --epochs 3 \
      --batch-size 16 \
      --lr 2e-5 \
      --device "$DEVICE" > "$MODEL_LOG" 2>&1; then
      echo "    ✓ Huấn luyện thành công: $CKPT_PATH"
  else
      echo "    ✗ Lỗi khi huấn luyện $MODEL_ID. Xem chi tiết tại: $MODEL_LOG"
      echo "$MODEL_ID" >> "${LOG_DIR}/failed_training.log"
      continue
  fi

  # Bước 2: Đo Pareto RQ2 quét lát cắt vector tìm d*
  echo ">>> [2/2] Đang quét lát cắt Pareto (RQ2)..."
  if python3 -u -m benchmark.evaluate_rqs \
      --checkpoint-path "$CKPT_PATH" \
      --output-dir "${RESULTS_DIR}/${SAFE_NAME}" \
      --device "$DEVICE" \
      --rq "2" >> "$MODEL_LOG" 2>&1; then
      echo "    ✓ Hoàn tất đo đạc RQ2 cho: $MODEL_ID"
  else
      echo "    ✗ Lỗi đo đạc RQ2 cho $MODEL_ID. Xem log: $MODEL_LOG"
  fi
done

echo "=================================================================="
echo "BƯỚC CUỐI: CHẠY BẢNG ĐỐI ĐẦU MASTER BENCHMARK TRÊN TOÀN BỘ MA TRẬN"
echo "=================================================================="
python3 -u -m benchmark.baseline_comparator \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --output-dir "$RESULTS_DIR" \
  --device "$DEVICE"

echo "=================================================================="
echo "✓ HOÀN TẤT 100% QUY TRÌNH THỰC NGHIỆM! BẢNG SỐ LIỆU ĐÃ LƯU TẠI:"
echo "  ${RESULTS_DIR}/master_baseline_comparison.csv"
echo "Thời gian kết thúc: $(date)"
echo "=================================================================="