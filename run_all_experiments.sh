#!/usr/bin/env bash
# ==============================================================================
# VIETLAWBERT: MASTER EXPERIMENT AUTOMATION HARNESS (13 BACKBONES)
# Tự động hóa: Kiểm tra Resume Checkpoint -> Huấn luyện MRL -> Đo RQ2 -> Master Benchmark
# ==============================================================================

REQUESTED_DEVICE="${1:-cuda}"

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

# 1. Khử lỗi đường dẫn /mnt/data trên Cloud VM và ép về thư mục an toàn
export DATA_STORAGE_ROOT="${HOME}/vietlawbert/data"
export PYTHONPATH="${HOME}/vietlawbert:${PYTHONPATH}"

# 2. Tự động nạp thư viện động cuDNN 9 & NCCL cho PyTorch 2.6
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cudnn, nvidia.nccl, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib:' + os.path.dirname(nvidia.nccl.__file__) + '/lib')" 2>/dev/null):$LD_LIBRARY_PATH

PARQUET_FILE="artifacts/triplets/hin_triplets.parquet"
RESULTS_DIR="benchmark/results"
CHECKPOINTS_DIR="checkpoints"
LOG_DIR="logs/experiments"
FINAL_MODEL_DIR="models/vietlawbert_mrl_final"

mkdir -p "$RESULTS_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR" "models" "${DATA_STORAGE_ROOT}/logs"

# Danh mục đầy đủ 13 mô hình nghiên cứu
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

echo "=================================================================="
echo "BẮT ĐẦU CHU KỲ HUẤN LUYỆN & TÌM DIMENSION TỐI ƯU (RESUME-AWARE)"
echo "Thiết bị thực thi: $DEVICE"
echo "Thời gian bắt đầu: $(date)"
echo "=================================================================="

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

  # BƯỚC 1: KIỂM TRA TÍNH NĂNG RESUME CHECKPOINT
  if [ -f "${CKPT_PATH}/vietlawbert_mrl.pt" ]; then
      echo "    -> [RESUME] Đã tìm thấy checkpoint hoàn chỉnh tại: $CKPT_PATH"
      echo "    -> Bỏ qua huấn luyện, chuyển tiếp ngay sang kiểm định!"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
      echo ">>> [1/2] Đang fine-tune MRL..."
      if python3 -u -m training.train_mrl \
          --model-name "$MODEL_ID" \
          --train-parquet "$PARQUET_FILE" \
          --output-dir "$CKPT_PATH" \
          --epochs 3 \
          --batch-size 16 \
          --gradient-accumulation-steps 2 \
          --lr 2e-5 \
          --precision bf16 \
          --device "$DEVICE" > "$MODEL_LOG" 2>&1; then
          echo "    ✓ Huấn luyện thành công: $CKPT_PATH"
          SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      else
          echo "    ✗ Lỗi khi huấn luyện $MODEL_ID. Xem chi tiết tại: $MODEL_LOG"
          echo "$MODEL_ID" >> "${LOG_DIR}/failed_training.log"
          continue
      fi
  fi

  # Tự động đồng bộ mô hình SOTA đề xuất (BGE-M3) sang Production
  if [ "$MODEL_ID" == "BAAI/bge-m3" ]; then
      echo "    -> Đồng bộ checkpoint BGE-M3 sang $FINAL_MODEL_DIR..."
      rm -rf "$FINAL_MODEL_DIR"
      cp -r "$CKPT_PATH" "$FINAL_MODEL_DIR"
      echo "    ✓ Đã tạo thành công mô hình phục vụ: $FINAL_MODEL_DIR"
  fi

  # BƯỚC 2: ĐO PARETO RQ2 (CẮT LÁT TÌM d*)
  if [ -f "$RQ2_RESULT" ]; then
      echo "    -> [RESUME] Đã có kết quả RQ2 cho $MODEL_ID. Bỏ qua đo đạc!"
  else
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
  fi
done

if [ "$SUCCESS_COUNT" -eq 0 ]; then
    echo "=================================================================="
    echo "✗ TẤT CẢ CÁC MÔ HÌNH ĐỀU THẤT BẠI! DỪNG TIẾN TRÌNH."
    echo "=================================================================="
    exit 1
fi

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
echo "  Thư mục phục vụ RAG: ${FINAL_MODEL_DIR}"
echo "Thời gian kết thúc: $(date)"
echo "=================================================================="
