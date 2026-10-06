#!/usr/bin/env bash
# ==============================================================================
# VIETLAWBERT: MASTER AUTONOMOUS EXPERIMENT HARNESS (13 BACKBONES)
# Flow: Chuẩn hóa môi trường Cloud -> Đồng bộ Code/Data -> Chạy Master Runner 
#       -> Giám sát thời gian thực -> Tải toàn bộ Checkpoints & CSV -> Tự ngắt Cloud VM
# ==============================================================================
set -e

CLOUD_IP="103.9.158.134"
CLOUD_PORT="22"
CLOUD_USER="admin"
CLOUD_PASS="Jj8iVuqrqvqa"
REMOTE="${CLOUD_USER}@${CLOUD_IP}"

LOCAL_DIR="/home/dell/Desktop/vietlawbert"
LOCAL_DATA="/mnt/data/vietlawbert_data"
REMOTE_DIR="/home/${CLOUD_USER}/vietlawbert"

echo "=================================================================="
echo "      VIETLAWBERT: KHỞI ĐỘNG ĐIỀU PHỐI CLOUD GPU TỰ ĐỘNG"
echo "      Mục tiêu: Huấn luyện & Đối chuẩn 13 Mô hình Backbone"
echo "      Máy chủ Cloud: ${REMOTE}:${CLOUD_PORT}"
echo "=================================================================="

# 0. KHỞI TẠO MÔI TRƯỜNG THIẾU YẾU TRÊN CLOUD GPU
echo ">>> [0/5] Đồng bộ gói môi trường, sửa mirror và nạp thư viện bổ trợ..."
ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} bash -c "'
  echo \"${CLOUD_PASS}\" | sudo -S sed -i \"s|http://apt-mirror.apt-mirror.svc.cluster.local/|http://|g\" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true
  echo \"${CLOUD_PASS}\" | sudo -S rm -f /etc/apt/sources.list.d/deadsnakes.list 2>/dev/null || true
  echo \"${CLOUD_PASS}\" | sudo -S apt-get update -y -qq
  echo \"${CLOUD_PASS}\" | sudo -S apt-get install -y -qq rsync tmux
  pip install --quiet sentencepiece protobuf
  pip install --quiet --force-reinstall nvidia-cudnn-cu12 nvidia-nccl-cu12
'"

# 1. ĐỒNG BỘ MÃ NGUỒN VÀ DỮ LIỆU LÊN CLOUD
echo ">>> [1/5] Đang đồng bộ mã nguồn mới nhất lên máy ảo Cloud..."
ssh -p ${CLOUD_PORT} ${REMOTE} "mkdir -p ${REMOTE_DIR}/data ${REMOTE_DIR}/artifacts/triplets ${REMOTE_DIR}/benchmark ${REMOTE_DIR}/checkpoints ${REMOTE_DIR}/models ${REMOTE_DIR}/logs/experiments"

# Đồng bộ mã nguồn (giữ nguyên thư mục checkpoints trên Cloud)
rsync -avP -e "ssh -p ${CLOUD_PORT}" \
  --exclude "venv" --exclude ".git" --exclude "__pycache__" \
  --exclude "models" --exclude "logs" \
  ${LOCAL_DIR}/ ${REMOTE}:${REMOTE_DIR}/

# Sửa triệt để biến DATA_STORAGE_ROOT trong .env trên Cloud để tránh lỗi /mnt/data
ssh -p ${CLOUD_PORT} ${REMOTE} "sed -i 's|DATA_STORAGE_ROOT=.*|DATA_STORAGE_ROOT=/home/admin/vietlawbert/data|g' ${REMOTE_DIR}/.env 2>/dev/null || true"

# 2. KIỂM TRA TRẠNG THÁI GPU
echo ">>> [2/5] Đang kiểm tra khả năng nhận diện GPU & PyTorch 2.6..."
ssh -p ${CLOUD_PORT} ${REMOTE} bash -c "'
  export LD_LIBRARY_PATH=\$(python3 -c \"import nvidia.cudnn, nvidia.nccl, os; print(os.path.dirname(nvidia.cudnn.__file__) + \\\"/lib:\\\" + os.path.dirname(nvidia.nccl.__file__) + \\\"/lib\\\")\" 2>/dev/null):\$LD_LIBRARY_PATH
  python3 -c \"import torch; print(f\\\"✓ PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}\\\")\"
'"

# 3. KÍCH HOẠT TIẾN TRÌNH TRÊN CLOUD BẰNG TMUX (TRÁNH NGHẼN SSH PIPE)
echo ">>> [3/5] Khởi động run_all_experiments.sh trong phiên tmux độc lập..."
ssh -p ${CLOUD_PORT} ${REMOTE} bash -c "'
  cd ${REMOTE_DIR}
  chmod +x run_all_experiments.sh
  rm -f MASTER_DONE.flag MASTER_FAILED.flag
  tmux kill-session -t master_exp 2>/dev/null || true

  # Kích hoạt phiên tmux chạy ngầm
  tmux new-session -d -s master_exp \"./run_all_experiments.sh cuda > logs/experiments/master_orchestrator.log 2>&1 && touch MASTER_DONE.flag || touch MASTER_FAILED.flag\"
'"

# 4. GIÁM SÁT TIẾN TRÌNH THỰC THI THỜI GIAN THỰC
echo ">>> [4/5] Đang giám sát chu trình huấn luyện & đánh giá 13 mô hình..."
echo "    (Tiến trình tự động bỏ qua 5 mô hình đã xong và cày tiếp các mô hình còn lại)"

while true; do
  STATUS=$(ssh -p ${CLOUD_PORT} -o ConnectTimeout=5 ${REMOTE} bash -c "'
    if [ -f ${REMOTE_DIR}/MASTER_DONE.flag ]; then
      echo \"DONE\"
    elif [ -f ${REMOTE_DIR}/MASTER_FAILED.flag ]; then
      echo \"FAILED\"
    else
      echo \"RUNNING\"
    fi
  '" 2>/dev/null || echo "RETRY")

  if [ "$STATUS" == "DONE" ]; then
    echo ""
    echo "=================================================================="
    echo "✓ MA TRẬN 13 MÔ HÌNH ĐÃ HOÀN TẤT HUẤN LUYỆN VÀ ĐỐI CHUẨN SOTA!"
    echo "=================================================================="
    break
  elif [ "$STATUS" == "FAILED" ]; then
    echo "✗ Lỗi phát sinh trong quá trình chạy ma trận 13 mô hình trên Cloud!"
    scp -P ${CLOUD_PORT} ${REMOTE}:${REMOTE_DIR}/logs/experiments/master_orchestrator.log ./master_error.log
    tail -n 40 ./master_error.log
    exit 1
  else
    CURRENT_MODEL=$(ssh -p ${CLOUD_PORT} -o ConnectTimeout=5 ${REMOTE} "grep -E 'TIẾN HÀNH CHO BACKBONE' ${REMOTE_DIR}/logs/experiments/master_orchestrator.log 2>/dev/null | tail -n 1" 2>/dev/null || true)
    LAST_LOG=$(ssh -p ${CLOUD_PORT} -o ConnectTimeout=5 ${REMOTE} "tail -n 1 ${REMOTE_DIR}/logs/experiments/master_orchestrator.log 2>/dev/null" 2>/dev/null || true)
    echo "[$(date +'%H:%M:%S')] ${CURRENT_MODEL} | ${LAST_LOG}"
    sleep 45
  fi
done

# 5. TẢI TOÀN BỘ THÀNH PHẨM VỀ DELL G7
echo ">>> [5/5] Bắt đầu tải toàn bộ thành phẩm thực nghiệm về máy trạm Dell G7..."

# 5.1 Tải toàn bộ bảng số liệu Benchmark (CSV, Paired t-test, Pareto RQ2)
mkdir -p "${LOCAL_DIR}/benchmark/results"
rsync -avP -e "ssh -p ${CLOUD_PORT}" \
  ${REMOTE}:${REMOTE_DIR}/benchmark/results/ \
  "${LOCAL_DIR}/benchmark/results/"

# 5.2 Tải toàn bộ nhật ký huấn luyện chi tiết của 13 mô hình
mkdir -p "${LOCAL_DIR}/logs/experiments"
rsync -avP -e "ssh -p ${CLOUD_PORT}" \
  ${REMOTE}:${REMOTE_DIR}/logs/experiments/ \
  "${LOCAL_DIR}/logs/experiments/"

# 5.3 Tải mô hình cốt lõi triển khai trực tuyến (VietLawBERT-MRL Final)
mkdir -p "${LOCAL_DIR}/models/vietlawbert_mrl_final"
rsync -avP -e "ssh -p ${CLOUD_PORT}" \
  ${REMOTE}:${REMOTE_DIR}/models/vietlawbert_mrl_final/ \
  "${LOCAL_DIR}/models/vietlawbert_mrl_final/"

# 5.4 Tải toàn bộ 13 checkpoints đã fine-tune
mkdir -p "${LOCAL_DIR}/checkpoints"
rsync -avP -e "ssh -p ${CLOUD_PORT}" \
  ${REMOTE}:${REMOTE_DIR}/checkpoints/ \
  "${LOCAL_DIR}/checkpoints/"

echo "=================================================================="
echo "✓ TOÀN BỘ DỮ LIỆU ĐÃ ĐƯỢC TẢI VỀ MÁY CỦA BẠN:"
echo "  1. Bảng đối đầu 13 mô hình: ${LOCAL_DIR}/benchmark/results/master_baseline_comparison.csv"
echo "  2. Bảng quét chiều Pareto:  ${LOCAL_DIR}/benchmark/results/*/rq2_mrl_pareto_analysis.csv"
echo "  3. Trọng số 13 mô hình:     ${LOCAL_DIR}/checkpoints/"
echo "  4. Model phục vụ RAG:       ${LOCAL_DIR}/models/vietlawbert_mrl_final/"
echo "=================================================================="

# Gửi lệnh ngắt nguồn máy ảo ngay lập tức
echo ">>> Đang gửi tín hiệu ngắt nguồn máy ảo Cloud..."
ssh -p ${CLOUD_PORT} ${REMOTE} "echo '${CLOUD_PASS}' | sudo -S poweroff" 2>/dev/null || true

echo -e "\a"
echo "=================================================================="
echo "✓ MÁY ẢO ĐÃ ĐƯỢC TẮT NGUỒN AN TOÀN."
echo "⚠  LƯU Ý: Hãy vào trang web VNSO bấm 'Xóa máy' để ngừng tính cước 100%!"
echo "=================================================================="
