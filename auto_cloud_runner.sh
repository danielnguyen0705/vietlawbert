#!/usr/bin/env bash
# ==============================================================================
# VIETLAWBERT: MASTER AUTONOMOUS EXPERIMENT HARNESS (13 BACKBONES)
# Vị trí: /home/dell/Desktop/vietlawbert/auto_cloud_runner.sh
# Flow: Chuẩn hóa môi trường Cloud -> Đồng bộ Code/Data -> Chạy Master Runner 
#       -> Giám sát thời gian thực -> Tải toàn bộ Checkpoints & CSV -> Tự ngắt Cloud VM
# ==============================================================================
set -e

# Hỗ trợ truyền tham số động: ./auto_cloud_runner.sh [IP] [PORT] [USER] [PASS]
CLOUD_IP="${1:-159.48.242.34}"
CLOUD_PORT="${2:-21607}"
CLOUD_USER="${3:-root}"
CLOUD_PASS="${4:-rri_4A2pzrbrg2E30sAT}"
REMOTE="${CLOUD_USER}@${CLOUD_IP}"

LOCAL_DIR="/home/dell/Desktop/vietlawbert"
# Tự động nhận diện nếu user là root thì đường dẫn là /root, còn lại là /home/user
if [ "$CLOUD_USER" = "root" ]; then
  REMOTE_DIR="/root/vietlawbert"
else
  REMOTE_DIR="/home/${CLOUD_USER}/vietlawbert"
fi

echo "=================================================================="
echo "      VIETLAWBERT: KHỞI ĐỘNG ĐIỀU PHỐI CLOUD GPU TỰ ĐỘNG"
echo "      Mục tiêu: Khai phá Triplet -> Huấn luyện MRL -> Đo RQ1-RQ4"
echo "      Máy chủ Cloud: ${REMOTE}:${CLOUD_PORT}"
echo "=================================================================="

# 0. KHỞI TẠO MÔI TRƯỜNG THIẾT YẾU TRÊN CLOUD GPU
echo ">>> [0/5] Đồng bộ gói môi trường, sửa mirror và nạp thư viện bổ trợ..."
ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} bash -c "'
  # Kiểm tra nếu là root thì chạy lệnh trực tiếp, không cần gọi sudo
  SUDO_CMD=\"\"
  if command -v sudo &>/dev/null; then
    SUDO_CMD=\"echo \\\"${CLOUD_PASS}\\\" | sudo -S\"
  fi

  eval \$SUDO_CMD sed -i \"s|http://apt-mirror.apt-mirror.svc.cluster.local/|http://|g\" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true
  eval \$SUDO_CMD apt-get update -y -qq || true
  eval \$SUDO_CMD apt-get install -y -qq rsync tmux
  pip install --quiet sentencepiece protobuf scipy pandas pyarrow
'"

# 1. ĐỒNG BỘ MÃ NGUỒN VÀ DỮ LIỆU LÊN CLOUD (CHẶN TRIỆT ĐỂ FILE NẶNG)
echo ">>> [1/5] Đang đồng bộ mã nguồn tinh gọn lên máy ảo Cloud..."
ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} "mkdir -p ${REMOTE_DIR}/data ${REMOTE_DIR}/artifacts/triplets ${REMOTE_DIR}/artifacts/benchmark ${REMOTE_DIR}/benchmark/results ${REMOTE_DIR}/checkpoints ${REMOTE_DIR}/models ${REMOTE_DIR}/logs/experiments"

# Đẩy mã nguồn và tệp triplets sạch (BỎ CỜ -z, BỔ SUNG ĐẦY ĐỦ CÁC CỜ --exclude)
rsync -avP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  --exclude "venv" \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude "models" \
  --exclude "checkpoints" \
  --exclude "logs" \
  --exclude "data" \
  --exclude "*.tar*" \
  --exclude "*.gz" \
  --exclude "*.zip" \
  --exclude "benchmark/results" \
  --exclude "artifacts/graph_embeddings_128d.parquet" \
  --exclude "artifacts/triplets/dry_run_triplets.parquet" \
  ${LOCAL_DIR}/ ${REMOTE}:${REMOTE_DIR}/

# Khử lỗi đường dẫn lưu trữ trên Cloud VM
ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} "sed -i 's|DATA_STORAGE_ROOT=.*|DATA_STORAGE_ROOT=/home/${CLOUD_USER}/vietlawbert/data|g' ${REMOTE_DIR}/.env 2>/dev/null || true"

# 2. KIỂM TRA TRẠNG THÁI GPU
echo ">>> [2/5] Đang kiểm tra khả năng nhận diện GPU & PyTorch..."
ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} bash -c "'
  export LD_LIBRARY_PATH=\$(python3 -c \"import nvidia.cudnn, nvidia.nccl, os; print(os.path.dirname(nvidia.cudnn.__file__) + \\\"/lib:\\\" + os.path.dirname(nvidia.nccl.__file__) + \\\"/lib\\\")\" 2>/dev/null):\$LD_LIBRARY_PATH
  python3 -c \"import torch; print(f\\\"✓ PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\\\")\"
'"

# 3. KÍCH HOẠT TIẾN TRÌNH TRÊN CLOUD BẰNG TMUX
echo ">>> [3/5] Khởi động run_all_experiments.sh trong phiên tmux độc lập..."
ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} bash -c "'
  cd ${REMOTE_DIR}
  chmod +x run_all_experiments.sh
  rm -f MASTER_DONE.flag MASTER_FAILED.flag
  tmux kill-session -t master_exp 2>/dev/null || true

  # Khóa VietLawBERT chạy trên nhân 0-15 và GPU 0, chừa nhân 16-63 cho tác vụ CPU phụ
  tmux new-session -d -s master_exp "taskset -c 0-15 ./run_all_experiments.sh cuda > logs/experiments/master_orchestrator.log 2>&1 && touch MASTER_DONE.flag || touch MASTER_FAILED.flag"
'"

# 4. GIÁM SÁT TIẾN TRÌNH THỰC THI THỜI GIAN THỰC
echo ">>> [4/5] Đang giám sát chu trình huấn luyện & đánh giá khoa học..."
while true; do
  STATUS=$(ssh -p ${CLOUD_PORT} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${REMOTE} bash -c "'
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
    echo "✓ TOÀN BỘ CHU TRÌNH HUẤN LUYỆN & ĐỐI CHUẨN ĐÃ HOÀN TẤT THÀNH CÔNG!"
    echo "=================================================================="
    break
  elif [ "$STATUS" == "FAILED" ]; then
    echo "✗ Lỗi phát sinh trong quá trình chạy thực nghiệm trên Cloud!"
    scp -P ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE}:${REMOTE_DIR}/logs/experiments/master_orchestrator.log ./master_error.log 2>/dev/null || true
    if [ -f ./master_error.log ]; then
      tail -n 40 ./master_error.log
    fi
    exit 1
  else
    CURRENT_MODEL=$(ssh -p ${CLOUD_PORT} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${REMOTE} "grep -E 'TIẾN HÀNH CHO BACKBONE|GIAI ĐOẠN' ${REMOTE_DIR}/logs/experiments/master_orchestrator.log 2>/dev/null | tail -n 1" 2>/dev/null || true)
    LAST_LOG=$(ssh -p ${CLOUD_PORT} -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${REMOTE} "tail -n 1 ${REMOTE_DIR}/logs/experiments/master_orchestrator.log 2>/dev/null" 2>/dev/null || true)
    echo "[$(date +'%H:%M:%S')] ${CURRENT_MODEL} | ${LAST_LOG}"
    sleep 45
  fi
done

# 5. TẢI TOÀN BỘ THÀNH PHẨM VỀ MÁY LOCAL
echo ">>> [5/5] Bắt đầu tải toàn bộ thành phẩm thực nghiệm về máy trạm local..."

# 5.1 Tải toàn bộ bảng kết quả Benchmark (CSV: Master baseline, RQ1, RQ2, RQ3, RQ4)
mkdir -p "${LOCAL_DIR}/benchmark/results"
rsync -avzP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  ${REMOTE}:${REMOTE_DIR}/benchmark/results/ \
  "${LOCAL_DIR}/benchmark/results/"

# 5.2 Tải toàn bộ nhật ký thực nghiệm chi tiết
mkdir -p "${LOCAL_DIR}/logs/experiments"
rsync -avzP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  ${REMOTE}:${REMOTE_DIR}/logs/experiments/ \
  "${LOCAL_DIR}/logs/experiments/"

# 5.3 Tải mô hình cốt lõi phục vụ RAG (vietlawbert_mrl_final)
mkdir -p "${LOCAL_DIR}/models/vietlawbert_mrl_final"
rsync -avzP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  ${REMOTE}:${REMOTE_DIR}/models/vietlawbert_mrl_final/ \
  "${LOCAL_DIR}/models/vietlawbert_mrl_final/"

# 5.4 Tải toàn bộ tập kiểm chuẩn Ground-Truth và tập Triplets đã sinh
mkdir -p "${LOCAL_DIR}/artifacts/benchmark" "${LOCAL_DIR}/artifacts/triplets"
rsync -avzP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  ${REMOTE}:${REMOTE_DIR}/artifacts/benchmark/ \
  "${LOCAL_DIR}/artifacts/benchmark/" || true

rsync -avzP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  ${REMOTE}:${REMOTE_DIR}/artifacts/triplets/ \
  "${LOCAL_DIR}/artifacts/triplets/" || true

# 5.5 Tải các checkpoints mô hình đã huấn luyện
mkdir -p "${LOCAL_DIR}/checkpoints"
rsync -avzP -e "ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no" \
  ${REMOTE}:${REMOTE_DIR}/checkpoints/ \
  "${LOCAL_DIR}/checkpoints/" || true

echo "=================================================================="
echo "✓ TOÀN BỘ THÀNH PHẨM ĐÃ ĐƯỢC ĐỒNG BỘ VỀ MÁY CỦA BẠN:"
echo "  1. Bảng đối đầu 13 mô hình : ${LOCAL_DIR}/benchmark/results/master_baseline_comparison.csv"
echo "  2. Bảng thực nghiệm RQ1-RQ4 : ${LOCAL_DIR}/benchmark/results/rq*.csv"
echo "  3. Trọng số mô hình RAG     : ${LOCAL_DIR}/models/vietlawbert_mrl_final/"
echo "  4. Tập kiểm chuẩn chuẩn hóa: ${LOCAL_DIR}/artifacts/benchmark/"
echo "=================================================================="

# BỔ SUNG: Không tự động tắt máy nếu bạn đang muốn tận dụng CPU làm việc khác
echo "=================================================================="
echo "✓ THÀNH PHẨM ĐÃ TẢI XONG VỀ MÁY DELL G7!"
echo "⚠  MÁY ẢO VẪN ĐANG MỞ để bạn tiếp tục chạy các tác vụ CPU."
echo "   Khi hoàn tất mọi việc, hãy vào Dashboard nhà cung cấp để Xóa Container ngừng tính cước."
echo "=================================================================="
# (Nếu sau này muốn tự tắt máy thì chỉ cần bỏ dấu # ở dòng dưới)
# ssh -p ${CLOUD_PORT} -o StrictHostKeyChecking=no ${REMOTE} "poweroff 2>/dev/null || true"

echo -e "\a"
echo "=================================================================="
echo "✓ MÁY ẢO ĐÃ ĐƯỢC TẮT NGUỒN AN TOÀN."
echo "⚠  LƯU Ý: Hãy kiểm tra bảng điều khiển máy ảo để xác nhận ngừng tính cước!"
echo "=================================================================="