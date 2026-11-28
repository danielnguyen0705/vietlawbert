#!/usr/bin/env bash
# Tải bộ VLQA (eval benchmark) về box từ Google Drive — chạy khi set up máy cloud mới.
# load_vlqa chỉ cần train.json + legal_corpus.json; 3 file còn lại tùy chọn (--full để tải luôn).
# VLQA_DIR: lấy từ biến môi trường (khớp VLQA_DIR trong .env), mặc định $HOME/VLQA.
set -euo pipefail

VLQA_DIR="${VLQA_DIR:-$HOME/VLQA}"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

mkdir -p "$VLQA_DIR"
command -v gdown >/dev/null 2>&1 || pip install -U gdown

dl() {  # $1=file-id  $2=tên file
    local out="$VLQA_DIR/$2"
    if [ -s "$out" ]; then echo "✓ đã có: $out"; return 0; fi
    echo ">>> tải $2 ..."
    gdown --fuzzy "$1" -O "$out"
    [ -s "$out" ] || { echo "✗ $2 tải về rỗng — kiểm quyền chia sẻ Drive ('Anyone with the link')."; return 1; }
}

# Bắt buộc cho load_vlqa
dl 1JITsikpVUtWwVzt75zZ7Exz5c9Smipf7 train.json
dl 1Cwr6OxzJELTTiFnWGP4UwfJ2Uzhpm_V9 legal_corpus.json

if [ "$FULL" = "1" ]; then
    dl 11z6ihLAAwBr4NIS1StMgyVpKyaOOO-KY legal_dataset.json
    dl 1CaXdvAyAfCIqtE51ymwI66a_Wi0oH80O private_test.json
    dl 1oydfeLYcOHp9kFa_QKx_y1VIkusQJ6at public_test.json
fi

echo "✓ VLQA sẵn tại: $VLQA_DIR (nhớ đặt VLQA_DIR=$VLQA_DIR trong .env của box)"
ls -la "$VLQA_DIR"
