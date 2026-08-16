# 1. Sử dụng Python bản nhẹ
FROM python:3.12-slim

# === ĐIỂM NÂNG CẤP MỚI ===
# Ép Python in log/print thẳng ra Terminal của Docker ngay lập tức 
# (Tránh việc code chạy xong 1 lúc lâu mới thấy log hiện ra do bị nghẽn buffer)
ENV PYTHONUNBUFFERED=1
# =========================

# 2. Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    gnupg \
    wget \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# 5. Image triển khai hiện chạy consumer CPU. Cài wheel CPU trước để tránh uv
# kéo theo hơn 1 GB CUDA/NVIDIA không dùng tới; requirements còn lại sẽ dùng lại
# bản torch đã thỏa mãn này.
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -r requirements.txt && playwright install --with-deps chromium

# 6. Copy code
COPY . .

# 7. Chạy app
CMD ["python", "src/main.py"]
