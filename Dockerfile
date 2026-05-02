# 1. Sử dụng Python bản nhẹ
FROM python:3.10-slim

# === ĐIỂM NÂNG CẤP MỚI ===
# Ép Python in log/print thẳng ra Terminal của Docker ngay lập tức 
# (Tránh việc code chạy xong 1 lúc lâu mới thấy log hiện ra do bị nghẽn buffer)
ENV PYTHONUNBUFFERED=1
# =========================

# 2. Cài đặt thư viện hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# 5. Nâng cấp pip và cài đặt với thời gian chờ (Timeout) cực lớn (1000 giây)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 6. Copy code
COPY . .

# 7. Chạy app
CMD ["python", "src/main.py"]