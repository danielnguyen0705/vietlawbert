# 1. Sử dụng Python bản nhẹ
FROM python:3.10-slim

# 2. Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy requirements
COPY requirements.txt .

# 5. Nâng cấp pip và cài đặt với thời gian chờ (Timeout) cực lớn (1000 giây)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 6. Copy code
COPY . .

# 7. Chạy app
CMD ["python", "src/main.py"]