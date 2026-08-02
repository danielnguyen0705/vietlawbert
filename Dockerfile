# 1. Sử dụng Python bản nhẹ
FROM python:3.12-slim

# === ĐIỂM NÂNG CẤP MỚI ===
# Ép Python in log/print thẳng ra Terminal của Docker ngay lập tức 
# (Tránh việc code chạy xong 1 lúc lâu mới thấy log hiện ra do bị nghẽn buffer)
ENV PYTHONUNBUFFERED=1
# =========================

# 2. Cài đặt thư viện hệ thống + Intel GPU runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Intel oneAPI runtime libraries for XPU support
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/intel-oneapi.list \
    && apt-get update \
    && apt-get install -y intel-oneapi-runtime-ccl intel-oneapi-runtime-dpcpp-cpp \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/tbb/latest/lib:${LD_LIBRARY_PATH}

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Copy requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# 5. Cài uv (fast installer) và cài dependencies qua uv pip
# uv nhanh hơn pip 10-100x, không cần --default-timeout hack
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.txt

# 6. Copy code
COPY . .

# 7. Chạy app
CMD ["python", "src/main.py"]