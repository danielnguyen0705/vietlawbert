FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    wget \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache -r requirements.txt && \
    playwright install-deps chromium && \
    playwright install chromium

COPY . .

CMD ["python", "main.py"]