FROM python:3.10-slim-buster

WORKDIR /app

# Install system dependencies needed by numpy, pillow, onnxruntime, etc.
# build-essential: Diperlukan untuk kompilasi
# libgomp1: Diperlukan oleh ONNX Runtime
# libjpeg-dev, zlib1g-dev: Diperlukan oleh Pillow (untuk format JPEG, PNG)
# libgl1-mesa-glx, libglib2.0-0: Diperlukan oleh OpenCV yang mungkin menjadi dependensi ONNX Runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Now, run pip install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY start.sh .
RUN chmod +x ./start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD curl --fail http://localhost:8000/ || exit 1

# Jalankan startup script
CMD ["./start.sh"]