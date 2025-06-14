# Dockerfile

FROM python:3.10-slim-buster

# Atur direktori kerja ke root (/)
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin SEMUA kode proyek Anda ke dalam direktori root (/) di dalam kontainer.
COPY . .

# Buat startup script dapat dieksekusi di root
RUN chmod +x /start.sh

EXPOSE 8000

# Healthcheck untuk memastikan aplikasi benar-benar berjalan sebelum dianggap 'healthy'
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD curl --fail http://localhost:8000/ || exit 1

# Perintah terakhir untuk menjalankan aplikasi Anda menggunakan startup script dari root
CMD ["/start.sh"]