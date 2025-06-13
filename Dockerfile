FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin folder model yang berisi file .h5 ke dalam image
# Baris ini tidak diperlukan jika kamu mengunduh, tapi tidak apa-apa jika dibiarkan
COPY ./model ./model
    
# Salin sisa kode aplikasi
COPY . .

EXPOSE 8000

# UBAH BARIS DI BAWAH INI
# Beri waktu 5 menit (300 detik) sebelum health check pertama dimulai
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD curl --fail http://localhost:8000/ || exit 1

# CMD tetap sama, menjalankan skrip download lalu server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]