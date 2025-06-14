#!/bin/bash
# start.sh

# Keluar dari skrip jika ada perintah yang gagal
set -e

# Jalankan skrip unduh. File berada di direktori kerja saat ini.
echo "Running model download script..."
python ./download_model.py

# Jalankan server aplikasi. Uvicorn akan menemukan 'app.py' di direktori kerja.
echo "Starting Uvicorn server..."
exec uvicorn app:app --host 0.0.0.0 --port 8000