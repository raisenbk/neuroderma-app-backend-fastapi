#!/bin/bash

# Keluar dari skrip jika ada perintah yang gagal
set -e

# Jalankan skrip unduh
echo "Running model download script..."
python download_model.py

# Jalankan server aplikasi
echo "Starting Uvicorn server..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
