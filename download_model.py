import os
from pathlib import Path
import gdown

MODEL_PATH = os.getenv("MODEL_PATH", "model/final_best_model_vgg19_finetuned.h5")
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID")

def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        print(f"Model sudah ada di {MODEL_PATH}")
        return

    if not MODEL_FILE_ID:
        print("MODEL_FILE_ID tidak ditemukan di environment variable.")
        return

    print("Mengunduh model dari Google Drive...")
    Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        print(f"Model berhasil diunduh ke {MODEL_PATH}")
    except Exception as e:
        print(f"Gagal mengunduh model: {e}")

if __name__ == "__main__":
    ensure_model_downloaded()