import os
from pathlib import Path
import gdown

MODEL_PATH_ONNX = os.getenv("MODEL_PATH_ONNX", "model/final_best_model_vgg19_finetuned.onnx")
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID")

def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH_ONNX):
        print(f"Model sudah ada di {MODEL_PATH_ONNX}")
        return

    if not MODEL_FILE_ID:
        print("MODEL_FILE_ID tidak ditemukan di environment variable.")
        return

    print("Mengunduh model dari Google Drive...")
    Path(os.path.dirname(MODEL_PATH_ONNX)).mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH_ONNX, quiet=False)
        print(f"Model berhasil diunduh ke {MODEL_PATH_ONNX}")
    except Exception as e:
        print(f"Gagal mengunduh model: {e}")

if __name__ == "__main__":
    ensure_model_downloaded()