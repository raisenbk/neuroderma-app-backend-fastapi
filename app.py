import os
import io
import json
import uvicorn
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Muat environment variables dari file .env
load_dotenv()

# Inisialisasi FastAPI App
app = FastAPI(
    title="API Deteksi Penyakit Kulit",
    description="API untuk mendeteksi penyakit kulit berdasarkan gambar menggunakan model VGG19.",
    version="1.0.0"
)

# Konfigurasi CORS (Cross-Origin Resource Sharing)
# Mengizinkan semua origin untuk kemudahan pengembangan, bisa disesuaikan nanti.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Pengaturan dari Environment Variables
MODEL_PATH = os.getenv('MODEL_PATH', 'model/final_best_model_vgg19_finetuned.h5')
IMAGE_SIZE_STR = os.getenv('IMAGE_SIZE', '(224, 224)')
IMAGE_SIZE = tuple(map(int, IMAGE_SIZE_STR.strip('()').split(',')))
CLASS_NAMES_STR = os.getenv('CLASS_NAMES', '["Chickenpox", "Measles", "Monkeypox", "Normal"]')
CLASS_NAMES = json.loads(CLASS_NAMES_STR)

# Variabel global untuk menampung model, diinisialisasi dengan None
model = None

@app.on_event("startup")
async def startup_event():
    """
    Fungsi ini akan dijalankan sekali saat aplikasi FastAPI dimulai.
    Ini adalah tempat terbaik untuk memuat model.
    """
    global model
    print("Memulai proses startup aplikasi...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: File model tidak ditemukan di '{MODEL_PATH}'.")
        print("Pastikan skrip download_model.py berhasil dijalankan sebelum server dimulai.")
        # Aplikasi akan tetap berjalan, tapi endpoint /predict akan mengembalikan error.
        return

    try:
        print(f"Mencoba memuat model dari: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model berhasil dimuat.")
    except Exception as e:
        print(f"Terjadi error fatal saat memuat model: {e}")
        # Model akan tetap None jika gagal dimuat.

@app.get("/")
def read_root():
    """Endpoint untuk health check."""
    return {"status": "ok", "message": "API is running."}

def predict_disease_from_image(image_bytes: bytes):
    """Fungsi untuk memproses gambar dan melakukan prediksi."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia atau gagal dimuat. Silakan periksa log server.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        predictions = model.predict(img_preprocessed)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        if predicted_class_index < len(CLASS_NAMES):
            disease_label = CLASS_NAMES[predicted_class_index]
        else:
            disease_label = "Unknown"

        suggestions_map = {
            "Monkeypox": [
                "Segera konsultasikan dengan dokter atau fasilitas kesehatan terdekat untuk konfirmasi.",
                "Isolasi diri untuk mencegah potensi penularan sampai ada diagnosis pasti.",
                "Hindari menggaruk ruam untuk mencegah infeksi sekunder.",
                "Jaga kebersihan diri dan lingkungan sekitar."
            ],
            "Chickenpox": [
                "Konsultasikan dengan dokter untuk diagnosis dan penanganan yang tepat.",
                "Istirahat yang cukup dan perbanyak minum cairan.",
                "Hindari menggaruk lepuh untuk mencegah bekas luka dan infeksi bakteri.",
                "Gunakan losion kalamin atau mandi dengan larutan oatmeal untuk meredakan gatal."
            ],
            "Measles": [
                "Segera hubungi dokter jika Anda atau anak Anda diduga menderita campak untuk penanganan.",
                "Pastikan penderita mendapatkan istirahat yang cukup dan asupan cairan yang memadai.",
                "Isolasi diri untuk mencegah penyebaran virus ke orang lain.",
                "Periksa status vaksinasi Anda dan keluarga, vaksin MMR sangat efektif mencegah campak."
            ],
            "Normal": [
                "Kulit Anda tampak normal berdasarkan analisis gambar ini.",
                "Lanjutkan menjaga kebersihan dan kesehatan kulit Anda.",
                "Gunakan tabir surya secara teratur untuk melindungi kulit dari paparan sinar UV.",
                "Jika Anda memiliki kekhawatiran lain tentang kulit Anda, jangan ragu untuk berkonsultasi dengan dokter kulit."
            ]
        }
        suggestions_list = suggestions_map.get(disease_label, ["Untuk informasi lebih lanjut, silakan konsultasikan dengan profesional medis."])

        return {"disease": disease_label, "confidence": confidence, "suggestions": suggestions_list}

    except Exception as e:
        print(f"Error selama prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}.")

@app.post("/predict")
async def create_prediction(file: UploadFile = File(..., description="File gambar kulit yang akan dideteksi")):
    """Endpoint utama untuk mengunggah gambar dan mendapatkan hasil prediksi."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    try:
        image_bytes = await file.read()
        result = predict_disease_from_image(image_bytes)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error tak terduga di server: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada server.")

if __name__ == "__main__":
    uvicorn_host = os.getenv("UVICORN_HOST", "0.0.0.0")
    uvicorn_port = int(os.getenv("UVICORN_PORT", 8000))
    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port)