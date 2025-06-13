from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI(
    title="API Deteksi Penyakit Kulit",
    description="API untuk mendeteksi penyakit kulit berdasarkan gambar menggunakan model VGG19.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "ok"}

# CORS
origins_str = os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000"]')
origins = json.loads(origins_str)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan origins kalau mau restrict
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Konfigurasi
MODEL_PATH = os.getenv('MODEL_PATH', 'model/final_best_model_vgg19_finetuned.h5')
image_size_str = os.getenv('IMAGE_SIZE', '(224, 224)')
IMAGE_SIZE = tuple(map(int, image_size_str.strip('()').split(',')))
class_names_str = os.getenv('CLASS_NAMES', '["Chickenpox", "Measles", "Monkeypox", "Normal"]')
CLASS_NAMES = json.loads(class_names_str)

# Lazy Load Model
model = None
model_loaded = False

def load_model_lazy():
    global model, model_loaded
    if not model_loaded:
        try:
            print(f"Trying to lazily load model from path: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            model_loaded = True
            print("Model berhasil dimuat secara lazy.")
        except Exception as e:
            print(f"Error lazy load model: {e}")
            raise HTTPException(status_code=500, detail="Gagal memuat model.")

# Fungsi prediksi
def predict_disease_from_image(image_bytes: bytes):
    load_model_lazy()  # pastikan model terload sebelum prediksi

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize(IMAGE_SIZE)

        img_array = np.array(img_resized)
        img_array_expanded = np.expand_dims(img_array, axis=0)

        img_preprocessed = preprocess_input(img_array_expanded)

        predictions = model.predict(img_preprocessed)

        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        disease_label = (
            CLASS_NAMES[predicted_class_index]
            if predicted_class_index < len(CLASS_NAMES)
            else "Unknown"
        )

        suggestions_map = {
            "Monkeypox": [
                "Segera konsultasikan dengan dokter atau fasilitas kesehatan terdekat untuk konfirmasi.",
                "Isolasi diri untuk mencegah potensi penularan sampai ada diagnosis pasti.",
                "Hindari menggaruk ruam untuk mencegah infeksi sekunder.",
                "Jaga kebersihan diri dan lingkungan sekitar.",
            ],
            "Chickenpox": [
                "Konsultasikan dengan dokter untuk diagnosis dan penanganan yang tepat.",
                "Istirahat yang cukup dan perbanyak minum cairan.",
                "Hindari menggaruk lepuh untuk mencegah bekas luka dan infeksi bakteri.",
                "Gunakan losion kalamin atau mandi dengan larutan oatmeal untuk meredakan gatal.",
            ],
            "Measles": [
                "Segera hubungi dokter jika Anda atau anak Anda diduga menderita campak untuk penanganan.",
                "Pastikan penderita mendapatkan istirahat yang cukup dan asupan cairan yang memadai.",
                "Isolasi diri untuk mencegah penyebaran virus ke orang lain.",
                "Periksa status vaksinasi Anda dan keluarga, vaksin MMR sangat efektif mencegah campak.",
            ],
            "Normal": [
                "Kulit Anda tampak normal berdasarkan analisis gambar ini.",
                "Lanjutkan menjaga kebersihan dan kesehatan kulit Anda.",
                "Gunakan tabir surya secara teratur untuk melindungi kulit dari paparan sinar UV.",
                "Jika Anda memiliki kekhawatiran lain tentang kulit Anda, jangan ragu untuk berkonsultasi dengan dokter kulit.",
            ],
        }

        suggestions_list = suggestions_map.get(disease_label, [
            "Untuk informasi lebih lanjut, silakan konsultasikan dengan profesional medis."
        ])

        return {
            "disease": disease_label,
            "confidence": confidence,
            "suggestions": suggestions_list,
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}. Pastikan gambar yang diunggah sesuai dan coba lagi.",
        )

# Endpoint prediksi
@app.post("/predict")
async def create_prediction(file: UploadFile = File(..., description="File gambar kulit yang akan dideteksi")):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar (JPEG, PNG, WEBP).")

    try:
        image_bytes = await file.read()
        result = predict_disease_from_image(image_bytes)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected server error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal pada server saat memproses permintaan Anda.")

# Run locally (optional)
if __name__ == "__main__":
    uvicorn_host = os.getenv("UVICORN_HOST", "0.0.0.0")
    uvicorn_port = int(os.getenv("UVICORN_PORT", 8000))
    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port)
