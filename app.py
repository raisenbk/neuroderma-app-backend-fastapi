from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
import json
from contextlib import asynccontextmanager 

load_dotenv()

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kode ini dieksekusi SAAT STARTUP
    print("Startup: Memulai proses pemuatan model...")
    try:
        # Import yang berat dilakukan di dalam lifespan agar tidak memperlambat impor awal
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.vgg19 import preprocess_input

        model_path = os.getenv('MODEL_PATH', 'model/final_best_model_vgg19_finetuned.h5')
        
        if os.path.exists(model_path):
            ml_resources["model"] = load_model(model_path)
            ml_resources["preprocess_input"] = preprocess_input
            print(f"Startup: Model dari {model_path} berhasil dimuat.")
        else:
            print(f"Startup Error: File model tidak ditemukan di {model_path}")
            # Anda bisa memutuskan untuk menghentikan aplikasi jika model tidak ada,
            # tapi untuk sekarang kita biarkan kosong.
    except Exception as e:
        print(f"Startup Error: Gagal memuat model. Error: {e}")

    yield 

    print("Shutdown: Membersihkan resource ML...")
    ml_resources.clear()
    print("Shutdown: Resource telah dibersihkan.")

app = FastAPI(
    title="API Deteksi Penyakit Kulit",
    description="API untuk mendeteksi penyakit kulit berdasarkan gambar menggunakan model VGG19.",
    version="1.0.0",
    lifespan=lifespan 
)


IMAGE_SIZE_STR = os.getenv('IMAGE_SIZE', '(224, 224)')
IMAGE_SIZE = tuple(map(int, IMAGE_SIZE_STR.strip('()').split(',')))
CLASS_NAMES_STR = os.getenv('CLASS_NAMES', '["Chickenpox", "Measles", "Monkeypox", "Normal"]')
CLASS_NAMES = json.loads(CLASS_NAMES_STR)
origins_str = os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000"]') 
origins = json.loads(origins_str) 

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --- Endpoints ---
@app.get("/wakeup")
def read_root():
    """Endpoint health check sederhana."""
    return {"status": "ok"}

def predict_disease_from_image(image_bytes: bytes):
    if "model" not in ml_resources or ml_resources["model"] is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia atau gagal dimuat saat startup. Periksa log server.")
    
    try:
        model = ml_resources["model"]
        preprocess_input = ml_resources["preprocess_input"]

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        predictions = model.predict(img_preprocessed)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        disease_label = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else "Unknown"
        
        suggestions_map = {
            "Monkeypox": ["Segera konsultasikan dengan dokter...", "..."],
            "Chickenpox": ["Konsultasikan dengan dokter...", "..."],
            "Measles": ["Segera hubungi dokter...", "..."],
            "Normal": ["Kulit Anda tampak normal...", "..."]
        } # (Saran disingkat untuk keringkasan)
        suggestions_list = suggestions_map.get(disease_label, ["Konsultasikan dengan profesional medis."])

        return {"disease": disease_label, "confidence": confidence, "suggestions": suggestions_list}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}.")

@app.post("/predict")
async def create_prediction(file: UploadFile = File(..., description="File gambar kulit yang akan dideteksi")):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")
    
    try:
        image_bytes = await file.read()
        return predict_disease_from_image(image_bytes)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected server error: {e}")
        raise HTTPException(status_code=500, detail="Kesalahan internal pada server.")

# Blok untuk menjalankan secara lokal (tidak berubah)
if __name__ == "__main__":
    uvicorn_host = os.getenv("UVICORN_HOST", "0.0.0.0")
    uvicorn_port = int(os.getenv("UVICORN_PORT", 8000))
    uvicorn.run("app:app", host=uvicorn_host, port=uvicorn_port, reload=True)