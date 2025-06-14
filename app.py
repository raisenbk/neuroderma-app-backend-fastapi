import os
import io
import json
import uvicorn
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import onnxruntime as ort 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="API Deteksi Penyakit Kulit",
    description="API untuk mendeteksi penyakit kulit berdasarkan gambar menggunakan model VGG19 (ONNX).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neuroderma-app-frontend-nextjs-wqae.vercel.app/"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MODEL_PATH_ONNX = os.getenv('MODEL_PATH_ONNX', 'model/final_best_model_vgg19_finetuned.onnx') 
IMAGE_SIZE_STR = os.getenv('IMAGE_SIZE', '(224, 224)')
IMAGE_SIZE = tuple(map(int, IMAGE_SIZE_STR.strip('()').split(',')))
CLASS_NAMES_STR = os.getenv('CLASS_NAMES', '["Chickenpox", "Measles", "Monkeypox", "Normal"]')
CLASS_NAMES = json.loads(CLASS_NAMES_STR)

ort_session = None
input_name = None

@app.on_event("startup")
async def startup_event():
    """
    Memuat model ONNX saat aplikasi dimulai.
    """
    global ort_session, input_name
    print("Memulai proses startup aplikasi...")
    if not os.path.exists(MODEL_PATH_ONNX):
        print(f"ERROR: File model tidak ditemukan di '{MODEL_PATH_ONNX}'.")
        return

    try:
        print(f"Mencoba memuat model ONNX dari: {MODEL_PATH_ONNX}")
        ort_session = ort.InferenceSession(MODEL_PATH_ONNX)
        input_name = ort_session.get_inputs()[0].name 
        print(f"Model ONNX berhasil dimuat. Nama input: {input_name}")
    except Exception as e:
        print(f"Terjadi error fatal saat memuat model ONNX: {e}")

@app.get("/")
def read_root():
    """Endpoint untuk health check."""
    return {"status": "ok", "message": "API is running."}

def preprocess_image(image_bytes: bytes):
    """
    Fungsi untuk memproses gambar agar sesuai dengan input model ONNX.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    return img_array_expanded.astype(np.float32)


def predict_disease_from_image(image_bytes: bytes):
    """Fungsi untuk memproses gambar dan melakukan prediksi menggunakan ONNX Runtime."""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia atau gagal dimuat.")

    try:
        img_preprocessed = preprocess_image(image_bytes)

        ort_inputs = {input_name: img_preprocessed}
        ort_outs = ort_session.run(None, ort_inputs)
        
        predictions = ort_outs[0] 
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