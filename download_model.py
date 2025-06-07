import os
import requests
from pathlib import Path

def download_from_gdrive(file_id, dest_path):
    print("Mengunduh model dari Google Drive...")

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: 
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id}, stream=True)

    token = get_confirm_token(response)
    if token:
        response = session.get(url, params={'id': file_id, 'confirm': token}, stream=True)

    Path(os.path.dirname(dest_path)).mkdir(parents=True, exist_ok=True)
    save_response_content(response, dest_path)
    print(f"Model berhasil diunduh ke {dest_path}")

MODEL_PATH = os.getenv('MODEL_PATH', 'model/final_best_model_vgg19_finetuned.h5')
MODEL_FILE_ID = os.getenv('MODEL_FILE_ID') 

if not os.path.exists(MODEL_PATH):
    if MODEL_FILE_ID:
        download_from_gdrive(MODEL_FILE_ID, MODEL_PATH)
    else:
        print("MODEL_FILE_ID tidak ditemukan di environment variable.")
