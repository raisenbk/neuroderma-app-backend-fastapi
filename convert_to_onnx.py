import tensorflow as tf
import tf2onnx

# Path ke model H5 Anda
keras_model_path = "model/final_best_model_vgg19_finetuned.h5"
# Path untuk menyimpan model ONNX baru
onnx_model_path = "model/final_best_model.onnx"

# Muat model Keras Anda
try:
    model = tf.keras.models.load_model(keras_model_path)
    print("Model Keras berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model Keras: {e}")
    exit()

# Konversi model ke format ONNX
# opset=13 adalah versi yang stabil dan umum digunakan
spec = (tf.TensorSpec(model.input.shape, model.input.dtype, name="input"),)
model_proto, _ = tf.keras.models.convert_keras(model, input_signature=spec, opset=13)

# Simpan model ONNX ke file
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model berhasil dikonversi dan disimpan di: {onnx_model_path}")