import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Sistem Deteksi APD K3", layout="wide")

st.title("🛡️ Monitoring Keselamatan Kerja (APD)")
st.sidebar.header("Konfigurasi Model")

# 1. Load Model
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di folder yang sama
    return YOLO("best.pt")

model = load_model()

# Daftar Kelas (Sesuai urutan Anda)
class_names = [
    'Fire', 'Gloves', 'Goggle', 'Mask', 'No helmet', 'No shoes', 
    'No suit', 'Person', 'Protective helmet', 'Protective shoes', 
    'Protective suit', 'Smoke'
]

# Slider untuk Confidence Threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Pilihan Sumber Input
source = st.sidebar.selectbox("Pilih Sumber Input", ("Unggah Gambar", "Webcam Langsung"))

if source == "Unggah Gambar":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Konversi file unggahan ke format OpenCV
        image = Image.open(uploaded_file)
        frame = np.array(image)
        
        # Jalankan Prediksi
        results = model.predict(source=frame, conf=conf_threshold)
        
        # Plot hasil deteksi
        res_plotted = results[0].plot()
        
        # Tampilkan Gambar
        st.image(res_plotted, caption="Hasil Deteksi APD", use_container_width=True)
        
        # Tampilkan Statistik Deteksi
        st.subheader("Detail Deteksi:")
        detections = results[0].boxes.cls.tolist()
        if detections:
            for class_id in set(detections):
                count = detections.count(class_id)
                st.write(f"- **{class_names[int(class_id)]}**: {count}")
        else:
            st.write("Tidak ada objek terdeteksi.")

elif source == "Webcam Langsung":
    st.warning("Gunakan tombol 'Stop' di browser untuk menghentikan kamera.")
    img_file_buffer = st.camera_input("Ambil Foto untuk Deteksi")

    if img_file_buffer is not None:
        # Baca gambar dari buffer kamera
        img = Image.open(img_file_buffer)
        frame = np.array(img)

        # Prediksi
        results = model.predict(source=frame, conf=conf_threshold)
        res_plotted = results[0].plot()

        # Tampilkan
        st.image(res_plotted, caption="Hasil Deteksi Real-time", use_container_width=True)