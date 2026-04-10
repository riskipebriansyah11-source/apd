import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Import OpenCV dengan error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV tidak tersedia, menggunakan fallback mode")

# Konfigurasi Halaman
st.set_page_config(page_title="Sistem Deteksi APD K3", layout="wide")

st.title("🛡️ Monitoring Keselamatan Kerja (APD)")
st.sidebar.header("Konfigurasi Model")

# 1. Load Model
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di folder yang sama
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

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

# Fungsi untuk melakukan prediksi dan menampilkan hasil
def process_and_display(image_array, image_input):
    # Jalankan Prediksi
    results = model.predict(source=image_array, conf=conf_threshold)
    
    # Plot hasil deteksi (YOLO sudah handle plotting)
    res_plotted = results[0].plot()
    
    # Tampilkan Gambar
    st.image(res_plotted, caption="Hasil Deteksi APD", use_container_width=True)
    
    # Tampilkan Statistik Deteksi
    st.subheader("📊 Detail Deteksi:")
    
    if len(results[0].boxes) > 0:
        detections = results[0].boxes.cls.tolist()
        # Hitung statistik per kelas
        from collections import Counter
        counts = Counter([class_names[int(cls)] for cls in detections])
        
        # Tampilkan dalam format tabel
        for item, count in counts.items():
            st.write(f"- **{item}**: {count}")
        
        # Tambahkan total objek
        st.write(f"**Total objek terdeteksi:** {len(detections)}")
    else:
        st.info("Tidak ada objek terdeteksi.")
    
    return results

if source == "Unggah Gambar":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Konversi file unggahan ke format yang bisa diproses YOLO
        image = Image.open(uploaded_file)
        frame = np.array(image)
        
        # Proses deteksi
        process_and_display(frame, uploaded_file)

elif source == "Webcam Langsung":
    st.info("📸 Klik tombol 'Ambil Foto' untuk melakukan deteksi APD")
    
    # Opsi untuk menggunakan kamera
    img_file_buffer = st.camera_input("Ambil Foto untuk Deteksi", key="webcam")

    if img_file_buffer is not None:
        # Baca gambar dari buffer kamera
        img = Image.open(img_file_buffer)
        frame = np.array(img)
        
        # Proses deteksi
        process_and_display(frame, img_file_buffer)

# Tambahkan informasi di sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Informasi Model:**
    - Model mendeteksi 12 kelas APD
    - Confidence threshold bisa diatur
    - Hasil deteksi akan ditampilkan dengan bounding box
    """
)

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Atur Confidence Threshold untuk hasil deteksi yang lebih akurat")
