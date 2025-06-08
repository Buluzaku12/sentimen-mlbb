import streamlit as st
import pandas as pd
import joblib
import re
import string
import os
import gdown
import matplotlib.pyplot as plt
import seaborn as sns

# Unduh model dan vectorizer jika belum ada
MODEL_PATH = "random_forest_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("📥 Mengunduh model dari Google Drive...")
    gdown.download("https://drive.google.com/file/d/1otEPccAqWgorX8hA2FU29h6kMAz2odDn/view?usp=drive_link", MODEL_PATH, quiet=False)

if not os.path.exists(VECTORIZER_PATH):
    st.info("📥 Mengunduh vectorizer dari Google Drive...")
    gdown.download("https://drive.google.com/file/d/1EAA9axbR68YmjomvUB9KfZX3PBJl6Q0a/view?usp=drive_link", VECTORIZER_PATH, quiet=False)

# Fungsi pembersihan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model dan vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# UI Overview
st.title("🔍 Analisis Sentimen Ulasan Mobile Legends")
st.subheader("🎮 Machine Learning + Streamlit App")

st.markdown("""
Aplikasi ini digunakan untuk menganalisis sentimen ulasan pengguna game **Mobile Legends: Bang Bang** berdasarkan data dari Google Play Store.

Model yang digunakan adalah **Random Forest Classifier** yang telah dilatih dengan data ulasan dan rating pengguna. Aplikasi ini memungkinkan Anda:

- ✍️ Menginput ulasan secara manual dan mendapatkan prediksi sentimennya.
- 📁 Mengunggah file CSV berisi kolom `ulasan` untuk diproses secara massal.
- 📊 Melihat hasil klasifikasi: Positif atau Negatif.

---

**Cara Penggunaan:**
1. Masukkan satu ulasan di kotak teks di bawah, lalu klik `Prediksi Sentimen`.
2. Atau upload file CSV berisi ulasan pada bagian bawah aplikasi.
""")

# Input teks manual
input_text = st.text_area("✍️ Masukkan Ulasan:")

if st.button("Prediksi Sentimen"):
    if input_text:
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        label = "🟢 Positif" if pred == 1 else "🔴 Negatif"
        st.success(f"Hasil Sentimen: **{label}**")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

# Upload CSV
st.markdown("---")
st.header("📁 Analisis CSV")
uploaded_file = st.file_uploader("Upload file CSV dengan kolom `ulasan`", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ulasan' not in df.columns:
        st.error("❌ Kolom 'ulasan' tidak ditemukan dalam file.")
    else:
        df['ulasan_bersih'] = df['ulasan'].astype(str).apply(clean_text)
        X_vect = vectorizer.transform(df['ulasan_bersih'])
        df['prediksi_sentimen'] = model.predict(X_vect)
        df['prediksi_label'] = df['prediksi_sentimen'].map({1: 'Positif', 0: 'Negatif'})

        st.success("✅ Analisis selesai. Berikut hasilnya:")
        st.dataframe(df[['ulasan', 'prediksi_label']])

        # Visualisasi hasil prediksi
        st.markdown("### 📊 Visualisasi Distribusi Sentimen")
        sentimen_counts = df['prediksi_label'].value_counts()

        fig, ax = plt.subplots()
        sns.barplot(x=sentimen_counts.index, y=sentimen_counts.values, ax=ax, palette="pastel")
        ax.set_xlabel("Label Sentimen")
        ax.set_ylabel("Jumlah Ulasan")
        ax.set_title("Distribusi Sentimen Ulasan Mobile Legends")
        st.pyplot(fig)

        # Tombol download hasil
        st.download_button("📥 Download Hasil",
                           df.to_csv(index=False),
                           file_name="hasil_prediksi.csv",
                           mime="text/csv")
