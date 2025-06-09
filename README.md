# 🧠 Analisis Sentimen Ulasan Mobile Legends 🎮

Aplikasi web interaktif berbasis **Streamlit** untuk melakukan **analisis sentimen** terhadap ulasan pengguna game **Mobile Legends: Bang Bang** yang diambil dari Google Play Store.

Model machine learning yang digunakan adalah **Random Forest Classifier**, dilatih menggunakan TF-IDF dari ulasan dan rating pengguna.

---

## 🌐 Coba Aplikasi Online

🚀 Klik untuk mengakses:  
👉 [https://sentimen-mlbb.streamlit.app](https://sentimen-mlbb-bjph9r7g7jfwu2dyyu2snp.streamlit.app/)

---

## 📦 Fitur Aplikasi

- ✍️ Input manual satu ulasan dan prediksi langsung sentimennya.
- 📁 Upload file CSV berisi kolom `ulasan` untuk prediksi massal.
- 📊 Tampilkan hasil klasifikasi dalam bentuk tabel dan grafik distribusi.
- 📥 Download hasil prediksi dalam format `.csv`.

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

1. Clone repositori ini:

```bash
git clone https://github.com/Buluzaku12/sentimen-mlbb.git
````

2. Install dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
streamlit run app.py
```

---

## 📁 Format File CSV

File CSV harus memiliki **kolom bernama `ulasan`**, contoh:

| ulasan                |
| --------------------- |
| game ini sangat seru  |
| lag parah bikin kesel |

---

## 🛠️ Requirements

```
streamlit
pandas
scikit-learn
joblib
matplotlib
seaborn
```

---

## 📚 Tentang Proyek

Proyek ini merupakan bagian dari praktikum machine learning dengan fokus pada **klasifikasi sentimen** menggunakan algoritma Random Forest dan visualisasi hasil melalui aplikasi web.

---

## 👨‍💻 Developer

* **Nama**: Advent, 
* **Tujuan**: Praktikum Machine Learning & Portofolio Analisis Sentimen

````
