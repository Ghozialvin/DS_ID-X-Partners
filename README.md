
# Credit Risk Prediction Model - ID/X Partners Final Project

## 📌 Project Overview

Proyek ini merupakan tugas akhir dari program *Internship Project-Based* di **ID/X Partners**. Fokus utama proyek ini adalah membangun model machine learning yang mampu memprediksi risiko kredit untuk membedakan antara calon peminjam yang berpotensi gagal bayar (*Bad Loan*) dan peminjam yang aman (*Good Loan*). Dengan adanya model ini, perusahaan pendanaan dapat meminimalkan kerugian finansial dan menjaga stabilitas portofolio kredit mereka.

## 📊 Dataset Discovery

*  **Sumber Data:** Dataset historis pinjaman periode 2007 - 2014.

*  **Dimensi Data:** 466.285 baris dan 75 fitur.

*  **Variabel Target:** Kolom `loan_status` yang dikategorikan menjadi:

*  **1 (Pinjaman Baik):** Mencakup status *Current, Fully Paid, In Grace Period*, dll.

*  **0 (Pinjaman Tidak Baik):** Mencakup status *Charged Off, Late, Default*, dll.

## 🛠️ Workflow & Methodology

1. **Data Preprocessing:**
* Pembersihan data dari nilai kosong (NaN) dan penghapusan fitur yang tidak relevan (ID, URL, deskripsi).

* Konversi fitur tanggal ke format *datetime* untuk ekstraksi fitur berbasis waktu.

* Penanganan fitur kategorikal dengan kardinalitas tinggi seperti `emp_title`.

2. **Feature Engineering:**
* Pembuatan fitur baru seperti `month_issue_since_crline` (durasi riwayat kredit) dan `month_last_pymnt_since_issue` (waktu pembayaran terakhir).

3. **Machine Learning Pipeline:**
* Imputasi data otomatis menggunakan `IterativeImputer` dan `SimpleImputer`.

* Normalisasi data numerik menggunakan `MinMaxScaler`.
* Encoding fitur kategorikal menggunakan `OrdinalEncoder` dan `LabelEncoder`

4.  **Model Selection:** Membandingkan beberapa algoritma: *Logistic Regression, Multinomial Naive Bayes,* dan *K-Neighbors Classifier*.

## 📈 Performance Results

Model terbaik yang terpilih adalah **Logistic Regression** dengan hasil evaluasi sebagai berikut:

* **Balanced Accuracy:** 87,1% 
* **F1 Score:** 98,3% 
* **ROC AUC Score:** 0.945 

## 📂 File Structure

```text
├── DA_IDX_credit_risk.ipynb      # Notebook utama proses analisis dan pemodelan
├── DA_IDX_credit_risk.py         # Script Python dari proses pemodelan
├── DA_IDX_credit_risk.pdf        # Laporan analisis lengkap
├── DA_IDX_credit_risk_Presentation.pptx # Slide presentasi proyek
├── credit_risk_model_logistic.pkl # Model final yang telah disimpan (Joblib)
└── README.md                     

```

## 🚀 How to Use

1. Clone repository ini.
2. Pastikan library `scikit-learn`, `pandas`, `numpy`, dan `joblib` telah terinstal.
3. Untuk menggunakan model yang sudah dilatih pada data baru:
```python
import joblib
model = joblib.load('credit_risk_model_logistic.pkl')
# Lakukan preprocessing pada data baru sesuai pipeline
# model.predict(data_baru)
