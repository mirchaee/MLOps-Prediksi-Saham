# MLOps: Sistem Prediksi Arah Pergerakan Harga Saham Harian

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Codespaces](https://img.shields.io/badge/Codespaces-Compatible-brightgreen.svg)](https://github.com/codespaces)

Proyek MLOps ini dikembangkan untuk memprediksi arah pergerakan harga saham harian (*Next-Day Price Direction*) pada emiten Bursa Efek Indonesia (BEI) berbasis *Continual Learning*. 

Sistem ini dirancang dengan infrastruktur MLOps modern untuk memantau performa model, menangani *data drift* dan *concept drift*, serta melakukan pembaruan model secara terautomasi agar hasil prediksi tetap relevan terhadap dinamika pasar finansial. Hasil prediksi bersifat eksperimental dan difokuskan sebagai alat bantu keputusan analitis.

---

## Metadata Proyek
* **Mata Kuliah:** Machine Learning Operations (CIF60050) - Kelas A
* **Dosen Pengampu:** Rizal Setya Perdana, S.Kom., M.Kom., Ph.D.
* **Nama Mahasiswa:** Roniarta Sibarani
* **NIM:** 245150200111036
* **Program Studi:** S1 Teknik Informatika
* **Fakultas / Universitas:** Fakultas Ilmu Komputer, Universitas Brawijaya

---

## Tujuan Proyek

Proyek ini bertujuan untuk:
1. **Automated Data Pipeline:** Penarikan data histori harga saham *End-of-Day* (EOD) secara berkala dan otomatis dari Yahoo Finance API.
2. **Feature Engineering:** Kalkulasi indikator teknikal (*Relative Strength Index*, *MACD*, *Bollinger Bands*) untuk mengekstraksi sinyal pergerakan tren pasar.
3. **Continual Learning Pipeline:** Pembangunan mekanisme *retraining* otomatis berbasis jadwal (mingguan) maupun *trigger* performa ($F1\text{-Score} < 0.58$) dan deteksi pergeseran data.
4. **Model Serving & Monitoring:** Menyediakan layanan prediksi melalui REST API serta pemantauan kesehatan model secara berkelanjutan.
5. **Reproducible Environment:** Menyediakan lingkungan pengembangan yang konsisten berbasis *cloud* menggunakan **GitHub Codespaces**.

---

## Struktur Direktori

Repositori ini disusun mengikuti konvensi standar MLOps industri agar kode, data, artefak model, serta eksperimen tersimpan secara terpisah, terstruktur, dan mudah dipelihara:

```text
MLOps-Prediksi-Saham/
├── .devcontainer/          # Konfigurasi otomatisasi lingkungan GitHub Codespaces
│   └── devcontainer.json
├── .github/                # Workflow GitHub Actions untuk CI/CD pipeline
│   └── workflows/
├── config/                 # File konfigurasi parameter, model, dan jalur data
├── data/
│   ├── raw/                # Data mentah EOD dari Yahoo Finance
│   ├── processed/          # Data pasca-pembersihan dan validasi
│   └── final/              # Dataset final dengan feature engineering (RSI, MACD, BB)
├── docs/                   # Laporan Lembar Kerja (LK) dan dokumentasi arsitektur
├── models/                 # Artefak model machine learning terlatih (.pkl / registry)
├── notebooks/              # Jupyter Notebooks untuk Exploratory Data Analysis (EDA)
├── scripts/                # Skrip operasional & otomatisasi pemeliharaan lokal
├── src/                    # Kode sumber modular utama
│   ├── api/                # Backend API (FastAPI) untuk serving prediksi
│   ├── data/               # Skrip penarikan data (ingest_data.py)
│   ├── features/           # Skrip kalkulasi fitur & indikator teknikal
│   ├── models/             # Skrip pelatihan, evaluasi, dan registry model
│   ├── monitoring/         # Skrip pemantauan Data Drift & penurunan performa
│   ├── environment_test.py # Skrip validasi kesiapan lingkungan kerja
│   └── initial_experiment.py# Skrip eksperimen & validasi awal pipeline
├── tests/                  # Pengujian unit (unit testing) dan integrasi
├── .gitignore              # Pengecualian pelacakan berkas Git
├── LICENSE                 # Lisensi open-source (MIT)
├── README.md               # Dokumentasi utama proyek
└── requirements.txt        # Daftar dependensi library Python

```

---

## Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.12
* **Environment & Containerization:** GitHub Codespaces, DevContainer, Docker
* **Data Ingestion & Analytics:** Pandas, NumPy, yfinance, `ta` (Technical Analysis Library)
* **Machine Learning & MLOps:** Scikit-Learn, XGBoost, MLflow, DVC
* **API & Serving:** FastAPI, Uvicorn
* **Testing & Quality Assurance:** Pytest, Pylance, AutoDocstring

---

## Menjalankan Proyek dengan GitHub Codespaces

Penggunaan **GitHub Codespaces** sangat direkomendasikan karena seluruh *environment* dan dependensi telah terkonfigurasi secara otomatis pada `.devcontainer/devcontainer.json`.

### 1. Membuat & Membuka Codespace

1. Buka repositori ini di GitHub.
2. Klik tombol **Code** di kanan atas $\rightarrow$ Pilih tab **Codespaces**.
3. Klik **Create codespace on main**.
4. Tunggu hingga proses *build container* selesai. Environment akan otomatis mengaktifkan Python 3.12 dan menginstal seluruh dependensi pada `requirements.txt`.

### 2. Verifikasi Environment

Setelah Codespace selesai dimuat, buka terminal dan jalankan skrip pengujian lingkungan:

```bash
python src/environment_test.py

```

*Jika seluruh library terinstal dengan benar tanpa error, terminal akan mengonfirmasi kesiapan sistem.*

### 3. Menjalankan Eksperimen Awal

Untuk menguji pipeline penarikan data saham awal (contoh emiten: `BBCA.JK`):

```bash
python src/initial_experiment.py

```

Atau buka file `notebooks/1.0-initial-eda.ipynb` menggunakan Jupyter Notebook di Codespaces untuk melihat hasil *Exploratory Data Analysis* (EDA) secara interaktif.

---

## Alur Pengembangan (GitHub Flow)

Pengembangan repositori ini menerapkan aturan standar **GitHub Flow**:

1. **Branching:** Membuat *branch* fitur/eksperimen baru dari `main` (contoh: `git switch -c feat/initial-eda`).
2. **Development & Commit:** Melakukan perubahan kode dan menyimpan *commit* dengan pesan yang informatif.
3. **Pull Request (PR):** Mengunggah *branch* ke GitHub dan membuat PR ke branch `main`.
4. **Merge:** Melakukan *merge* ke branch `main` setelah seluruh pengujian terverifikasi.

---

## Lisensi

Hak Cipta © 2026 **Roniarta Sibarani**. Dilisensikan di bawah [MIT License](https://www.google.com/search?q=LICENSE).

```

```