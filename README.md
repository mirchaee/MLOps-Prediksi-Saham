# MLOps: Sistem Prediksi Arah Pergerakan Harga Saham Harian

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Codespaces](https://img.shields.io/badge/Codespaces-Compatible-brightgreen.svg)](https://github.com/codespaces)
[![Daily Ingestion](https://github.com/mirchaee/MLOps-Prediksi-Saham/actions/workflows/ingest_data.yml/badge.svg)](https://github.com/mirchaee/MLOps-Prediksi-Saham/actions/workflows/ingest_data.yml)

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
1. **Automated Data Pipeline:** Penarikan data histori harga saham *End-of-Day* (EOD) secara berkala dan otomatis dari Yahoo Finance API, dijadwalkan lewat GitHub Actions CRON (Senin–Jumat, 17:00 WIB).
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
├── .github/
│   └── workflows/
│       └── ingest_data.yml # CRON job: ingestion -> cleaning -> feature engineering -> DVC
├── config/                 # File konfigurasi parameter, model, dan jalur data
├── data/
│   ├── raw/                # Data mentah EOD dari Yahoo Finance (ignored oleh Git, di-track DVC)
│   ├── processed/          # Data pasca-pembersihan dan validasi
│   └── final/              # Dataset final dengan feature engineering (RSI, MACD, BB)
├── docs/                   # Laporan Lembar Kerja (LK) dan dokumentasi arsitektur
├── models/                 # Artefak model machine learning terlatih (.pkl / registry)
├── notebooks/              # Jupyter Notebooks untuk Exploratory Data Analysis (EDA)
├── scripts/                # Skrip operasional & validasi lingkungan (bukan bagian pipeline inti)
│   ├── environment_test.py     # Validasi kesiapan lingkungan kerja
│   └── initial_experiment.py   # Eksperimen & validasi awal pipeline
├── src/                    # Kode sumber modular utama
│   ├── api/                # Backend API (FastAPI) untuk serving prediksi
│   ├── data/
│   │   ├── ingest_data.py  # Extract: tarik data OHLCV dari Yahoo Finance (yfinance)
│   │   └── clean_data.py   # Transform: null-check, duplicate-check, sort chronological
│   ├── features/
│   │   └── build_features.py # Transform: RSI-14, MACD, Bollinger Bands, target biner
│   ├── models/              # Skrip pelatihan, evaluasi, dan registry model
│   └── monitoring/          # Skrip pemantauan Data Drift & penurunan performa
├── tests/                  # Pengujian unit (unit testing) dan integrasi
├── .gitignore              # Pengecualian pelacakan berkas Git (termasuk data mentah -> DVC)
├── LICENSE                 # Lisensi open-source (MIT)
├── README.md               # Dokumentasi utama proyek
└── requirements.txt        # Daftar dependensi library Python
```

---

## Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.12
* **Environment & Containerization:** GitHub Codespaces, DevContainer, Docker
* **Data Ingestion & Analytics:** Pandas, NumPy, yfinance
* **Data Versioning & Orkestrasi:** DVC, GitHub Actions (scheduled CRON)
* **Machine Learning & MLOps:** Scikit-Learn, XGBoost, MLflow
* **API & Serving:** FastAPI, Uvicorn
* **Testing & Quality Assurance:** Pytest, Pylance, AutoDocstring

---

## Menjalankan Proyek dengan GitHub Codespaces

Penggunaan **GitHub Codespaces** sangat direkomendasikan karena seluruh *environment* dan dependensi telah terkonfigurasi secara otomatis pada `.devcontainer/devcontainer.json`.

### 1. Membuat & Membuka Codespace

1. Buka repositori ini di GitHub.
2. Klik tombol **Code** di kanan atas → pilih tab **Codespaces**.
3. Klik **Create codespace on main**.
4. Tunggu hingga proses *build container* selesai. Environment akan otomatis mengaktifkan Python 3.12 dan menginstal seluruh dependensi pada `requirements.txt`.

### 2. Verifikasi Environment

Setelah Codespace selesai dimuat, buka terminal dan jalankan skrip pengujian lingkungan:

```bash
python scripts/environment_test.py
```

*Jika seluruh library terinstal dengan benar tanpa error, terminal akan mengonfirmasi kesiapan sistem.*

### 3. Menjalankan Eksperimen Awal

Untuk menguji pipeline penarikan data saham awal (contoh emiten: `BBCA.JK`):

```bash
python scripts/initial_experiment.py
```

Atau buka file `notebooks/1.0-initial-eda.ipynb` menggunakan Jupyter Notebook di Codespaces untuk melihat hasil *Exploratory Data Analysis* (EDA) secara interaktif.

### 4. Menjalankan Pipeline Data (ETL) Secara Manual

Pipeline data dijalankan otomatis setiap hari kerja pukul 17:00 WIB lewat GitHub Actions, tetapi bisa juga dijalankan manual untuk pengembangan/debugging:

```bash
python src/data/ingest_data.py       # Extract: tarik data OHLCV terbaru dari Yahoo Finance
python src/data/clean_data.py        # Transform: bersihkan data (null, duplikat, urutan tanggal)
python src/features/build_features.py # Transform: hitung RSI/MACD/Bollinger Bands + label target
```

Setelah dataset final terbentuk di `data/final/BBCA_features.csv`, catat versinya dengan DVC:

```bash
dvc add data/final/BBCA_features.csv
git add data/final/BBCA_features.csv.dvc
git commit -m "feat(data): update dataset versi $(date +'%Y-%m-%d')"
git push
```

### 5. Memantau Otomasi Pipeline

Status ingestion harian dapat dipantau pada tab **Actions** repositori ini, workflow `Daily Stock Data Ingestion`. Workflow dapat dipicu manual kapan saja lewat tombol **Run workflow** tanpa menunggu jadwal CRON.

---

## Alur Pengembangan (GitHub Flow)

Pengembangan repositori ini menerapkan aturan standar **GitHub Flow**:

1. **Branching:** Membuat *branch* fitur/eksperimen baru dari `main` (contoh: `git switch -c feat/initial-eda`).
2. **Development & Commit:** Melakukan perubahan kode dan menyimpan *commit* dengan pesan yang informatif.
3. **Pull Request (PR):** Mengunggah *branch* ke GitHub dan membuat PR ke branch `main`.
4. **Merge:** Melakukan *merge* ke branch `main` setelah seluruh pengujian terverifikasi.

---

## Lisensi

Hak Cipta © 2026 **Roniarta Sibarani**. Dilisensikan di bawah [MIT License](LICENSE).