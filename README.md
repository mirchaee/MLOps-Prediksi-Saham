# MLOps: Sistem Prediksi Arah Pergerakan Harga Saham Harian

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Codespaces](https://img.shields.io/badge/Codespaces-Compatible-brightgreen.svg)](https://github.com/codespaces)

Proyek ini merupakan implementasi pipeline Machine Learning Operations (MLOps) *end-to-end* berbasis *Continual Learning* untuk memprediksi arah pergerakan harga saham harian (*Next-Day Price Direction*) pada emiten Bursa Efek Indonesia (BEI).

---

## Metadata Proyek
* **Mata Kuliah:** Machine Learning Operations (CIF60050)
* **Mahasiswa:** Roniarta Sibarani
* **NIM:** 245150200111036
* **Program Studi:** S1 Teknik Informatika
* **Fakultas / Universitas:** Fakultas Ilmu Komputer, Universitas Brawijaya

---

## Struktur Direktori (Cookiecutter Standard)

```text
├── .devcontainer/       # Konfigurasi otomatisasi GitHub Codespaces
├── config/              # Berkas konfigurasi parameter & jalur data
├── data/                # Manajemen data terstruktur
│   ├── raw/             # Data mentah EOD dari Yahoo Finance
│   ├── processed/       # Data pasca-pembersihan & validasi
│   └── final/           # Dataset dengan feature engineering (RSI, MACD, BB)
├── models/              # Artefak model terlatih & registry
├── notebooks/           # Jupyter Notebooks untuk EDA & eksperimen awal
├── src/                 # Kode sumber modular utama
│   ├── data/            # Skrip penarikan data (ingest_data.py)
│   ├── features/        # Skrip transformasi fitur (build_features.py)
│   └── models/          # Skrip pelatihan & evaluasi (train.py)
├── .gitignore           # File pengecualian pelacakan Git
├── LICENSE              # Lisensi proyek (MIT)
├── README.md            # Dokumentasi utama proyek
└── requirements.txt     # Daftar dependensi library Python

🚀 Panduan Membuka Lingkungan Kerja (GitHub Codespaces)
Proyek ini telah dikonfigurasi menggunakan DevContainer agar dapat langsung dijalankan tanpa perlu melakukan instalasi manual di komputer lokal.

Klik tombol Code di bagian atas repositori ini.

Pilih tab Codespaces lalu klik Create codespace on main.

Tunggu proses build container selesai. Lingkungan kerja akan otomatis menginstal Python 3.12 beserta seluruh dependensi library yang tercantum pada requirements.txt.

🌿 Strategi Percabangan (GitHub Flow)
Pengembangan repositori ini menggunakan aturan GitHub Flow:

main: Percabangan utama yang berisi kode terverifikasi dan siap produksi.

feat/*: Percabangan fitur/eksperimen (contoh: feat/initial-eda). Setiap perubahan dimasukkan ke main melalui Pull Request (PR) setelah lolos pengujian.
