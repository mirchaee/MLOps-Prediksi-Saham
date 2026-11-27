# MLOps: Sistem Prediksi Arah Pergerakan Harga Saham Harian

[![Python
3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub
Codespaces](https://img.shields.io/badge/Codespaces-Compatible-brightgreen.svg)](https://github.com/codespaces)
[![Daily
Ingestion](https://github.com/mirchaee/MLOps-Prediksi-Saham/actions/workflows/ingest_data.yml/badge.svg)](https://github.com/mirchaee/MLOps-Prediksi-Saham/actions/workflows/ingest_data.yml)

Proyek MLOps ini dikembangkan untuk memprediksi arah pergerakan harga
saham harian (*Next-Day Price Direction*) pada emiten Bursa Efek
Indonesia (BEI) berbasis *Continual Learning*.

Sistem ini dirancang dengan infrastruktur MLOps modern untuk memantau
performa model, menangani *data drift* dan *concept drift*, serta
melakukan pembaruan model secara terautomasi agar hasil prediksi tetap
relevan terhadap dinamika pasar finansial. Hasil prediksi bersifat
eksperimental dan difokuskan sebagai alat bantu keputusan analitis.

------------------------------------------------------------------------

## Metadata Proyek

- **Mata Kuliah:** Machine Learning Operations (CIF60050) - Kelas A
- **Dosen Pengampu:** Rizal Setya Perdana, S.Kom., M.Kom., Ph.D.
- **Nama Mahasiswa:** Roniarta Sibarani
- **NIM:** 245150200111036
- **Program Studi:** S1 Teknik Informatika
- **Fakultas / Universitas:** Fakultas Ilmu Komputer, Universitas
  Brawijaya

------------------------------------------------------------------------

## Tujuan Proyek

Proyek ini bertujuan untuk:

1.  **Automated Data Pipeline:** Penarikan data histori harga saham
    *End-of-Day* (EOD) secara berkala dan otomatis dari Yahoo Finance
    API, dijadwalkan lewat GitHub Actions CRON (Senin–Jumat, 17:00 WIB).
2.  **Feature Engineering:** Kalkulasi indikator teknikal (*Relative
    Strength Index*, *MACD*, *Bollinger Bands*) untuk mengekstraksi
    sinyal pergerakan tren pasar.
3.  **Continual Learning Pipeline:** Pembangunan mekanisme *retraining*
    otomatis berbasis jadwal (mingguan) maupun *trigger* performa
    ($F1\text{-Score} < 0.58$) dan deteksi pergeseran data.
4.  **Model Serving & Monitoring:** Menyediakan layanan prediksi melalui
    REST API serta pemantauan kesehatan model secara berkelanjutan.
5.  **Reproducible Environment:** Menyediakan lingkungan pengembangan
    yang konsisten berbasis *cloud* menggunakan **GitHub Codespaces**.

------------------------------------------------------------------------

## Struktur Direktori

Repositori ini disusun mengikuti konvensi standar MLOps industri agar
kode, data, artefak model, serta eksperimen tersimpan secara terpisah,
terstruktur, dan mudah dipelihara:

``` text
MLOps-Prediksi-Saham/
├── .devcontainer/
├── .dvc/
├── .github/
│   └── workflows/
│       ├── ingest_data.yml
│       └── mlops-automation.yaml
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── docs/
├── mlartifacts/
├── models/
├── notebooks/
├── scripts/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── features/
│   ├── models/
│   └── monitoring/
├── tests/
├── Dockerfile
├── compose.yaml
├── mlflow.db
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

------------------------------------------------------------------------

## Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python 3.12
- **Environment & Containerization:** GitHub Codespaces, DevContainer,
  Docker
- **Data Ingestion & Analytics:** Pandas, NumPy, yfinance
- **Data Versioning & Orkestrasi:** DVC, GitHub Actions (scheduled CRON)
- **Machine Learning & MLOps:** Scikit-Learn, XGBoost, MLflow
- **API & Serving:** FastAPI, Uvicorn
- **Testing & Quality Assurance:** Pytest, Pylance, AutoDocstring

------------------------------------------------------------------------

## Menjalankan Proyek dengan GitHub Codespaces

Penggunaan **GitHub Codespaces** sangat direkomendasikan karena seluruh
*environment* dan dependensi telah terkonfigurasi secara otomatis pada
`.devcontainer/devcontainer.json`.

### 1. Membuat & Membuka Codespace

1.  Buka repositori ini di GitHub.
2.  Klik tombol **Code** di kanan atas → pilih tab **Codespaces**.
3.  Klik **Create codespace on main**.
4.  Tunggu hingga proses *build container* selesai. Environment akan
    otomatis mengaktifkan Python 3.12 dan menginstal seluruh dependensi
    pada `requirements.txt`.

### 2. Verifikasi Environment

Setelah Codespace selesai dimuat, buka terminal dan jalankan skrip
pengujian lingkungan:

``` bash
python scripts/environment_test.py
```

*Jika seluruh library terinstal dengan benar tanpa error, terminal akan
mengonfirmasi kesiapan sistem.*

### 3. Menjalankan Eksperimen Awal

Untuk menguji pipeline penarikan data saham awal (contoh emiten:
`BBCA.JK`):

``` bash
python scripts/initial_experiment.py
```

Atau buka file `notebooks/1.0-initial-eda.ipynb` menggunakan Jupyter
Notebook di Codespaces untuk melihat hasil *Exploratory Data Analysis*
(EDA) secara interaktif.

### 4. Menjalankan Pipeline Data (ETL) Secara Manual

Pipeline data dijalankan otomatis setiap hari kerja pukul 17:00 WIB
lewat GitHub Actions, tetapi bisa juga dijalankan manual untuk
pengembangan/debugging:

``` bash
python src/data/ingest_data.py       # Extract: tarik data OHLCV terbaru dari Yahoo Finance (dengan retry saat error koneksi)
python src/data/preprocess.py        # Transform: bersihkan data mentah (null, duplikat, urutan tanggal)
python src/features/build_features.py # Transform: hitung RSI/MACD/Bollinger Bands + label target
```

**Detail skrip ingestion (`ingest_data.py`):**

- Mengambil sliding window 250 hari perdagangan terakhir dari Yahoo
  Finance API untuk ticker `BBCA.JK`.
- Dilengkapi mekanisme *retry* (maksimal 3 percobaan, jeda 5 detik)
  untuk menangani error koneksi/timeout.
- Bersifat non-destruktif: setiap eksekusi menghasilkan snapshot baru
  bertimestamp (`data/raw/BBCA_raw_YYYYMMDD.csv`) tanpa menimpa data
  lama, serta memperbarui file `data/raw/BBCA_raw.csv` sebagai referensi
  data terbaru.

**Detail skrip preprocessing (`preprocess.py`):**

- Membaca data mentah terbaru dari `data/raw/BBCA_raw.csv`.
- Melakukan null-check pada kolom OHLCV, pengecekan baris duplikat
  berdasarkan tanggal, dan pengurutan data secara kronologis.
- Menyimpan hasil ke `data/processed/BBCA_processed.csv`, siap digunakan
  pada tahap ekstraksi fitur (`build_features.py`).

Setelah dataset final terbentuk di `data/final/BBCA_features.csv`, catat
versinya dengan DVC:

``` bash
dvc add data/final/BBCA_features.csv
git add data/final/BBCA_features.csv.dvc
git commit -m "feat(data): update dataset versi $(date +'%Y-%m-%d')"
git push
```

### 5. Memantau Otomasi Pipeline

Status ingestion harian dapat dipantau pada tab **Actions** repositori
ini, workflow `Daily Stock Data Ingestion`. Workflow dapat dipicu manual
kapan saja lewat tombol **Run workflow** tanpa menunggu jadwal CRON.

------------------------------------------------------------------------

## Manajemen Data Versioning Menggunakan DVC

### 1. Tujuan Implementasi

Data Version Control (DVC) digunakan untuk mengelola perubahan dataset
saham BBCA.JK tanpa menyimpan seluruh versi file CSV secara langsung di
Git. DVC menyimpan konten dataset melalui cache dan mencatat metadata
hash pada file `.dvc`, sedangkan Git digunakan untuk menyimpan histori
perubahan metadata tersebut.

Mekanisme ini mendukung *data lineage*, audit perubahan dataset, dan
*reproducibility* dalam pengembangan pipeline continual learning.

### 2. Alur Pembaruan Dataset

Proses pembaruan dataset dilakukan melalui tahapan:

1.  **Data Ingestion:** Mengambil data saham BBCA.JK terbaru dari Yahoo
    Finance melalui `src/data/ingest_data.py`.
2.  **Preprocessing:** Melakukan null-check, duplicate-check, dan
    chronological sorting menggunakan `src/data/preprocess.py`.
3.  **Feature Engineering:** Menghitung RSI-14, MACD, Bollinger Bands,
    serta target klasifikasi melalui `src/features/build_features.py`.
4.  **Data Versioning:** Mencatat perubahan dataset raw dan final
    menggunakan DVC.
5.  **Git Version Control:** Menyimpan metadata `.dvc` ke Git untuk
    menghubungkan versi dataset dengan histori pengembangan.

Untuk menjalankan pembaruan dataset secara lokal melalui PowerShell:

``` powershell
.\.venv\Scripts\python.exe src/data/ingest_data.py
.\.venv\Scripts\python.exe src/data/preprocess.py
.\.venv\Scripts\python.exe src/features/build_features.py

.\.venv\Scripts\dvc.exe add data/raw/BBCA_raw.csv
.\.venv\Scripts\dvc.exe add data/final/BBCA_features.csv

git add data/raw/BBCA_raw.csv.dvc data/final/BBCA_features.csv.dvc
git commit -m "feat(dvc): update BBCA dataset"
```

Setiap pembaruan hanya dicatat sebagai versi baru setelah perubahan
dataset diverifikasi.

### 3. Hasil Implementasi Versioning

Implementasi LK-05 berhasil menghasilkan dua versi dataset:

| Parameter         | Versi 1 (V1)            | Versi 2 (V2)   |
|-------------------|-------------------------|----------------|
| Tanggal terakhir  | 25 September 2026       | 9 Oktober 2026 |
| Jumlah raw data   | 250 baris               | 250 baris      |
| Jumlah final data | 230 baris               | 230 baris      |
| Hash raw (MD5)    | `cc774d62...`           | `be4e1b69...`  |
| Hash final (MD5)  | `44b016fe...`           | `b2b54b88...`  |
| Git Commit        | `fb3a664`               | `2e2ed64`      |
| Git Tag           | `lk05-data-v1-complete` | `lk05-data-v2` |

Pembaruan dataset menghasilkan 10 tanggal perdagangan baru dan
mengeluarkan 10 tanggal terlama. Jumlah observasi tetap 250 karena
pipeline menerapkan mekanisme sliding window.

### 4. Audit Perubahan Dataset

Perbandingan kedua versi dilakukan menggunakan:

``` powershell
.\.venv\Scripts\dvc.exe diff --show-hash lk05-data-v1-complete lk05-data-v2
```

Hasil audit:

``` text
Modified:
    44b016fe..b2b54b88  data\final\BBCA_features.csv
    cc774d62..be4e1b69  data\raw\BBCA_raw.csv

files summary: 2 modified
```

Hasil tersebut menunjukkan bahwa DVC berhasil mengidentifikasi perubahan
pada dataset raw dan final, serta mencatat perbedaan hash antara kedua
versi.

### 5. Reproducibility

Penerapan DVC memungkinkan setiap versi dataset diidentifikasi melalui
kombinasi Git commit, metadata `.dvc`, dan konten dataset yang tersimpan
di cache atau remote storage.

Versi V1 dan V2 dapat ditelusuri melalui Git tag untuk mendukung proses
audit dan pengujian ulang model pada tahap pengembangan berikutnya.

Git menyimpan metadata, bukan seluruh konten dataset. Oleh karena itu,
pemulihan dataset membutuhkan DVC cache atau remote storage yang masih
menyimpan versi terkait.

Konfigurasi DVC remote merupakan mekanisme tambahan untuk melakukan
backup dan sinkronisasi dataset. Penggunaan remote berbasis direktori
lokal dapat mendukung pengujian mekanisme push dan pull, sedangkan
penyimpanan cloud diperlukan untuk akses lintas perangkat tanpa
bergantung pada laptop pengembangan.

### Pengujian DVC Remote Storage

Pada LK-05, DVC dikonfigurasi menggunakan *local filesystem remote* yang
disimpan di luar direktori repositori Git.

Pengujian dilakukan melalui perintah berikut:

``` powershell
.\.venv\Scripts\dvc.exe push --all-tags
.\.venv\Scripts\dvc.exe status --cloud --all-tags
.\.venv\Scripts\dvc.exe pull
```

Hasil pengujian menunjukkan bahwa empat objek data berhasil dikirim ke
remote, status cache dan remote telah tersinkronisasi, serta perintah
`dvc pull` berhasil dijalankan.

Konfigurasi remote disimpan secara lokal melalui `.dvc/config.local`
sehingga tidak dibagikan melalui Git. Implementasi ini memvalidasi
mekanisme sinkronisasi dataset, sedangkan penggunaan cloud storage dan
pengujian pemulihan lintas perangkat merupakan pengembangan lanjutan.

## Alur Pengembangan (GitHub Flow)

Pengembangan repositori ini menerapkan aturan standar **GitHub Flow**:

1.  **Branching:** Membuat *branch* fitur/eksperimen baru dari `main`
    (contoh: `git switch -c feat/initial-eda`).
2.  **Development & Commit:** Melakukan perubahan kode dan menyimpan
    *commit* dengan pesan yang informatif.
3.  **Pull Request (PR):** Mengunggah *branch* ke GitHub dan membuat PR
    ke branch `main`.
4.  **Merge:** Melakukan *merge* ke branch `main` setelah seluruh
    pengujian terverifikasi.

## Model Aktif untuk Inferensi

Berdasarkan hasil evaluasi dan proses registrasi menggunakan MLflow
Model Registry, model yang saat ini digunakan untuk inferensi adalah
**Random Forest Version 1** dari registered model
`bbca-direction-classifier`.

Model Version 1 dipilih karena menghasilkan performa evaluasi validation
yang lebih baik dibandingkan Version 2.

| Metrik    | Version 1 | Version 2 |
|-----------|----------:|----------:|
| Accuracy  |    0.6364 |    0.6061 |
| Precision |    1.0000 |    1.0000 |
| Recall    |    0.2000 |    0.1333 |
| F1-score  |    0.3333 |    0.2353 |

Version 1 memiliki nilai accuracy, recall, dan F1-score yang lebih
tinggi, sementara precision kedua versi sama. Oleh karena itu, Version 1
ditetapkan sebagai model Production dan menjadi versi aktif yang
digunakan untuk proses inferensi.

Informasi model aktif:

- **Registered Model:** `bbca-direction-classifier`
- **Active Version:** `1`
- **Stage:** `Production`
- **Status:** `READY`
- **Model URI:** `models:/bbca-direction-classifier/1`
- **Dataset:** `lk05-data-v2`

Model dapat dimuat kembali melalui MLflow menggunakan Model URI
tersebut. Pengujian inferensi telah berhasil dilakukan menggunakan lima
sampel input dengan sepuluh fitur.

------------------------------------------------------------------------

------------------------------------------------------------------------

## LK-08

### Alur Otomatisasi

Workflow `.github/workflows/mlops-automation.yaml` memiliki tiga job:

1.  **Automated Testing:** menjalankan unit test menggunakan pytest.
2.  **Training and Evaluation:** mengambil dataset dari DVC remote,
    melatih Random Forest, mencatat eksperimen ke MLflow, dan menghitung
    F1-score.
3.  **Model Registry:** mendaftarkan kandidat ke Staging hanya jika
    F1-score memenuhi threshold `0.58`.

Workflow berjalan pada push ke branch yang dikonfigurasi dan pull
request menuju `main`. Pada pull request, hanya job Testing yang
dijalankan. Kredensial DVC dan MLflow disimpan sebagai GitHub Secret
`DAGSHUB_TOKEN`.

### Hasil Pengujian

Eksekusi pertama pada 8 November 2026 menghasilkan:

| Komponen                | Hasil     |
|-------------------------|-----------|
| Automated Testing       | SUCCESS   |
| Training and Evaluation | SUCCESS   |
| Model Registry          | SUCCESS   |
| Unit test               | 14 PASSED |
| Validation F1-score     | 0.1250    |
| Minimum F1-score        | 0.5800    |
| Keputusan model         | REJECTED  |
| Registrasi model baru   | SKIPPED   |

Seluruh job berhasil dijalankan, tetapi kandidat model ditolak karena
F1-score tidak mencapai threshold. Status SUCCESS pada job Registry
berarti proses validasi berjalan tanpa error, bukan berarti model
berhasil didaftarkan.

Model Production Version 1 pada registry lokal LK-07 tetap
dipertahankan. Registry DagsHub merupakan lingkungan terpisah dan tidak
otomatis mewarisi model dari registry lokal.

### Catatan Evaluasi

Dataset baseline berisi 230 observasi dengan validation set 33
observasi. Ukuran data tersebut masih membatasi evaluasi performa.
Threshold `0.58` tetap dipertahankan agar model yang belum memenuhi
kriteria tidak dipaksakan masuk Staging.

Hasil LK-08 menunjukkan bahwa keberhasilan pipeline secara teknis harus
dibedakan dari kelayakan model berdasarkan metrik evaluasi.

------------------------------------------------------------------------

## Menjalankan Seluruh Sistem dengan Docker Compose

Pastikan Docker Desktop aktif dan terminal berada di direktori utama
proyek. Jalankan satu perintah berikut untuk membangun image serta
menjalankan MLflow Server dan FastAPI:

``` powershell
docker compose up -d --build
```

Periksa status layanan:

``` powershell
docker compose ps
```

Setelah kedua layanan berstatus `healthy`, akses:

- MLflow: `http://localhost:5055`
- Dokumentasi API: `http://localhost:8000/docs`

Untuk menghentikan layanan:

``` powershell
docker compose down
```


------------------------------------------------------------------------

Serving Model dan Simulasi Scaling

Pengujian dilakukan di Windows melalui VS Code PowerShell dan Docker Desktop.
Model Production yang digunakan adalah `bbca-direction-classifier` versi `1`.
MLflow menyimpan model dalam image `mlops-bbca:lk10`.
Service `model-api` menjalankan tiga replika secara default melalui
`deploy.replicas`, dengan satu worker per replika.

### Membangun image model

Jalankan dari direktori utama proyek dengan Docker Desktop aktif dan CLI MLflow
tersedia di lingkungan Python lokal. Registry LK-09 harus dapat mengakses
`mlflow.db` dan `mlartifacts/` milik proyek.

Jika hanya perlu menjalankan layanan LK-09, gunakan
`docker compose up -d --build mlflow-server api-service`.
Jalankan `model-api` setelah image LK-10 tersedia.

```powershell
docker compose up -d --wait mlflow-server
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5055'
mlflow models build-docker -m 'models:/bbca-direction-classifier/1' -n 'mlops-bbca:lk10' --env-manager local
```

Pada pengujian LK-10, build pertama mengalami timeout saat mengunduh PyArrow.
Build yang berhasil menggunakan image LK-09 yang sudah memiliki dependensi
dan memasang timeout pip yang lebih panjang. Parameter `base_image` diberikan
melalui API Python MLflow karena CLI lokal tidak menyediakan `--base-image`.

```powershell
# Pastikan image mlops-bbca:lk09 sudah ada. Bila belum, jalankan
# docker compose build mlflow-server
@'
FROM mlops-bbca:lk09
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8 DISABLE_NGINX=true
'@ | docker build -t mlops-bbca:lk10-base -

$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5055'
python -c "from mlflow.models import build_docker; build_docker(model_uri='models:/bbca-direction-classifier/1', name='mlops-bbca:lk10', env_manager='local', base_image='mlops-bbca:lk10-base')"
```

### Menjalankan service dan mengakses endpoint

```powershell
docker compose config --quiet
docker compose up -d --wait model-api
docker compose ps model-api
```

Setiap replika mendengarkan di port container `8080`. Compose memilih port
Windows yang tersedia dari rentang `18081–18099` dan mengikatnya ke
`127.0.0.1`. Gunakan `docker compose port --index` untuk melihat alamat
replika saat ini. Endpoint `GET /ping` memeriksa kesiapan API, dan
`POST /invocations` menerima JSON untuk inferensi.

```powershell
$address = docker compose port --index 1 model-api 8080
$sample = '.\mlartifacts\1\models\m-73bb59608aa74fd9973bb39a64c4f070\artifacts\serving_input_example.json'
curl.exe -sS -w '\nHTTP %{http_code}\n' -H 'Content-Type: application/json' --data-binary "@$sample" "http://$address/invocations"
```

Untuk contoh lima baris tersebut, ketiga replika yang diuji mengembalikan
HTTP 200 dan `{"predictions": [1, 0, 1, 0, 1]}`.
Label `1` berarti Up dan label `0` berarti Non-Up.

### Mengubah jumlah replika

```powershell
docker compose up -d --wait --scale model-api=4 model-api
docker compose ps model-api
docker compose port --index 4 model-api 8080

docker compose up -d --wait --scale model-api=3 model-api
docker compose ps model-api
```

Opsi `--scale` mengganti jumlah replika untuk perintah yang dijalankan.
Konfigurasi default tetap tiga. Setiap replika memerlukan satu port host
yang tersedia di rentang Compose.

------------------------------------------------------------------------

## Lisensi

Hak Cipta © 2026 **Roniarta Sibarani**. Dilisensikan di bawah [MIT
License](LICENSE).
