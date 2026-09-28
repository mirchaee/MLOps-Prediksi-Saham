"""Skrip pembersihan (preprocessing) data mentah hasil ingestion.

Membersihkan data OHLCV mentah agar konsisten dan siap masuk ke
tahap ekstraksi fitur (feature engineering): null-check, pengecekan
duplikasi tanggal, dan pengurutan kronologis.
"""

import os

import pandas as pd

RAW_PATH = os.path.join("data", "raw", "BBCA_raw.csv")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "BBCA_processed.csv")

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_data(input_path: str = RAW_PATH) -> pd.DataFrame:
    """Membersihkan data mentah dari `input_path`.

    Tahapan: (1) null-check pada kolom OHLCV, (2) duplicate-date
    check, (3) pengurutan kronologis berdasarkan tanggal. Hasil
    disimpan ke `data/processed/BBCA_processed.csv`.
    """
    print(f"=== [PREPROCESSING] Memproses {input_path} ===")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"[ERROR] File input tidak ditemukan: {input_path}. "
            "Jalankan ingest_data.py terlebih dahulu."
        )

    df = pd.read_csv(input_path, parse_dates=["Date"])

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    removed_null = before - len(df)
    print(f"[NULL-CHECK] {removed_null} baris nilai kosong dihapus")

    before = len(df)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    removed_dup = before - len(df)
    print(f"[DUPLICATE-CHECK] {removed_dup} baris duplikat dihapus")

    df = df.sort_values("Date").reset_index(drop=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"[SUCCESS] Data bersih disimpan ke: {PROCESSED_PATH}")
    print(f"Total Baris Data Bersih: {len(df)}")

    return df


if __name__ == "__main__":
    clean_data()