import os

import pandas as pd

RAW_PATH = os.path.join("data", "raw", "BBCA_raw.csv")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "BBCA_processed.csv")


def clean_data(input_path: str = RAW_PATH) -> pd.DataFrame:
    """Membersihkan data mentah: null-check, duplicate-date check, sort chronological."""
    print(f"=== [CLEANING] Memproses {input_path} ===")

    df = pd.read_csv(input_path, parse_dates=["Date"])

    # 1. Null-check pada kolom OHLCV
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    before = len(df)
    df = df.dropna(subset=ohlcv_cols)
    print(f"[NULL-CHECK] {before - len(df)} baris dengan nilai kosong dihapus")

    # 2. Duplicate-date check
    before = len(df)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    print(f"[DUPLICATE-CHECK] {before - len(df)} baris duplikat tanggal dihapus")

    # 3. Sort chronological
    df = df.sort_values("Date").reset_index(drop=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"[SUCCESS] Data bersih disimpan ke: {PROCESSED_PATH}")
    print(f"Total Baris Data Bersih: {len(df)}")

    return df


if __name__ == "__main__":
    clean_data()
