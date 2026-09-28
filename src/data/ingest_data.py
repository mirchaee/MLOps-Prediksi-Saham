"""Skrip ingestion data harga saham harian dari Yahoo Finance API.

Mendukung pengambilan data secara berkala (non-destruktif, snapshot
diberi timestamp) serta penanganan error koneksi dengan mekanisme
retry sederhana.
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

TICKER = "BBCA.JK"
WINDOW_DAYS = 250  
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def download_with_retry(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """Menarik data dari yfinance dengan retry saat terjadi error koneksi.

    Mencoba ulang hingga `max_retries` kali dengan jeda antar percobaan
    apabila terjadi error jaringan/timeout, agar skrip tetap tangguh
    saat dijalankan otomatis (mis. lewat GitHub Actions CRON).
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INGESTION] Percobaan {attempt}/{max_retries}...")
            df = yf.download(ticker, start=start, end=end)
            if df.empty:
                raise ValueError(f"Data kosong untuk ticker {ticker}")
            return df
        except Exception as err:  # noqa: BLE001
            last_error = err
            print(f"[WARNING] Gagal mengambil data: {err}")
            if attempt < max_retries:
                print(f"Mencoba lagi dalam {RETRY_DELAY_SECONDS} detik...")
                time.sleep(RETRY_DELAY_SECONDS)

    raise ConnectionError(
        f"[ERROR] Gagal menarik data {ticker} setelah "
        f"{max_retries} percobaan. Error terakhir: {last_error}"
    )


def fetch_stock_data(
    ticker: str = TICKER,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Menarik data historis harian dari Yahoo Finance API.

    Data disimpan dua kali: sebagai snapshot bertimestamp (immutable,
    tidak menimpa data lama) dan sebagai file "latest" untuk dipakai
    tahap pemrosesan berikutnya.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(window_days * 1.6))

    print(
        f"=== [INGESTION] Menarik data {ticker} "
        f"({window_days} hari perdagangan terakhir) ==="
    )

    df = download_with_retry(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    df = df.tail(window_days).reset_index(drop=True)
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    date_tag = end_date.strftime("%Y%m%d")
    snapshot_path = os.path.join(raw_dir, f"BBCA_raw_{date_tag}.csv")
    latest_path = os.path.join(raw_dir, "BBCA_raw.csv")

    df.to_csv(snapshot_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"[SUCCESS] Snapshot disimpan ke: {snapshot_path}")
    print(f"[SUCCESS] File latest diperbarui: {latest_path}")
    print(f"Total Baris Data: {len(df)}")
    print("\n--- Cuplikan 5 Data Terakhir ---")
    print(
        df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail()
    )

    return df


if __name__ == "__main__":
    fetch_stock_data()