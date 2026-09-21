import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

TICKER = "BBCA.JK"
WINDOW_DAYS = 250  # sliding window ~250 hari perdagangan terakhir


def fetch_stock_data(ticker: str = TICKER, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Menarik data historis harian dari Yahoo Finance API dalam rentang sliding window."""
    end_date = datetime.now()
    # dilebihkan ke kalender karena window_days dihitung hari perdagangan, bukan hari kalender
    start_date = end_date - timedelta(days=int(window_days * 1.6))

    print(f"=== [INGESTION] Menarik data {ticker} ({window_days} hari perdagangan terakhir) ===")

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )

    if df.empty:
        raise ValueError(f"[ERROR] Gagal menarik data untuk ticker {ticker}!")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # Ambil window_days baris paling akhir saja
    df = df.tail(window_days).reset_index(drop=True)

    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Nama file konsisten dengan tanggal snapshot + file "latest" untuk dipakai tahap berikutnya
    date_tag = end_date.strftime("%Y%m%d")
    snapshot_path = os.path.join(raw_dir, f"BBCA_raw_{date_tag}.csv")
    latest_path = os.path.join(raw_dir, "BBCA_raw.csv")

    df.to_csv(snapshot_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"[SUCCESS] Snapshot disimpan ke: {snapshot_path}")
    print(f"[SUCCESS] File latest diperbarui: {latest_path}")
    print(f"Total Baris Data: {len(df)}")
    print("\n--- Cuplikan 5 Data Terakhir ---")
    print(df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail())

    return df


if __name__ == "__main__":
    fetch_stock_data()
