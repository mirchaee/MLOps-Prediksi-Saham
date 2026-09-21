import os

import pandas as pd

PROCESSED_PATH = os.path.join("data", "processed", "BBCA_processed.csv")
FINAL_DIR = os.path.join("data", "final")
FINAL_PATH = os.path.join(FINAL_DIR, "BBCA_features.csv")


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(close: pd.Series, window: int = 20, num_std: int = 2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def build_features(input_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Menghitung indikator teknikal dan target biner. Semua fitur hanya pakai data hingga hari T
    (tidak melihat T+1) untuk mencegah data leakage."""
    print(f"=== [FEATURE ENGINEERING] Memproses {input_path} ===")

    df = pd.read_csv(input_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["RSI_14"] = compute_rsi(df["Close"], period=14)
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Lower"] = compute_bollinger_bands(df["Close"])

    # Target biner: 1 = Up jika Close(T+1) > Close(T), dihitung dari shift(-1) lalu dibuang barisnya
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Baris terakhir tidak punya Close(T+1) -> buang agar tidak ada label palsu
    df = df.iloc[:-1]

    # Baris awal yang indikatornya masih NaN (belum cukup window) juga dibuang
    df = df.dropna().reset_index(drop=True)

    os.makedirs(FINAL_DIR, exist_ok=True)
    df.to_csv(FINAL_PATH, index=False)

    print(f"[SUCCESS] Dataset final disimpan ke: {FINAL_PATH}")
    print(f"Total Baris Data Final: {len(df)}")
    print(f"Distribusi Target -> Up: {df['Target'].sum()} | Non-Up: {len(df) - df['Target'].sum()}")

    return df

if __name__ == "__main__":
    build_features()
