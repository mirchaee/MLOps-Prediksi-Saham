
import argparse
import hashlib
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DATA_PATH = Path("data/final/BBCA_features.csv")
EXPERIMENT_NAME = "bbca-price-direction"
TRACKING_URI = "http://127.0.0.1:5055"
DATASET_VERSION = "lk05-data-v2"

EXPECTED_MD5 = "b2b54b8853b623e24328369d34d31128"


def load_data():
    """Memuat dan memvalidasi dataset dari LK-05."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    dataset_hash = hashlib.md5(
        DATA_PATH.read_bytes()
    ).hexdigest()

    if dataset_hash != EXPECTED_MD5:
        raise ValueError(
            "Hash dataset berbeda dari V2 LK-05. "
            "Periksa DVC sebelum training."
        )

    df = pd.read_csv(DATA_PATH)

    if not {"Date", "Target"}.issubset(df.columns):
        raise ValueError("Kolom Date atau Target tidak ditemukan.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Terdapat duplikasi tanggal.")

    if df.isna().any().any():
        raise ValueError("Dataset memiliki missing values.")

    if not df["Target"].isin([0, 1]).all():
        raise ValueError("Target harus berupa kelas 0 atau 1.")

    df["Target"] = df["Target"].astype(int)

    feature_cols = [
        col for col in df.columns
        if col not in ["Date", "Target"]
    ]

    df[feature_cols] = df[feature_cols].apply(
        pd.to_numeric, errors="raise"
    )

    return df, feature_cols, dataset_hash


def split_data(df, feature_cols):
    """Membagi data secara kronologis tanpa random shuffle."""

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Satu baris gap pada tiap batas split
    # untuk menghindari kebocoran label T+1.
    train_df = df.iloc[:train_end - 1]
    val_df = df.iloc[train_end:val_end - 1]
    test_df = df.iloc[val_end:]

    if min(len(train_df), len(val_df), len(test_df)) < 10:
        raise ValueError("Jumlah data untuk split terlalu sedikit.")

    X_train = train_df[feature_cols]
    y_train = train_df["Target"]

    X_val = val_df[feature_cols]
    y_val = val_df["Target"]

    X_test = test_df[feature_cols]
    y_test = test_df["Target"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(n_estimators, max_depth):
    """Melatih model dan mencatat eksperimen ke MLflow."""

    df, features, dataset_hash = load_data()

    X_train, y_train, X_val, y_val, X_test, y_test = (
        split_data(df, features)
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"RF_n{n_estimators}_depth{max_depth}"

    with mlflow.start_run(run_name=run_name) as run:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": "balanced",
            "random_state": 42,
            "dataset_version": DATASET_VERSION,
            "dataset_md5": dataset_hash,
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples_reserved": len(X_test),
            "feature_count": len(features),
        })

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = {
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_precision": precision_score(
                y_val, y_pred, zero_division=0
            ),
            "val_recall": recall_score(
                y_val, y_pred, zero_division=0
            ),
            "val_f1_score": f1_score(
                y_val, y_pred, zero_division=0
            ),
        }

        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_val,
            y_pred,
            labels=[0, 1],
            display_labels=["Non-Up", "Up"],
            ax=ax,
            colorbar=False,
        )

        ax.set_title("Validation Confusion Matrix")
        fig.tight_layout()

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "confusion_matrix.png"
            fig.savefig(image_path, dpi=150)
            mlflow.log_artifact(
                str(image_path),
                artifact_path="evaluation"
            )

        plt.close(fig)

        sample = X_train.head(5)
        signature = infer_signature(
            sample, model.predict(sample)
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=sample,
            serialization_format="cloudpickle",
        )

        print("\n=== HASIL TRAINING ===")
        print(f"Run ID        : {run.info.run_id}")
        print(f"Model         : Random Forest")
        print(f"Dataset       : {DATASET_VERSION}")
        print(f"Train samples : {len(X_train)}")
        print(f"Val samples   : {len(X_val)}")
        print(f"Test reserved : {len(X_test)}")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nEksperimen berhasil dicatat di MLflow.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators", type=int, default=100
    )

    parser.add_argument(
        "--max_depth", type=int, default=4
    )

    args = parser.parse_args()

    train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
