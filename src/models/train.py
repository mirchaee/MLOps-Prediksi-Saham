
"""LK-08: Automated Training BBCA dengan DVC dan MLflow."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

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
DVC_POINTER = Path("data/final/BBCA_features.csv.dvc")

EXPERIMENT_NAME = "bbca-price-direction"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5055",
)


def file_md5(path: Path) -> str:
    """Menghitung hash MD5 dataset."""
    digest = hashlib.md5()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_dvc_pointer(
    csv_path: Path = DATA_PATH,
    pointer_path: Path = DVC_POINTER,
) -> str:
    """Memastikan dataset sesuai metadata DVC aktif."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {csv_path}. "
            "Jalankan dvc pull terlebih dahulu."
        )

    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"Metadata DVC tidak ditemukan: {pointer_path}"
        )

    metadata = yaml.safe_load(
        pointer_path.read_text(encoding="utf-8")
    )

    outs = (
        metadata.get("outs", [])
        if isinstance(metadata, dict)
        else []
    )

    if (
        len(outs) != 1
        or not isinstance(outs[0], dict)
        or outs[0].get("hash", "md5") != "md5"
    ):
        raise ValueError("Struktur metadata DVC tidak valid.")

    expected_md5 = outs[0].get("md5")

    if not expected_md5:
        raise ValueError("Hash MD5 tidak ditemukan di DVC.")

    actual_md5 = file_md5(csv_path)

    if actual_md5 != expected_md5:
        raise ValueError(
            "Hash dataset tidak cocok dengan metadata DVC. "
            f"Expected: {expected_md5}, "
            f"Actual: {actual_md5}"
        )

    return actual_md5


def load_data():
    """Memuat dataset yang telah diverifikasi DVC."""

    dataset_hash = validate_dvc_pointer()

    df = pd.read_csv(DATA_PATH)

    if not {"Date", "Target"}.issubset(df.columns):
        raise ValueError(
            "Kolom Date atau Target tidak ditemukan."
        )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="raise"
    )

    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Terdapat duplikasi tanggal.")

    if df.isna().any().any():
        raise ValueError("Dataset memiliki missing values.")

    if not df["Target"].isin([0, 1]).all():
        raise ValueError("Target harus bernilai 0 atau 1.")

    df["Target"] = df["Target"].astype(int)

    feature_cols = [
        col for col in df.columns
        if col not in ["Date", "Target"]
    ]

    if not feature_cols:
        raise ValueError("Tidak ada fitur untuk training.")

    df[feature_cols] = df[feature_cols].apply(
        pd.to_numeric, errors="raise"
    )

    return df, feature_cols, dataset_hash


def split_data(df, feature_cols):
    """Split 70/15/15 dengan gap untuk label T+1."""

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end - 1]
    val_df = df.iloc[train_end:val_end - 1]
    test_df = df.iloc[val_end:]

    if min(
        len(train_df),
        len(val_df),
        len(test_df),
    ) < 10:
        raise ValueError(
            "Jumlah data untuk split terlalu sedikit."
        )

    X_train = train_df[feature_cols]
    y_train = train_df["Target"]

    X_val = val_df[feature_cols]
    y_val = val_df["Target"]

    X_test = test_df[feature_cols]
    y_test = test_df["Target"]

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )


def train_model(
    n_estimators: int,
    max_depth: int,
    output: Path = Path("outputs/training_result.json"),
):
    """Melatih model dan menyimpan hasil ke MLflow."""

    df, features, dataset_hash = load_data()

    X_train, y_train, X_val, y_val, X_test, _ = (
        split_data(df, features)
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = (
        f"RF_n{n_estimators}_depth{max_depth}_LK08"
    )

    with mlflow.start_run(run_name=run_name) as run:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = {
            "val_accuracy": float(
                accuracy_score(y_val, y_pred)
            ),
            "val_precision": float(
                precision_score(
                    y_val, y_pred, zero_division=0
                )
            ),
            "val_recall": float(
                recall_score(
                    y_val, y_pred, zero_division=0
                )
            ),
            "val_f1_score": float(
                f1_score(
                    y_val, y_pred, zero_division=0
                )
            ),
        }

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": "balanced",
            "random_state": 42,
            "dataset_md5": dataset_hash,
            "dataset_version": f"dvc-md5:{dataset_hash}",
            "dataset_dvc_pointer": str(DVC_POINTER),
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples_reserved": len(X_test),
            "feature_count": len(features),
        })

        mlflow.log_metrics(metrics)

        git_sha = os.getenv("GITHUB_SHA", "local")

        mlflow.set_tags({
            "source": "LK08-Automated-Training",
            "git_sha": git_sha,
        })

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
            image_path = (
                Path(tmpdir) / "confusion_matrix.png"
            )

            fig.savefig(image_path, dpi=150)

            mlflow.log_artifact(
                str(image_path),
                artifact_path="evaluation",
            )

        plt.close(fig)

        sample = X_train.head(5)

        signature = infer_signature(
            sample,
            model.predict(sample),
        )

        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=sample,
            serialization_format="cloudpickle",
        )

        model_uri = logged_model.model_uri

        if not model_uri:
            raise RuntimeError(
                "MLflow tidak mengembalikan model URI."
            )

        result = {
            "run_id": run.info.run_id,
            "model_uri": str(model_uri),
            "dataset_md5": dataset_hash,
            "git_sha": git_sha,
            "metrics": metrics,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }

        output.parent.mkdir(
            parents=True, exist_ok=True
        )

        output.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print("\n=== LK08 AUTOMATED TRAINING ===")
        print(f"Tracking URI : {TRACKING_URI}")
        print(f"Run ID       : {run.info.run_id}")
        print(f"Model URI    : {model_uri}")
        print(f"Dataset MD5  : {dataset_hash}")
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples  : {len(X_val)}")
        print(f"Test reserved: {len(X_test)}")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print(f"Training output: {output}")

        return result



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="LK08 Automated Training BBCA"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/training_result.json"),
    )

    args = parser.parse_args()

    train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        output=args.output,
    )
