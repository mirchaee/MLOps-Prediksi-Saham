
import numpy as np

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.train import load_data, split_data


THRESHOLDS = (0.35, 0.40, 0.45, 0.50)

CANDIDATES = {
    "RF_100_depth4": RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "RF_500_depth10": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        ),
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        min_samples_leaf=8,
        learning_rate=0.05,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    ),
}


def cross_validate_model(estimator, X, y):
    splitter = TimeSeriesSplit(
        n_splits=4,
        test_size=25,
        gap=1,
    )

    probabilities = np.full(len(X), np.nan)

    for train_idx, val_idx in splitter.split(X):
        model = clone(estimator)

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        positive_idx = list(model.classes_).index(1)

        probabilities[val_idx] = model.predict_proba(
            X.iloc[val_idx]
        )[:, positive_idx]

    valid = np.isfinite(probabilities)

    return y.to_numpy()[valid], probabilities[valid]


def compare_models():
    df, features, dataset_hash = load_data()

    X_train, y_train, _, _, _, _ = split_data(
        df, features
    )

    results = []
    baseline_f1 = None

    for name, estimator in CANDIDATES.items():
        y_true, probabilities = cross_validate_model(
            estimator,
            X_train,
            y_train,
        )

        if baseline_f1 is None:
            baseline_predictions = np.ones(
                len(y_true), dtype=int
            )

            baseline_f1 = f1_score(
                y_true,
                baseline_predictions,
                zero_division=0,
            )

        for threshold in THRESHOLDS:
            predictions = (
                probabilities >= threshold
            ).astype(int)

            results.append({
                "model": name,
                "threshold": threshold,
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "predicted_up": int(predictions.sum()),
                "samples": len(y_true),
            })

    results.sort(
        key=lambda row: (
            row["balanced_accuracy"],
            row["f1"],
        ),
        reverse=True,
    )

    print("\n=== MODEL COMPARISON ===")
    print(f"Dataset MD5: {dataset_hash}")
    print(f"Training observations: {len(X_train)}")
    print(f"Always-Up baseline F1: {baseline_f1:.4f}")
    print("Always-Up baseline balanced accuracy: 0.5000")

    print(
        "\nModel | Threshold | F1 | "
        "Balanced Acc | Precision | Recall | Pred Up"
    )

    for row in results:
        print(
            f"{row['model']} | "
            f"{row['threshold']:.2f} | "
            f"{row['f1']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['predicted_up']}/{row['samples']}"
        )

    return results


if __name__ == "__main__":
    compare_models()
