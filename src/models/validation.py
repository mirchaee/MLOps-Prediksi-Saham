
"""Validasi performa model sebelum registrasi ke MLflow Staging."""

import math


def validate_model(
    val_f1: float,
    threshold: float = 0.58,
    production_f1: float | None = None,
) -> tuple[bool, str]:
    """
    Memvalidasi F1-score model baru.

    Model dinyatakan lolos jika:
    1. F1-score >= threshold.
    2. Jika baseline Production tersedia dan sebanding,
       F1-score baru harus lebih tinggi.
    """

    def check_score(value, name):
        try:
            score = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} harus berupa angka.")

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{name} harus berada dalam rentang 0–1.")

        return score

    val_f1 = check_score(val_f1, "Validation F1")
    threshold = check_score(threshold, "Threshold")

    if val_f1 < threshold:
        return (
            False,
            f"REJECTED: F1 {val_f1:.4f} < threshold {threshold:.4f}",
        )

    if production_f1 is not None:
        production_f1 = check_score(
            production_f1, "Production F1"
        )

        if val_f1 <= production_f1:
            return (
                False,
                f"REJECTED: F1 {val_f1:.4f} tidak lebih baik "
                f"dari Production {production_f1:.4f}",
            )

    return (
        True,
        f"PASSED: F1 {val_f1:.4f} memenuhi kriteria validasi.",
    )
