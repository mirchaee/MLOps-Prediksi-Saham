
"""Unit test untuk Model Validation Gate LK-08."""

import pytest

from src.models.validation import validate_model


@pytest.mark.parametrize(
    "val_f1, threshold, production_f1, expected",
    [
        (0.65, 0.58, None, True),
        (0.58, 0.58, None, True),
        (0.45, 0.58, None, False),
        (0.65, 0.58, 0.70, False),
        (0.65, 0.58, 0.50, True),
        (0.65, 0.58, 0.65, False),
    ],
)
def test_validation_gate(
    val_f1, threshold, production_f1, expected
):
    """Memastikan keputusan validasi sesuai aturan."""
    passed, message = validate_model(
        val_f1=val_f1,
        threshold=threshold,
        production_f1=production_f1,
    )

    assert passed is expected
    assert message.startswith(
        "PASSED" if expected else "REJECTED"
    )


@pytest.mark.parametrize(
    "invalid_f1",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_f1_rejected(invalid_f1):
    """Nilai F1 di luar rentang valid harus ditolak."""
    with pytest.raises(ValueError):
        validate_model(val_f1=invalid_f1)


def test_production_comparison():
    """Model yang lebih buruk dari Production ditolak."""
    passed, message = validate_model(
        val_f1=0.60,
        threshold=0.58,
        production_f1=0.70,
    )

    assert passed is False
    assert "Production" in message
