
import pandas as pd
import pytest

from src.models.train import split_data


@pytest.fixture
def sample_data():
    """Dataset sintetis untuk pengujian tanpa data saham asli."""
    return pd.DataFrame({
        "feature": range(100),
        "Target": [i % 2 for i in range(100)],
    })


def test_chronological_split(sample_data):
    """Memastikan pembagian train, validation, test benar."""
    X_train, y_train, X_val, y_val, X_test, y_test = (
        split_data(sample_data, ["feature"])
    )

    assert len(X_train) == 69
    assert len(X_val) == 14
    assert len(X_test) == 15

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_no_data_leakage(sample_data):
    """Memastikan tidak ada overlap dan terdapat gap T+1."""
    X_train, _, X_val, _, X_test, _ = (
        split_data(sample_data, ["feature"])
    )

    assert X_train["feature"].max() == 68
    assert X_val["feature"].min() == 70

    assert X_val["feature"].max() == 83
    assert X_test["feature"].min() == 85

    assert X_train["feature"].max() < X_val["feature"].min()
    assert X_val["feature"].max() < X_test["feature"].min()


def test_small_dataset_rejected():
    """Dataset terlalu kecil harus menghasilkan error."""
    df = pd.DataFrame({
        "feature": range(20),
        "Target": [i % 2 for i in range(20)],
    })

    with pytest.raises(ValueError, match="terlalu sedikit"):
        split_data(df, ["feature"])
