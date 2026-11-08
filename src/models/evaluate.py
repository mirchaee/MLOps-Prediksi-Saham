
import argparse
import json
import math
import os
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.models.train import validate_dvc_pointer
from src.models.validation import validate_model


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5055",
)

TRAINING_RESULT = Path("outputs/training_result.json")
EVALUATION_RESULT = Path("outputs/evaluation_result.json")


def evaluate_training(
    input_path: Path = TRAINING_RESULT,
    output_path: Path = EVALUATION_RESULT,
    threshold: float = 0.58,
):
    """Memvalidasi hasil training berdasarkan metrik MLflow."""

    training = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    run_id = training["run_id"]
    dataset_md5 = validate_dvc_pointer()

    if training["dataset_md5"] != dataset_md5:
        raise ValueError("Dataset training tidak sesuai DVC.")

    client = MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"MLflow run belum selesai: {run.info.status}"
        )

    if run.data.params.get("dataset_md5") != dataset_md5:
        raise ValueError(
            "Dataset pada MLflow tidak sesuai DVC."
        )

    val_f1 = run.data.metrics.get("val_f1_score")

    if val_f1 is None:
        raise ValueError(
            "Metrik val_f1_score tidak ditemukan."
        )

    recorded_f1 = training["metrics"]["val_f1_score"]

    if not math.isclose(
        float(val_f1),
        float(recorded_f1),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Metrik training tidak sesuai dengan MLflow."
        )

    passed, message = validate_model(
        val_f1=float(val_f1),
        threshold=threshold,
    )

    result = {
        "run_id": run_id,
        "model_uri": training["model_uri"],
        "dataset_md5": dataset_md5,
        "val_f1_score": float(val_f1),
        "threshold": threshold,
        "passed": passed,
        "status": "PASSED" if passed else "REJECTED",
        "message": message,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    client.set_tag(
        run_id,
        "lk08.validation_status",
        result["status"],
    )

    github_output = os.getenv("GITHUB_OUTPUT")

    if github_output:
        with Path(github_output).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                f"gate_passed={str(passed).lower()}\n"
            )

    print("\n=== LK08 MODEL EVALUATION ===")
    print(f"Run ID       : {run_id}")
    print(f"Dataset MD5  : {dataset_md5}")
    print(f"F1-score     : {val_f1:.4f}")
    print(f"Threshold    : {threshold:.4f}")
    print(f"Status       : {result['status']}")
    print(message)
    print(f"Evaluation   : {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LK08 Automated Model Evaluation"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=TRAINING_RESULT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATION_RESULT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.58,
    )

    args = parser.parse_args()

    evaluate_training(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
    )
