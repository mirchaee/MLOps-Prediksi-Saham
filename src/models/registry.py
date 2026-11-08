
import argparse
import json
import math
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.models.train import validate_dvc_pointer
from src.models.validation import validate_model


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5055",
)

MODEL_NAME = "bbca-direction-classifier"
EVALUATION_PATH = Path("outputs/evaluation_result.json")
REGISTRY_PATH = Path("outputs/registry_result.json")


def register_model(
    input_path: Path = EVALUATION_PATH,
    output_path: Path = REGISTRY_PATH,
):
    evaluation = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    run_id = evaluation["run_id"]
    model_uri = evaluation["model_uri"]
    dataset_md5 = validate_dvc_pointer()

    if evaluation["dataset_md5"] != dataset_md5:
        raise ValueError("Dataset evaluasi tidak sesuai DVC.")

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise RuntimeError("MLflow run belum selesai.")

    if run.data.params.get("dataset_md5") != dataset_md5:
        raise ValueError("Dataset MLflow tidak sesuai DVC.")

    val_f1 = run.data.metrics.get("val_f1_score")

    if val_f1 is None:
        raise ValueError("F1-score tidak ditemukan di MLflow.")

    if not math.isclose(
        float(val_f1),
        float(evaluation["val_f1_score"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Metrik evaluasi tidak sesuai MLflow.")

    threshold = float(evaluation["threshold"])

    passed, message = validate_model(
        val_f1=float(val_f1),
        threshold=threshold,
    )

    expected_status = "PASSED" if passed else "REJECTED"

    if (
        evaluation.get("passed") is not passed
        or evaluation.get("status") != expected_status
    ):
        raise ValueError("Keputusan evaluasi tidak konsisten.")

    result = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "model_uri": model_uri,
        "dataset_md5": dataset_md5,
        "val_f1_score": float(val_f1),
        "threshold": threshold,
        "registered": False,
        "version": None,
        "stage": None,
        "status": "SKIPPED",
    }

    if passed:
        if not model_uri.startswith("models:/m-"):
            raise ValueError("Model URI MLflow 3 tidak valid.")

        model_id = model_uri.removeprefix("models:/")
        logged_model = client.get_logged_model(model_id)

        if logged_model.source_run_id != run_id:
            raise ValueError(
                "Artefak model tidak berasal dari run evaluasi."
            )

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_registry_uri(TRACKING_URI)

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
            await_registration_for=120,
        )

        version = registered.version

        transitioned = client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )

        if transitioned.current_stage != "Staging":
            raise RuntimeError(
                "Transisi model ke Staging gagal."
            )

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "dataset_md5",
            dataset_md5,
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "validation_f1",
            str(val_f1),
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version,
            "git_sha",
            run.data.tags.get("git_sha", "unknown"),
        )

        result.update({
            "registered": True,
            "version": str(version),
            "stage": "Staging",
            "status": "REGISTERED",
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== LK08 MODEL REGISTRY ===")
    print(f"Run ID       : {run_id}")
    print(f"Model        : {MODEL_NAME}")
    print(f"F1-score     : {val_f1:.4f}")
    print(f"Threshold    : {threshold:.4f}")
    print(f"Status       : {result['status']}")
    print(f"Version      : {result['version']}")
    print(f"Stage        : {result['stage']}")
    print(message)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LK08 Automated Model Registry"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=EVALUATION_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=REGISTRY_PATH,
    )

    args = parser.parse_args()

    register_model(
        input_path=args.input,
        output_path=args.output,
    )
