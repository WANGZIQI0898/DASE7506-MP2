"""Evaluate PCA-regularized listwise selectors on train/validation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, read_jsonl
from train_v2 import evaluate_predictions, train_listwise, unique_answer_counts
from train_verifier import build_labels
from verifier_features import FEATURE_NAMES, build_structural_features


def main() -> None:
    tick = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_references = read_jsonl(CODE_DIR / "data" / "gsm8k_train.jsonl")
    validation_references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")
    train_labels = build_labels(train_bank, train_references)
    validation_labels = build_labels(validation_bank, validation_references)
    train_answer_counts = unique_answer_counts(train_bank)
    validation_answer_counts = unique_answer_counts(validation_bank)
    train_embeddings = train_arrays["embeddings"]
    validation_embeddings = validation_arrays["embeddings"]
    train_structural = build_structural_features(train_bank, train_embeddings)
    validation_structural = build_structural_features(
        validation_bank, validation_embeddings
    )
    flat_train = train_embeddings.reshape(len(train_bank) * 4, -1)
    flat_validation = validation_embeddings.reshape(len(validation_bank) * 4, -1)

    records = []
    best = None
    for components in (32, 64, 128, 256):
        model_tick = time.perf_counter()
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            whiten=True,
            random_state=20261026,
        )
        pca_train = pca.fit_transform(flat_train).astype(np.float32)
        pca_validation = pca.transform(flat_validation).astype(np.float32)
        train_features = np.concatenate(
            (pca_train, train_structural.reshape(len(train_bank) * 4, -1)),
            axis=1,
        ).reshape(len(train_bank), 4, -1)
        validation_features = np.concatenate(
            (
                pca_validation,
                validation_structural.reshape(len(validation_bank) * 4, -1),
            ),
            axis=1,
        ).reshape(len(validation_bank), 4, -1)
        component_mean = train_features.mean(axis=(0, 1), keepdims=True)
        component_scale = np.maximum(
            train_features.std(axis=(0, 1), keepdims=True), 1e-6
        )
        train_features = (train_features - component_mean) / component_scale
        validation_features = (
            validation_features - component_mean
        ) / component_scale
        weights = train_listwise(
            train_features,
            train_labels,
            (train_answer_counts > 1).astype(np.float64),
            regularization=0.001,
            epochs=80,
            learning_rate=0.02,
            seed=20261026,
        )
        scores = np.einsum("qcf,f->qc", validation_features, weights)
        evaluation = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            scores.argmax(axis=1),
            validation_answer_counts,
        )
        record = {
            "variant": f"pca_listwise_{components}",
            "components": components,
            "accuracy": evaluation["accuracy"],
            "correct": evaluation["correct"],
            "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": evaluation[
                "accuracy_by_unique_answers"
            ],
            "training_seconds": round(time.perf_counter() - model_tick, 4),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if best is None or (
            record["non_unanimous_accuracy"], record["accuracy"]
        ) > (
            best["record"]["non_unanimous_accuracy"],
            best["record"]["accuracy"],
        ):
            best = {
                "record": record,
                "pca": pca,
                "mean": component_mean,
                "scale": component_scale,
                "weights": weights,
            }

    model_dir = CODE_DIR / "runs" / "pca_listwise_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "model.npz",
        pca_mean=best["pca"].mean_,
        pca_components=best["pca"].components_,
        pca_scale=best["pca"].explained_variance_**0.5,
        feature_mean=best["mean"].reshape(-1),
        feature_scale=best["scale"].reshape(-1),
        weights=best["weights"],
    )
    result = {
        "split": "validation",
        "test_used": False,
        "records": records,
        "selected_variant": best["record"]["variant"],
        "structural_feature_names": list(FEATURE_NAMES),
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "pca-listwise-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
