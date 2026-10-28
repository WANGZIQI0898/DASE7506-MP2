"""Train a low-dimensional nonlinear candidate verifier on hard training groups."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, read_jsonl
from train_v2 import evaluate_predictions, unique_answer_counts
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

    pca = PCA(
        n_components=128,
        whiten=True,
        svd_solver="randomized",
        random_state=20261026,
    )
    pca_train = pca.fit_transform(
        train_embeddings.reshape(len(train_bank) * 4, -1)
    )
    pca_validation = pca.transform(
        validation_embeddings.reshape(len(validation_bank) * 4, -1)
    )
    train_x = np.concatenate(
        (pca_train, train_structural.reshape(len(train_bank) * 4, -1)), axis=1
    ).reshape(len(train_bank), 4, -1)
    validation_x = np.concatenate(
        (
            pca_validation,
            validation_structural.reshape(len(validation_bank) * 4, -1),
        ),
        axis=1,
    ).reshape(len(validation_bank), 4, -1)
    active = train_answer_counts > 1
    flat_train_x = train_x[active].reshape(-1, train_x.shape[2])
    flat_train_y = train_labels[active].reshape(-1)
    positive_rate = float(flat_train_y.mean())
    sample_weight = np.where(
        flat_train_y == 1,
        0.5 / positive_rate,
        0.5 / (1.0 - positive_rate),
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=20261026,
    )
    classifier.fit(flat_train_x, flat_train_y, sample_weight=sample_weight)
    validation_probabilities = classifier.predict_proba(
        validation_x.reshape(-1, validation_x.shape[2])
    )[:, 1].reshape(len(validation_bank), 4)
    evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        validation_probabilities.argmax(axis=1),
        validation_answer_counts,
    )
    model_dir = CODE_DIR / "runs" / "tabular_boosting_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "pca.npz",
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        pca_scale=pca.explained_variance_**0.5,
    )
    import joblib

    joblib.dump(classifier, model_dir / "model.joblib", compress=3)
    result = {
        "split": "validation",
        "test_used": False,
        "variant": "pca128_hist_gradient_boosting",
        "accuracy": evaluation["accuracy"],
        "correct": evaluation["correct"],
        "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
        "accuracy_by_unique_answers": evaluation["accuracy_by_unique_answers"],
        "training_rows": int(len(flat_train_y)),
        "feature_count": int(train_x.shape[2]),
        "model_bytes": (model_dir / "model.joblib").stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "tabular-boosting-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                **result,
                "structural_feature_names": list(FEATURE_NAMES),
                "pca_components": 128,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
