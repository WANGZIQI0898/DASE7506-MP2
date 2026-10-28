"""Train listwise and set-encoder models with semantic consistency features."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, read_jsonl
from similarity_features import SIMILARITY_FEATURE_NAMES, build_similarity_features
from train_set_mlp import forward, train as train_set_encoder
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
    train_counts = unique_answer_counts(train_bank)
    validation_counts = unique_answer_counts(validation_bank)
    train_embeddings = train_arrays["embeddings"]
    validation_embeddings = validation_arrays["embeddings"]
    train_structural = build_structural_features(train_bank, train_embeddings)
    validation_structural = build_structural_features(
        validation_bank, validation_embeddings
    )
    train_similarity = build_similarity_features(train_bank, train_embeddings)
    validation_similarity = build_similarity_features(
        validation_bank, validation_embeddings
    )
    raw_train = np.concatenate(
        (
            train_embeddings.reshape(len(train_bank) * 4, -1),
            train_structural.reshape(len(train_bank) * 4, -1),
            train_similarity.reshape(len(train_bank) * 4, -1),
        ),
        axis=1,
    )
    raw_validation = np.concatenate(
        (
            validation_embeddings.reshape(len(validation_bank) * 4, -1),
            validation_structural.reshape(len(validation_bank) * 4, -1),
            validation_similarity.reshape(len(validation_bank) * 4, -1),
        ),
        axis=1,
    )
    mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        raw_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    train_x = ((raw_train - mean) / scale).astype(np.float32).reshape(
        len(train_bank), 4, -1
    )
    validation_x = ((raw_validation - mean) / scale).astype(np.float32).reshape(
        len(validation_bank), 4, -1
    )

    weights = train_listwise(
        train_x,
        train_labels,
        (train_counts > 1).astype(np.float64),
        regularization=0.001,
        epochs=80,
        learning_rate=0.02,
        seed=20261026,
    )
    listwise_scores = np.einsum("qcf,f->qc", validation_x, weights)
    listwise_evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        listwise_scores.argmax(axis=1),
        validation_counts,
    )
    set_parameters, _ = train_set_encoder(
        train_x,
        train_labels,
        train_counts > 1,
        hidden_size=32,
        epochs=35,
        learning_rate=0.001,
        regularization=0.0003,
        dropout_rate=0.3,
        seed=20261026,
    )
    set_scores = forward(validation_x, set_parameters)[0]
    set_evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        set_scores.argmax(axis=1),
        validation_counts,
    )
    records = [
        {
            "variant": "listwise_with_similarity",
            "accuracy": listwise_evaluation["accuracy"],
            "correct": listwise_evaluation["correct"],
            "non_unanimous_accuracy": listwise_evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": listwise_evaluation[
                "accuracy_by_unique_answers"
            ],
        },
        {
            "variant": "set_encoder_with_similarity",
            "accuracy": set_evaluation["accuracy"],
            "correct": set_evaluation["correct"],
            "non_unanimous_accuracy": set_evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": set_evaluation[
                "accuracy_by_unique_answers"
            ],
        },
    ]
    result = {
        "split": "validation",
        "test_used": False,
        "records": records,
        "structural_feature_names": list(FEATURE_NAMES),
        "similarity_feature_names": list(SIMILARITY_FEATURE_NAMES),
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    model_dir = CODE_DIR / "runs" / "similarity_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "models.npz",
        feature_mean=mean,
        feature_scale=scale,
        listwise_weights=weights,
        **{name: value for name, value in set_parameters.items()},
    )
    output_path = CODE_DIR / "runs" / "similarity-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
