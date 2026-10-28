"""Evaluate low-variance linear listwise ensembles on validation only."""

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
from train_set_mlp import forward
from train_v2 import evaluate_predictions, train_listwise, unique_answer_counts
from train_verifier import build_labels
from verifier_features import FEATURE_NAMES, build_structural_features


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def main() -> None:
    tick = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_references = read_jsonl(CODE_DIR / "data" / "gsm8k_train.jsonl")
    validation_references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")

    train_embeddings = train_arrays["embeddings"].reshape(len(train_bank) * 4, -1)
    validation_embeddings = validation_arrays["embeddings"].reshape(
        len(validation_bank) * 4, -1
    )
    train_structural = build_structural_features(
        train_bank, train_arrays["embeddings"]
    ).reshape(len(train_bank) * 4, -1)
    validation_structural = build_structural_features(
        validation_bank, validation_arrays["embeddings"]
    ).reshape(len(validation_bank) * 4, -1)
    raw_train = np.concatenate((train_embeddings, train_structural), axis=1)
    raw_validation = np.concatenate(
        (validation_embeddings, validation_structural), axis=1
    )
    mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        raw_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    train_features = ((raw_train - mean) / scale).astype(np.float32).reshape(
        len(train_bank), 4, -1
    )
    validation_features = ((raw_validation - mean) / scale).astype(np.float32).reshape(
        len(validation_bank), 4, -1
    )
    train_labels = build_labels(train_bank, train_references)
    validation_labels = build_labels(validation_bank, validation_references)
    train_answer_counts = unique_answer_counts(train_bank)
    validation_answer_counts = unique_answer_counts(validation_bank)
    question_weights = (train_answer_counts > 1).astype(np.float64)

    seeds = list(range(20261026, 20261031))
    linear_weights = []
    linear_probabilities = []
    records = []
    for seed in seeds:
        model_tick = time.perf_counter()
        weights = train_listwise(
            train_features,
            train_labels,
            question_weights,
            regularization=0.001,
            epochs=80,
            learning_rate=0.02,
            seed=seed,
        )
        scores = np.einsum("qcf,f->qc", validation_features, weights)
        probabilities = softmax(scores)
        choices = scores.argmax(axis=1)
        evaluation = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            choices,
            validation_answer_counts,
        )
        records.append(
            {
                "variant": f"listwise_seed_{seed}",
                "accuracy": evaluation["accuracy"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "accuracy_by_unique_answers": evaluation[
                    "accuracy_by_unique_answers"
                ],
                "training_seconds": round(time.perf_counter() - model_tick, 4),
            }
        )
        linear_probabilities.append(probabilities)
        linear_weights.append(weights)
        print(json.dumps(records[-1]), flush=True)

    linear_probability_mean = np.mean(linear_probabilities, axis=0)
    ensemble_choices = linear_probability_mean.argmax(axis=1)
    ensemble_evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        ensemble_choices,
        validation_answer_counts,
    )

    mlp_path = CODE_DIR / "runs" / "set_mlp_candidate" / "weights.npz"
    with np.load(mlp_path, allow_pickle=False) as stored:
        mlp_parameters = {
            name: stored[name].copy() for name in ("w1", "b1", "w2", "b2")
        }
        mlp_mean = stored["feature_mean"]
        mlp_scale = stored["feature_scale"]
    mlp_features = ((raw_validation[:, : len(mlp_mean)] - mlp_mean) / mlp_scale).astype(
        np.float32
    )
    mlp_scores, _ = forward(
        mlp_features.reshape(len(validation_bank), 4, -1), mlp_parameters
    )
    mlp_probabilities = softmax(mlp_scores)

    fusion_records = []
    for mlp_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        probabilities = (
            (1.0 - mlp_weight) * linear_probability_mean
            + mlp_weight * mlp_probabilities
        )
        choices = probabilities.argmax(axis=1)
        evaluation = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            choices,
            validation_answer_counts,
        )
        fusion_records.append(
            {
                "variant": f"linear_ensemble_mlp_fusion_{mlp_weight:.2f}",
                "mlp_weight": mlp_weight,
                "accuracy": evaluation["accuracy"],
                "correct": evaluation["correct"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "accuracy_by_unique_answers": evaluation[
                    "accuracy_by_unique_answers"
                ],
                "per_question_correct": evaluation["per_question_correct"],
            }
        )

    control_bank, control_arrays = load_inputs("validation")
    control_embeddings = control_arrays["embeddings"].reshape(
        len(control_bank) * 4, -1
    )
    control_features = np.concatenate(
        (
            control_embeddings,
            build_structural_features(control_bank, control_arrays["embeddings"]).reshape(
                len(control_bank) * 4, -1
            ),
        ),
        axis=1,
    )
    with np.load(
        CODE_DIR / "runs" / "verifier_v1" / "weights.npz", allow_pickle=False
    ) as control_model:
        control_weights = control_model["weights"]
        control_mean = control_model["feature_mean"]
        control_scale = control_model["feature_scale"]
    control_scores = (
        ((control_features - control_mean) / control_scale) @ control_weights
    ).reshape(len(control_bank), 4)
    control_choices = control_scores.argmax(axis=1)
    control_evaluation = evaluate_predictions(
        "validation",
        control_bank,
        validation_labels,
        control_choices,
        validation_answer_counts,
    )
    control_correct = np.asarray(
        control_evaluation["per_question_correct"], dtype=np.int8
    )
    rng = np.random.default_rng(20261026)
    bootstrap_indices = rng.integers(
        0, len(control_correct), size=(5000, len(control_correct))
    )
    for record in fusion_records:
        correct = np.asarray(record.pop("per_question_correct"), dtype=np.int8)
        differences = (
            correct[bootstrap_indices].mean(axis=1)
            - control_correct[bootstrap_indices].mean(axis=1)
        ) * 100
        record["minus_v1_percentage_points"] = round(
            100 * float(correct.mean() - control_correct.mean()), 4
        )
        record["minus_v1_ci95"] = [
            round(float(np.quantile(differences, 0.025)), 4),
            round(float(np.quantile(differences, 0.975)), 4),
        ]

    ensemble_record = {
        "variant": "listwise_linear_5_seed_probability_mean",
        "accuracy": ensemble_evaluation["accuracy"],
        "correct": ensemble_evaluation["correct"],
        "non_unanimous_accuracy": ensemble_evaluation["non_unanimous_accuracy"],
        "accuracy_by_unique_answers": ensemble_evaluation[
            "accuracy_by_unique_answers"
        ],
    }
    output = {
        "split": "validation",
        "test_used": False,
        "single_seed_records": records,
        "linear_ensemble": ensemble_record,
        "fusion_records": fusion_records,
        "v1_control": {
            "accuracy": control_evaluation["accuracy"],
            "non_unanimous_accuracy": control_evaluation["non_unanimous_accuracy"],
        },
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "ensemble-v2-development.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model_dir = CODE_DIR / "runs" / "linear_ensemble_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "weights.npz",
        weights=np.stack(linear_weights),
        feature_mean=mean,
        feature_scale=scale,
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "five_seed_listwise_linear_probability_mean",
                "seeds": seeds,
                "epochs": 80,
                "learning_rate": 0.02,
                "regularization": 0.001,
                "embedding_count": train_embeddings.shape[1],
                "structural_feature_names": list(FEATURE_NAMES),
                "validation_accuracy": ensemble_evaluation["accuracy"],
                "non_unanimous_accuracy": ensemble_evaluation[
                    "non_unanimous_accuracy"
                ],
                "accuracy_by_unique_answers": ensemble_evaluation[
                    "accuracy_by_unique_answers"
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "linear_ensemble": ensemble_record,
                "fusion_records": fusion_records,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
