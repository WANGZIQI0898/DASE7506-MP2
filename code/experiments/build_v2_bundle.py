"""Build and validate the fixed equal-probability v2 ensemble."""

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
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels
from verifier_features import FEATURE_NAMES, build_structural_features


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def main() -> None:
    tick = time.perf_counter()
    bank, arrays = load_inputs("validation")
    references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")
    labels = build_labels(bank, references)
    answer_counts = unique_answer_counts(bank)
    embeddings = arrays["embeddings"]
    structural = build_structural_features(bank, embeddings)
    raw_features = np.concatenate(
        (
            embeddings.reshape(len(bank) * 4, -1),
            structural.reshape(len(bank) * 4, -1),
        ),
        axis=1,
    )

    source_linear = CODE_DIR / "runs" / "linear_ensemble_candidate" / "weights.npz"
    source_mlp = CODE_DIR / "runs" / "set_mlp_candidate" / "weights.npz"
    with np.load(source_linear, allow_pickle=False) as source:
        linear_weights = source["weights"][0].copy()
        linear_mean = source["feature_mean"].copy()
        linear_scale = source["feature_scale"].copy()
    with np.load(source_mlp, allow_pickle=False) as source:
        mlp_parameters = {
            name: source[name].copy() for name in ("w1", "b1", "w2", "b2")
        }
        mlp_mean = source["feature_mean"].copy()
        mlp_scale = source["feature_scale"].copy()

    linear_scores = (
        ((raw_features - linear_mean) / linear_scale) @ linear_weights
    ).reshape(len(bank), 4)
    linear_probabilities = softmax(linear_scores)
    mlp_scores, _ = forward(
        ((raw_features - mlp_mean) / mlp_scale).reshape(len(bank), 4, -1),
        mlp_parameters,
    )
    mlp_probabilities = softmax(mlp_scores)
    ensemble_probabilities = 0.5 * linear_probabilities + 0.5 * mlp_probabilities
    choices = ensemble_probabilities.argmax(axis=1)
    evaluation = evaluate_predictions(
        "validation", bank, labels, choices, answer_counts
    )

    with np.load(
        CODE_DIR / "runs" / "verifier_v1" / "weights.npz", allow_pickle=False
    ) as v1_model:
        v1_weights = v1_model["weights"]
        v1_mean = v1_model["feature_mean"]
        v1_scale = v1_model["feature_scale"]
    v1_probability = (
        ((raw_features - v1_mean) / v1_scale) @ v1_weights
    ).reshape(len(bank), 4)
    v1_choices = v1_probability.argmax(axis=1)
    v1_evaluation = evaluate_predictions(
        "validation", bank, labels, v1_choices, answer_counts
    )
    ensemble_correct = np.asarray(
        evaluation["per_question_correct"], dtype=np.int8
    )
    v1_correct = np.asarray(v1_evaluation["per_question_correct"], dtype=np.int8)
    rng = np.random.default_rng(20261026)
    bootstrap_indices = rng.integers(0, len(bank), size=(5000, len(bank)))
    bootstrap_differences = (
        ensemble_correct[bootstrap_indices].mean(axis=1)
        - v1_correct[bootstrap_indices].mean(axis=1)
    ) * 100
    non_unanimous = answer_counts > 1
    bootstrap_non_unanimous = (
        ensemble_correct[bootstrap_indices][:, non_unanimous].mean(axis=1)
        - v1_correct[bootstrap_indices][:, non_unanimous].mean(axis=1)
    ) * 100

    model_dir = CODE_DIR / "runs" / "verifier_v2"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "linear_weights.npz",
        weights=linear_weights,
        feature_mean=linear_mean,
        feature_scale=linear_scale,
    )
    np.savez_compressed(
        model_dir / "set_encoder_weights.npz",
        **mlp_parameters,
        feature_mean=mlp_mean,
        feature_scale=mlp_scale,
    )
    config = {
        "model_version": 2,
        "architecture": "equal_probability_linear_listwise_and_set_encoder",
        "fusion_weights": {"linear_listwise": 0.5, "set_encoder": 0.5},
        "linear_component": {
            "training_seed": 20261026,
            "epochs": 80,
            "learning_rate": 0.02,
            "regularization": 0.001,
        },
        "set_encoder_component": {
            "training_seed": 20261026,
            "hidden_size": 32,
            "epochs": 35,
            "learning_rate": 0.001,
            "regularization": 0.0003,
            "dropout": 0.3,
        },
        "embedding_count": embeddings.shape[2],
        "structural_feature_names": list(FEATURE_NAMES),
        "validation": {
            "accuracy": evaluation["accuracy"],
            "correct": evaluation["correct"],
            "n": evaluation["n"],
            "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
            "non_unanimous_n": evaluation["non_unanimous_n"],
            "accuracy_by_unique_answers": evaluation[
                "accuracy_by_unique_answers"
            ],
        },
        "versus_v1": {
            "overall_percentage_points": round(
                evaluation["accuracy"] - v1_evaluation["accuracy"], 4
            ),
            "overall_bootstrap_ci95": [
                round(float(np.quantile(bootstrap_differences, 0.025)), 4),
                round(float(np.quantile(bootstrap_differences, 0.975)), 4),
            ],
            "non_unanimous_percentage_points": round(
                evaluation["non_unanimous_accuracy"]
                - v1_evaluation["non_unanimous_accuracy"],
                4,
            ),
            "non_unanimous_bootstrap_ci95": [
                round(float(np.quantile(bootstrap_non_unanimous, 0.025)), 4),
                round(float(np.quantile(bootstrap_non_unanimous, 0.975)), 4),
            ],
        },
        "test_used_for_selection": False,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_path = CODE_DIR / "runs" / "v2-bundle-validation.json"
    output_path.write_text(
        json.dumps(
            {
                "config": config,
                "validation_choices": choices.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
