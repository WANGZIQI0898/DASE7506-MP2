"""Develop listwise and equation-aware candidate rankers on train/validation."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, parse_answer, read_jsonl, score_choices
from equation_features import EQUATION_FEATURE_NAMES, build_equation_features
from train_verifier import (
    build_labels,
    build_pair_differences,
    train_pairwise_ranker,
)
from verifier_features import FEATURE_NAMES, build_structural_features


def train_listwise(
    features: np.ndarray,
    labels: np.ndarray,
    question_weights: np.ndarray,
    regularization: float,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[2], dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    batch_size = 256
    total_steps = epochs * math.ceil(len(features) / batch_size)
    step = 0
    targets = labels / labels.sum(axis=1, keepdims=True)

    for _ in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = features[indices].astype(np.float64)
            batch_targets = targets[indices]
            batch_weights = question_weights[indices]
            logits = batch_x @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            score_gradient = (
                probabilities - batch_targets
            ) * batch_weights[:, None]
            gradient = np.einsum("qcf,qc->f", batch_x, score_gradient)
            gradient /= batch_weights.sum()
            gradient += regularization * weights

            step += 1
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            weights -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )
    return weights.astype(np.float32)


def unique_answer_counts(bank: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            len({parse_answer(candidate["text"]) for candidate in row["candidates"]})
            for row in bank
        ],
        dtype=np.int8,
    )


def evaluate_predictions(
    split: str,
    bank: list[dict],
    labels: np.ndarray,
    choices: np.ndarray,
    answer_counts: np.ndarray,
) -> dict:
    official = score_choices(split, [row["id"] for row in bank], choices.tolist())
    correct = labels[np.arange(len(bank)), choices].astype(np.int8)
    subsets = {}
    for count in (1, 2, 3, 4):
        mask = answer_counts == count
        if mask.any():
            subsets[str(count)] = {
                "n": int(mask.sum()),
                "accuracy": round(100 * float(correct[mask].mean()), 4),
            }
    non_unanimous = answer_counts > 1
    return {
        "accuracy": round(official["accuracy"], 4),
        "correct": official["correct"],
        "n": official["n"],
        "accuracy_by_unique_answers": subsets,
        "non_unanimous_accuracy": round(
            100 * float(correct[non_unanimous].mean()), 4
        ),
        "non_unanimous_n": int(non_unanimous.sum()),
        "per_question_correct": correct.tolist(),
    }


def hard_question_weights(answer_counts: np.ndarray) -> np.ndarray:
    mapping = np.asarray([0.0, 0.0, 1.0, 1.5, 2.0], dtype=np.float64)
    return mapping[answer_counts]


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
    train_equations = build_equation_features(
        train_bank, train_arrays["embeddings"]
    ).reshape(len(train_bank) * 4, -1)
    validation_equations = build_equation_features(
        validation_bank, validation_arrays["embeddings"]
    ).reshape(len(validation_bank) * 4, -1)

    raw_train_v2 = np.concatenate(
        (train_embeddings, train_structural, train_equations), axis=1
    )
    raw_validation_v2 = np.concatenate(
        (validation_embeddings, validation_structural, validation_equations), axis=1
    )
    raw_train_v1 = raw_train_v2[:, : train_embeddings.shape[1] + len(FEATURE_NAMES)]
    raw_validation_v1 = raw_validation_v2[
        :, : validation_embeddings.shape[1] + len(FEATURE_NAMES)
    ]

    mean = raw_train_v2.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        raw_train_v2.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    train_x_v1 = ((raw_train_v1 - mean[: raw_train_v1.shape[1]]) / scale[: raw_train_v1.shape[1]]).astype(np.float32)
    validation_x_v1 = ((raw_validation_v1 - mean[: raw_train_v1.shape[1]]) / scale[: raw_train_v1.shape[1]]).astype(np.float32)
    train_x_v2 = ((raw_train_v2 - mean) / scale).astype(np.float32)
    validation_x_v2 = ((raw_validation_v2 - mean) / scale).astype(np.float32)

    train_labels = build_labels(train_bank, train_references)
    validation_labels = build_labels(validation_bank, validation_references)
    train_answer_counts = unique_answer_counts(train_bank)
    validation_answer_counts = unique_answer_counts(validation_bank)
    weights_hard = hard_question_weights(train_answer_counts)
    weights_unweighted = (train_answer_counts > 1).astype(np.float64)

    epochs = 80
    learning_rate = 0.02
    regularization = 0.001
    seed = 20261026
    records = []
    predictions = {}

    v1_differences = build_pair_differences(
        train_x_v1.reshape(len(train_bank), 4, -1), train_labels
    )
    v1_weights = train_pairwise_ranker(
        v1_differences, regularization, epochs, learning_rate, seed
    )
    v1_scores = np.einsum(
        "qcf,f->qc",
        validation_x_v1.reshape(len(validation_bank), 4, -1),
        v1_weights,
    )
    v1_choices = v1_scores.argmax(axis=1)
    predictions["pairwise_v1_control"] = v1_choices
    record = {
        "variant": "pairwise_v1_control",
        "feature_count": int(v1_weights.size),
        **evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            v1_choices,
            validation_answer_counts,
        ),
    }
    records.append(record)
    print(json.dumps({k: v for k, v in record.items() if k != "per_question_correct"}), flush=True)

    variants = {
        "listwise_v1_unweighted": {
            "features": train_x_v1,
            "validation_features": validation_x_v1,
            "method": "listwise",
            "question_weights": weights_unweighted,
        },
        "pairwise_v2_equations": {
            "features": train_x_v2,
            "validation_features": validation_x_v2,
            "method": "pairwise",
            "question_weights": weights_unweighted,
        },
        "listwise_v2_unweighted": {
            "features": train_x_v2,
            "validation_features": validation_x_v2,
            "method": "listwise",
            "question_weights": weights_unweighted,
        },
        "listwise_v2_hard_weighted": {
            "features": train_x_v2,
            "validation_features": validation_x_v2,
            "method": "listwise",
            "question_weights": weights_hard,
        },
    }
    for name, variant in variants.items():
        train_matrix = variant["features"].reshape(len(train_bank), 4, -1)
        if variant["method"] == "pairwise":
            differences = build_pair_differences(train_matrix, train_labels)
            weights = train_pairwise_ranker(
                differences, regularization, epochs, learning_rate, seed
            )
        else:
            weights = train_listwise(
                train_matrix,
                train_labels,
                variant["question_weights"],
                regularization,
                epochs,
                learning_rate,
                seed,
            )
        validation_scores = np.einsum(
            "qcf,f->qc",
            variant["validation_features"].reshape(len(validation_bank), 4, -1),
            weights,
        )
        choices = validation_scores.argmax(axis=1)
        predictions[name] = choices
        record = {
            "variant": name,
            "feature_count": int(weights.size),
            **evaluate_predictions(
                "validation",
                validation_bank,
                validation_labels,
                choices,
                validation_answer_counts,
            ),
        }
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "per_question_correct"}), flush=True)

    control_correct = np.asarray(records[0]["per_question_correct"], dtype=np.int8)
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, len(control_correct), size=(5000, len(control_correct))
    )
    for record in records:
        correct = np.asarray(record.pop("per_question_correct"), dtype=np.int8)
        differences = (
            correct[bootstrap_indices].mean(axis=1)
            - control_correct[bootstrap_indices].mean(axis=1)
        ) * 100
        record["minus_pairwise_v1_percentage_points"] = round(
            100 * float(correct.mean() - control_correct.mean()), 4
        )
        record["minus_pairwise_v1_ci95"] = [
            round(float(np.quantile(differences, 0.025)), 4),
            round(float(np.quantile(differences, 0.975)), 4),
        ]

    output = {
        "split": "validation",
        "test_used": False,
        "fixed_config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "seed": seed,
            "hard_weights": {"2": 1.0, "3": 1.5, "4": 2.0},
        },
        "feature_names": {
            "embedding": train_embeddings.shape[1],
            "structural": list(FEATURE_NAMES),
            "equation": list(EQUATION_FEATURE_NAMES),
        },
        "records": records,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "v2-development.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": [
                    {
                        key: record[key]
                        for key in (
                            "variant",
                            "accuracy",
                            "non_unanimous_accuracy",
                            "minus_pairwise_v1_percentage_points",
                            "minus_pairwise_v1_ci95",
                        )
                    }
                    for record in records
                ],
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
