"""Fuse answer-group ranking with a fixed-seed candidate listwise ranker."""

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
from equation_features import build_equation_features
from train_group_ranker import build_group_dataset
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels
from verifier_features import build_structural_features


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def main() -> None:
    tick = time.perf_counter()
    bank, arrays = load_inputs("validation")
    references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")
    labels = build_labels(bank, references)
    answer_counts = unique_answer_counts(bank)
    embeddings = arrays["embeddings"]
    structural = build_structural_features(bank, embeddings)
    equations = build_equation_features(bank, embeddings)

    with np.load(
        CODE_DIR / "runs" / "verifier_v1" / "weights.npz", allow_pickle=False
    ) as v1_model:
        v1_weights = v1_model["weights"]
        v1_mean = v1_model["feature_mean"]
        v1_scale = v1_model["feature_scale"]
    candidate_features = np.concatenate(
        (
            embeddings.reshape(len(bank) * 4, -1),
            structural.reshape(len(bank) * 4, -1),
        ),
        axis=1,
    )
    v1_scores = (
        ((candidate_features - v1_mean) / v1_scale) @ v1_weights
    ).reshape(len(bank), 4)

    with np.load(
        CODE_DIR / "runs" / "linear_ensemble_candidate" / "weights.npz",
        allow_pickle=False,
    ) as listwise_model:
        listwise_weights = listwise_model["weights"][0]
        listwise_mean = listwise_model["feature_mean"]
        listwise_scale = listwise_model["feature_scale"]
    listwise_scores = (
        ((candidate_features - listwise_mean) / listwise_scale) @ listwise_weights
    ).reshape(len(bank), 4)
    listwise_probabilities = softmax(listwise_scores)

    groups = build_group_dataset(
        bank, embeddings, structural, equations, v1_scores, labels
    )
    with np.load(
        CODE_DIR / "runs" / "group_ranker_candidate" / "weights.npz",
        allow_pickle=False,
    ) as group_model:
        group_weights = group_model["group_weights"]
        group_mean = group_model["group_feature_mean"]
        group_scale = group_model["group_feature_scale"]
    group_scores = (
        (groups["features"] - group_mean) / group_scale
    ) @ group_weights

    group_indices_by_question = {}
    for group_index, (q_index, members) in enumerate(groups["groups"]):
        group_indices_by_question.setdefault(q_index, []).append(
            (group_index, members)
        )
    group_probabilities = {}
    listwise_group_probabilities = {}
    for q_index, items in group_indices_by_question.items():
        indices = np.asarray([item[0] for item in items], dtype=np.int64)
        group_probabilities[q_index] = softmax(group_scores[indices])
        listwise_group_probabilities[q_index] = np.asarray(
            [
                listwise_probabilities[q_index, members].sum()
                for _, members in items
            ],
            dtype=np.float32,
        )

    def choices_for_alpha(alpha: float) -> np.ndarray:
        choices = np.zeros(len(bank), dtype=np.int64)
        for q_index, items in group_indices_by_question.items():
            fused = (
                (1.0 - alpha) * group_probabilities[q_index]
                + alpha * listwise_group_probabilities[q_index]
            )
            members = items[int(fused.argmax())][1]
            choices[q_index] = members[
                int(listwise_probabilities[q_index, members].argmax())
            ]
        return choices

    records = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        choices = choices_for_alpha(alpha)
        evaluation = evaluate_predictions(
            "validation", bank, labels, choices, answer_counts
        )
        records.append(
            {
                "variant": f"group_listwise_probability_fusion_{alpha:.2f}",
                "listwise_group_weight": alpha,
                "accuracy": evaluation["accuracy"],
                "correct": evaluation["correct"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "accuracy_by_unique_answers": evaluation[
                    "accuracy_by_unique_answers"
                ],
                "per_question_correct": evaluation["per_question_correct"],
            }
        )

    v1_choices = v1_scores.argmax(axis=1)
    v1_evaluation = evaluate_predictions(
        "validation", bank, labels, v1_choices, answer_counts
    )
    listwise_choices = listwise_scores.argmax(axis=1)
    listwise_evaluation = evaluate_predictions(
        "validation", bank, labels, listwise_choices, answer_counts
    )
    control_correct = np.asarray(
        listwise_evaluation["per_question_correct"], dtype=np.int8
    )
    rng = np.random.default_rng(20261026)
    bootstrap_indices = rng.integers(
        0, len(control_correct), size=(5000, len(control_correct))
    )
    for record in records:
        correct = np.asarray(record.pop("per_question_correct"), dtype=np.int8)
        differences = (
            correct[bootstrap_indices].mean(axis=1)
            - control_correct[bootstrap_indices].mean(axis=1)
        ) * 100
        record["minus_single_listwise_percentage_points"] = round(
            100 * float(correct.mean() - control_correct.mean()), 4
        )
        record["minus_single_listwise_ci95"] = [
            round(float(np.quantile(differences, 0.025)), 4),
            round(float(np.quantile(differences, 0.975)), 4),
        ]

    result = {
        "split": "validation",
        "test_used": False,
        "v1": {
            "accuracy": v1_evaluation["accuracy"],
            "non_unanimous_accuracy": v1_evaluation["non_unanimous_accuracy"],
        },
        "single_listwise_control": {
            "accuracy": listwise_evaluation["accuracy"],
            "non_unanimous_accuracy": listwise_evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": listwise_evaluation[
                "accuracy_by_unique_answers"
            ],
        },
        "fusion_records": records,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "group-fusion-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
