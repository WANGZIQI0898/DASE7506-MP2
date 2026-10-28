"""Rank groups of candidates by identical final answer, then select within a group."""

from __future__ import annotations

from collections import defaultdict
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

from common import load_inputs, parse_answer, read_jsonl
from equation_features import EQUATION_FEATURE_NAMES, build_equation_features
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels, build_pair_differences, train_pairwise_ranker
from verifier_features import FEATURE_NAMES, build_structural_features


def build_group_dataset(
    bank: list[dict],
    embeddings: np.ndarray,
    structural: np.ndarray,
    equations: np.ndarray,
    candidate_scores: np.ndarray,
    labels: np.ndarray | None,
) -> dict:
    features = []
    groups = []
    targets = []
    question_indices = []
    for q_index, row in enumerate(bank):
        members = defaultdict(list)
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        for candidate_index, answer in enumerate(answers):
            key = answer if answer is not None else "__unparseable__"
            members[key].append(candidate_index)
        for answer, indices in members.items():
            index_array = np.asarray(indices, dtype=np.int64)
            group_feature = np.concatenate(
                (
                    embeddings[q_index, index_array].mean(axis=0),
                    embeddings[q_index, index_array].max(axis=0),
                    structural[q_index, index_array].mean(axis=0),
                    structural[q_index, index_array].max(axis=0),
                    equations[q_index, index_array].mean(axis=0),
                    np.asarray(
                        [
                            len(indices) / 4.0,
                            float(answer != "__unparseable__"),
                            float(candidate_scores[q_index, index_array].max()),
                            float(candidate_scores[q_index, index_array].mean()),
                        ],
                        dtype=np.float32,
                    ),
                )
            ).astype(np.float32)
            features.append(group_feature)
            groups.append((q_index, index_array))
            question_indices.append(q_index)
            if labels is not None:
                targets.append(float(labels[q_index, index_array].any()))
    return {
        "features": np.asarray(features, dtype=np.float32),
        "groups": groups,
        "targets": np.asarray(targets, dtype=np.int8) if labels is not None else None,
        "question_indices": np.asarray(question_indices, dtype=np.int64),
        "feature_names": (
            [f"mean_embedding_{i}" for i in range(embeddings.shape[2])]
            + [f"max_embedding_{i}" for i in range(embeddings.shape[2])]
            + [f"mean_{name}" for name in FEATURE_NAMES]
            + [f"max_{name}" for name in FEATURE_NAMES]
            + [f"mean_{name}" for name in EQUATION_FEATURE_NAMES]
            + ["group_size", "parseable_group", "max_candidate_score", "mean_candidate_score"]
        ),
    }


def group_pair_differences(
    features: np.ndarray,
    targets: np.ndarray,
    question_indices: np.ndarray,
) -> np.ndarray:
    differences = []
    for q_index in np.unique(question_indices):
        question_mask = question_indices == q_index
        local_indices = np.flatnonzero(question_mask)
        positives = local_indices[targets[local_indices] == 1]
        negatives = local_indices[targets[local_indices] == 0]
        for positive in positives:
            for negative in negatives:
                differences.append(
                    (features[positive] - features[negative])
                    / max(np.sqrt(len(positives) * len(negatives)), 1.0)
                )
    return np.asarray(differences, dtype=np.float32)


def evaluate_group_choices(
    split: str,
    bank: list[dict],
    labels: np.ndarray,
    dataset: dict,
    candidate_scores: np.ndarray,
    group_scores: np.ndarray,
    answer_counts: np.ndarray,
) -> tuple[np.ndarray, dict]:
    best_group_by_question = {}
    for group_index, (q_index, _) in enumerate(dataset["groups"]):
        score = group_scores[group_index]
        if q_index not in best_group_by_question or score > best_group_by_question[q_index][0]:
            best_group_by_question[q_index] = (score, group_index)
    choices = np.zeros(len(bank), dtype=np.int64)
    for q_index in range(len(bank)):
        _, group_index = best_group_by_question[q_index]
        _, members = dataset["groups"][group_index]
        choices[q_index] = members[np.argmax(candidate_scores[q_index, members])]
    evaluation = evaluate_predictions(
        split, bank, labels, choices, answer_counts
    )
    return choices, evaluation


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
    train_equations = build_equation_features(train_bank, train_embeddings)
    validation_equations = build_equation_features(
        validation_bank, validation_embeddings
    )

    with np.load(
        CODE_DIR / "runs" / "verifier_v1" / "weights.npz", allow_pickle=False
    ) as v1_model:
        v1_weights = v1_model["weights"]
        v1_mean = v1_model["feature_mean"]
        v1_scale = v1_model["feature_scale"]
    train_candidate_features = np.concatenate(
        (
            train_embeddings.reshape(len(train_bank) * 4, -1),
            train_structural.reshape(len(train_bank) * 4, -1),
        ),
        axis=1,
    )
    validation_candidate_features = np.concatenate(
        (
            validation_embeddings.reshape(len(validation_bank) * 4, -1),
            validation_structural.reshape(len(validation_bank) * 4, -1),
        ),
        axis=1,
    )
    train_candidate_scores = (
        ((train_candidate_features - v1_mean) / v1_scale) @ v1_weights
    ).reshape(len(train_bank), 4)
    validation_candidate_scores = (
        ((validation_candidate_features - v1_mean) / v1_scale) @ v1_weights
    ).reshape(len(validation_bank), 4)

    train_groups = build_group_dataset(
        train_bank,
        train_embeddings,
        train_structural,
        train_equations,
        train_candidate_scores,
        train_labels,
    )
    validation_groups = build_group_dataset(
        validation_bank,
        validation_embeddings,
        validation_structural,
        validation_equations,
        validation_candidate_scores,
        validation_labels,
    )
    feature_mean = train_groups["features"].mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    feature_scale = np.maximum(
        train_groups["features"].std(axis=0, dtype=np.float64).astype(np.float32),
        1e-6,
    )
    train_group_features = (
        (train_groups["features"] - feature_mean) / feature_scale
    ).astype(np.float32)
    validation_group_features = (
        (validation_groups["features"] - feature_mean) / feature_scale
    ).astype(np.float32)

    differences = group_pair_differences(
        train_group_features,
        train_groups["targets"],
        train_groups["question_indices"],
    )
    group_weights = train_pairwise_ranker(
        differences,
        regularization=0.001,
        epochs=80,
        learning_rate=0.02,
        seed=20261026,
    )
    train_group_scores = train_group_features @ group_weights
    validation_group_scores = validation_group_features @ group_weights
    train_choices, train_evaluation = evaluate_group_choices(
        "train",
        train_bank,
        train_labels,
        train_groups,
        train_candidate_scores,
        train_group_scores,
        train_answer_counts,
    )
    validation_choices, validation_evaluation = evaluate_group_choices(
        "validation",
        validation_bank,
        validation_labels,
        validation_groups,
        validation_candidate_scores,
        validation_group_scores,
        validation_answer_counts,
    )

    result = {
        "split": "validation",
        "test_used": False,
        "train": {
            "accuracy": train_evaluation["accuracy"],
            "non_unanimous_accuracy": train_evaluation["non_unanimous_accuracy"],
        },
        "validation": {
            "accuracy": validation_evaluation["accuracy"],
            "correct": validation_evaluation["correct"],
            "non_unanimous_accuracy": validation_evaluation[
                "non_unanimous_accuracy"
            ],
            "accuracy_by_unique_answers": validation_evaluation[
                "accuracy_by_unique_answers"
            ],
        },
        "group_statistics": {
            "train_groups": len(train_groups["groups"]),
            "validation_groups": len(validation_groups["groups"]),
            "training_pairs": len(differences),
            "feature_count": int(group_weights.size),
        },
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "group-ranker-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model_dir = CODE_DIR / "runs" / "group_ranker_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "weights.npz",
        group_weights=group_weights,
        group_feature_mean=feature_mean,
        group_feature_scale=feature_scale,
    )
    config = {
        **result,
        "group_feature_names": train_groups["feature_names"],
        "regularization": 0.001,
        "epochs": 80,
        "learning_rate": 0.02,
        "seed": 20261026,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
