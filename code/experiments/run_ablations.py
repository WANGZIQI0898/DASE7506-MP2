"""Run fixed-seed validation ablations for verifier v1."""

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

from common import load_inputs, read_jsonl, score_choices
from train_verifier import (
    build_labels,
    build_pair_differences,
    candidate_scores,
    train_pairwise_ranker,
)
from verifier_features import FEATURE_NAMES, build_structural_features


EPOCHS = 80
LEARNING_RATE = 0.02
REGULARIZATION = 0.001
SEED = 20261026


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
    train_x = ((raw_train - mean) / scale).astype(np.float32)
    validation_x = ((raw_validation - mean) / scale).astype(np.float32)

    embedding_columns = set(range(train_embeddings.shape[1]))
    structural_columns = {
        train_embeddings.shape[1] + i for i in range(len(FEATURE_NAMES))
    }
    feature_positions = {
        name: train_embeddings.shape[1] + i
        for i, name in enumerate(FEATURE_NAMES)
    }
    consensus_columns = {
        feature_positions[name]
        for name in ("answer_group_size", "answer_is_top_group", "answer_group_count")
    }
    length_style_columns = {
        feature_positions[name]
        for name in (
            "log_characters",
            "log_words",
            "log_lines",
            "log_digits",
            "log_equals",
            "log_math_operators",
            "log_math_blocks",
            "length_rank",
            "relative_length",
        )
    }
    question_context_columns = {
        feature_positions[name]
        for name in ("question_word_coverage", "question_number_coverage")
    }
    all_columns = embedding_columns | structural_columns
    variants = {
        "embedding_only": embedding_columns,
        "structure_only": structural_columns,
        "full": all_columns,
        "without_answer_consensus": all_columns - consensus_columns,
        "without_length_style": all_columns - length_style_columns,
        "without_question_context": all_columns - question_context_columns,
    }

    labels = build_labels(train_bank, train_references)
    validation_labels = build_labels(validation_bank, validation_references)
    choices_by_variant = {}
    records = []
    for name, column_set in variants.items():
        columns = np.asarray(sorted(column_set), dtype=np.int64)
        variant_train = train_x[:, columns]
        differences = build_pair_differences(
            variant_train.reshape(len(train_bank), 4, -1), labels
        )
        weights = train_pairwise_ranker(
            differences,
            REGULARIZATION,
            EPOCHS,
            LEARNING_RATE,
            SEED,
        )
        train_scores = candidate_scores(
            variant_train.reshape(len(train_bank), 4, -1), weights
        )
        validation_scores = candidate_scores(
            validation_x[:, columns].reshape(len(validation_bank), 4, -1), weights
        )
        choices = validation_scores.argmax(axis=1)
        choices_by_variant[name] = choices
        train_eval = score_choices(
            "train",
            [row["id"] for row in train_bank],
            train_scores.argmax(axis=1).tolist(),
        )
        validation_eval = score_choices(
            "validation",
            [row["id"] for row in validation_bank],
            choices.tolist(),
        )
        correct = validation_labels[
            np.arange(len(validation_bank)), choices
        ].astype(np.int8)
        records.append(
            {
                "variant": name,
                "feature_count": int(len(columns)),
                "train_accuracy": round(train_eval["accuracy"], 4),
                "validation_accuracy": round(validation_eval["accuracy"], 4),
                "validation_correct": validation_eval["correct"],
                "per_question_correct": correct.tolist(),
            }
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in records[-1].items()
                    if key != "per_question_correct"
                }
            ),
            flush=True,
        )

    full_correct = np.asarray(
        next(
            record["per_question_correct"]
            for record in records
            if record["variant"] == "full"
        ),
        dtype=np.int8,
    )
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(
        0, len(full_correct), size=(5000, len(full_correct))
    )
    for record in records:
        variant_correct = np.asarray(record.pop("per_question_correct"), dtype=np.int8)
        differences = (
            full_correct[bootstrap_indices].mean(axis=1)
            - variant_correct[bootstrap_indices].mean(axis=1)
        ) * 100
        record["full_minus_variant_percentage_points"] = round(
            float((full_correct.mean() - variant_correct.mean()) * 100), 4
        )
        record["full_minus_variant_ci95"] = [
            round(float(np.quantile(differences, 0.025)), 4),
            round(float(np.quantile(differences, 0.975)), 4),
        ]

    result = {
        "split": "validation",
        "n": len(validation_bank),
        "fixed_config": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "regularization": REGULARIZATION,
            "seed": SEED,
        },
        "variants": records,
        "bootstrap_samples": 5000,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "ablation-verifier-v1.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": [
                    {
                        key: value
                        for key, value in record.items()
                        if key
                        in (
                            "variant",
                            "feature_count",
                            "validation_accuracy",
                            "full_minus_variant_percentage_points",
                            "full_minus_variant_ci95",
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
