"""Train and compare first-pass candidate ranking models on train/validation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import load_inputs, parse_answer, read_jsonl, score_choices
from verifier_features import FEATURE_NAMES, build_structural_features


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def build_labels(bank: list[dict], references: list[dict]) -> np.ndarray:
    labels = np.zeros((len(bank), 4), dtype=np.int8)
    for q_index, (row, reference) in enumerate(zip(bank, references, strict=True)):
        gold = parse_answer(reference["answer"])
        for i, candidate in enumerate(row["candidates"]):
            labels[q_index, i] = parse_answer(candidate["text"]) == gold
    return labels


def build_pair_differences(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    differences = []
    for q_index in range(len(features)):
        positives = np.flatnonzero(labels[q_index])
        negatives = np.flatnonzero(~labels[q_index].astype(bool))
        if len(positives) == 0 or len(negatives) == 0:
            continue
        for positive in positives:
            for negative in negatives:
                delta = features[q_index, positive] - features[q_index, negative]
                differences.append(delta / max(len(positives) * len(negatives), 1) ** 0.5)
    if not differences:
        raise ValueError("No positive-negative candidate pairs")
    return np.asarray(differences, dtype=np.float32)


def train_pairwise_ranker(
    differences: np.ndarray,
    regularization: float,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros(differences.shape[1], dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    batch_size = min(4096, len(differences))
    steps_per_epoch = max(1, math_ceil_div(len(differences), batch_size))
    total_steps = epochs * steps_per_epoch
    step = 0

    for _ in range(epochs):
        order = rng.permutation(len(differences))
        for start in range(0, len(order), batch_size):
            batch = differences[order[start : start + batch_size]].astype(np.float64)
            logits = batch @ weights
            # d softplus(-logit) / d logit = -sigmoid(-logit)
            negative_gradient = _sigmoid(-logits)
            gradient = -(batch.T @ negative_gradient) / len(batch)
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


def math_ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def candidate_scores(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("qcf,f->qc", features, weights)


def evaluate(
    split: str,
    bank: list[dict],
    labels: np.ndarray,
    choices: np.ndarray,
) -> dict:
    score = score_choices(split, [row["id"] for row in bank], choices.tolist())
    unique_answers = Counter(
        len({parse_answer(candidate["text"]) for candidate in row["candidates"]})
        for row in bank
    )
    correct = np.asarray(
        [labels[i, choices[i]] for i in range(len(bank))], dtype=np.int8
    )
    subsets = {}
    for answer_count in sorted(unique_answers):
        mask = np.asarray(
            [
                len({parse_answer(candidate["text"]) for candidate in row["candidates"]})
                == answer_count
                for row in bank
            ]
        )
        subsets[str(answer_count)] = {
            "n": int(mask.sum()),
            "accuracy": round(100 * float(correct[mask].mean()), 4),
        }
    return {
        "accuracy": round(score["accuracy"], 4),
        "correct": score["correct"],
        "n": score["n"],
        "accuracy_by_unique_answers": subsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "runs" / "verifier_v1")
    args = parser.parse_args()

    tick = time.perf_counter()
    train_bank, train_arrays = load_inputs("train")
    validation_bank, validation_arrays = load_inputs("validation")
    train_refs = read_jsonl(CODE_DIR / "data" / "gsm8k_train.jsonl")
    validation_refs = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")

    train_structural = build_structural_features(
        train_bank, train_arrays["embeddings"]
    ).reshape(len(train_bank) * 4, -1)
    validation_structural = build_structural_features(
        validation_bank, validation_arrays["embeddings"]
    ).reshape(len(validation_bank) * 4, -1)
    train_embeddings = train_arrays["embeddings"].reshape(len(train_bank) * 4, -1)
    validation_embeddings = validation_arrays["embeddings"].reshape(
        len(validation_bank) * 4, -1
    )

    raw_train = np.concatenate((train_embeddings, train_structural), axis=1)
    raw_validation = np.concatenate((validation_embeddings, validation_structural), axis=1)
    mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    train_x = ((raw_train - mean) / scale).astype(np.float32)
    validation_x = ((raw_validation - mean) / scale).astype(np.float32)

    labels = build_labels(train_bank, train_refs)
    validation_labels = build_labels(validation_bank, validation_refs)

    embedding_columns = np.arange(len(train_embeddings[0]))
    structure_columns = np.arange(
        len(train_embeddings[0]), train_x.shape[1]
    )
    variants = {
        "embedding": embedding_columns,
        "structure": structure_columns,
        "combined": np.arange(train_x.shape[1]),
    }
    regularizations = (0.0003, 0.001, 0.003)
    records = []
    best = None

    for variant, columns in variants.items():
        variant_x = train_x[:, columns]
        differences = build_pair_differences(
            variant_x.reshape(len(train_bank), 4, -1), labels
        )
        for regularization in regularizations:
            model_tick = time.perf_counter()
            weights = train_pairwise_ranker(
                differences,
                regularization,
                args.epochs,
                args.learning_rate,
                args.seed,
            )
            train_scores = candidate_scores(
                variant_x.reshape(len(train_bank), 4, -1), weights
            )
            validation_x_variant = validation_x[:, columns].reshape(
                len(validation_bank), 4, -1
            )
            validation_scores = candidate_scores(validation_x_variant, weights)
            train_choices = train_scores.argmax(axis=1)
            validation_choices = validation_scores.argmax(axis=1)
            train_eval = evaluate(
                "train", train_bank, labels, train_choices
            )
            validation_eval = evaluate(
                "validation",
                validation_bank,
                validation_labels,
                validation_choices,
            )
            record = {
                "variant": variant,
                "regularization": regularization,
                "feature_count": len(columns),
                "train_accuracy": train_eval["accuracy"],
                "validation_accuracy": validation_eval["accuracy"],
                "validation_accuracy_by_unique_answers": validation_eval[
                    "accuracy_by_unique_answers"
                ],
                "training_seconds": round(time.perf_counter() - model_tick, 4),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

            if best is None or record["validation_accuracy"] > best["record"]["validation_accuracy"]:
                best = {
                    "record": record,
                    "variant": variant,
                    "columns": columns,
                    "weights": weights,
                }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "weights.npz"
    np.savez_compressed(
        weights_path,
        weights=best["weights"],
        feature_mean=mean[best["columns"]],
        feature_scale=scale[best["columns"]],
        embedding_count=np.asarray(len(embedding_columns), dtype=np.int32),
    )
    config = {
        "model_version": 1,
        "selected_variant": best["variant"],
        "embedding_count": int(len(embedding_columns)),
        "structural_feature_names": list(FEATURE_NAMES),
        "selected_feature_count": int(len(best["columns"])),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "regularization": best["record"]["regularization"],
        "seed": args.seed,
        "train_accuracy": best["record"]["train_accuracy"],
        "validation_accuracy": best["record"]["validation_accuracy"],
        "validation_accuracy_by_unique_answers": best["record"][
            "validation_accuracy_by_unique_answers"
        ],
        "all_records": records,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
        "weights_bytes": weights_path.stat().st_size,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
