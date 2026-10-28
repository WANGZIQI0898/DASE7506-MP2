"""Train a small permutation-equivariant set scorer on four candidates."""

from __future__ import annotations

import argparse
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

from common import load_inputs, read_jsonl
from equation_features import EQUATION_FEATURE_NAMES, build_equation_features
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels
from verifier_features import FEATURE_NAMES, build_structural_features


def initialise_parameters(
    input_size: int, hidden_size: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(
            0.0, math.sqrt(2.0 / input_size), size=(input_size, hidden_size)
        ).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "w2": rng.normal(
            0.0, math.sqrt(1.0 / (3 * hidden_size)), size=(3 * hidden_size,)
        ).astype(np.float32),
        "b2": np.zeros(1, dtype=np.float32),
    }


def forward(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    dropout_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hidden_linear = x @ parameters["w1"] + parameters["b1"]
    hidden = np.maximum(hidden_linear, 0.0)
    if dropout_mask is not None:
        hidden *= dropout_mask
    hidden_mean = hidden.mean(axis=1, keepdims=True)
    hidden_max = hidden.max(axis=1, keepdims=True)
    combined = np.concatenate(
        (
            hidden,
            np.broadcast_to(hidden_mean, hidden.shape),
            np.broadcast_to(hidden_max, hidden.shape),
        ),
        axis=2,
    )
    scores = combined @ parameters["w2"] + parameters["b2"]
    cache = {
        "x": x,
        "hidden_linear": hidden_linear,
        "hidden": hidden,
        "dropout_mask": dropout_mask,
        "combined": combined,
    }
    return scores, cache


def backward(
    score_gradient: np.ndarray,
    cache: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    regularization: float,
) -> dict[str, np.ndarray]:
    hidden = cache["hidden"]
    hidden_size = hidden.shape[2]
    grad_w2 = np.einsum("qcf,qc->f", cache["combined"], score_gradient)
    grad_b2 = np.asarray([score_gradient.sum()], dtype=np.float32)
    grad_combined = score_gradient[:, :, None] * parameters["w2"][None, None, :]
    grad_hidden = grad_combined[:, :, :hidden_size].copy()
    grad_mean = grad_combined[:, :, hidden_size : 2 * hidden_size].sum(axis=1)
    grad_max = grad_combined[:, :, 2 * hidden_size :].sum(axis=1)
    grad_hidden += grad_mean[:, None, :] / 4.0
    maxima = hidden.argmax(axis=1)
    question_indices = np.arange(hidden.shape[0])
    hidden_indices = np.arange(hidden_size)
    grad_hidden[question_indices[:, None], maxima, hidden_indices[None, :]] += grad_max
    grad_hidden_linear = grad_hidden * (cache["hidden_linear"] > 0)
    if cache["dropout_mask"] is not None:
        grad_hidden_linear *= cache["dropout_mask"]
    flat_x = cache["x"].reshape(-1, cache["x"].shape[2])
    flat_grad = grad_hidden_linear.reshape(-1, hidden_size)
    grad_w1 = flat_x.T @ flat_grad
    grad_b1 = flat_grad.sum(axis=0)
    grad_w1 += regularization * parameters["w1"]
    grad_w2 += regularization * parameters["w2"]
    return {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}


def train(
    features: np.ndarray,
    labels: np.ndarray,
    active_questions: np.ndarray,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    regularization: float,
    dropout_rate: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[float]]:
    parameters = initialise_parameters(features.shape[2], hidden_size, seed)
    moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    variances = {name: np.zeros_like(value) for name, value in parameters.items()}
    targets = labels / labels.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    batch_size = 256
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0
    losses = []

    for _ in range(epochs):
        order = rng.permutation(np.flatnonzero(active_questions))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if dropout_rate:
                dropout_mask = (
                    rng.random((len(indices), 4, hidden_size)) >= dropout_rate
                ).astype(np.float32) / (1.0 - dropout_rate)
            else:
                dropout_mask = None
            scores, cache = forward(features[indices], parameters, dropout_mask)
            shifted = scores - scores.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            batch_targets = targets[indices]
            epoch_loss -= float(
                np.sum(batch_targets * np.log(probabilities + 1e-12))
                / len(indices)
            )
            score_gradient = (probabilities - batch_targets) / len(indices)
            gradients = backward(
                score_gradient.astype(np.float32), cache, parameters, regularization
            )
            step += 1
            for name in parameters:
                moments[name] = beta1 * moments[name] + (1 - beta1) * gradients[name]
                variances[name] = (
                    beta2 * variances[name] + (1 - beta2) * gradients[name] ** 2
                )
                corrected_moment = moments[name] / (1 - beta1**step)
                corrected_variance = variances[name] / (1 - beta2**step)
                parameters[name] -= (
                    learning_rate
                    * corrected_moment
                    / (np.sqrt(corrected_variance) + epsilon)
                )
        losses.append(epoch_loss)
    return parameters, losses


def predict(
    features: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    scores, _ = forward(features, parameters)
    return scores, scores.argmax(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regularization", type=float, default=0.0003)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output-tag", default="set_mlp_candidate")
    args = parser.parse_args()

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

    raw_train_v1 = np.concatenate((train_embeddings, train_structural), axis=1)
    raw_validation_v1 = np.concatenate(
        (validation_embeddings, validation_structural), axis=1
    )
    raw_train_v2 = np.concatenate((raw_train_v1, train_equations), axis=1)
    raw_validation_v2 = np.concatenate(
        (raw_validation_v1, validation_equations), axis=1
    )
    mean = raw_train_v2.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        raw_train_v2.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    features_v1 = {
        "train": ((raw_train_v1 - mean[: raw_train_v1.shape[1]]) / scale[: raw_train_v1.shape[1]])
        .astype(np.float32)
        .reshape(len(train_bank), 4, -1),
        "validation": (
            (raw_validation_v1 - mean[: raw_validation_v1.shape[1]])
            / scale[: raw_validation_v1.shape[1]]
        )
        .astype(np.float32)
        .reshape(len(validation_bank), 4, -1),
    }
    features_v2 = {
        "train": ((raw_train_v2 - mean) / scale)
        .astype(np.float32)
        .reshape(len(train_bank), 4, -1),
        "validation": ((raw_validation_v2 - mean) / scale)
        .astype(np.float32)
        .reshape(len(validation_bank), 4, -1),
    }

    train_labels = build_labels(train_bank, train_references)
    validation_labels = build_labels(validation_bank, validation_references)
    train_answer_counts = unique_answer_counts(train_bank)
    validation_answer_counts = unique_answer_counts(validation_bank)
    active_questions = train_answer_counts > 1

    records = []
    best = None
    for name, feature_set in (("mlp_v1_features", features_v1), ("mlp_v2_equations", features_v2)):
        model_tick = time.perf_counter()
        parameters, losses = train(
            feature_set["train"],
            train_labels,
            active_questions,
            args.hidden_size,
            args.epochs,
            args.learning_rate,
            args.regularization,
            args.dropout,
            args.seed,
        )
        train_scores, train_choices = predict(feature_set["train"], parameters)
        validation_scores, validation_choices = predict(
            feature_set["validation"], parameters
        )
        train_record = evaluate_predictions(
            "train",
            train_bank,
            train_labels,
            train_choices,
            train_answer_counts,
        )
        validation_record = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            validation_choices,
            validation_answer_counts,
        )
        record = {
            "variant": name,
            "feature_count": int(feature_set["train"].shape[2]),
            "hidden_size": args.hidden_size,
            "parameter_count": int(
                parameters["w1"].size
                + parameters["b1"].size
                + parameters["w2"].size
                + parameters["b2"].size
            ),
            "train_accuracy": train_record["accuracy"],
            "accuracy": validation_record["accuracy"],
            "non_unanimous_accuracy": validation_record["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": validation_record["accuracy_by_unique_answers"],
            "validation_correct": validation_record["correct"],
            "training_seconds": round(time.perf_counter() - model_tick, 4),
            "initial_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "per_question_correct": validation_record["per_question_correct"],
        }
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "per_question_correct"}), flush=True)
        if best is None or (
            record["non_unanimous_accuracy"], record["accuracy"]
        ) > (
            best["record"]["non_unanimous_accuracy"],
            best["record"]["accuracy"],
        ):
            best = {
                "record": record,
                "features": feature_set,
                "parameters": parameters,
            }

    for record in records:
        del record["per_question_correct"]
    output_path = CODE_DIR / "runs" / f"set-mlp-development-{args.output_tag}.json"
    output = {
        "split": "validation",
        "test_used": False,
        "fixed_config": vars(args),
        "records": records,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model_dir = CODE_DIR / "runs" / args.output_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "weights.npz",
        **best["parameters"],
        feature_mean=mean[: best["features"]["train"].shape[2]],
        feature_scale=scale[: best["features"]["train"].shape[2]],
    )
    model_config = {
        **vars(args),
        "selected_variant": best["record"]["variant"],
        "feature_names": {
            "embedding_count": train_embeddings.shape[1],
            "structure": list(FEATURE_NAMES),
            "equation": (
                list(EQUATION_FEATURE_NAMES)
                if best["record"]["variant"] == "mlp_v2_equations"
                else []
            ),
            "architecture": "candidate hidden-mean hidden-max listwise scorer",
        },
        "train_accuracy": best["record"]["train_accuracy"],
        "validation_accuracy": best["record"]["accuracy"],
        "non_unanimous_accuracy": best["record"]["non_unanimous_accuracy"],
        "accuracy_by_unique_answers": best["record"]["accuracy_by_unique_answers"],
        "parameter_count": best["record"]["parameter_count"],
    }
    (model_dir / "config.json").write_text(
        json.dumps(model_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": records,
                "best": best["record"]["variant"],
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
