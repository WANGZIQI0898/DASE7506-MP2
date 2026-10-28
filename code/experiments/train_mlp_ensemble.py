"""Train a fixed five-seed set-encoder probability ensemble."""

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
from train_set_mlp import predict, train
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
    active_questions = train_answer_counts > 1

    seeds = list(range(20261026, 20261031))
    parameters_by_seed = []
    probabilities = []
    records = []
    for seed in seeds:
        model_tick = time.perf_counter()
        parameters, _ = train(
            train_features,
            train_labels,
            active_questions,
            hidden_size=32,
            epochs=35,
            learning_rate=0.001,
            regularization=0.0003,
            dropout_rate=0.3,
            seed=seed,
        )
        scores, choices = predict(validation_features, parameters)
        probability = softmax(scores)
        evaluation = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            choices,
            validation_answer_counts,
        )
        records.append(
            {
                "seed": seed,
                "accuracy": evaluation["accuracy"],
                "correct": evaluation["correct"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "accuracy_by_unique_answers": evaluation[
                    "accuracy_by_unique_answers"
                ],
                "training_seconds": round(time.perf_counter() - model_tick, 4),
            }
        )
        parameters_by_seed.append(parameters)
        probabilities.append(probability)
        print(json.dumps(records[-1]), flush=True)

    ensemble_probabilities = np.mean(probabilities, axis=0)
    ensemble_choices = ensemble_probabilities.argmax(axis=1)
    ensemble_evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        ensemble_choices,
        validation_answer_counts,
    )
    result = {
        "split": "validation",
        "test_used": False,
        "single_seed_records": records,
        "ensemble": {
            "accuracy": ensemble_evaluation["accuracy"],
            "correct": ensemble_evaluation["correct"],
            "non_unanimous_accuracy": ensemble_evaluation[
                "non_unanimous_accuracy"
            ],
            "accuracy_by_unique_answers": ensemble_evaluation[
                "accuracy_by_unique_answers"
            ],
        },
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "mlp-ensemble-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model_dir = CODE_DIR / "runs" / "mlp_ensemble_candidate"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "weights.npz",
        w1=np.stack([parameters["w1"] for parameters in parameters_by_seed]),
        b1=np.stack([parameters["b1"] for parameters in parameters_by_seed]),
        w2=np.stack([parameters["w2"] for parameters in parameters_by_seed]),
        b2=np.stack([parameters["b2"] for parameters in parameters_by_seed]),
        feature_mean=mean,
        feature_scale=scale,
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "five_seed_dropout_set_encoder_probability_mean",
                "seeds": seeds,
                "hidden_size": 32,
                "epochs": 35,
                "learning_rate": 0.001,
                "regularization": 0.0003,
                "dropout": 0.3,
                "embedding_count": train_embeddings.shape[1],
                "structural_feature_names": list(FEATURE_NAMES),
                **result["ensemble"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
