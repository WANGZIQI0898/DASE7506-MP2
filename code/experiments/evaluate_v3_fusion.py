"""Evaluate transformer fusion with the frozen v2 ensemble on validation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, read_jsonl
from train_set_mlp import forward
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels
from transformer_ranker import CandidateTransformer
from verifier_features import build_structural_features


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def transformer_probabilities(
    features: np.ndarray, model_dir: Path
) -> tuple[np.ndarray, dict]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = CandidateTransformer(
        input_size=features.shape[2],
        hidden_size=config["hidden_size"],
        layers=config["layers"],
        heads=config["heads"],
        feedforward_size=config["feedforward_size"],
        dropout=config["dropout"],
    )
    state = torch.load(model_dir / "state_dict.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with np.load(model_dir / "feature_scaler.npz", allow_pickle=False) as scaler:
        mean = scaler["feature_mean"]
        scale = scaler["feature_scale"]
    standardized = ((features.reshape(-1, features.shape[2]) - mean) / scale).reshape(
        features.shape
    )
    with torch.inference_mode():
        scores = model(torch.from_numpy(standardized.astype(np.float32))).numpy()
    return softmax(scores), config


def main() -> None:
    torch.set_num_threads(4)
    bank, arrays = load_inputs("validation")
    references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")
    labels = build_labels(bank, references)
    answer_counts = unique_answer_counts(bank)
    embeddings = arrays["embeddings"]
    structural = build_structural_features(bank, embeddings)
    raw = np.concatenate(
        (
            embeddings.reshape(len(bank) * 4, -1),
            structural.reshape(len(bank) * 4, -1),
        ),
        axis=1,
    )

    v2_dir = CODE_DIR / "runs" / "verifier_v2"
    with np.load(v2_dir / "linear_weights.npz", allow_pickle=False) as model:
        linear_scores = (
            ((raw - model["feature_mean"]) / model["feature_scale"]) @ model["weights"]
        ).reshape(len(bank), 4)
    with np.load(v2_dir / "set_encoder_weights.npz", allow_pickle=False) as model:
        parameters = {name: model[name] for name in ("w1", "b1", "w2", "b2")}
        encoder_features = (
            (raw - model["feature_mean"]) / model["feature_scale"]
        ).reshape(len(bank), 4, -1)
    encoder_scores = forward(encoder_features, parameters)[0]
    v2_probabilities = 0.5 * softmax(linear_scores) + 0.5 * softmax(encoder_scores)

    transformer_parts = []
    transformer_records = []
    for tag in (
        "transformer_v3_candidate",
        "transformer_1l_d02",
        "transformer_2l_d035_wd5",
    ):
        model_dir = CODE_DIR / "runs" / tag
        features = np.concatenate(
            (
                embeddings.reshape(len(bank) * 4, -1),
                structural.reshape(len(bank) * 4, -1),
            ),
            axis=1,
        ).reshape(len(bank), 4, -1)
        probabilities, config = transformer_probabilities(features, model_dir)
        transformer_parts.append(probabilities)
        choices = probabilities.argmax(axis=1)
        evaluation = evaluate_predictions(
            "validation", bank, labels, choices, answer_counts
        )
        transformer_records.append(
            {
                "variant": tag,
                "accuracy": evaluation["accuracy"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "best_epoch": config["best_epoch"],
            }
        )
    transformer_mean = np.mean(transformer_parts, axis=0)

    records = []
    for transformer_weight in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0):
        probabilities = (
            (1.0 - transformer_weight) * v2_probabilities
            + transformer_weight * transformer_mean
        )
        choices = probabilities.argmax(axis=1)
        evaluation = evaluate_predictions(
            "validation", bank, labels, choices, answer_counts
        )
        records.append(
            {
                "variant": f"v2_transformer_ensemble_{transformer_weight:.2f}",
                "transformer_weight": transformer_weight,
                "accuracy": evaluation["accuracy"],
                "correct": evaluation["correct"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "accuracy_by_unique_answers": evaluation[
                    "accuracy_by_unique_answers"
                ],
            }
        )
    result = {
        "split": "validation",
        "test_used": False,
        "transformer_models": transformer_records,
        "fusion_records": records,
    }
    output_path = CODE_DIR / "runs" / "v3-fusion-development.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
