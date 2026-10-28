"""Frozen v2 ensemble submission interface."""

from pathlib import Path

import numpy as np

from verifier_features import build_structural_features


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _set_encoder_scores(
    features: np.ndarray, parameters: dict[str, np.ndarray]
) -> np.ndarray:
    hidden_linear = features @ parameters["w1"] + parameters["b1"]
    hidden = np.maximum(hidden_linear, 0.0)
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
    return combined @ parameters["w2"] + parameters["b2"]


def predict(bank, arrays, checkpoint):
    """Return the argmax of an equal-weight two-model probability ensemble."""
    if checkpoint is None:
        raise ValueError("A checkpoint directory is required")
    checkpoint_path = Path(checkpoint)
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Bank and embeddings are not aligned")
    structural = build_structural_features(bank, embeddings)
    raw = np.concatenate(
        (
            embeddings.reshape(len(bank) * 4, -1),
            structural.reshape(len(bank) * 4, -1),
        ),
        axis=1,
    )

    with np.load(checkpoint_path / "linear_weights.npz", allow_pickle=False) as model:
        linear_weights = model["weights"]
        linear_mean = model["feature_mean"]
        linear_scale = model["feature_scale"]
    linear_scores = (
        ((raw - linear_mean) / linear_scale) @ linear_weights
    ).reshape(len(bank), 4)

    with np.load(
        checkpoint_path / "set_encoder_weights.npz", allow_pickle=False
    ) as model:
        parameters = {name: model[name] for name in ("w1", "b1", "w2", "b2")}
        encoder_mean = model["feature_mean"]
        encoder_scale = model["feature_scale"]
    encoder_features = ((raw - encoder_mean) / encoder_scale).reshape(
        len(bank), 4, -1
    )
    encoder_scores = _set_encoder_scores(encoder_features, parameters)

    probabilities = 0.5 * _softmax(linear_scores) + 0.5 * _softmax(encoder_scores)
    return probabilities.argmax(axis=1).tolist()
