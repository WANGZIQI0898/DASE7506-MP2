"""Train a small four-candidate transformer on train/validation only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, read_jsonl
from train_v2 import evaluate_predictions, unique_answer_counts
from train_verifier import build_labels
from transformer_ranker import CandidateTransformer
from verifier_features import FEATURE_NAMES, build_structural_features


def evaluate_scores(
    scores: np.ndarray,
    bank: list[dict],
    labels: np.ndarray,
    answer_counts: np.ndarray,
) -> dict:
    choices = scores.argmax(axis=1)
    result = evaluate_predictions(
        "validation", bank, labels, choices, answer_counts
    )
    return result


def selection_key(evaluation: dict) -> tuple[float, float, float]:
    by_count = evaluation["accuracy_by_unique_answers"]
    hard_score = (
        0.35 * by_count["2"]["accuracy"]
        + 0.4 * by_count["3"]["accuracy"]
        + 0.25 * by_count["4"]["accuracy"]
    )
    return (
        evaluation["non_unanimous_accuracy"],
        hard_score,
        evaluation["accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output-tag", default="transformer_candidate")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    raw_train = np.concatenate(
        (
            train_embeddings.reshape(len(train_bank) * 4, -1),
            train_structural.reshape(len(train_bank) * 4, -1),
        ),
        axis=1,
    )
    raw_validation = np.concatenate(
        (
            validation_embeddings.reshape(len(validation_bank) * 4, -1),
            validation_structural.reshape(len(validation_bank) * 4, -1),
        ),
        axis=1,
    )
    feature_mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(
        raw_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    train_features = torch.from_numpy(
        ((raw_train - feature_mean) / feature_scale)
        .astype(np.float32)
        .reshape(len(train_bank), 4, -1)
    )
    validation_features = torch.from_numpy(
        ((raw_validation - feature_mean) / feature_scale)
        .astype(np.float32)
        .reshape(len(validation_bank), 4, -1)
    )
    train_targets = torch.from_numpy(
        train_labels.astype(np.float32)
        / train_labels.sum(axis=1, keepdims=True)
    )
    active_indices = np.flatnonzero(train_answer_counts > 1)

    model = CandidateTransformer(
        input_size=train_features.shape[2],
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        feedforward_size=args.feedforward_size,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    rng = np.random.default_rng(args.seed)
    best_state = None
    best_evaluation = None
    best_key = None
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_order = rng.permutation(active_indices)
        total_loss = 0.0
        for start in range(0, len(epoch_order), args.batch_size):
            indices = torch.from_numpy(
                epoch_order[start : start + args.batch_size]
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_features[indices])
            log_probabilities = torch.log_softmax(logits, dim=1)
            loss = -(
                train_targets[indices] * log_probabilities
            ).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            validation_scores = model(validation_features).cpu().numpy()
            train_scores = model(train_features).cpu().numpy()
        validation_evaluation = evaluate_scores(
            validation_scores,
            validation_bank,
            validation_labels,
            validation_answer_counts,
        )
        train_choices = train_scores.argmax(axis=1)
        train_correct = train_labels[
            np.arange(len(train_bank)), train_choices
        ].mean()
        key = selection_key(validation_evaluation)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(epoch_order), 6),
                "train_accuracy": round(100 * float(train_correct), 4),
                "validation_accuracy": validation_evaluation["accuracy"],
                "non_unanimous_accuracy": validation_evaluation[
                    "non_unanimous_accuracy"
                ],
                "accuracy_by_unique_answers": validation_evaluation[
                    "accuracy_by_unique_answers"
                ],
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_evaluation = validation_evaluation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model_dir = CODE_DIR / "runs" / args.output_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "state_dict.pt")
    np.savez_compressed(
        model_dir / "feature_scaler.npz",
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    config = {
        **vars(args),
        "architecture": "two_layer_four_candidate_transformer",
        "feature_names": {
            "embedding_count": train_embeddings.shape[2],
            "structural": list(FEATURE_NAMES),
        },
        "best_epoch": best_epoch,
        "validation": best_evaluation,
        "history": history,
        "training_seconds": round(time.perf_counter() - tick, 4),
        "test_used": False,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_validation": best_evaluation,
                "best_epoch": config["best_epoch"],
                "training_seconds": config["training_seconds"],
                "model_dir": str(model_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
