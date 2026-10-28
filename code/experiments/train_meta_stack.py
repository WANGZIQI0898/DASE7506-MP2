"""Cross-fitted stacking of four complementary candidate experts."""

from __future__ import annotations

from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "experiments"))

from common import load_inputs, parse_answer, read_jsonl
from train_set_mlp import forward, train as train_set_encoder
from train_v2 import evaluate_predictions, train_listwise, unique_answer_counts
from train_verifier import (
    build_labels,
    build_pair_differences,
    train_pairwise_ranker,
)
from transformer_ranker import CandidateTransformer
from verifier_features import build_structural_features


EXPERT_NAMES = ("pairwise", "linear_listwise", "set_encoder", "transformer")


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def fit_transform_scaler(
    train_raw: np.ndarray, other_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        train_raw.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    return (
        ((train_raw - mean) / scale).astype(np.float32),
        ((other_raw - mean) / scale).astype(np.float32),
        mean,
        scale,
    )


def train_transformer_fold(
    features: np.ndarray,
    labels: np.ndarray,
    answer_counts: np.ndarray,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = CandidateTransformer(
        input_size=features.shape[2],
        hidden_size=64,
        layers=2,
        heads=4,
        feedforward_size=128,
        dropout=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    features_tensor = torch.from_numpy(features)
    targets = torch.from_numpy(
        labels.astype(np.float32) / labels.sum(axis=1, keepdims=True)
    )
    active = np.flatnonzero(answer_counts > 1)
    rng = np.random.default_rng(seed)
    for _ in range(7):
        model.train()
        order = rng.permutation(active)
        for start in range(0, len(order), 64):
            indices = torch.from_numpy(order[start : start + 64])
            optimizer.zero_grad(set_to_none=True)
            logits = model(features_tensor[indices])
            loss = -(
                targets[indices] * torch.log_softmax(logits, dim=1)
            ).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    return copy.deepcopy(model.state_dict())


def transformer_predict(
    state: dict[str, torch.Tensor], features: np.ndarray
) -> np.ndarray:
    model = CandidateTransformer(
        input_size=features.shape[2],
        hidden_size=64,
        layers=2,
        heads=4,
        feedforward_size=128,
        dropout=0.2,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(features.astype(np.float32))).numpy()


def full_validation_expert_probabilities(
    raw_features: np.ndarray, question_count: int
) -> np.ndarray:
    probabilities = np.zeros((question_count, 4, len(EXPERT_NAMES)), dtype=np.float32)
    with np.load(
        CODE_DIR / "runs" / "verifier_v1" / "weights.npz", allow_pickle=False
    ) as model:
        scores = (
            ((raw_features - model["feature_mean"]) / model["feature_scale"])
            @ model["weights"]
        ).reshape(question_count, 4)
    probabilities[:, :, 0] = softmax(scores)

    with np.load(
        CODE_DIR / "runs" / "verifier_v2" / "linear_weights.npz",
        allow_pickle=False,
    ) as model:
        scores = (
            ((raw_features - model["feature_mean"]) / model["feature_scale"])
            @ model["weights"]
        ).reshape(question_count, 4)
    probabilities[:, :, 1] = softmax(scores)

    with np.load(
        CODE_DIR / "runs" / "verifier_v2" / "set_encoder_weights.npz",
        allow_pickle=False,
    ) as model:
        parameters = {name: model[name] for name in ("w1", "b1", "w2", "b2")}
        features = (
            (raw_features - model["feature_mean"]) / model["feature_scale"]
        ).reshape(question_count, 4, -1)
    probabilities[:, :, 2] = softmax(forward(features, parameters)[0])

    transformer_dir = CODE_DIR / "runs" / "transformer_v3_candidate"
    with np.load(
        transformer_dir / "feature_scaler.npz", allow_pickle=False
    ) as scaler:
        features = (
            (raw_features - scaler["feature_mean"]) / scaler["feature_scale"]
        ).reshape(question_count, 4, -1)
    state = torch.load(
        transformer_dir / "state_dict.pt", map_location="cpu", weights_only=True
    )
    probabilities[:, :, 3] = softmax(transformer_predict(state, features))
    return probabilities


def build_meta_features(
    bank: list[dict],
    structural: np.ndarray,
    expert_probabilities: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    question_count, candidate_count, expert_count = expert_probabilities.shape
    features = []
    names = []
    for expert in range(expert_count):
        names.extend(
            [
                f"{EXPERT_NAMES[expert]}_probability",
                f"{EXPERT_NAMES[expert]}_probability_rank",
                f"{EXPERT_NAMES[expert]}_answer_group_probability",
                f"{EXPERT_NAMES[expert]}_answer_group_rank",
            ]
        )
    names.extend(
        [
            "expert_answer_agreement",
            "parsed_candidate",
            "answer_group_size",
            "answer_group_is_top",
            "answer_group_count",
            "relative_length",
            "length_rank",
            "question_number_coverage",
        ]
    )
    result = np.zeros(
        (question_count, candidate_count, len(names)), dtype=np.float32
    )
    for q_index, row in enumerate(bank):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        answer_counts = Counter(answer for answer in answers if answer is not None)
        top_count = max(answer_counts.values(), default=0)
        group_keys = [
            answer if answer is not None else "__missing__" for answer in answers
        ]
        expert_choices = expert_probabilities[q_index].argmax(axis=1)
        expert_answers = [answers[index] for index in expert_choices]
        lengths = np.asarray([len(candidate["text"]) for candidate in row["candidates"]])
        length_rank = np.argsort(np.argsort(lengths)) / 3.0
        column = 0
        for expert in range(expert_count):
            probability = expert_probabilities[q_index, :, expert]
            probability_rank = np.argsort(np.argsort(probability)) / 3.0
            group_probability = {}
            for key in set(group_keys):
                members = [i for i, candidate_key in enumerate(group_keys) if candidate_key == key]
                group_probability[key] = float(probability[members].sum())
            ordered_group_probabilities = sorted(
                set(group_probability.values()), reverse=True
            )
            group_rank = {
                value: rank / max(len(ordered_group_probabilities) - 1, 1)
                for rank, value in enumerate(ordered_group_probabilities)
            }
            for i in range(candidate_count):
                result[q_index, i, column : column + 4] = (
                    probability[i],
                    probability_rank[i],
                    group_probability[group_keys[i]],
                    group_rank[group_probability[group_keys[i]]],
                )
            column += 4
        for i, answer in enumerate(answers):
            agreement = sum(expert_answer == answer for expert_answer in expert_answers)
            group_size = answer_counts.get(answer, 0) if answer is not None else 0
            result[q_index, i, column:] = (
                agreement / expert_count,
                float(answer is not None),
                group_size / 4.0,
                float(group_size > 0 and group_size == top_count),
                len(answer_counts) / 4.0,
                (lengths[i] - lengths.mean()) / max(lengths.std(), 1.0),
                length_rank[i],
                structural[q_index, i, 13],
            )
    return result, names


def main() -> None:
    torch.set_num_threads(4)
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

    oof = np.zeros(
        (len(train_bank), 4, len(EXPERT_NAMES)), dtype=np.float32
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261026)
    for fold, (fit_indices, held_indices) in enumerate(
        splitter.split(train_bank, train_answer_counts), start=1
    ):
        fold_tick = time.perf_counter()
        train_raw = raw_train[fit_indices]
        held_raw = raw_train[held_indices]
        fit_x, held_x, _, _ = fit_transform_scaler(train_raw, held_raw)
        fit_x = fit_x.reshape(len(fit_indices), 4, -1)
        held_x = held_x.reshape(len(held_indices), 4, -1)
        fit_labels = train_labels[fit_indices]
        fit_counts = train_answer_counts[fit_indices]
        fit_question_weights = (fit_counts > 1).astype(np.float64)

        differences = build_pair_differences(fit_x, fit_labels)
        pairwise_weights = train_pairwise_ranker(
            differences, 0.001, 80, 0.02, 20261026
        )
        oof[held_indices, :, 0] = softmax(
            np.einsum("qcf,f->qc", held_x, pairwise_weights)
        )

        linear_weights = train_listwise(
            fit_x,
            fit_labels,
            fit_question_weights,
            0.001,
            80,
            0.02,
            20261026,
        )
        oof[held_indices, :, 1] = softmax(
            np.einsum("qcf,f->qc", held_x, linear_weights)
        )

        set_parameters, _ = train_set_encoder(
            fit_x,
            fit_labels,
            fit_counts > 1,
            hidden_size=32,
            epochs=35,
            learning_rate=0.001,
            regularization=0.0003,
            dropout_rate=0.3,
            seed=20261026,
        )
        set_scores = forward(held_x, set_parameters)[0]
        oof[held_indices, :, 2] = softmax(set_scores)

        transformer_state = train_transformer_fold(
            fit_x, fit_labels, fit_counts, 20261026
        )
        transformer_scores = transformer_predict(transformer_state, held_x)
        oof[held_indices, :, 3] = softmax(transformer_scores)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "fit_questions": len(fit_indices),
                    "held_questions": len(held_indices),
                    "seconds": round(time.perf_counter() - fold_tick, 4),
                }
            ),
            flush=True,
        )

    full_validation_probabilities = full_validation_expert_probabilities(
        raw_validation, len(validation_bank)
    )
    train_meta, meta_names = build_meta_features(
        train_bank, train_structural, oof
    )
    validation_meta, _ = build_meta_features(
        validation_bank, validation_structural, full_validation_probabilities
    )
    flat_train_meta = train_meta.reshape(-1, train_meta.shape[2])
    flat_validation_meta = validation_meta.reshape(-1, validation_meta.shape[2])
    meta_mean = flat_train_meta.mean(axis=0)
    meta_scale = np.maximum(flat_train_meta.std(axis=0), 1e-6)
    standardized_train_meta = (
        (flat_train_meta - meta_mean) / meta_scale
    ).reshape(train_meta.shape)
    standardized_validation_meta = (
        (flat_validation_meta - meta_mean) / meta_scale
    ).reshape(validation_meta.shape)

    records = []
    candidates = {}
    meta_weights = train_listwise(
        standardized_train_meta,
        train_labels,
        (train_answer_counts > 1).astype(np.float64),
        regularization=0.001,
        epochs=100,
        learning_rate=0.02,
        seed=20261026,
    )
    scores = np.einsum(
        "qcf,f->qc", standardized_validation_meta, meta_weights
    )
    evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        scores.argmax(axis=1),
        validation_answer_counts,
    )
    records.append(
        {
            "variant": "meta_listwise_linear",
            "accuracy": evaluation["accuracy"],
            "correct": evaluation["correct"],
            "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": evaluation["accuracy_by_unique_answers"],
        }
    )
    candidates["meta_listwise_linear"] = {
        "weights": meta_weights,
        "probabilities": softmax(scores),
    }

    for regularization in (0.1, 0.5, 2.0):
        classifier = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            solver="liblinear",
            max_iter=300,
            random_state=20261026,
        )
        classifier.fit(flat_train_meta, train_labels.reshape(-1))
        probabilities = classifier.predict_proba(flat_validation_meta)[:, 1].reshape(
            len(validation_bank), 4
        )
        evaluation = evaluate_predictions(
            "validation",
            validation_bank,
            validation_labels,
            probabilities.argmax(axis=1),
            validation_answer_counts,
        )
        name = f"meta_logistic_C{regularization}"
        records.append(
            {
                "variant": name,
                "accuracy": evaluation["accuracy"],
                "correct": evaluation["correct"],
                "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
                "accuracy_by_unique_answers": evaluation[
                    "accuracy_by_unique_answers"
                ],
            }
        )
        candidates[name] = {"classifier": classifier, "probabilities": probabilities}

    boosting = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=20261026,
    )
    boosting.fit(flat_train_meta, train_labels.reshape(-1))
    probabilities = boosting.predict_proba(flat_validation_meta)[:, 1].reshape(
        len(validation_bank), 4
    )
    evaluation = evaluate_predictions(
        "validation",
        validation_bank,
        validation_labels,
        probabilities.argmax(axis=1),
        validation_answer_counts,
    )
    records.append(
        {
            "variant": "meta_gradient_boosting",
            "accuracy": evaluation["accuracy"],
            "correct": evaluation["correct"],
            "non_unanimous_accuracy": evaluation["non_unanimous_accuracy"],
            "accuracy_by_unique_answers": evaluation["accuracy_by_unique_answers"],
        }
    )
    candidates["meta_gradient_boosting"] = {
        "classifier": boosting,
        "probabilities": probabilities,
    }

    output = {
        "split": "validation",
        "test_used": False,
        "meta_feature_names": meta_names,
        "records": records,
        "elapsed_seconds": round(time.perf_counter() - tick, 4),
    }
    output_path = CODE_DIR / "runs" / "meta-stack-development.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    artifact_dir = CODE_DIR / "runs" / "meta_stack_candidate"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(
        records,
        key=lambda item: (item["non_unanimous_accuracy"], item["accuracy"]),
    )["variant"]
    best = candidates[best_name]
    np.savez_compressed(
        artifact_dir / "meta_data.npz",
        oof_probabilities=oof,
        validation_probabilities=full_validation_probabilities,
        meta_mean=meta_mean,
        meta_scale=meta_scale,
    )
    if "classifier" in best:
        joblib.dump(best["classifier"], artifact_dir / "meta_model.joblib", compress=3)
    else:
        np.savez_compressed(
            artifact_dir / "meta_model.npz", weights=best["weights"]
        )
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {
                "selected_variant": best_name,
                "meta_feature_names": meta_names,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
