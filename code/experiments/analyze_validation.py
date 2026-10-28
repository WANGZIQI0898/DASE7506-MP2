"""Validation-only statistical and qualitative analysis for verifier v1."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from baseline import build_choices
from common import load_inputs, parse_answer, read_jsonl, score_choices
from verifier_features import FEATURE_NAMES, build_structural_features


def bootstrap_difference(
    method_correct: np.ndarray, baseline_correct: np.ndarray, seed: int = 20261026
) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(method_correct), size=(5000, len(method_correct)))
    differences = (
        method_correct[indices].mean(axis=1) - baseline_correct[indices].mean(axis=1)
    ) * 100
    return {
        "mean_percentage_points": round(float(differences.mean()), 4),
        "ci95_percentage_points": [
            round(float(np.quantile(differences, 0.025)), 4),
            round(float(np.quantile(differences, 0.975)), 4),
        ],
        "bootstrap_samples": 5000,
    }


def main() -> None:
    checkpoint = CODE_DIR / "runs" / "verifier_v1"
    prediction_path = CODE_DIR / "runs" / "validation-predictions-verifier-v1.json"
    bank, arrays = load_inputs("validation")
    references = read_jsonl(CODE_DIR / "data" / "gsm8k_validation.jsonl")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    choices = np.asarray(predictions["selected"], dtype=np.int64)

    embeddings = arrays["embeddings"]
    structural = build_structural_features(bank, embeddings).reshape(len(bank) * 4, -1)
    features = np.concatenate(
        (embeddings.reshape(len(bank) * 4, -1), structural), axis=1
    )
    with np.load(checkpoint / "weights.npz", allow_pickle=False) as model:
        weights = model["weights"]
        mean = model["feature_mean"]
        scale = model["feature_scale"]
    scores = (((features - mean) / scale) @ weights).reshape(len(bank), 4)

    labels = np.zeros((len(bank), 4), dtype=np.int8)
    for q_index, (row, reference) in enumerate(zip(bank, references, strict=True)):
        gold = parse_answer(reference["answer"])
        labels[q_index] = [
            parse_answer(candidate["text"]) == gold
            for candidate in row["candidates"]
        ]
    method_correct = labels[np.arange(len(bank)), choices]

    baseline_choices = np.asarray(
        build_choices(bank, "vote_longest"), dtype=np.int64
    )
    baseline_correct = labels[np.arange(len(bank)), baseline_choices]
    baseline_evaluation = score_choices(
        "validation", [row["id"] for row in bank], baseline_choices.tolist()
    )

    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    quantile_edges = np.quantile(margins, [0.2, 0.4, 0.6, 0.8])
    bin_indices = np.digitize(margins, quantile_edges)
    calibration = []
    for bin_index in range(5):
        mask = bin_indices == bin_index
        calibration.append(
            {
                "bin": bin_index + 1,
                "n": int(mask.sum()),
                "mean_score_margin": round(float(margins[mask].mean()), 4),
                "accuracy": round(100 * float(method_correct[mask].mean()), 4),
            }
        )

    labels_flat = labels.reshape(-1)
    signal_correlations = []
    for column, name in enumerate(FEATURE_NAMES):
        values = structural[:, column]
        if values.std() == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(values, labels_flat)[0, 1])
        signal_correlations.append(
            {
                "feature": name,
                "mean_correct": round(float(values[labels_flat == 1].mean()), 5),
                "mean_incorrect": round(float(values[labels_flat == 0].mean()), 5),
                "correlation_with_correctness": round(correlation, 5),
                "ranker_weight": round(float(weights[-len(FEATURE_NAMES) + column]), 5),
            }
        )
    signal_correlations.sort(
        key=lambda item: abs(item["correlation_with_correctness"]), reverse=True
    )

    error_mask = method_correct == 0
    errors = []
    error_categories = Counter()
    errors_by_unique_answers = Counter()
    for q_index in np.flatnonzero(error_mask):
        row = bank[q_index]
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        counts = Counter(answer for answer in answers if answer is not None)
        gold = parse_answer(references[q_index]["answer"])
        selected_answer = answers[choices[q_index]]
        selected_frequency = counts.get(selected_answer, 0)
        gold_frequency = counts.get(gold, 0)
        top_frequency = max(counts.values(), default=0)
        if selected_answer is None:
            category = "selected_unparseable_candidate"
        elif selected_frequency == top_frequency and gold_frequency == top_frequency:
            category = "tied_top_group_wrong_choice"
        elif selected_frequency > gold_frequency:
            category = "followed_more_votes_than_correct_answer"
        elif selected_frequency < gold_frequency:
            category = "selected_less_supported_answer"
        else:
            category = "equal_support_non_top_groups"
        error_categories[category] += 1
        errors_by_unique_answers[len(set(answers))] += 1
        errors.append(
            {
                "id": row["id"],
                "question": row["question"],
                "selected": int(choices[q_index]),
                "selected_answer": selected_answer,
                "correct_answer": gold,
                "candidate_answers": answers,
                "answer_counts": dict(counts),
                "selected_answer_frequency": selected_frequency,
                "correct_answer_frequency": gold_frequency,
                "top_answer_frequency": top_frequency,
                "category": category,
                "score_margin": round(float(margins[q_index]), 4),
                "selected_text": row["candidates"][choices[q_index]]["text"],
            }
        )
    errors.sort(key=lambda item: item["score_margin"], reverse=True)

    result = {
        "n": len(bank),
        "method": {
            "correct": int(method_correct.sum()),
            "accuracy": round(100 * float(method_correct.mean()), 4),
        },
        "vote_longest_baseline": {
            "correct": baseline_evaluation["correct"],
            "accuracy": round(baseline_evaluation["accuracy"], 4),
        },
        "paired_bootstrap_method_minus_baseline": bootstrap_difference(
            method_correct, baseline_correct
        ),
        "confidence_quintiles": calibration,
        "structural_signal_correlations": signal_correlations,
        "wrong_question_count": int(error_mask.sum()),
        "error_categories": dict(error_categories.most_common()),
        "errors_by_unique_answer_count": {
            str(key): value for key, value in sorted(errors_by_unique_answers.items())
        },
        "highest_confidence_errors": errors[:15],
    }
    output_path = CODE_DIR / "runs" / "validation-analysis-verifier-v1.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "method": result["method"],
                "baseline": result["vote_longest_baseline"],
                "paired_bootstrap": result["paired_bootstrap_method_minus_baseline"],
                "confidence_quintiles": result["confidence_quintiles"],
                "top_signals": result["structural_signal_correlations"][:8],
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
