"""Development-only baselines for answer selection.

This script deliberately accepts only train and validation. Test predictions
must be produced later by the frozen submission through predict.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import parse_answer, read_jsonl, score_choices


def vote_choice(row: dict, tie_breaker: str) -> int:
    candidates = row["candidates"]
    answers = [parse_answer(candidate["text"]) for candidate in candidates]
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return 0

    best_count = max(counts.values())
    eligible = [i for i, answer in enumerate(answers) if counts.get(answer) == best_count]
    if tie_breaker == "first":
        return eligible[0]
    if tie_breaker == "longest":
        return max(eligible, key=lambda i: len(candidates[i]["text"]))
    if tie_breaker == "shortest":
        return min(eligible, key=lambda i: len(candidates[i]["text"]))
    raise ValueError(f"Unknown tie breaker: {tie_breaker}")


def build_choices(bank: list[dict], method: str) -> list[int]:
    if method == "random":
        rng = np.random.default_rng(20261026)
        return rng.integers(0, 4, size=len(bank)).tolist()
    if method == "first":
        return [0] * len(bank)
    if method == "last":
        return [3] * len(bank)
    if method == "shortest":
        return [
            min(range(4), key=lambda i: len(row["candidates"][i]["text"]))
            for row in bank
        ]
    if method == "longest":
        return [
            max(range(4), key=lambda i: len(row["candidates"][i]["text"]))
            for row in bank
        ]
    if method.startswith("vote_"):
        return [vote_choice(row, method.removeprefix("vote_")) for row in bank]
    raise ValueError(f"Unknown method: {method}")


def top_answer_group_diagnostics(bank: list[dict], references: list[dict]) -> dict:
    parseable_counts = Counter()
    top_tie_count = 0
    top_group_contains_correct = 0
    top_tie_contains_correct = 0

    for row, reference in zip(bank, references, strict=True):
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        gold = parse_answer(reference["answer"])
        parseable_counts[sum(answer is not None for answer in answers)] += 1

        counts = Counter(answer for answer in answers if answer is not None)
        if not counts:
            continue
        best_count = max(counts.values())
        eligible = [i for i, answer in enumerate(answers) if counts.get(answer) == best_count]
        contains_correct = any(answers[i] == gold for i in eligible)
        top_group_contains_correct += int(contains_correct)

        has_tie = len({answers[i] for i in eligible}) > 1
        top_tie_count += int(has_tie)
        top_tie_contains_correct += int(has_tie and contains_correct)

    return {
        "parseable_candidate_count": dict(sorted(parseable_counts.items())),
        "top_vote_tie_questions": top_tie_count,
        "top_group_contains_correct": top_group_contains_correct,
        "top_vote_tie_contains_correct": top_tie_contains_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bank_path = CODE_DIR / "data" / "candidates" / f"gsm8k_{args.split}.jsonl"
    reference_path = CODE_DIR / "data" / f"gsm8k_{args.split}.jsonl"
    bank = read_jsonl(bank_path)
    references = read_jsonl(reference_path)
    ids = [row["id"] for row in bank]

    methods = (
        "random",
        "first",
        "last",
        "shortest",
        "longest",
        "vote_first",
        "vote_longest",
        "vote_shortest",
    )
    scores = {}
    for method in methods:
        choices = build_choices(bank, method)
        scores[method] = score_choices(args.split, ids, choices)

    diagnostics = top_answer_group_diagnostics(bank, references)
    diagnostics["top_group_oracle_accuracy"] = (
        100 * diagnostics["top_group_contains_correct"] / len(bank)
    )
    diagnostics["top_vote_tie_oracle_accuracy"] = (
        100 * diagnostics["top_vote_tie_contains_correct"] / len(bank)
    )

    result = {
        "split": args.split,
        "n": len(bank),
        "baselines": {
            method: round(score["accuracy"], 4) for method, score in scores.items()
        },
        "diagnostics": diagnostics,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
