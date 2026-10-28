"""Cheap, reference-free features for ranking candidate solutions."""

from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from common import parse_answer


FEATURE_NAMES = (
    "parsed_final_answer",
    "boxed_final_answer",
    "hash_answer_marker",
    "answer_label_marker",
    "log_characters",
    "log_words",
    "log_lines",
    "log_digits",
    "log_equals",
    "log_math_operators",
    "log_math_blocks",
    "log_fraction_tokens",
    "question_word_coverage",
    "question_number_coverage",
    "log_answer_mentions",
    "answer_group_size",
    "answer_is_top_group",
    "answer_group_count",
    "length_rank",
    "relative_length",
)

_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_ANSWER_LABEL_RE = re.compile(r"(?:final\s+answer|answer\s*:)", re.IGNORECASE)
_MATH_OPERATOR_RE = re.compile(r"[+\-*/×÷]")


def _words(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text)}


def _numbers(text: str) -> set[str]:
    return {token.replace(",", "") for token in _NUMBER_RE.findall(text)}


def _log_count(value: int) -> float:
    return math.log1p(value)


def build_structural_features(
    bank: list[dict], embeddings: np.ndarray
) -> np.ndarray:
    """Return one feature row per candidate, aligned with [N, 4] bank order."""
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Bank and embeddings are not aligned")

    result = np.zeros((len(bank), 4, len(FEATURE_NAMES)), dtype=np.float32)
    for q_index, row in enumerate(bank):
        texts = [candidate["text"] for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]
        parsed_counts = Counter(answer for answer in answers if answer is not None)
        top_count = max(parsed_counts.values(), default=0)
        answer_group_count = len(parsed_counts)

        word_sets = [_words(text) for text in texts]
        question_words = _words(row["question"])
        question_numbers = _numbers(row["question"])
        lengths = np.asarray([len(text) for text in texts], dtype=np.float32)
        length_order = np.argsort(np.argsort(lengths))
        length_mean = float(lengths.mean())
        length_std = max(float(lengths.std()), 1.0)

        for i, (text, answer, words) in enumerate(zip(texts, answers, word_sets, strict=True)):
            group_size = parsed_counts.get(answer, 0) if answer is not None else 0
            candidate_numbers = _numbers(text)
            answer_mentions = text.count(str(answer)) if answer is not None else 0
            result[q_index, i] = (
                float(answer is not None),
                float("\\boxed" in text),
                float("####" in text),
                float(_ANSWER_LABEL_RE.search(text) is not None),
                _log_count(len(text)),
                _log_count(len(_WORD_RE.findall(text))),
                _log_count(text.count("\n") + 1),
                _log_count(len(_NUMBER_RE.findall(text))),
                _log_count(text.count("=")),
                _log_count(len(_MATH_OPERATOR_RE.findall(text))),
                _log_count(text.count("$$") // 2),
                _log_count(text.count("\\frac")),
                len(question_words & words) / max(len(question_words), 1),
                len(question_numbers & candidate_numbers) / max(len(question_numbers), 1),
                _log_count(answer_mentions),
                group_size / 4.0,
                float(group_size > 0 and group_size == top_count),
                answer_group_count / 4.0,
                float(length_order[i]) / 3.0,
                (float(lengths[i]) - length_mean) / length_std,
            )
    return result
