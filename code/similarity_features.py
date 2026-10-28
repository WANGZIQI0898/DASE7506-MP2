"""Reference-free similarity features for answer-group consistency."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from common import parse_answer


SIMILARITY_FEATURE_NAMES = (
    "embedding_norm",
    "cosine_to_all_centroid",
    "cosine_to_group_centroid",
    "mean_cosine_same_answer",
    "max_cosine_same_answer",
    "mean_cosine_different_answer",
    "max_cosine_different_answer",
    "same_minus_different_cosine",
    "distance_to_group_centroid",
    "group_internal_consistency",
    "centroid_similarity_to_top_group",
    "distance_to_top_group_centroid",
)


def build_similarity_features(
    bank: list[dict], embeddings: np.ndarray
) -> np.ndarray:
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Bank and embeddings are not aligned")
    result = np.zeros(
        (len(bank), 4, len(SIMILARITY_FEATURE_NAMES)), dtype=np.float32
    )
    for q_index, row in enumerate(bank):
        vectors = embeddings[q_index].astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors / np.maximum(norms[:, None], 1e-12)
        similarities = unit @ unit.T
        answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
        groups = defaultdict(list)
        for i, answer in enumerate(answers):
            groups[answer].append(i)
        group_sizes = {key: len(indices) for key, indices in groups.items()}
        top_groups = {
            key for key, size in group_sizes.items() if size == max(group_sizes.values())
        }
        all_centroid = unit.mean(axis=0)
        all_centroid /= max(np.linalg.norm(all_centroid), 1e-12)
        centroids = {}
        for key, indices in groups.items():
            centroid = unit[indices].mean(axis=0)
            centroids[key] = centroid / max(np.linalg.norm(centroid), 1e-12)
        top_centroid = np.mean(
            [centroids[key] for key in top_groups], axis=0
        )
        top_centroid /= max(np.linalg.norm(top_centroid), 1e-12)

        for i, answer in enumerate(answers):
            group = groups[answer]
            other_group_members = [j for j in group if j != i]
            different = [j for j in range(4) if answers[j] != answer]
            same_similarities = [float(similarities[i, j]) for j in other_group_members]
            different_similarities = [float(similarities[i, j]) for j in different]
            if len(group) > 1:
                within_group = [
                    float(similarities[a, b])
                    for pos, a in enumerate(group)
                    for b in group[pos + 1 :]
                ]
                group_consistency = float(np.mean(within_group))
            else:
                group_consistency = 0.0
            result[q_index, i] = (
                math.log1p(float(norms[i])),
                float(unit[i] @ all_centroid),
                float(unit[i] @ centroids[answer]),
                float(np.mean(same_similarities)) if same_similarities else 0.0,
                max(same_similarities, default=0.0),
                float(np.mean(different_similarities)) if different_similarities else 0.0,
                max(different_similarities, default=0.0),
                (
                    float(np.mean(same_similarities)) if same_similarities else 0.0
                )
                - (
                    float(np.mean(different_similarities))
                    if different_similarities
                    else 0.0
                ),
                float(np.linalg.norm(unit[i] - centroids[answer])),
                group_consistency,
                float(centroids[answer] @ top_centroid),
                float(np.linalg.norm(centroids[answer] - top_centroid)),
            )
    return result
