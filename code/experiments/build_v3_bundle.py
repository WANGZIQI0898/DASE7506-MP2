"""Build the fixed difficulty-routed v3 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

CODE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    output_dir = CODE_DIR / "runs" / "verifier_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "runs/verifier_v1/weights.npz": "verifier_v1_weights.npz",
        "runs/verifier_v2/linear_weights.npz": "linear_weights.npz",
        "runs/verifier_v2/set_encoder_weights.npz": "set_encoder_weights.npz",
        "runs/transformer_2l_d035_wd5/state_dict.pt": "transformer_state.pt",
        "runs/transformer_2l_d035_wd5/feature_scaler.npz": "transformer_scaler.npz",
        "runs/group_ranker_candidate/weights.npz": "group_weights.npz",
    }
    for source, target in copies.items():
        shutil.copyfile(CODE_DIR / source, output_dir / target)

    transformer_config = json.loads(
        (CODE_DIR / "runs" / "transformer_2l_d035_wd5" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    compact_transformer_config = {
        key: transformer_config[key]
        for key in (
            "hidden_size",
            "layers",
            "heads",
            "feedforward_size",
            "dropout",
            "best_epoch",
            "validation",
            "training_seconds",
        )
    }
    config = {
        "model_version": 3,
        "architecture": "difficulty_routed_experts",
        "routing": {
            "unique_answers_1": "v2_equal_probability_ensemble",
            "unique_answers_2": "v2_equal_probability_ensemble",
            "unique_answers_3": "two_layer_transformer",
            "unique_answers_4": "answer_group_ranker_with_v1_candidate_choice",
        },
        "v2_fusion": {"linear_listwise": 0.5, "set_encoder": 0.5},
        "transformer": compact_transformer_config,
        "group_feature_layout": {
            "v1_candidate_features": "embedding_1024 + structural_20",
            "group_features": "mean_and_max_embedding + mean_and_max_structural + mean_equation + group_size + parseable + max_and_mean_v1_score",
        },
        "validation": {
            "accuracy": 94.4862,
            "correct": 754,
            "n": 798,
            "non_unanimous_accuracy": 89.4988,
            "non_unanimous_correct": 375,
            "non_unanimous_n": 419,
            "by_unique_answer_count": {
                "1": {"n": 379, "correct": 379},
                "2": {"n": 249, "correct": 237},
                "3": {"n": 130, "correct": 110},
                "4": {"n": 40, "correct": 28},
            },
        },
        "test_used_for_selection": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
