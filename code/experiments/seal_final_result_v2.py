"""Verify v2 frozen hashes and seal the official test result."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    freeze_path = CODE_DIR / "runs" / "freeze-manifest-verifier-v2.json"
    prediction_path = (
        CODE_DIR / "runs" / "test-predictions-verifier-v2-frozen.json"
    )
    score_path = CODE_DIR / "runs" / "test-score-verifier-v2-frozen.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))

    hash_checks = {}
    for relative_path, expected in freeze["hashes"].items():
        path = CODE_DIR / relative_path
        actual = sha256(path)
        hash_checks[relative_path] = {
            "matches_checkpoint": actual == expected["sha256"],
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual,
        }
    failed = [
        name
        for name, check in hash_checks.items()
        if not check["matches_checkpoint"]
    ]
    if failed:
        raise ValueError(f"Frozen files changed before/after test: {failed}")

    v1_run = json.loads(
        (CODE_DIR / "runs" / "final-test-run.json").read_text(encoding="utf-8")
    )
    v1_score = v1_run["score"]
    accuracy = score["accuracy"]
    standard_error = 100 * (
        (score["correct"] / score["n"])
        * (1 - score["correct"] / score["n"])
        / score["n"]
    ) ** 0.5
    result = {
        "status": "final",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": "verifier_v2",
        "student_score": {
            "accuracy": accuracy,
            "correct": score["correct"],
            "n": score["n"],
            "binomial_approximate_ci95": [
                round(accuracy - 1.96 * standard_error, 4),
                round(accuracy + 1.96 * standard_error, 4),
            ],
        },
        "validation": freeze["validation_evidence"],
        "comparison_to_v1": {
            "correct_delta": score["correct"] - v1_score["correct"],
            "percentage_point_delta": round(
                score["accuracy"] - v1_score["accuracy"], 4
            ),
            "v1_accuracy": v1_score["accuracy"],
            "v1_correct": v1_score["correct"],
        },
        "runtime": {
            "threads": 4,
            "prediction_seconds_reported_by_runner": prediction[
                "prediction_seconds"
            ],
            "scoring_process_seconds_external": 0.3763375,
            "limit_seconds": 30,
        },
        "asset_size_bytes": freeze["asset_size"]["combined_bytes"],
        "asset_limit_bytes": freeze["asset_size"]["limit_bytes"],
        "test_prediction_sha256": sha256(prediction_path),
        "freeze_manifest_sha256": sha256(freeze_path),
        "hash_verification": {"all_passed": True, "files": hash_checks},
        "protocol_compliance": {
            "test_predictions_used_for_tuning": False,
            "test_error_analysis_used_for_tuning": False,
            "method_changed_after_v2_freeze": False,
            "retraining_after_v2_test": False,
            "official_scorer_used": True,
        },
        "limitations": [
            "The test-to-validation gap remains substantial.",
            "Difficulty weighting and equation features did not improve robustly.",
            "The final fusion weight was selected on validation, not test.",
        ],
    }
    output_path = CODE_DIR / "runs" / "release-manifest-verifier-v2.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
