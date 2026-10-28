"""Verify frozen hashes and seal the one-time official test result."""

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
    freeze_path = CODE_DIR / "runs" / "freeze-manifest-verifier-v1.json"
    final_run_path = CODE_DIR / "runs" / "final-test-run.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    final_run = json.loads(final_run_path.read_text(encoding="utf-8"))
    hash_checks = {}
    for relative_path, expected in freeze["hashes"].items():
        path = CODE_DIR / relative_path
        actual = sha256(path)
        hash_checks[relative_path] = {
            "matches_checkpoint": actual == expected["sha256"],
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual,
        }
    if not all(check["matches_checkpoint"] for check in hash_checks.values()):
        failed = [
            name for name, check in hash_checks.items() if not check["matches_checkpoint"]
        ]
        raise ValueError(f"Frozen files changed before/after test: {failed}")

    score = final_run["score"]
    accuracy = score["accuracy"]
    standard_error = 100 * (
        (score["correct"] / score["n"])
        * (1 - score["correct"] / score["n"])
        / score["n"]
    ) ** 0.5
    config = json.loads(
        (CODE_DIR / "runs" / "verifier_v1" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    subset = config["validation_accuracy_by_unique_answers"]
    nontrivial_n = sum(subset[key]["n"] for key in ("2", "3", "4"))
    nontrivial_correct = sum(
        subset[key]["n"] * subset[key]["accuracy"] / 100
        for key in ("2", "3", "4")
    )
    release = {
        "status": "final",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": "verifier_v1",
        "student_score": {
            "accuracy": accuracy,
            "correct": score["correct"],
            "n": score["n"],
            "binomial_approximate_ci95": [
                round(accuracy - 1.96 * standard_error, 4),
                round(accuracy + 1.96 * standard_error, 4),
            ],
        },
        "validation": {
            "accuracy": config["validation_accuracy"],
            "non_unanimous_subset_accuracy": round(
                100 * nontrivial_correct / nontrivial_n, 4
            ),
            "non_unanimous_subset_n": nontrivial_n,
        },
        "runtime": {
            "threads": final_run["threads"],
            "prediction_process_seconds": final_run[
                "prediction_process_seconds_external"
            ],
            "scoring_process_seconds": final_run[
                "scoring_process_seconds_external"
            ],
            "total_seconds": final_run["total_evaluation_seconds_external"],
            "limit_seconds": 30,
        },
        "asset_size_bytes": freeze["asset_size"]["combined_bytes"],
        "asset_limit_bytes": freeze["asset_size"]["limit_bytes"],
        "test_prediction_sha256": final_run["prediction_sha256"],
        "freeze_manifest_sha256": sha256(freeze_path),
        "hash_verification": {
            "all_passed": True,
            "files": hash_checks,
        },
        "protocol_compliance": {
            "test_predictions_used_for_tuning": False,
            "test_error_analysis_used_for_tuning": False,
            "method_changed_after_freeze": False,
            "retraining_after_test": False,
            "official_scorer_used": True,
        },
        "limitations": [
            "Validation contains unanimous-answer questions that are absent from test.",
            "The validation-to-test accuracy gap indicates a difficult generalization shift.",
            "Top score margins are not monotonic estimates of correctness probability.",
        ],
    }
    output_path = CODE_DIR / "runs" / "release-manifest-verifier-v1.json"
    output_path.write_text(
        json.dumps(release, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(release, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
