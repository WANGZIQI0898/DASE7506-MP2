"""Verify v3 freeze hashes and record its final test outcome without tuning."""

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
    freeze_path = CODE_DIR / "runs" / "freeze-manifest-verifier-v3.json"
    final_run_path = CODE_DIR / "runs" / "final-test-run-verifier-v3.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    final_run = json.loads(final_run_path.read_text(encoding="utf-8"))
    v2_release = json.loads(
        (CODE_DIR / "runs" / "release-manifest-verifier-v2.json").read_text(
            encoding="utf-8"
        )
    )

    hash_checks = {}
    for relative_path, expected in freeze["hashes"].items():
        actual = sha256(CODE_DIR / relative_path)
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
        raise ValueError(f"Frozen v3 files changed before/after test: {failed}")

    score = final_run["score"]
    v2_score = v2_release["student_score"]
    result = {
        "status": "evaluated_candidate",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": "verifier_v3",
        "validation_selected": True,
        "test": score,
        "comparison_to_v2": {
            "correct_delta": score["correct"] - v2_score["correct"],
            "percentage_point_delta": round(
                score["accuracy"] - v2_score["accuracy"], 4
            ),
            "v2_accuracy": v2_score["accuracy"],
            "v2_correct": v2_score["correct"],
        },
        "runtime": {
            "threads": final_run["threads"],
            "prediction_process_seconds_external": final_run[
                "prediction_process_seconds_external"
            ],
            "scoring_process_seconds_external": final_run[
                "scoring_process_seconds_external"
            ],
            "total_evaluation_seconds_external": final_run[
                "total_evaluation_seconds_external"
            ],
            "limit_seconds": 30,
        },
        "asset_size_bytes": freeze["asset_size"]["combined_bytes"],
        "asset_limit_bytes": freeze["asset_size"]["limit_bytes"],
        "test_prediction_sha256": final_run["prediction_sha256"],
        "freeze_manifest_sha256": sha256(freeze_path),
        "hash_verification": {"all_passed": True, "files": hash_checks},
        "protocol_compliance": {
            "test_predictions_used_for_training": False,
            "test_error_analysis_used_for_tuning": False,
            "method_changed_after_v3_freeze": False,
            "retraining_after_v3_test": False,
            "official_scorer_used": True,
        },
        "selection_recommendation": {
            "best_observed_official_score": "verifier_v2",
            "accuracy": v2_score["accuracy"],
            "reason": "v3 improved validation accuracy but did not transfer to test.",
        },
    }
    output_path = CODE_DIR / "runs" / "release-manifest-verifier-v3.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
