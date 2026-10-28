"""Run frozen verifier v2 and the official scorer exactly once."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

CODE_DIR = Path(__file__).resolve().parents[1]
PYTHON = CODE_DIR / ".venv" / "Scripts" / "python.exe"
PREDICTIONS = CODE_DIR / "runs" / "test-predictions-verifier-v2-frozen.json"
SCORE = CODE_DIR / "runs" / "test-score-verifier-v2-frozen.json"
REPORT = CODE_DIR / "runs" / "final-test-run-verifier-v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], environment: dict[str, str]) -> tuple[float, str]:
    tick = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=CODE_DIR,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return time.perf_counter() - tick, completed.stdout.strip()


def main() -> None:
    if PREDICTIONS.exists() or SCORE.exists() or REPORT.exists():
        raise FileExistsError("v2 final-test outputs already exist; refusing to rerun")
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "4"

    overall_tick = time.perf_counter()
    prediction_seconds, prediction_stdout = run(
        [
            str(PYTHON),
            "predict.py",
            "--method",
            "submission_v2.py",
            "--checkpoint",
            "runs/verifier_v2",
            "--split",
            "test",
            "--output",
            str(PREDICTIONS.relative_to(CODE_DIR)).replace("\\", "/"),
        ],
        environment,
    )
    if not PREDICTIONS.is_file():
        raise FileNotFoundError("Prediction process did not create its output")
    prediction_hash = sha256(PREDICTIONS)
    scoring_seconds, scoring_stdout = run(
        [
            str(PYTHON),
            "score_predictions.py",
            "--split",
            "test",
            "--predictions",
            str(PREDICTIONS.relative_to(CODE_DIR)).replace("\\", "/"),
            "--output",
            str(SCORE.relative_to(CODE_DIR)).replace("\\", "/"),
        ],
        environment,
    )
    total_seconds = time.perf_counter() - overall_tick
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": "verifier_v2",
        "split": "test",
        "threads": 4,
        "prediction_process_seconds_external": round(prediction_seconds, 6),
        "scoring_process_seconds_external": round(scoring_seconds, 6),
        "total_evaluation_seconds_external": round(total_seconds, 6),
        "prediction_sha256": prediction_hash,
        "prediction_stdout": prediction_stdout,
        "scoring_stdout": scoring_stdout,
        "score": json.loads(SCORE.read_text(encoding="utf-8")),
    }
    REPORT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
