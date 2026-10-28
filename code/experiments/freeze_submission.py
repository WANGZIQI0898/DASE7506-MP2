"""Create a hash-based freeze record before final test evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checkpoint_dir = CODE_DIR / "runs" / "verifier_v1"
    files = [
        CODE_DIR / "submission.py",
        CODE_DIR / "verifier_features.py",
        CODE_DIR / "common.py",
        CODE_DIR / "requirements.txt",
        CODE_DIR / "experiments" / "baseline.py",
        CODE_DIR / "experiments" / "train_verifier.py",
        CODE_DIR / "experiments" / "analyze_validation.py",
        CODE_DIR / "experiments" / "run_ablations.py",
        CODE_DIR / "experiments" / "run_final_test.py",
        checkpoint_dir / "weights.npz",
        checkpoint_dir / "config.json",
        CODE_DIR / "runs" / "validation-predictions-verifier-v1.json",
        CODE_DIR / "runs" / "validation-score-verifier-v1.json",
        CODE_DIR / "runs" / "ablation-verifier-v1.json",
        CODE_DIR / "runs" / "validation-analysis-verifier-v1.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen evidence is incomplete: {missing}")

    with np.load(checkpoint_dir / "weights.npz", allow_pickle=False) as model:
        array_bytes = sum(array.nbytes for array in model.values())
    config_bytes = (checkpoint_dir / "config.json").stat().st_size
    config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    test_prediction = CODE_DIR / "runs" / "test-predictions-frozen.json"

    manifest = {
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": "verifier_v1",
        "selected_model": {
            key: config[key]
            for key in (
                "selected_variant",
                "embedding_count",
                "structural_feature_names",
                "selected_feature_count",
                "epochs",
                "learning_rate",
                "regularization",
                "seed",
                "train_accuracy",
                "validation_accuracy",
            )
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "configured_threads": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "asset_size": {
            "uncompressed_array_bytes": array_bytes,
            "config_bytes": config_bytes,
            "combined_bytes": array_bytes + config_bytes,
            "limit_bytes": 16 * 2**20,
        },
        "hashes": {
            str(path.relative_to(CODE_DIR)).replace("\\", "/"): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "validation_evidence": {
            "accuracy": 92.10526315789474,
            "correct": 735,
            "n": 798,
        },
        "test_protocol": {
            "status": "not_run_at_freeze",
            "planned_prediction": str(test_prediction.relative_to(CODE_DIR)).replace(
                "\\", "/"
            ),
            "selection_uses_test_references": False,
        },
    }
    output_path = CODE_DIR / "runs" / "freeze-manifest-verifier-v1.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"freeze_manifest": str(output_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
