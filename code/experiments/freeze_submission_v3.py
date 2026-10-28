"""Create a hash-based freeze record for verifier v3."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checkpoint_dir = CODE_DIR / "runs" / "verifier_v3"
    files = [
        CODE_DIR / "submission_v3.py",
        CODE_DIR / "transformer_ranker.py",
        CODE_DIR / "verifier_features.py",
        CODE_DIR / "equation_features.py",
        CODE_DIR / "common.py",
        CODE_DIR / "requirements.txt",
        CODE_DIR / "requirements-training.txt",
        CODE_DIR / "experiments" / "train_v2.py",
        CODE_DIR / "experiments" / "train_set_mlp.py",
        CODE_DIR / "experiments" / "train_transformer.py",
        CODE_DIR / "experiments" / "train_group_ranker.py",
        CODE_DIR / "experiments" / "build_v3_bundle.py",
        checkpoint_dir / "config.json",
        checkpoint_dir / "verifier_v1_weights.npz",
        checkpoint_dir / "linear_weights.npz",
        checkpoint_dir / "set_encoder_weights.npz",
        checkpoint_dir / "transformer_state.pt",
        checkpoint_dir / "transformer_scaler.npz",
        checkpoint_dir / "group_weights.npz",
        CODE_DIR / "runs" / "validation-predictions-verifier-v3.json",
        CODE_DIR / "runs" / "validation-score-verifier-v3.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen evidence is incomplete: {missing}")

    uncompressed_bytes = 0
    for name in (
        "verifier_v1_weights.npz",
        "linear_weights.npz",
        "set_encoder_weights.npz",
        "transformer_scaler.npz",
        "group_weights.npz",
    ):
        with np.load(checkpoint_dir / name, allow_pickle=False) as model:
            uncompressed_bytes += sum(array.nbytes for array in model.values())
    state = torch.load(
        checkpoint_dir / "transformer_state.pt",
        map_location="cpu",
        weights_only=True,
    )
    uncompressed_bytes += sum(
        tensor.numel() * tensor.element_size()
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )
    config_bytes = (checkpoint_dir / "config.json").stat().st_size
    config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    manifest = {
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": "verifier_v3",
        "selected_model": config,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "configured_threads": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "asset_size": {
            "uncompressed_array_bytes": uncompressed_bytes,
            "config_bytes": config_bytes,
            "combined_bytes": uncompressed_bytes + config_bytes,
            "limit_bytes": 16 * 2**20,
        },
        "hashes": {
            str(path.relative_to(CODE_DIR)).replace("\\", "/"): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "validation_evidence": config["validation"],
        "test_protocol": {
            "status": "not_run_at_v3_freeze",
            "planned_prediction": "runs/test-predictions-verifier-v3-frozen.json",
            "selection_uses_test_references": False,
        },
    }
    output_path = CODE_DIR / "runs" / "freeze-manifest-verifier-v3.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
