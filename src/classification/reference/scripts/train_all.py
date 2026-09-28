"""Documented convenience entry point; stages remain separately resumable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, str(root / "scripts" / "build_datasets.py")],
        [sys.executable, str(root / "scripts" / "train_ml.py"), "--max-train-rows", "60000"],
        [sys.executable, str(root / "scripts" / "train_dl.py"), "--max-train-rows", "60000"],
        [sys.executable, str(root / "scripts" / "train_hybrid.py")],
        [sys.executable, str(root / "scripts" / "evaluate_all.py")],
        [sys.executable, str(root / "scripts" / "build_comparison.py")],
        [sys.executable, str(root / "scripts" / "freeze_final_generation.py")],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
