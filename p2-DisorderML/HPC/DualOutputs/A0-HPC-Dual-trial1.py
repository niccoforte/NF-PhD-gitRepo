#!/usr/bin/env python3
"""HPO-informed dual baseline; reuses the tested runner and its result collection."""

import hashlib
from pathlib import Path
import runpy
import sys


# Explicit CLI arguments override this preset, including --run-label.
# Rationale and the four independent HPO sources are in README.md.
DEFAULT_ARGS = [
    "--nsims", "all", "--epochs", "450", "--batch", "2",
    "--lr", "1e-4", "--weight-decay", "1e-8",
    "--field-d-model", "256", "--field-n-heads", "4",
    "--field-n-layers", "4", "--field-ff-mult", "4", "--field-dropout", "0.20",
    "--curve-d-model", "256", "--curve-n-heads", "4",
    "--curve-n-layers", "2", "--curve-ff-mult", "4", "--curve-dropout", "0.15",
    "--curve-pool", "mean", "--curve-cls-token",
    "--encoder-act", "relu", "--pos-encoding", "learned",
    "--loss", "mse", "--scheduler-factor", "0.7", "--scheduler-patience", "12",
    "--early-stop-patience", "75", "--eval-split", "val",
]


def main(argv=None):
    script = Path(__file__)
    runner = runpy.run_path(str(script.with_name("A0-HPC-Dual-test.py")))
    return runner["main"](
        DEFAULT_ARGS + list(sys.argv[1:] if argv is None else argv),
        preset={"name": "trial1", "script": script.name,
                "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "basis": "Independent UT/FT field and full-201 field-to-curve HPO; not dual HPO"},
    )


if __name__ == "__main__":
    main()
