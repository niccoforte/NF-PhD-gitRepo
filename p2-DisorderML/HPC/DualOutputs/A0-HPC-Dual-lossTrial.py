#!/usr/bin/env python3
"""Controlled dual loss ablation, retaining the HPO-informed Trial 1 architecture."""
from pathlib import Path
import runpy
import sys


def main(argv=None):
    here = Path(__file__).parent
    preset = runpy.run_path(str(here/"A0-HPC-Dual-trial1.py"))["DEFAULT_ARGS"]
    args = preset + ["--field-loss-variant","baseline","--fixed-selection-score"]
    args += list(sys.argv[1:] if argv is None else argv)
    return runpy.run_path(str(here/"A0-HPC-Dual-test.py"))["main"](
        args, preset={"name":"loss-ablation", "basis":"Trial 1 architecture; fixed balanced validation selection"})


if __name__ == "__main__":
    main()
