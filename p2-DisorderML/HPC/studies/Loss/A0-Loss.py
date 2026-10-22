"""Case1(i) field/curve loss ablations; production defaults, unchanged selection."""
from pathlib import Path
import runpy
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    runner = Path(__file__).resolve().parents[2]/"DualOutputs/A0-HPC-Dual-test.py"
    return runpy.run_path(str(runner))["main"](args + ["--experiment", "baseline",
        "--independence-case", "1i", "--followup-study", "losses"])


if __name__ == "__main__":
    main()
