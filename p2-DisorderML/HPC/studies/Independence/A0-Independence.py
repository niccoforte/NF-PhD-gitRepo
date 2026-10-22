"""Eight-case sharing comparison; production defaults come from the common runner."""
from pathlib import Path
import runpy
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--independence-case" or a.startswith("--independence-case=") for a in args):
        raise SystemExit("Choose an explicit --independence-case for the Independence study.")
    runner = Path(__file__).resolve().parents[2]/"DualOutputs/A0-HPC-Dual-test.py"
    return runpy.run_path(str(runner))["main"](args + ["--experiment", "baseline"])


if __name__ == "__main__":
    main()
