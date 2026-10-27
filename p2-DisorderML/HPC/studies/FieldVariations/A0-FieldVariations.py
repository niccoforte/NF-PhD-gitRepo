"""Thirteen isolated independent field fits: feature, residual, box and Gaussian."""
import argparse
import gc
from pathlib import Path
import runpy

CASES = (("crack_face", "FT"), ("residual", "UT"), ("residual", "FT"),
         ("box2", "FT"), ("box4", "FT"),
         *((f"gaussian_a{a}_w{w}", m) for a in (1, 3) for w in (.15, .25) for m in ("UT", "FT")))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json", required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--mode", choices=("UT", "FT"), default="FT")
    p.add_argument("--variant", choices=sorted({v for v, _ in CASES}), default="crack_face")
    p.add_argument("--epochs", type=int, default=450)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nsims", default="all")
    p.add_argument("--batch", type=int)
    p.add_argument("--data-path", default="HPC")
    p.add_argument("--run-root")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args(argv)
    a.nsims = None if a.nsims == "all" else int(a.nsims)
    if a.epochs < 1 or (a.batch is not None and a.batch < 1) or (a.nsims is not None and a.nsims < 1):
        p.error("Epochs, batch and bounded nsims must be positive")
    if not a.preflight and (a.variant, a.mode) not in CASES:
        p.error("Crack-face and crack-tip box runs are FT only")
    return a


def main(argv=None):
    import json
    import torch
    from resources import MLfield_variations
    a = parse_args(argv)
    base = runpy.run_path(str(Path(__file__).resolve().parents[1]/"Locality/A0-Locality.py"))
    def run(case):
        return base["run"](case, configure=MLfield_variations.configure_field_variation,
            study="field_variations", source_paths=(__file__, MLfield_variations.__file__))
    if not a.preflight:
        return run(a)
    full = argparse.Namespace(**vars(a)); full.nsims = None
    data = base["load_data"](full)
    base["check_reference"](data, json.loads(Path(a.reference_model_json).read_text()), True)
    del data; gc.collect()
    for variant, mode in CASES:
        case = argparse.Namespace(**vars(a))
        case.variant, case.mode, case.nsims, case.epochs = variant, mode, 64, 1
        case.run_label = a.run_label+"-"+variant+"-"+mode
        run(case); gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("FIELD_VARIATIONS_PREFLIGHT_OK: full-data contract and13 GPU train/export/reload cases", flush=True)


if __name__ == "__main__":
    main()
