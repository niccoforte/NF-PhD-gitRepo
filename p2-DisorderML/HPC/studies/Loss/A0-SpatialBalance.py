"""FT-only spatial coefficient screen; unchanged independent final field model."""
import argparse
import gc
from pathlib import Path
import runpy

COEFFICIENTS = (0., .001, .003, .01, .1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json", required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--spatial-coefficient", type=float, choices=COEFFICIENTS, default=.001)
    p.add_argument("--data-path", default="HPC")
    p.add_argument("--run-root")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=450)
    p.add_argument("--nsims", default="all")
    p.add_argument("--batch", type=int)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args(argv)
    a.nsims = None if a.nsims == "all" else int(a.nsims)
    if a.epochs < 1 or (a.batch is not None and a.batch < 1) or (a.nsims is not None and a.nsims < 1):
        p.error("Positive training budgets required")
    a.mode = "FT"
    a.variant = "spatial_"+str(a.spatial_coefficient).replace(".", "p")
    return a


def configure(network, data, args):
    from resources.MLdual import DualLoss
    from resources.MLfield import field_loss_from_data
    # Both loss modules expose the same component logs. Only FT executes/trains;
    # the inactive UT loss has zero added weights and no active targets.
    losses = {m: field_loss_from_data(data, m,
        spatial_weight=args.spatial_coefficient if m == "FT" else 0.,
        temporal_weight=0., localization_gain=0.) for m in ("UT", "FT")}
    return DualLoss(field_loss=losses)


def main(argv=None):
    import json
    import torch
    from resources import MLfield
    a = parse_args(argv)
    base = runpy.run_path(str(Path(__file__).parents[1]/"Locality/A0-Locality.py"))

    def run(case):
        return base["run"](case, configure=configure, study="spatial_balance",
            source_paths=(__file__, MLfield.__file__))

    if not a.preflight:
        return run(a)
    full = argparse.Namespace(**vars(a)); full.nsims = None
    data = base["load_data"](full)
    base["check_reference"](data, json.loads(Path(a.reference_model_json).read_text()), True)
    del data; gc.collect()
    for coefficient in COEFFICIENTS:
        case = argparse.Namespace(**vars(a))
        case.spatial_coefficient = coefficient
        case.variant = "spatial_"+str(coefficient).replace(".", "p")
        case.nsims, case.epochs = 64, 1
        case.run_label = a.run_label+"-"+case.variant
        run(case); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("SPATIAL_BALANCE_PREFLIGHT_OK: full identities and five FT train/export/reload cases", flush=True)


if __name__ == "__main__":
    main()
