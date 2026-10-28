"""Independent field-only MSE versus MAE and Smooth-L1 screen."""
import argparse
import gc
from pathlib import Path
import runpy

VARIANTS=("control","mae","smooth_l1_0p1","smooth_l1_1")


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json",required=True)
    p.add_argument("--run-label",required=True)
    p.add_argument("--mode",choices=("UT","FT"),default="UT")
    p.add_argument("--variant",choices=VARIANTS,default="control")
    p.add_argument("--data-path",default="HPC")
    p.add_argument("--run-root")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=450)
    p.add_argument("--nsims",default="all")
    p.add_argument("--batch",type=int)
    p.add_argument("--allow-cpu",action="store_true")
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args(argv);a.nsims=None if a.nsims=="all" else int(a.nsims)
    if a.epochs<1 or (a.batch is not None and a.batch<1) or (a.nsims is not None and a.nsims<1):
        p.error("Positive training budgets required")
    return a


def configure(network,data,args):
    from resources.MLdual import DualLoss
    from resources.MLfunc import MaskedFieldMSELoss
    from resources.MLfield import MaskedRobustFieldLoss
    loss=MaskedFieldMSELoss() if args.variant=="control" else MaskedRobustFieldLoss(
        kind="mae" if args.variant=="mae" else "smooth_l1",
        beta=.1 if args.variant=="smooth_l1_0p1" else 1.)
    return DualLoss(field_loss=loss)


def main(argv=None):
    import json
    import torch
    from resources import MLfield
    a=parse_args(argv)
    base=runpy.run_path(str(Path(__file__).parents[1]/"Locality/A0-Locality.py"))
    def run(case):
        return base["run"](case,configure=configure,study="robust_field_loss",source_paths=(__file__,MLfield.__file__))
    if not a.preflight:return run(a)
    full=argparse.Namespace(**vars(a));full.nsims=None
    data=base["load_data"](full)
    base["check_reference"](data,json.loads(Path(a.reference_model_json).read_text()),True)
    del data;gc.collect()
    for mode in ("UT","FT"):
        for variant in VARIANTS:
            case=argparse.Namespace(**vars(a));case.mode=mode;case.variant=variant
            case.nsims,case.epochs=64,1;case.run_label=f"{a.run_label}-{variant}-{mode}"
            run(case);gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    print("ROBUST_FIELD_PREFLIGHT_OK: full IDs/scales and eight train/export/reload cases",flush=True)


if __name__=="__main__":main()
