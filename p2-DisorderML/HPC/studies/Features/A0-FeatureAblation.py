"""Fifteen independent CNN2/MSE physical-context contribution fits."""
import argparse
import gc
import json
from pathlib import Path
import runpy
from resources.MLfeature_ablation import CASES


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json",required=True)
    p.add_argument("--run-label",required=True)
    p.add_argument("--mode",choices=("UT","FT"),default="UT")
    p.add_argument("--variant",choices=sorted({v for m,v in CASES}),default="control")
    p.add_argument("--data-path",default="HPC")
    p.add_argument("--run-root")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=450)
    p.add_argument("--nsims",default="all")
    p.add_argument("--batch",type=int)
    p.add_argument("--allow-cpu",action="store_true")
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args(argv);a.nsims=None if a.nsims=="all" else int(a.nsims)
    if a.epochs<1 or (a.batch is not None and a.batch<1) or (a.nsims is not None and a.nsims<1):p.error("Positive budgets required")
    if not a.preflight and (a.mode,a.variant) not in CASES:p.error("Task-specific feature group is inactive for this task")
    return a


def main(argv=None):
    import torch
    from resources import MLfeature_ablation
    a=parse_args(argv);base=runpy.run_path(str(Path(__file__).parents[1]/"Locality/A0-Locality.py"))
    def run(case):return base["run"](case,configure=MLfeature_ablation.configure_context_ablation,
        study="nodal_feature_ablation",source_paths=(__file__,MLfeature_ablation.__file__))
    if not a.preflight:return run(a)
    full=argparse.Namespace(**vars(a));full.nsims=None
    data=base["load_data"](full);base["check_reference"](data,json.loads(Path(a.reference_model_json).read_text()),True)
    del data;gc.collect()
    for mode,variant in CASES:
        case=argparse.Namespace(**vars(a));case.mode,case.variant,case.nsims,case.epochs=mode,variant,64,1
        case.run_label=f"{a.run_label}-{variant}-{mode}";run(case);gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print("FEATURE_PREFLIGHT_OK: full identity contract and15 train/export/reload cases",flush=True)


if __name__=="__main__":main()
