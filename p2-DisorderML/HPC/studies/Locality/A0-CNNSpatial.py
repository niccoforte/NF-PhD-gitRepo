"""Isolated CNN2 plus task-specific spatial loss; reuse final-field trainer."""
import argparse
import gc
import json
from pathlib import Path
import runpy


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json",required=True)
    p.add_argument("--run-label",required=True)
    p.add_argument("--mode",choices=("UT","FT"),default="UT")
    p.add_argument("--variant",choices=("cnn2","cnn3","cnn2_spatial"),default="cnn2_spatial")
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
    return a


def configure(network,data,args):
    from resources.MLlocality import configure_field_locality
    from resources.MLfield import field_loss_from_data,fcc_initial_crack_region
    from resources.MLdual import DualLoss
    configure_field_locality(network,data,"cnn2" if args.variant=="cnn2_spatial" else args.variant)
    if args.variant!="cnn2_spatial":return
    region=fcc_initial_crack_region(data.metadata["canonical_coords"],data.node_masks["FT"])[0] if args.mode=="FT" else None
    losses={m:field_loss_from_data(data,m,spatial_weight=(.1 if m=="UT" else .001) if m==args.mode else 0.,
        spatial_node_mask=region if m=="FT" and m==args.mode else None,temporal_weight=0.,localization_gain=0.) for m in ("UT","FT")}
    return DualLoss(field_loss=losses)


def main(argv=None):
    import torch
    from resources import MLfield
    a=parse_args(argv);base=runpy.run_path(str(Path(__file__).with_name("A0-Locality.py")))
    def run(case):return base["run"](case,configure=configure,study="cnn_spatial_depth",source_paths=(__file__,MLfield.__file__))
    if not a.preflight:return run(a)
    full=argparse.Namespace(**vars(a));full.nsims=None
    data=base["load_data"](full);base["check_reference"](data,json.loads(Path(a.reference_model_json).read_text()),True)
    del data;gc.collect()
    for mode in ("UT","FT"):
        for variant in ("cnn2","cnn3","cnn2_spatial"):
            case=argparse.Namespace(**vars(a));case.mode,case.variant,case.nsims,case.epochs=mode,variant,64,1
            case.run_label=f"{a.run_label}-{variant}-{mode}";run(case);gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    print("CNN_SPATIAL_PREFLIGHT_OK: full identities and six train/export/reload cases",flush=True)


if __name__=="__main__":main()
