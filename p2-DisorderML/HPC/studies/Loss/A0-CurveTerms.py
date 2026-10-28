"""Independent UT/FT true-field curve-loss terms, with fixed MSE selection."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import numpy as np
import torch
from resources.MLcurve_terms import IndependentCurveTrainer,VARIANTS,load_curve_term_model
from resources.MLdual import DualStageTransformer,_json_safe,_move_to_device


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json",required=True)
    p.add_argument("--run-label",required=True)
    p.add_argument("--mode",choices=("UT","FT"),default="UT")
    p.add_argument("--variant",choices=VARIANTS,default="mse")
    p.add_argument("--data-path",default=os.environ.get("ML_DATA_ROOT","HPC"))
    p.add_argument("--run-root",type=Path,default=os.environ.get("ML_RUN_ROOT"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=450)
    p.add_argument("--nsims",default="all")
    p.add_argument("--batch",type=int)
    p.add_argument("--allow-cpu",action="store_true")
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args(argv);a.nsims=None if a.nsims=="all" else int(a.nsims)
    if a.epochs<1 or (a.nsims is not None and a.nsims<1) or (a.batch is not None and a.batch<1):
        p.error("Positive training budgets required")
    return a


def reference_data(args,descriptor):
    helper=runpy.run_path(str(Path(__file__).parents[1]/"Locality/A0-Locality.py"))
    data=helper["load_data"](args);helper["check_reference"](data,descriptor,args.nsims is None)
    if not descriptor["model_config"]["curve_model"].get("independent_tasks"):
        raise ValueError("Require independent curve stages")
    return data


def run(args):
    from resources.MLmodels import _mp_run_root,_mp_slugify
    from resources import MLcurve_terms
    if not torch.cuda.is_available() and not args.allow_cpu:raise RuntimeError("Production requires CUDA")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    source=Path(args.reference_model_json);descriptor=json.loads(source.read_text());data=reference_data(args,descriptor)
    container=DualStageTransformer.from_config(descriptor["model_config"])
    stage=container.curve_model.tasks[args.mode];del container
    training=dict(descriptor["training"]);training["batch"]=args.batch or training["batch"]
    training["lr"]*=training["curve_lr_factor"]
    trainer=IndependentCurveTrainer(stage,data,args.mode,args.variant,training,"cuda" if torch.cuda.is_available() else "cpu")
    label=_mp_slugify(args.run_label,max_len=96,preserve_case=True)
    relative=Path(args.mode)/"FieldToCurve/Transformer"/label
    directory=Path(args.run_root or _mp_run_root())/relative
    archive=Path(os.environ["ML_ARCHIVE_ROOT"])/relative if os.environ.get("ML_ARCHIVE_ROOT") else None
    for path in (directory,archive):
        if path is not None and path.exists():raise FileExistsError("Refuse overwrite: "+str(path))
    directory.mkdir(parents=True)
    metadata=dict(study="independent_true_curve_terms",variant=args.variant,mode=args.mode,seed=args.seed,
        curve_training_source="true",curve_evaluation_source="true",run_config=vars(args),
        source_revision=os.environ.get("ML_SOURCE_REVISION"),slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(Path(MLcurve_terms.__file__).read_bytes()).hexdigest(),
        run_layout=dict(task=args.mode,output_kind="FieldToCurve",model="Transformer",run_name=label),
        selection="fixed physical curve MSE / TRAIN-mean MSE",status="training")
    torch.manual_seed(args.seed)
    config=descriptor["metadata"]["run_config"]
    trainer.fit(args.epochs,config["early_stop_patience"],config["early_stop_delta"],directory,metadata)
    trainer.export(directory,metadata)
    loaded,_=load_curve_term_model(directory,trainer.device)
    batch=_move_to_device(next(iter(trainer.loaders["val"])),trainer.device)
    trainer.stage.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(batch["field"],batch["task_features"],batch["node_mask"])[args.mode],
            trainer.predict_batch(batch),rtol=1e-5,atol=1e-6)
    metadata.update(status="complete",checkpoint_reload="passed")
    trainer.save(directory,metadata)
    (directory/"run_metadata.json").write_text(json.dumps(_json_safe(metadata),indent=2))
    metrics_path=directory/"results/metrics.json";metrics=json.loads(metrics_path.read_text());metrics["metadata"]=_json_safe(metadata)
    metrics_path.write_text(json.dumps(metrics,indent=2))
    print("CURVE_TERMS_RUN_OK",args.mode,args.variant,directory,flush=True)


def main(argv=None):
    a=parse_args(argv)
    if not a.preflight:return run(a)
    full=argparse.Namespace(**vars(a));full.nsims=None
    data=reference_data(full,json.loads(Path(a.reference_model_json).read_text()));del data;gc.collect()
    for mode in ("UT","FT"):
        for variant in VARIANTS:
            case=argparse.Namespace(**vars(a));case.mode=mode;case.variant=variant;case.nsims=64;case.epochs=1
            case.run_label=f"{a.run_label}-{variant}-{mode}"
            run(case);gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    print("CURVE_TERMS_PREFLIGHT_OK: full IDs/scales and sixteen independent GPU train/export/reload cases",flush=True)


if __name__=="__main__":main()
