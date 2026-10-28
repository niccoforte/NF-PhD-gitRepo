"""Independent STATUS/displacement screen; full production data and budgets."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from resources.MLdual import DUAL_DATA, DualStageTransformer, _json_safe, _move_to_device
from resources.MLdamage import StatusFieldNetwork, StatusTrainer, align_status_bank, load_status_model, select_threshold, status_metrics

VARIANTS=("control","damage","weak","strong")


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-model-json",required=True)
    p.add_argument("--run-label",required=True)
    p.add_argument("--mode",choices=("UT","FT"),default="UT")
    p.add_argument("--variant",choices=VARIANTS,default="control")
    p.add_argument("--data-path",default=os.environ.get("ML_DATA_ROOT","HPC"))
    p.add_argument("--damage-root",type=Path,help="Directory with aligned MULTI allD banks; defaults to data-path/MLdata")
    p.add_argument("--run-root",type=Path,default=os.environ.get("ML_RUN_ROOT"))
    p.add_argument("--seed",type=int,default=42);p.add_argument("--epochs",type=int,default=450)
    p.add_argument("--nsims",default="all");p.add_argument("--allow-cpu",action="store_true")
    p.add_argument("--damage-coefficient",type=float,help="Explicit auxiliary BCE weight; historical presets remain unchanged")
    p.add_argument("--preflight-profile",choices=("screen","balance"),default="screen")
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args(argv);a.nsims=None if a.nsims=="all" else int(a.nsims)
    if a.epochs<1 or (a.nsims is not None and a.nsims<1):p.error("Positive budgets required")
    if a.damage_coefficient is not None and (a.variant not in ("weak","strong") or not np.isfinite(a.damage_coefficient) or a.damage_coefficient <= 0):
        p.error("Positive damage-coefficient requires weak/strong auxiliary variant")
    return a


def load_data(args, descriptor):
    data=DUAL_DATA.from_files(path=args.data_path,nsims=args.nsims,split_frac=.9,split_seed=42,
        range_split=(args.nsims is None,False),load_split=False,save_split=False,
        LAT="FCC",nnx=20,dis="disNodes",dN=.2,d_data="in",freq=False,
        field_input_config={"components":("U1","U2"),"drop_frame0":True,"layout":"auto"})
    if descriptor["metadata"]["curve_training_source"]!="mixed" or not descriptor["model_config"]["field_model"]["independent_tasks"]:
        raise ValueError("Require accepted independent Mixed reference")
    if args.nsims is None:
        current=data.to_metadata()
        for key in ("sample_ids","normalizers","node_masks","reference_task_features"):
            a,b=_json_safe(current[key]),descriptor["data"][key]
            if key=="sample_ids":a={k:list(map(str,v)) for k,v in a.items()};b={k:list(map(str,v)) for k,v in b.items()}
            if a!=b:raise ValueError("Full-data reference mismatch: "+key)
    return data


def export_results(trainer, evaluation, directory, metadata, provenance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from resources.MLmetrics import field_performance_diagnostics, save_field_motion_diagnostics
    mode=trainer.mode;data=trainer.data;folder=directory/"results";folder.mkdir()
    arrays=dict(sample_ids=evaluation["sample_ids"],canonical_coords=np.asarray(data.metadata["canonical_coords"]),
                edge_index=trainer.network.edges.cpu().numpy())
    metrics=dict(kind="independent-status-field",evaluation_split="val",prediction_space="physical",
                 diagnostics=dict(field={}),metadata=metadata,balance=trainer.balance,status_provenance=provenance)
    if trainer.variant!="damage":
        pred=data.inverse_field(mode,evaluation["field"])
        valid=data.splits["val"]["field_mask"][mode]
        truth=np.where(valid,data.inverse_field(mode,data.splits["val"]["field"][mode]),np.nan)
        train=np.where(data.splits["train"]["field_mask"][mode],data.inverse_field(mode,data.splits["train"]["field"][mode]),np.nan)
        present=data.node_masks[mode]; frames=data.metadata["field_frame_values"][mode]
        components=data.metadata["field_components"][mode]
        diag=field_performance_diagnostics(pred[:,present],truth[:,present],
            field_shape=(len(frames),int(present.sum()),len(components)),frame_values=frames,components=components,
            node_labels=np.flatnonzero(present),node_coords=np.asarray(data.metadata["canonical_coords"])[present],train_truth=train[:,present])
        metrics["diagnostics"]["field"][mode]=diag["summary"]
        for name in ("sample_metrics","frame_metrics","component_metrics","node_metrics"):
            table=diag[name].copy()
            if name=="sample_metrics":table["sample_id"]=evaluation["sample_ids"]
            table.to_csv(folder/f"{mode}_val_field_{name}.csv",index=False)
        arrays.update({f"{mode}_val_field_outputs":pred,f"{mode}_val_field_truth":truth,
            f"{mode}_val_field_mask":valid,f"{mode}_node_mask":present,
            f"{mode}_frame_values":np.asarray(frames),f"{mode}_components":np.asarray(components)})
    if "probability" in evaluation:
        probability,target,mask=(evaluation[k] for k in ("probability","damage","damage_mask"))
        threshold=select_threshold(probability,target,mask)
        metrics["damage"]=dict(fixed_05=evaluation["damage_metrics"],
            validation_f1=status_metrics(probability,target,mask,threshold),
            threshold_policy="F1 selected on validation, optimistic diagnostic; freeze before locked test")
        arrays.update(damage_probability=probability,damage_truth=target.astype(np.uint8),damage_mask=mask)
        # Exact AP remains the checkpoint metric; a compact PR curve is display only.
        from sklearn.metrics import precision_recall_curve
        precision,recall,cut=precision_recall_curve(target[mask],probability[mask])
        picks=np.unique(np.linspace(0,len(cut)-1,min(400,len(cut))).astype(int))
        pd.DataFrame(dict(threshold=cut[picks],precision=precision[picks],recall=recall[picks])).to_csv(folder/"damage-pr-curve.csv",index=False)
    np.savez(folder/"predictions.npz",**arrays)
    (folder/"metrics.json").write_text(json.dumps(_json_safe(metrics),indent=2))
    data.to_json(directory/"model_data.json")
    pd.DataFrame(trainer.history).to_csv(directory/"loss_history.csv",index=False)
    if trainer.variant!="damage":save_field_motion_diagnostics(folder,data,mode)
    fig,ax=plt.subplots(figsize=(8,4))
    table=pd.DataFrame(trainer.history)
    for name in ("train_loss","val_loss","val_field","val_damage"):
        ax.plot(table.epoch,table[name].where(table[name]>0),label=name)
    ax.set(yscale="log",xlabel="Epoch",ylabel="Training objective (log)",title=f"{mode}: {trainer.variant}")
    ax.legend();fig.tight_layout();fig.savefig(folder/"loss-history.png",dpi=170);plt.close(fig)


def run(args):
    from resources.MLmodels import _mp_run_root, _mp_slugify
    if not torch.cuda.is_available() and not args.allow_cpu:raise RuntimeError("Production requires CUDA")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    descriptor=json.loads(Path(args.reference_model_json).read_text());data=load_data(args,descriptor)
    root=args.damage_root or Path(data.metadata["source_path"])/"MLdata"
    if not (root/f"MULTI-{args.mode}-disNodes-allD.npz").exists() and root.name!="MLdata":root=root.parent
    status,edges,provenance=align_status_bank(root/f"MULTI-{args.mode}-disNodes-allD.npz",data,args.mode)
    # Construct the exact reference container before extracting one independent
    # stage, preserving both tasks' common initialization convention.
    container=DualStageTransformer.from_config(descriptor["model_config"])
    stage=container.field_model.tasks[args.mode];del container
    network=StatusFieldNetwork(stage,args.mode,edges,data.metadata["canonical_coords"],data.normalizers["geometry"],args.variant)
    label=_mp_slugify(args.run_label,max_len=96,preserve_case=True)
    directory=Path(args.run_root or _mp_run_root())/"MULTI/Damage/Transformer"/label
    archive=Path(os.environ["ML_ARCHIVE_ROOT"])/"MULTI/Damage/Transformer"/label if os.environ.get("ML_ARCHIVE_ROOT") else None
    for path in (directory,archive):
        if path is not None and path.exists():raise FileExistsError("Refuse overwrite: "+str(path))
    directory.mkdir(parents=True)
    training=dict(descriptor["training"])
    trainer=StatusTrainer(network,data,status,training,args.mode,args.variant,"cuda" if torch.cuda.is_available() else "cpu",
        damage_coefficient=args.damage_coefficient)
    metadata=dict(study="status",mode=args.mode,variant=args.variant,seed=args.seed,run_config=vars(args),
        source_revision=os.environ.get("ML_SOURCE_REVISION"),slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        run_layout=dict(task="MULTI",output_kind="Damage",model="Transformer",run_name=label),
        reference_sha256=hashlib.sha256(Path(args.reference_model_json).read_bytes()).hexdigest(),
        damage_provenance=provenance,selection="validation average precision" if args.variant=="damage" else "fixed field physical MSE / train-mean MSE",status="training")
    torch.manual_seed(args.seed)
    evaluation=trainer.fit(args.epochs,descriptor["metadata"]["run_config"]["early_stop_patience"],
        descriptor["metadata"]["run_config"]["early_stop_delta"],directory,metadata)
    export_results(trainer,evaluation,directory,metadata,provenance)
    reloaded,_=load_status_model(directory,trainer.device)
    batch=_move_to_device(next(iter(trainer.loaders["val"])),trainer.device)
    with torch.no_grad():
        actual,expected=trainer.network(batch),reloaded(batch)
        for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=1e-5,atol=1e-6)
    metadata.update(status="complete",checkpoint_reload="passed")
    trainer.save(directory,metadata)
    metrics_path=directory/"results/metrics.json"
    saved_metrics=json.loads(metrics_path.read_text());saved_metrics["metadata"]=_json_safe(metadata)
    metrics_path.write_text(json.dumps(saved_metrics,indent=2))
    (directory/"run_metadata.json").write_text(json.dumps(_json_safe(metadata),indent=2))
    print("STATUS_RUN_OK",args.mode,args.variant,directory,flush=True)


def main(argv=None):
    args=parse_args(argv)
    if not args.preflight:return run(args)
    # Full bank/data alignment and training-class checks precede tiny GPU cases.
    descriptor=json.loads(Path(args.reference_model_json).read_text());full=argparse.Namespace(**vars(args));full.nsims=None
    data=load_data(full,descriptor)
    from resources.MLdamage import class_balance
    root=args.damage_root or Path(data.metadata["source_path"])/"MLdata"
    for mode in ("UT","FT"):
        status,_,_=align_status_bank(root/f"MULTI-{mode}-disNodes-allD.npz",data,mode)
        print("STATUS_FULL_ALIGNMENT_OK",mode,class_balance(status["train"]["target"],status["train"]["mask"]),flush=True)
        del status
    del data;gc.collect()
    cases = ([(m,v,42,None) for m in ("UT","FT") for v in VARIANTS] if args.preflight_profile=="screen" else
             [("UT",v,seed,None) for seed in (43,44) for v in ("control","strong")]+
             [("FT","weak",42,weight) for weight in (.001,.01)])
    for index,(mode,variant,seed,weight) in enumerate(cases):
        case=argparse.Namespace(**vars(args));case.mode=mode;case.variant=variant;case.seed=seed
        case.damage_coefficient=weight;case.epochs=1;case.nsims=64
        case.run_label=f"{args.run_label}-{index}-{variant}-{mode}"
        run(case);gc.collect();torch.cuda.empty_cache()
    print(f"STATUS_PREFLIGHT_OK: full alignment and {len(cases)} GPU train/export/reload cases",flush=True)


if __name__=="__main__":main()
