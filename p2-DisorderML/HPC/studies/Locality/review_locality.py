"""Read completed single-seed locality fields; no training or curve inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS=("control","degree","graph","gated_graph","adjacency","geometry","cnn1","cnn2")
LABELS=("Control","Degree","GNN","Gated GNN","Adjacency","Edge geometry","CNN1","CNN2")


def verify_download(root,manifest):
    files,total,runs=0,0,set()
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        expected,name=line.split("  ",1);path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError("Download path escapes result root")
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(4194304),b""):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError("Checksum mismatch: "+name)
        files+=1;total+=path.stat().st_size;runs.add(Path(name).parts[0])
    if len(runs)!=16:raise ValueError("Expected all sixteen Locality runs")
    return dict(files=files,bytes=total,runs=len(runs),sha256="all matched")


def collect(root,suite):
    rows=[];contracts={}
    for mode in ("UT","FT"):
        for variant in VARIANTS:
            path=root/f"{suite}-{variant}-{mode}-s42"
            meta=json.loads((path/"run_metadata.json").read_text())
            metrics=json.loads((path/"results/metrics.json").read_text())
            data=json.loads((path/"model_data.json").read_text())
            if meta["status"]!="complete" or meta["checkpoint_reload"]!="passed" or metrics["evaluation_split"]!="val":
                raise ValueError("Require complete validation/reload artifacts")
            if meta["variant"]!=variant or meta["mode"]!=mode:raise ValueError("Run identity mismatch")
            for key in ("sample_ids","normalizers","node_masks"):
                value=hashlib.sha256(json.dumps(data[key],sort_keys=True).encode()).hexdigest()
                if contracts.setdefault(key,value)!=value:raise ValueError("Changed common data: "+key)
            with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
                if list(map(str,a["sample_ids"]))!=list(map(str,data["sample_ids"]["val"])):raise ValueError("IDs differ")
                pred,truth=a[f"{mode}_val_field_outputs"],a[f"{mode}_val_field_truth"]
                valid=a[f"{mode}_val_field_mask"].astype(bool)&a[f"{mode}_node_mask"][None,:,None].astype(bool)&np.isfinite(truth)
                for tag,array in (("truth",truth),("mask",valid)):
                    value=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    if contracts.setdefault((mode,tag),value)!=value:raise ValueError("Targets or masks differ")
                rmse=np.sqrt(np.mean((pred[valid].astype(float)-truth[valid])**2))
            diag=metrics["diagnostics"]["field"][mode]
            if not np.isclose(rmse,diag["rmse"],rtol=3e-6):raise ValueError("Saved RMSE does not reproduce")
            if diag["mean_field_baseline_source"]!="train_mean_field":raise ValueError("Require TRAIN-fitted mean")
            baseline=diag["mean_field_baseline_rmse"]
            if not np.isfinite(baseline) or baseline<=0 or not np.isclose(baseline,contracts.setdefault((mode,"baseline"),baseline),rtol=1e-8):raise ValueError("Mean-field denominator changed")
            if not np.isclose(diag["skill_vs_mean_field_rmse"],1-rmse/baseline,atol=1e-7):raise ValueError("Mean-relative skill differs")
            motion=pd.read_csv(path/f"results/{mode}_val_field_motion.csv")
            rows.append(dict(mode=mode,variant=variant,rmse=rmse,skill=100*diag["skill_vs_mean_field_rmse"],
                local_jump=motion.local_jump_nrmse_percent.mean(),timing=motion.jump_interval_mae.mean(),
                collapse_ratio=diag["collapse_ratio"],baseline_rmse=diag["mean_field_baseline_rmse"],
                hours=meta["training_seconds"]/3600,parameters=meta["active_parameters"],job=meta["slurm_job_id"],
                revision=meta["source_revision"],best_epoch=metrics["best_epoch"],run=str(path)))
    table=pd.DataFrame(rows)
    controls=table[table.variant=="control"].set_index("mode")
    table["change"]=100*(table.rmse/table["mode"].map(controls.rmse)-1)
    return table



def main(argv=None):
    from resources.MLstudy_reports import (collect_fields, comparison_figures, loss_figures,
        architecture_figure, field_examples, figure)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--suite",default="locality-261026-r1")
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/locality-study")
    p.add_argument("--damage-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--download-manifest",type=Path)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    if a.download_manifest:
        receipt=verify_download(a.root,a.download_manifest)
        (a.out/"download-verification.json").write_text(json.dumps(receipt,indent=2))
    legacy=collect(a.root,a.suite)
    api=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    entries=[dict(mode=m,variant=v,label=label,path=a.root/f"{a.suite}-{v}-{m}-s42")
        for m in ("UT","FT") for v,label in zip(VARIANTS,LABELS)]
    table,regions,frames=collect_fields(entries,a.damage_root,api)
    table.to_csv(a.out/"metrics.csv",index=False);regions.to_csv(a.out/"confirmed-fracture-regions.csv",index=False)
    frames.to_csv(a.out/"frame-metrics.csv",index=False);legacy.to_csv(a.out/"motion-and-capacity.csv",index=False)
    comparison_figures(table,regions,a.out);loss_figures(table,a.out,{"UT":"cnn2","FT":"cnn2"})
    architecture_figure(a.out,"Isolated local processing before global attention:\nGNN / gated GNN / CNN1 / CNN2\nOR degree / pairwise attention bias\nNo candidate adopted yet")
    for mode in ("UT","FT"):
        path=a.root/f"{a.suite}-cnn2-{mode}-s42";descriptor=json.loads((path/"model_data.json").read_text())
        field_examples(path,mode,a.out,"CNN2",a.damage_root,api["training_mean"](a.damage_root,descriptor,mode))
    image=lambda name,caption:figure(a.out,name,caption)
    text=["# Locality: independent field predictions\n\n## Recap and verdict\n\n",
        "All 16 fits completed. CNN2 remains the strongest confirmation candidate: global RMSE −2.39% UT / −12.47% FT, confirmed fracture endpoints −4.41% / −13.22%, strut relative motion −5.79% / −13.56%. One seed 42; no adopted winner or claimed curve improvement. Same plain 384/8/3 field Transformer, split, MSE and stopping; each change is isolated.\n\n",
        image("architecture-comparison.png","Control versus isolated locality changes; CNN2 is a candidate"),
        "## Global performance versus control\n\nNegative is better. Matched validation IDs, masks, physical targets and train-fitted scales are checked.\n\n",
        image("global-control.png","Global RMSE changes"),
        "## Global performance versus TRAIN mean\n\nSkill = 100×(1−model RMSE/TRAIN-mean RMSE); zero matches mean-predictor error. Because the reference error is fixed, this score is mathematically linked to RMSE. It does not by itself show that prediction arrays moved towards the mean. The variation ratio is a separate collapse diagnostic.\n\n",
        image("global-mean.png","Accuracy beyond the mean field"),image("diversity.png","Predicted versus true specimen variation"),
        image("UT-mean-field-examples.png","UT TRAIN-mean reference comparison, not a duplicate error gallery"),
        image("FT-mean-field-examples.png","FT TRAIN-mean reference comparison"),
        "## STATUS fracture regions versus control\n\nActual intact→failed strut endpoints and signed strut motion at onset ±1 retained frame. STATUS is retrospective evaluation only. The larger FT box is a separate overlapping scale.\n\n",
        image("fracture-control.png","Confirmed event changes for every locality variant"),
        "## STATUS fracture regions versus TRAIN mean\n\nExactly the same regions and masks apply to both predictors; reconstructed global mean-baseline error must reproduce saved metrics.\n\n",
        image("fracture-mean.png","Fracture-region accuracy beyond TRAIN mean"),
        "## Best and worst examples\n\nCNN2 seed 42, rank by complete valid field RMSE. U2 at largest true regional increment; full UT, crack-tip FT. Four columns show truth, prediction, signed error and absolute percentage error. Tiny/zero truth is grey; the percentage colour limit is P95, with no aggregate filtering.\n\n",
        image("UT-field-examples.png","UT CNN2 four-column gallery"),image("FT-field-examples.png","FT CNN2 four-column gallery"),
        "## Training time\n\nFinal generators only, no OOF or downstream curves; queue time excluded. Different stopping and parameter counts affect cost.\n\n",
        image("runtime.png","Recorded training hours"),
        "## Loss histories\n\nThe first figure compares validation across all variants; the second compares train/validation for control and CNN2, diagnosing overfitting. These have different purposes. Log axes; selected checkpoints supply predictions, not overfit final epochs.\n\n",
        image("loss-validation.png","All validation histories"),image("loss-train-validation.png","Selected training versus validation"),
        "## What the modules change\n\nGNN uses initial strut neighbours; gated GNN starts with a small residual contribution. Adjacency bias changes attention scores while retaining long-range attention; geometry bias adds initial edge direction/length. CNN uses occupied sites on the reference grid, depthwise 3×3 plus pointwise residual processing before global attention; vacant sites remain masked. Grid neighbours are a spatial prior, not asserted strut connectivity. Degree adds only degree/8. All preserve common base initialization.\n\n",
        "UT has distributed competing localization paths, while FT has a fixed initial crack. Local processing may therefore suit FT more strongly; this is a hypothesis. UT overfitting appears in the plain control too. Neither more layers nor identical UT/FT settings are assumed universally best. Confirmation and a separate CNN3 screen are now authorised; use PROJECT_STATUS/job receipts for live state.\n\n",
        "[Fixed metrics](metrics.csv) · [Fracture entries and denominators](confirmed-fracture-regions.csv) · [Motion/capacity provenance](motion-and-capacity.csv). Validation only; locked test unchanged.\n"]
    from resources.MLstudy_reports import submission_note
    text.append(submission_note(a.out,"confirmation-jobs.tsv","CNN confirmation/depth"))
    (a.out/"README.md").write_text("".join(text),encoding="utf-8")
    from resources.MLstudy_reports import insert_variation_section
    insert_variation_section(a.out/"README.md",entries)
    print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
