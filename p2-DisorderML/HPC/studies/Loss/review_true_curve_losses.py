"""Historical matched true-field curve-loss comparison; validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS=("dual-recovery-261004-curve_true","dual-peak-261008-curve_true-combined_no_location","dual-peak-261008-curve_true-combined_soft")
LABELS=("MSE","Combined, no location","Combined + soft location")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/loss-study")
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True);rows=[];contract={}
    for name,label in zip(RUNS,LABELS):
        run=a.root/name;cfg=json.loads((run/"model.json").read_text());data=json.loads((run/"model_data.json").read_text())
        m=json.loads((run/"results/metrics.json").read_text())
        if json.loads((run/"run_metadata.json").read_text()).get("status")!="complete" or cfg["training"].get("curve_only_source")!="true":
            raise ValueError("Require completed true-field training/evaluation")
        if m["evaluation_split"]!="val" or m["run_config"]["experiment"]!="curve_true":raise ValueError("Expected true-source validation")
        for key,value in (("architecture",cfg["model_config"]),("training",cfg["training"]),("splits",data["sample_ids"]),("scales",data["normalizers"])):
            # best epoch/loss differ by design; all optimizer/source settings stay fixed.
            if key=="training":value={k:v for k,v in value.items() if k not in ("best_epoch","best_loss")}
            digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            if contract.setdefault(key,digest)!=digest:raise ValueError("Historical controls differ: "+key)
        with np.load(run/"results/predictions.npz",allow_pickle=False) as z:
            if list(map(str,z["sample_ids"]))!=list(map(str,data["sample_ids"]["val"])):raise ValueError("IDs differ")
            for mode in ("UT","FT"):
                y,pred=z[f"{mode}_val_curve_truth"],z[f"{mode}_val_curve_outputs"]
                digest=hashlib.sha256(y.tobytes()).hexdigest()
                if contract.setdefault(mode,digest)!=digest:raise ValueError("Curve targets differ")
                error=np.sqrt(np.mean((pred.astype(float)-y)**2));diag=m["diagnostics"]["curve"][mode]
                if not np.isclose(error,diag["rmse"],rtol=3e-6):raise ValueError("RMSE differs")
                baseline=diag["mean_curve_baseline_rmse"]
                if diag.get("mean_curve_baseline_source")!="train_mean_curve" or not np.isfinite(baseline) or baseline<=0:
                    raise ValueError("Require a verified positive TRAIN-mean curve baseline")
                if not np.isclose(diag["skill_vs_mean_curve_rmse"],1-error/baseline,atol=1e-7):raise ValueError("Mean-relative skill differs")
                t=pd.read_csv(run/f"results/{mode}_val_curve_sample_metrics.csv")
                rows.append(dict(label=label,mode=mode,rmse=error,skill=100*diag["skill_vs_mean_curve_rmse"],
                    peak_height_percent=100*abs(t.peak_error).mean()/abs(t.true_peak).mean(),
                    peak_location_percent=100*abs(t.peak_x_error).mean()/np.ptp(z[f"{mode}_curve_x_values"]),
                    work_percent=100*abs(t.energy_error).mean()/abs(t.true_energy).mean(),run=name))
    table=pd.DataFrame(rows);baseline=table[table.label=="MSE"].set_index("mode").rmse
    table["change_percent"]=100*(table.rmse/table["mode"].map(baseline)-1)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,mode in zip(axes,("UT","FT")):
        t=table[table["mode"]==mode]
        ax.bar(range(3),t.skill,color="#3e7b94");ax.set(title=mode,ylabel="RMSE reduction vs training mean (%)",xticks=range(3),xticklabels=[v.replace(", ","\n").replace(" + ","\n+ ") for v in LABELS]);ax.grid(axis="y",alpha=.2)
    fig.suptitle("True-field → curve: historical shared curve Transformer, seed42\nTrained and evaluated on true fields; higher is better; not disorder-only deployment")
    fig.tight_layout();fig.savefig(a.out/"true-field-curve-losses.png",dpi=170);plt.close(fig)
    text=["# Custom curve losses with true fields\n\n**Yes: this test already exists in the earlier shared curve-Transformer workflow.** All three runs use the same architecture, optimizer/source settings, split and793 validation truths, checked here. One seed42; no field training or locked-test evaluation. These are true-field training AND true-field evaluation, not Mixed or predicted-field deployment.\n\n",
        f"![Historical true-field curve losses](<{(a.out/'true-field-curve-losses.png').resolve().as_posix()}>)\n\n",
        "| Loss | Task | RMSE change vs MSE | RMSE reduction vs training mean | Peak height / location / work error (%) |\n|---|---|---:|---:|---|\n"]
    for r in table.itertuples():text.append(f"| {r.label} | {r.mode} | {r.change_percent:+.2f}% | {r.skill:.2f}% | {r.peak_height_percent:.2f} / {r.peak_location_percent:.2f} / {r.work_percent:.2f} |\n")
    text += ["\nThe combined presets do not improve global RMSE even with true inputs here. Therefore predicted-field noise cannot be the sole explanation for their poorer RMSE. This does not prove MSE is best for every peak/work objective or every coefficient. Retain exact hard metrics and distinguish a possible peak trade-off from overall improvement.\n\n",
        "The current combined formulation changes what errors matter: full-curve MSE plus zone MSE0.5 with weights1/5/0.2, derivative0.05, peak height0.25 and work0.1; location is0 or0.02. It can trade post-peak accuracy for the middle region or peak terms. Those coefficients and beta20 were not optimized in this controlled comparison, and checkpoint selection remains fixed overall curve error. A joint training objective and the checkpoint criterion need not reach their minima at the same epoch. These are hypotheses explaining a trade-off, not evidence that more loss terms supply more input information.\n\n",
        "These models share the UT/FT curve encoder and use historical shared-stage training, so they do not settle a fresh Case1(i) curve-loss comparison. If further testing is justified, reuse the existing independent true-field MSE fits and compare the same two combined presets, with predicted-field deployment and true-field oracle evaluation reported separately. No new jobs were submitted by this review.\n\n",
        "[Metrics and source runs](true-field-curve-losses.csv). Saved descriptors, arrays and truths are authoritative; older slide labels are not the control.\n"]
    table.to_csv(a.out/"true-field-curve-losses.csv",index=False);(a.out/"true-field-curves.md").write_text("".join(text),encoding="utf-8")
    print(table.to_string(index=False))


if __name__=="__main__":main()
