"""Refresh the completed sharing comparison; saved validation arrays only."""
import argparse
import json
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from resources.MLstudy_reports import (pipeline_regions,pipeline_global_figures,
    field_examples,curve_examples,figure)


def collect_metrics(root,out):
    cases=("1i","1ii","2ai","2aii","2bi","2bii","3i","3ii")
    rows=[];provenance=[]
    for case in cases:
        for seed in (42,43,44):
            path=root/f"independence-261013-r1-{case}p-s{seed}"
            meta=json.loads((path/"run_metadata.json").read_text())
            metrics=json.loads((path/"results/metrics.json").read_text())
            if meta.get("status")!="complete" or metrics.get("evaluation_split")!="val":
                raise ValueError("Require completed validation artifacts")
            provenance.append(dict(case=case,seed=seed,job=meta.get("slurm_job_id"),
                fit_hours=meta["training_seconds"]/3600,groups=meta.get("group_selection"),
                parameter_counts=meta.get("parameter_counts"),source_revision=meta.get("source_revision")))
            for kind in ("field","curve"):
                for mode in ("UT","FT"):
                    d=metrics["diagnostics"][kind][mode]
                    baseline=d[f"mean_{kind}_baseline_rmse"]
                    if d[f"mean_{kind}_baseline_source"]!=f"train_mean_{kind}" or baseline<=0:
                        raise ValueError("Require a TRAIN-only baseline")
                    suffix="ii" if case.endswith("ii") else "i"
                    rows.append(dict(case=case,label=case[:-len(suffix)]+"("+suffix+")",seed=seed,kind=kind,mode=mode,rmse=d["rmse"],
                        skill=100*(1-d["rmse"]/baseline),collapse_ratio=d["collapse_ratio"]))
    table=pd.DataFrame(rows);base=table[table["case"]=="1i"].set_index(["seed","kind","mode"]).rmse
    table["rmse_change_vs_1i_percent"]=[100*(r.rmse/base.loc[r.seed,r.kind,r.mode]-1) for r in table.itertuples()]
    table.to_csv(out/"per-run-metrics.csv",index=False)
    # Existing detailed source receipts retain their original fields.
    if not (out/"provenance.json").exists():
        (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
    return table,json.loads((out/"provenance.json").read_text())


def refresh(root,out,data_root):
    out.mkdir(parents=True,exist_ok=True)
    table,provenance=collect_metrics(root,out)
    cases=list(dict.fromkeys(table["case"]))
    runs=[dict(variant=c,seed=s,path=root/f"independence-261013-r1-{c}p-s{s}")
          for c in cases for s in (42,43,44)]
    api=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    pipeline_regions(runs,"1i",data_root,api,out)
    display=table.rename(columns={"case":"variant","skill":"skill_percent","rmse_change_vs_1i_percent":"change_percent"})
    pipeline_global_figures(display,out)
    for case in ("1i","3ii"):
        path=root/f"independence-261013-r1-{case}p-s42";data=json.loads((path/"model_data.json").read_text())
        for mode in ("UT","FT"):
            field_examples(path,mode,out,f"Case {case}",data_root,api["training_mean"](data_root,data,mode),prefix=case+"-")
            curve_examples(path,mode,out,f"Case {case}, predicted-field deployment",data_root,prefix=case+"-")
    costs=[]
    for row in provenance:
        path=root/f"independence-261013-r1-{row['case']}p-s{row['seed']}"
        meta=json.loads((path/"run_metadata.json").read_text());bank=meta.get("field_bank")
        bank_h=json.loads((root/Path(bank).name/"run_metadata.json").read_text())["training_seconds"]/3600 if bank else 0
        costs.append(dict(case=row["case"],seed=row["seed"],fit_h=meta["training_seconds"]/3600,bank_h=bank_h))
    cost=pd.DataFrame(costs);cost.to_csv(out/"training-cost.csv",index=False)
    values=cost.groupby("case",sort=False)[["fit_h","bank_h"]].mean().reindex(cases)
    fig,ax=plt.subplots(figsize=(12,5));ax.bar(range(len(cases)),values.fit_h,label="Case fit");ax.bar(range(len(cases)),values.bank_h,bottom=values.fit_h,label="Reusable OOF bank")
    ax.set(ylabel="Mean training hours per seed",xticks=range(len(cases)),xticklabels=cases,title="Prospective workflow costs; a reused bank is included in each applicable case")
    ax.legend();fig.tight_layout();fig.savefig(out/"training-cost.png",dpi=170);plt.close(fig)
    for name,columns in (("validation",("val",)),("train-validation",("train","val"))):
        fig,axes=plt.subplots(2,2,figsize=(15,9))
        for row,kind in enumerate(("field","curve")):
            for col,mode in enumerate(("UT","FT")):
                ax=axes[row,col]
                for index,case in enumerate(("1i","3ii")):
                    path=root/f"independence-261013-r1-{case}p-s42"
                    if case=="1i" and kind=="field":path=root/"independence-261013-r1-bank-1i-s42"
                    history=path/"loss_history.csv"
                    if not history.exists():history=path/"results/loss_history.csv"
                    h=pd.read_csv(history)
                    if case=="1i" and "training_group" in h:h=h[h.training_group==f"{kind}_{mode}"]
                    for split in columns:
                        values=h[f"{split}_raw_{kind}_{mode}"]
                        ax.plot(h.epoch,values.where(values>0),"--" if split=="train" else "-",color=plt.get_cmap("tab10")(index),label=case+" "+split)
                ax.set(title=f"{mode} {kind}",yscale="log",xlabel="Stage training epoch",ylabel="Normalized MSE (log)");ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.suptitle("Selected architecture histories: independent final fields use their own bank training history")
        fig.tight_layout();fig.savefig(out/f"loss-{name}.png",dpi=170);plt.close(fig)
    from resources.MLstudy_reports import serial_schematic
    serial_schematic(out,"sharing-architecture.png",[
        ("Pre-study joint DUAL",["Shared UT/FT\nfield encoder","Predicted\nUT / FT fields","Shared UT/FT\ncurve encoder"],"One joint loss; curve gradients reach the field stage"),
        ("Selected Case1(i)",["Separate UT/FT\nfield models","Predicted\nUT / FT fields","Separate UT/FT\ncurve models"],"Disconnected stage training; OOF curve inputs in this study")])
    image=lambda name,caption:figure(out,name,caption)
    text=["# Independence comparison — completed validation review\n\n## Recap and verdict\n\n",
        "All 24 case fits and six field banks completed. Case 1(i), four independent models, remains the practical architecture choice in this fixed-capacity study. Three seeds; 7,137 TRAIN / 793 validation / 881 locked-test specimens. These original curve fits train on OOF predicted fields only, before the later Mixed source study. No locked-test evaluation. Shared-field cases lose FT accuracy; joint stage training can produce near-mean FT curves even without UT/FT parameter sharing. This does not prove a gradient-conflict mechanism or that every possible DUAL architecture fails.\n\n",
        "**Architecture decision:** pre-study joint DUAL shares UT/FT encoders and trains fields/curves together. Selected Case 1(i) has separate UT-field, FT-field, UT-curve and FT-curve models; field training and curve training are disconnected. Deployment still passes each predicted field to its curve model. The subsequent input study changes curve training to Predicted+True.\n\n",
        image("sharing-architecture.png","Joint control versus selected independent architecture"),
        "## Global performance versus Case 1(i)\n\nNegative is better; paired same-seed RMSE changes. Per-task block capacity is matched, not total parameters. Staged cases include extra OOF fits and stationary curve inputs, so these are workflow comparisons.\n\n",image("global-control.png","Four-output global changes"),
        "## Global performance versus TRAIN mean\n\nSkill=100×(1−model RMSE/TRAIN-mean RMSE); zero matches the mean predictor. Whiskers are seed SD, not confidence intervals.\n\n",image("global-mean.png","Accuracy beyond mean fields and curves"),
        image("output-variation.png","Specimen variation is separate from RMSE skill; matching spread does not establish accuracy"),
        "## Confirmed STATUS fracture regions versus Case 1(i)\n\nActual intact→failed strut endpoints and relative motion at onset ±1 retained frame. This newly added retrospective evaluation does not feed damage to the predictors. The broader FT box is separate.\n\n",image("status-fracture-regions.png","Actual fracture-event errors"),
        "## Confirmed STATUS fracture regions versus TRAIN mean\n\nSame validation masks and regions; reconstructed TRAIN mean must reproduce saved global baseline error.\n\n",image("fracture-mean.png","Fracture-region accuracy beyond the mean field"),
        "## Best and worst examples\n\nFixed seed 42; each output ranks by complete valid sample RMSE. Large four-column galleries use initial disordered positions, full UT and crack-tip FT. Percentage displays omit tiny/zero truth only; aggregate metrics are unfiltered. TRAIN-mean reference panels are a separate baseline diagnostic.\n\n"]
    for case in ("1i","3ii"):
        text.append(f"### Case {case}\n\n")
        for mode in ("UT","FT"):
            text.extend([image(f"{case}-{mode}-field-examples.png",f"{case} {mode} fields"),image(f"{case}-{mode}-curve-examples.png",f"{case} {mode} curves")])
    text += ["## Training time\n\nTraining costs exclude queue/export time. A bank reused between cases is counted in each prospective workflow, not as additional actual spending. Detailed parameter/group/checkpoint provenance remains linked.\n\n",image("training-cost.png","Training costs including reusable banks"),
        "## Loss histories\n\nValidation-only compares selected architectures; train/validation diagnoses overfitting. Independent fields use final-generator bank histories, rather than constant frozen-field scores logged during curve training.\n\n",image("loss-validation.png","Selected validation histories"),image("loss-train-validation.png","Selected training and validation histories"),
        "## Limits and fixed safeguards\n\nThe accepted 2% tolerance is a relative RMSE safeguard, separately per output. Actual fracture errors, local motion and curve peak/work errors remain additional evidence. No new aggregate acceptance threshold is introduced. Hyperparameters were anchored to existing HPO, not re-optimized per architecture; the original DUAL HPO used different tuned task weights.\n\n",image("local-and-curve-safeguards.png","Existing local-motion and curve safeguards"),
        figure(out,"curve-regions.png","Target peak/post-peak windows versus control and TRAIN mean; separate from spatial STATUS regions"),
        "[Global metrics](per-run-metrics.csv) · [Confirmed fracture entries/denominators](confirmed-fracture-regions.csv) · [Motion/curve metrics](local-and-curve-metrics.csv) · [Provenance](provenance.json) · [Original validation checks](validation.txt). No retraining or inference.\n"]
    (out/"README.md").write_text("".join(text),encoding="utf-8")
    print("REPORT",out/"README.md",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/independence-review")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    a=p.parse_args();refresh(a.root,a.out,a.data_root)
