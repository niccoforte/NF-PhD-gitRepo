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
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/loss-study/historical-shared-true-field")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
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
    deepen(a.root,a.out,table,a.data_root)


def deepen(root,out,table,data_root):
    """Detailed read-only source diagnostic, deliberately separate from Mixed."""
    from resources.MLstudy_reports import training_curve_mean,measure_curve_regions,percent_error,figure
    records={};regions=[];cost=[]
    for name,label in zip(RUNS,LABELS):
        path=root/name;data=json.loads((path/"model_data.json").read_text())
        meta=json.loads((path/"run_metadata.json").read_text());metrics=json.loads((path/"results/metrics.json").read_text())
        cost.append(dict(label=label,hours=meta["training_seconds"]/3600,run=name))
        with np.load(path/"results/predictions.npz",allow_pickle=False) as z:
            for mode in ("UT","FT"):
                y,p,x=z[f"{mode}_val_curve_truth"],z[f"{mode}_val_curve_outputs"],z[f"{mode}_curve_x_values"]
                mean=training_curve_mean(data_root,data,mode,y,z["sample_ids"],x)
                baseline=np.sqrt(np.mean((y-mean)**2))
                if not np.isclose(baseline,metrics["diagnostics"]["curve"][mode]["mean_curve_baseline_rmse"],rtol=3e-6):
                    raise ValueError("Reconstructed TRAIN-mean curve baseline differs")
                records[label,mode]=dict(y=y.copy(),p=p.copy(),x=x.copy(),mean=mean,ids=z["sample_ids"].copy(),path=path)
                for row in measure_curve_regions(p,y,x,mean):regions.append(dict(label=label,mode=mode,**row))
    regions=pd.DataFrame(regions);control=regions[regions.label=="MSE"].set_index(["mode","region"]).rmse
    regions["change_percent"]=[100*(r.rmse/control.loc[r.mode,r.region]-1) for r in regions.itertuples()]
    regions.to_csv(out/"true-field-curve-regions.csv",index=False);pd.DataFrame(cost).to_csv(out/"true-field-curve-cost.csv",index=False)
    for name,column,title in (("global-control","change_percent","Global RMSE change vs matched MSE — lower is better"),
                              ("global-mean","skill","RMSE reduction vs TRAIN mean — higher is better")):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,mode in zip(axes,("UT","FT")):
            part=table[table["mode"]==mode];bars=ax.bar(range(3),part[column],color="#477f9f");ax.bar_label(bars,fmt="%.2f",padding=3)
            ax.set(title=mode,ylabel="Percent",xticks=range(3),xticklabels=[v.replace(", ","\n") for v in LABELS]);ax.axhline(0,color="black",lw=.8);ax.grid(axis="y",alpha=.2)
        fig.suptitle(title);fig.tight_layout();fig.savefig(out/f"true-field-{name}.png",dpi=170);plt.close(fig)
    for name,column,title in (("regions-control","change_percent","Peak and post-peak errors vs MSE — lower is better"),
                              ("regions-mean","skill_percent","Peak and post-peak RMSE reduction vs TRAIN mean — higher is better")):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,mode in zip(axes,("UT","FT")):
            for j,region in enumerate(("Peak window ±5% span","Pre-peak","Post-peak to recorded end")):
                values=regions[(regions["mode"]==mode)&(regions.region==region)].set_index("label").loc[list(LABELS)]
                ax.plot(np.arange(3)+(j-1)*.12,values[column],"o",label=region)
            ax.set(title=mode,ylabel="Percent",xticks=range(3),xticklabels=[v.replace(", ","\n") for v in LABELS]);ax.axhline(0,color="black",lw=.8);ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.suptitle(title);fig.tight_layout();fig.savefig(out/f"true-field-{name}.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,mode in zip(axes,("UT","FT")):
        part=table[table["mode"]==mode]
        for column,label in (("peak_height_percent","Peak-height MAE / mean true peak"),("peak_location_percent","Hard-location MAE / x span"),("work_percent","Work MAE / mean true work")):
            ax.plot(range(3),part[column],"o-",label=label)
        ax.set(title=mode,ylabel="Percent — lower is better",xticks=range(3),xticklabels=[v.replace(", ","\n") for v in LABELS]);ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"true-field-mechanical-errors.png",dpi=170);plt.close(fig)
    for mode in ("UT","FT"):
        r=records["MSE",mode];y,p,x,mean=(r[k] for k in ("y","p","x","mean"));errors=np.sqrt(np.mean((p-y)**2,axis=1));picks=[errors.argmin(),errors.argmax()]
        unit=1 if mode=="UT" else 1000.;ylabel="Stress (MPa)" if mode=="UT" else "Force (kN)"
        fig,axes=plt.subplots(2,4,figsize=(21,10))
        for row,index in enumerate(picks):
            ape=percent_error(p[index],y[index],np.isfinite(y[index]),np.sqrt(np.mean(mean**2)))
            for col,ax in enumerate(axes[row]):
                if col==0:ax.plot(x,y[index]/unit,color="black",label="Truth")
                if col==1:
                    ax.plot(x,y[index]/unit,color="black",label="Truth");ax.plot(x,p[index]/unit,"--",label="Prediction");ax.plot(x,mean/unit,":",label="TRAIN mean",alpha=.7);ax.bar(x,(p[index]-y[index])/unit,width=np.median(np.diff(x))*.95,alpha=.18,color="#ad6158",label="Signed error")
                if col==2:ax.bar(x,(p[index]-y[index])/unit,width=np.median(np.diff(x))*.95,alpha=.3,label="Signed residual");ax.axhline(0,color="black",lw=.8)
                if col==3:ax.plot(x,ape,label="Absolute error / |truth|")
                ax.set(title=("Truth","Prediction + reference","Signed error","Percent error")[col],xlabel="Strain" if mode=="UT" else "Load-line displacement (mm)",ylabel="Percent" if col==3 else ylabel)
                ax.title.set_fontsize(16);ax.tick_params(labelsize=12);ax.xaxis.label.set_size(13);ax.yaxis.label.set_size(13);ax.set_xlim(float(x[0]),float(x[-1]));ax.legend(fontsize=10);ax.grid(alpha=.15)
            axes[row,0].text(0,1.15,f"{'Best' if row==0 else 'Worst'} ID {r['ids'][index]}",transform=axes[row,0].transAxes,fontsize=15)
        fig.suptitle(f"{mode}: historical MSE true-field model; best/worst by whole-curve RMSE\nTRAIN-mean reference is fixed; percent errors omit tiny/zero truth only",fontsize=18)
        fig.tight_layout(rect=(0,0,1,.93),h_pad=3);fig.savefig(out/f"true-field-{mode}-examples.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));bars=ax.bar(range(3),[r['hours'] for r in cost]);ax.bar_label(bars,fmt="%.2f",padding=3);ax.set(ylabel="Curve-stage training hours",xticks=range(3),xticklabels=LABELS,title="One shared UT/FT curve fit per preset; frozen fields not retrained")
    fig.tight_layout();fig.savefig(out/"true-field-runtime.png",dpi=170);plt.close(fig)
    for tag,cols in (("validation",("val",)),("train-validation",("train","val"))):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,mode in zip(axes,("UT","FT")):
            for index,(name,label) in enumerate(zip(RUNS,LABELS)):
                if len(cols)>1 and label not in ("MSE","Combined + soft location"):continue
                h=pd.read_csv(root/name/"results/loss_history.csv")
                for split in cols:
                    values=h[f"{split}_raw_curve_{mode}"]
                    ax.plot(h.epoch,values.where(values>0),"--" if split=="train" else "-",color=plt.get_cmap("tab10")(index),label=label+" "+split)
            ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Saved curve objective (log)");ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f"true-field-loss-{tag}.png",dpi=170);plt.close(fig)
    from resources.MLstudy_reports import serial_schematic
    nodes=["True fields\n+ saved context","Shared task-conditioned\ncurve Transformer","UT / FT\n201-point curves"]
    serial_schematic(out,"true-field-architecture.png",[("Historical MSE control",nodes,"True-field training and evaluation; curve MSE"),
        ("Tested combined objectives",nodes,"Same architecture/source; added losses, not adopted")])
    image=lambda name,caption:figure(out,name,caption)
    text=["# Historical custom curve losses with true fields\n\n## Recap and verdict\n\n",
        "Keep this as a separately labelled study: all three models share a UT/FT curve encoder and train AND evaluate on true fields. They are not the current four-independent/Mixed workflow. One seed 42, 793 validation specimens; field predictions do not change. MSE retains the lowest global error. The combined no-location/soft-location presets worsen global RMSE 18.04%/17.19% UT and 30.47%/30.92% FT. This is evidence against these bundles at their tested coefficients, not proof that every individual term is unhelpful.\n\n",
        "**Architecture:** true fields + saved context → shared task-conditioned curve Transformer → UT/FT curve heads. The architecture is unchanged across MSE/combined/combined+soft-location; only curve objectives change. The field generator is frozen and bypassed for these reported curves.\n\n",
        image("true-field-architecture.png","Historical source and architecture; only objectives change"),
        "## Global performance versus control\n\n",image("true-field-global-control.png","Historical global changes versus matched MSE"),
        "## Global performance versus TRAIN mean\n\nReconstructed per-point means use recorded TRAIN IDs only. CSV truths/grid and global mean-baseline errors are checked against the saved arrays/metrics.\n\n",image("true-field-global-mean.png","Historical accuracy beyond the mean curve"),
        "## Fracture-relevant curve regions versus control\n\nThese are target-defined curve regions: ±5% of the loading span around the hard true peak, pre-peak, and post-peak through the recorded end. Masks overlap around the peak by design. They are not spatial STATUS-labelled node regions or measured fracture-initiation times. The study changes only curve prediction; reporting changed field/fracture-node accuracy would be misleading.\n\n",image("true-field-regions-control.png","Peak and post-peak regional changes"),
        "Both combined presets worsen peak-window RMSE by about15% in UT and9–13% in FT; post-peak RMSE worsens17–18% UT andabout31% FT. The soft term modestly improves FT peak-window error relative to the already combined preset, but still does not beat MSE.\n\n",
        "## Fracture-relevant curve regions versus TRAIN mean\n\nThe identical target-defined windows are applied to predictions and the TRAIN mean. Region counts and target-RMS denominators remain in the CSV.\n\n",image("true-field-regions-mean.png","Regional accuracy beyond the mean curve"),
        "## Best and worst examples\n\nUse the lowest-global-error MSE control, fixed seed 42. Full-curve RMSE ranks specimens. Tiny truth below 1% of TRAIN-mean RMS is excluded only from percentage displays, never aggregate metrics.\n\n",image("true-field-UT-examples.png","UT truth/prediction/error/percentage examples"),image("true-field-FT-examples.png","FT truth/prediction/error/percentage examples"),
        "## Training time\n\nQueue and frozen upstream field-training costs are excluded. One jointly trained curve stage per preset; costs are not counted twice for UT/FT.\n\n",image("true-field-runtime.png","Observed curve training cost"),
        "## Loss histories\n\nValidation-only compares all three objectives; selected train/validation histories compare MSE and combined+soft for overfitting. Weighted totals are not comparable accuracy metrics. All plotted positive losses use log axes.\n\n",image("true-field-loss-validation.png","Historical validation objectives"),image("true-field-loss-train-validation.png","Selected training and validation objectives"),
        "## Hard mechanical curve metrics and loss attribution\n\n",image("true-field-mechanical-errors.png","Hard peak-height/location and work errors"),
        "The bundle uses curve MSE1, zone MSE0.5 (weights1/5/0.2), first derivative0.05, peak height0.25, work0.1; the location coefficient is0 or0.02 with sharpness20. These terms were not individually ablated. Adding soft location to the existing bundle isolates that addition conditionally; it does not isolate the other terms or demonstrate their individually optimal coefficients. FT integrated force-displacement work is a proxy, not K_JIC. No new property definition or physical fracture-time alignment is inferred.\n\n",
        "The next useful curve test, after field improvements, is MSE plus one modest term at a time with independent true-field controls, followed by Mixed/predicted-field evaluation. No curve-term learning job is submitted by this review.\n\n",
        "[Global/hard metrics](true-field-curve-losses.csv) · [Regional errors, counts and denominators](true-field-curve-regions.csv) · [Measured costs](true-field-curve-cost.csv). No inference, retraining, locked-test evaluation or upstream export.\n"]
    table.to_csv(out/"true-field-curve-losses.csv",index=False);(out/"README.md").write_text("".join(text),encoding="utf-8")
    print(regions[["label","mode","region","change_percent","skill_percent"]].to_string(index=False))


if __name__=="__main__":main()
