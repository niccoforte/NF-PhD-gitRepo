"""Review all 48 independent true-field curve-term fits, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resources.MLstudy_reports import (verify_study_inventory,training_curve_mean,
    measure_curve_regions,prediction_variation_section,curve_examples,figure,serial_schematic)

VARIANTS=("mse","range_mse","zone","derivative","peak_height","work","peak_location","combined")
LABELS=("MSE","Range-MSE","+ Zone","+ Derivative","+ Peak height","+ Work","+ Soft location","Combined")


def term_verdict(table,losses,out):
    """Use fixed hard metrics and measured component values, not loss rankings."""
    means=table.groupby(["mode","variant"]).mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(18,6))
    for ax,mode in zip(axes,("UT","FT")):
        part=losses[(losses["mode"]==mode)&~losses.component.isin(("mse","range_mse"))]
        groups=list(part.groupby(["variant","component"],sort=False))
        for index,((variant,component),group) in enumerate(groups):
            ax.scatter(np.full(len(group),index),group.ratio_to_mse,color="#397b95")
        ax.set(title=mode,yscale="log",ylabel="Weighted term / normalized MSE (log)",xticks=range(len(groups)),xticklabels=[f"{v}\n{c}" for (v,c),_ in groups])
        ax.tick_params(axis="x",rotation=50,labelsize=9);ax.axhline(1,color="black",linestyle="--");ax.grid(axis="y",alpha=.2)
    fig.suptitle("Actual added-loss budget at the chosen validation checkpoint\nThree seeds; loss-value ratios are not gradient magnitudes",fontsize=16)
    fig.tight_layout();fig.savefig(Path(out)/"loss-budget.png",dpi=170);plt.close(fig)
    derivative=losses[(losses.variant=="derivative")&(losses.component=="derivative")].groupby("mode").ratio_to_mse.mean()
    paragraphs=["**Practical verdict: retain MSE as the general curve default.** The derivative and combined presets worsen global RMSE in every seed in both tasks. "
        f"Derivative-only changes average {means.loc['UT','derivative'].change_percent:+.2f}% UT / {means.loc['FT','derivative'].change_percent:+.2f}% FT. "
        f"Its weighted value is {derivative['UT']:.2f}× / {derivative['FT']:.2f}× the normalized MSE at the selected checkpoints. "
        "This is a strong loss-balance warning and a plausible explanation for the bundle's deterioration, not a measured gradient diagnosis. Do not reuse this derivative coefficient unchanged; this does not reject derivative supervision at every possible scale."]
    for mode in ("UT","FT"):
        base,peak=means.loc[mode,"mse"],means.loc[mode,"peak_height"]
        paragraphs.append(f"**{mode} peak-height term:** hard amplitude error {base.peak_height_percent:.3f}% → {peak.peak_height_percent:.3f}% of mean true peak "
            f"({100*(peak.peak_height_percent/base.peak_height_percent-1):+.1f}% relative); global RMSE {peak.change_percent:+.2f}%. "
            f"Hard-location error {base.peak_location_percent:.3f}% → {peak.peak_location_percent:.3f}% of loading span. "
            "Keep this as a feature-specific candidate, with the location safeguard visible; it is not a universal replacement.")
    paragraphs.append("UT zone weighting produces small average global/peak improvements, but its global sign varies across seeds and work error rises. Range-MSE, work and soft-location arms do not establish a consistent global replacement. Soft-location supervision did not improve the reported hard-location error at this coefficient. The important distinction from the historical bundle is that some isolated terms can improve a requested feature while the derivative-dominated bundle fails. These true-input findings still need a controlled Predicted+True/predicted-input check before deployment adoption; no new job is submitted here.")
    return "\n\n".join(paragraphs)+"\n\n"


def matched_changes(table,keys,control="mse"):
    """Same task, seed and optional region; never compare an unpaired mean."""
    baseline=table[table.variant==control].set_index(keys).rmse
    if not baseline.index.is_unique:raise ValueError("Duplicate curve control")
    return np.asarray([100*(row.rmse/baseline.loc[tuple(getattr(row,k) for k in keys)]-1) for row in table.itertuples()])


def seed_plot(table,out,name,column,ylabel,regions=None):
    fig,axes=plt.subplots(1,2,figsize=(19,6.5))
    for ax,mode in zip(axes,("UT","FT")):
        for j,region in enumerate(regions or (None,)):
            part=table[table["mode"]==mode]
            if region is not None:part=part[part.region==region]
            for index,variant in enumerate(VARIANTS):
                values=part[part.variant==variant][column].to_numpy();x=index+(j-1)*.14 if regions else index
                color=plt.get_cmap("tab10")(j)
                ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt="o",capsize=4,color=color,label=region if regions and index==0 else None)
                ax.scatter(x+np.linspace(-.035,.035,len(values)),values,s=16,color=color,alpha=.7)
                if regions is None:
                    ax.annotate(f"{values.mean():.2f}",(x,values.mean()),xytext=(3,9),textcoords="offset points",fontsize=10)
        ax.set(title=mode,ylabel=ylabel,xticks=range(8),xticklabels=LABELS);ax.tick_params(axis="x",rotation=28,labelsize=11);ax.grid(axis="y",alpha=.2);ax.axhline(0,color="black",lw=.8);ax.margins(y=.22)
        if regions:ax.legend(fontsize=10,loc="upper center",bbox_to_anchor=(.5,-.22))
    fig.suptitle("Independent true-field curve models: three training seeds\nLarge markers = means; small points = seeds; whiskers = seed SD, not confidence intervals",fontsize=16)
    fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=170);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2"))
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/loss-study/independent-true-field-terms")
    a=p.parse_args(argv);out=a.out;out.mkdir(parents=True,exist_ok=True)
    receipt=verify_study_inventory(a.root,out/"download-inventory.json",48)
    (out/"download-verification.json").write_text(json.dumps(receipt,indent=2))
    rows=[];regions=[];entries=[];contracts={};losses=[]
    for mode in ("UT","FT"):
        for variant in VARIANTS:
            for seed in (42,43,44):
                path=a.root/mode/"FieldToCurve/Transformer"/f"curve-terms-261028-{variant}-{mode}-s{seed}"
                cfg=json.loads((path/"model.json").read_text());data=json.loads((path/"model_data.json").read_text())
                meta=json.loads((path/"run_metadata.json").read_text());metrics=json.loads((path/"results/metrics.json").read_text())
                if (cfg["kind"],cfg["mode"],cfg["variant"],meta["seed"])!=("independent-curve-loss-terms",mode,variant,seed):raise ValueError("Curve run identity differs")
                if metrics["evaluation_split"]!="val" or metrics["curve_training_source"]!="true" or metrics["curve_evaluation_source"]!="true":raise ValueError("Not a true-field validation experiment")
                for key,value in (("model",cfg["model_config"]),("training",cfg["training"]),("splits",data["sample_ids"]),("scalers",data["normalizers"]),("metadata",data["metadata"]),("reference",meta["reference_sha256"])):
                    digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
                    if contracts.setdefault((mode,key),digest)!=digest:raise ValueError("Changed curve contract: "+key)
                with np.load(path/"results/predictions.npz",allow_pickle=False) as z:
                    ids=z["sample_ids"].astype(str);y=z[f"{mode}_val_curve_truth"].astype(float);pred=z[f"{mode}_val_curve_outputs"].astype(float);x=z[f"{mode}_curve_x_values"]
                    if len(set(ids))!=len(ids) or list(ids)!=list(map(str,data["sample_ids"]["val"])):raise ValueError("Curve IDs differ")
                    digest=hashlib.sha256(y.tobytes()).hexdigest()
                    if contracts.setdefault((mode,"truth"),digest)!=digest:raise ValueError("Curve physical truths differ")
                    mean=training_curve_mean(a.data_root,data,mode,y,ids,x)
                    rmse=np.sqrt(np.mean((pred-y)**2));den=np.sqrt(np.mean((mean-y)**2));diag=metrics["diagnostics"]["curve"][mode]
                    if diag["mean_curve_baseline_source"]!="train_mean_curve" or not np.allclose([rmse,den],[diag["rmse"],diag["mean_curve_baseline_rmse"]],rtol=3e-6):raise ValueError("Curve errors/mean do not reproduce")
                    if not np.isclose(cfg["selection_denominator"],den**2,rtol=3e-6):raise ValueError("Selection denominator differs")
                    table=pd.read_csv(path/f"results/{mode}_val_curve_sample_metrics.csv")
                    if list(table.sample_id.astype(str))!=list(ids):raise ValueError("Per-specimen curve IDs differ")
                    peak_p,peak_y=pred.max(1),y.max(1);loc_p,loc_y=x[pred.argmax(1)],x[y.argmax(1)]
                    energy_p,energy_y=np.trapz(pred,x,axis=1),np.trapz(y,x,axis=1)
                    for actual,saved in ((peak_p-peak_y,table.peak_error),(loc_p-loc_y,table.peak_x_error),(energy_p-energy_y,table.energy_error)):
                        if not np.allclose(actual,saved,rtol=1e-5,atol=max(np.max(np.abs(actual))*1e-6,1e-6)):raise ValueError("Per-specimen curve feature error differs")
                    hard=dict(peak_height_percent=100*np.mean(abs(peak_p-peak_y))/np.mean(abs(peak_y)),peak_location_percent=100*np.mean(abs(loc_p-loc_y))/np.ptp(x),work_percent=100*np.mean(abs(energy_p-energy_y))/np.mean(abs(energy_y)))
                    for record in measure_curve_regions(pred,y,x,mean):regions.append(dict(mode=mode,seed=seed,variant=variant,**record))
                h=pd.read_csv(path/"loss_history.csv");selected=h[h.epoch==cfg["best_epoch"]].iloc[0]
                weighted=[c for c in h if c.startswith("val_weighted_")]
                if not np.allclose(h[weighted].sum(axis=1),h.val_loss,rtol=1e-5,atol=1e-8):raise ValueError("Curve loss components do not reconstruct")
                for c in weighted:losses.append(dict(mode=mode,seed=seed,variant=variant,component=c.removeprefix("val_weighted_"),weighted_value=selected[c],normalized_mse=selected.val_raw_mse,ratio_to_mse=selected[c]/selected.val_raw_mse))
                rows.append(dict(mode=mode,seed=seed,variant=variant,rmse=rmse,mean_rmse=den,skill_percent=100*(1-rmse/den),hours=meta["training_seconds"]/3600,epochs=len(h),best_epoch=cfg["best_epoch"],parameters=meta["active_parameters"],job=meta["slurm_job_id"],revision=meta["source_revision"],path=str(path),**hard))
                entries.append(dict(mode=mode,seed=seed,variant=variant,path=path,kinds=["curve"]))
                print("CURVE_REVIEW",mode,variant,seed,flush=True)
    table=pd.DataFrame(rows);regional=pd.DataFrame(regions)
    table["change_percent"]=matched_changes(table,["mode","seed"])
    regional["change_percent"]=matched_changes(regional,["mode","seed","region"])
    loss_table=pd.DataFrame(losses)
    table.to_csv(out/"metrics.csv",index=False);regional.to_csv(out/"regions.csv",index=False);loss_table.to_csv(out/"loss-components.csv",index=False)
    verdict=term_verdict(table,loss_table,out)
    seed_plot(table,out,"global-control","change_percent","RMSE change vs same-seed MSE (%) — lower is better")
    seed_plot(table,out,"global-mean","skill_percent","RMSE reduction vs TRAIN mean (%) — higher is better")
    windows=("Peak window ±5% span","Pre-peak","Post-peak to recorded end")
    seed_plot(regional,out,"regions-control","change_percent","Regional RMSE change vs same-seed MSE (%)",windows)
    seed_plot(regional,out,"regions-mean","skill_percent","Regional RMSE reduction vs TRAIN mean (%)",windows)
    seed_plot(table,out,"runtime","hours","Curve-stage training hours (queue excluded)")
    for metric in ("peak_height_percent","peak_location_percent","work_percent"):
        seed_plot(table,out,metric,metric,"MAE / mean true peak (%)" if metric=="peak_height_percent" else "Hard location MAE / loading span (%)" if metric=="peak_location_percent" else "Work MAE / mean true work (%)")
    spread=prediction_variation_section(entries,out)
    selected={mode:table[table["mode"]==mode].groupby("variant").change_percent.mean().idxmin() for mode in ("UT","FT")}
    for mode in ("UT","FT"):
        row=table[(table["mode"]==mode)&(table.variant==selected[mode])&(table.seed==42)].iloc[0]
        curve_examples(row.path,mode,out,f"{row.variant} true-field model, seed42",a.data_root)
    for filename,columns in (("validation",("selection_score",)),("objectives",("val_loss",)),("train-validation",("train_loss","val_loss"))):
        fig,axes=plt.subplots(1,2,figsize=(18,6))
        for ax,mode in zip(axes,("UT","FT")):
            for index,variant in enumerate(VARIANTS):
                if len(columns)>1 and variant not in ("mse",selected[mode]):continue
                row=table[(table["mode"]==mode)&(table.variant==variant)&(table.seed==42)].iloc[0];h=pd.read_csv(Path(row.path)/"loss_history.csv")
                for col in columns:ax.plot(h.epoch,h[col].where(h[col]>0),"--" if col.startswith("train") else "-",color=plt.get_cmap("tab10")(index),label=LABELS[index]+" "+col)
                if len(columns)>1:ax.axvline(row.best_epoch,color=plt.get_cmap("tab10")(index),alpha=.3,linestyle=":")
            ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Fixed physical MSE / TRAIN-mean MSE (log)" if filename=="validation" else "Saved optimization objective (log)");ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f"loss-{filename}.png",dpi=170);plt.close(fig)
    nodes=["True U1/U2 fields\n+ saved context","One independent\ncurve Transformer","201-point\nmechanical curve"]
    serial_schematic(out,"architecture.png",[("Matched MSE control",nodes,"Independent UT/FT fits; true-field train AND validation"),("One-term candidate",nodes,"Same architecture; one added term or range-base replacement")])
    summary=table.groupby(["mode","variant"],sort=False).agg(global_change=("change_percent","mean"),seed_sd=("change_percent","std"),skill_percent=("skill_percent","mean"),peak_height_percent=("peak_height_percent","mean"),peak_location_percent=("peak_location_percent","mean"),work_percent=("work_percent","mean"))
    summary.to_csv(out/"decision-metrics.csv")
    image=lambda name,caption:figure(out,name+".png",caption)
    text=["# Independent true-field curve-loss terms\n\n## Recap, main findings and verdict\n\n",
        "All48 fits completed and passed checkpoint reload: eight objectives × UT/FT × seeds42/43/44. These are fully separate UT and FT curve models, trained AND validated on true fields. The field model is bypassed and unchanged. This is a curve information-content/objective test, not the accepted Predicted+True pipeline's deployment performance.\n\n",
        "**No default is automatically changed.** Compare fixed global, peak/post-peak and hard peak/work errors together; a term is not proven useless by one unsuccessful coefficient. Whiskers show three-seed SD, not confidence intervals; the locked test remains untouched.\n\n"]
    for mode in ("UT","FT"):
        part=summary.loc[mode];best=part.global_change.idxmin();v=part.loc[best]
        text.append(f"**{mode}:** lowest mean global RMSE is **{best}**, {v.global_change:+.2f}% versus matched MSE; TRAIN-mean RMSE skill {v.skill_percent:.2f}%. The peak/work figures below determine whether this global ranking also preserves the requested curve features.\n\n")
    text += [verdict,image("architecture","Unchanged independent curve architecture; only supervision changes"),
        "## Global performance versus same-seed MSE control\n\n",image("global-control","Three-seed global error changes"),
        "## Global performance versus TRAIN mean\n\nMeans are fitted per curve point using TRAIN IDs only; both saved physical targets and mean-baseline errors reproduce from the upstream CSVs. Zero skill matches the mean-curve predictor.\n\n",image("global-mean","Accuracy beyond predicting the average curve"),spread,
        "## Fracture-relevant curve regions versus control\n\nTarget-defined hard peak ±5% of loading span, pre-peak, and post-peak to the recorded end. These overlapping peak-adjacent windows are not STATUS onset times or spatial fracture nodes. No changed field prediction exists in this experiment. Eligible specimen/point counts and denominators are retained in regions.csv.\n\n",image("regions-control","Peak and pre/post-peak RMSE changes"),
        "## Fracture-relevant curve regions versus TRAIN mean\n\nIdentical target-defined windows are used for the model and fixed training mean.\n\n",image("regions-mean","Regional accuracy beyond average curves"),
        "## Hard peak and work metrics\n\nPeak height uses mean absolute peak error / mean absolute true peak; hard location uses mean absolute loading-coordinate error / full loading span; work uses mean absolute curve-integral error / mean absolute true integral. These are ratios of means. FT force–displacement integral is a work diagnostic, not K_JIC.\n\n",image("peak_height_percent","Hard peak-height errors"),image("peak_location_percent","Hard peak-location errors"),image("work_percent","Curve integral errors"),
        "## Best and worst examples\n\nUse each task's lowest mean-global-error objective at seed42, including MSE if it wins. Rank by full-curve RMSE; no population superiority is inferred from examples. Translucent bars use the same x axis. Percentage error omits tiny/zero truth below1% of TRAIN-mean RMS only for display, never aggregate metrics.\n\n",image("UT-curve-examples","UT truth, prediction, signed residual and percentage error"),image("FT-curve-examples","FT truth, prediction, signed residual and percentage error"),
        "## Training time\n\nOne active curve Transformer per job, excluding queue and all upstream field training.\n\n",image("runtime","Three-seed curve-stage training hours"),
        "## Loss histories\n\nRepresentative seed42 histories: fixed physical validation error is comparable; differing total objectives are not an accuracy ranking. Selected control/candidate train and validation objectives are separate overfitting diagnostics, with chosen checkpoints marked. All positive losses use log axes. Population performance above uses all three seeds.\n\n",image("loss-validation","Fixed validation errors for every objective"),image("loss-objectives","Saved validation objective totals"),image("loss-train-validation","Selected train and validation histories"),
        "## Exact isolated terms and limits\n\n| Arm | Change from normalized MSE |\n|---|---|\n| MSE | Matched control |\n| range_mse | Replace base by historical per-target-range MSE |\n| zone | Add0.5 × zone MSE; weights1/5/0.2 |\n| derivative | Add0.05 × first-difference term |\n| peak_height | Add0.25 × peak-height term |\n| work | Add0.1 × area/work term |\n| peak_location | Add0.02 × soft-versus-soft location; beta20 |\n| combined | Add all five terms; interaction comparison |\n\n",
        "Terms reconstruct physical curves differentiably; the base stays unchanged except range_mse. All runs select by the same fixed physical MSE/TRAIN-mean MSE criterion. Component CSVs verify weighted terms sum to the archived objective and expose their sizes at the chosen checkpoint; these are loss values, not gradient ratios. This tests existing terms at specified coefficients, not optimal coefficients or predicted-field robustness. No new inference, training, HPO, property definition or winner promotion.\n\n",image("loss-budget","Measured added-term budget relative to the MSE base"),
        "[Global/hard metrics and provenance](metrics.csv) · [Regional counts/denominators](regions.csv) · [Decision summary](decision-metrics.csv) · [Selected loss components](loss-components.csv) · [Checksum verification](download-verification.json). The historical shared-encoder true-field report remains [separately labelled](../historical-shared-true-field/README.md).\n"]
    (out/"README.md").write_text("".join(text),encoding="utf-8");print(summary.to_string());print("REPORT",out/"README.md")


if __name__=="__main__":main()
