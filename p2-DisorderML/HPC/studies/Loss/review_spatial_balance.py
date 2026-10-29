"""Review completed FT balance and existing paired UT spatial safeguards."""
import argparse
import json
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resources.MLstudy_reports import (collect_fields,repeated_field_figures,
    verify_download,figure,replace_study_followup,prediction_variation_section,field_examples)


def paired_interval(base,changed,weights=None,rmse=False,repeats=2000):
    """Resample training seeds and paired validation IDs, never unpaired rows."""
    base,changed=np.asarray(base,float),np.asarray(changed,float)
    weights=np.ones_like(base) if weights is None else np.asarray(weights,float)
    if base.shape!=changed.shape or base.shape!=weights.shape or base.ndim!=2:
        raise ValueError("Require aligned [seed,specimen] values/weights")
    if not np.isfinite(base).all() or not np.isfinite(changed).all() or np.any(weights<0):
        raise ValueError("Invalid paired evidence")
    def estimate(b,c,w):
        b=(b*w).sum(-1)/w.sum(-1);c=(c*w).sum(-1)/w.sum(-1)
        return (100*(np.sqrt(c/b)-1) if rmse else c-b).mean(-1)
    rng=np.random.default_rng(42);draws=[];n_seeds,n=base.shape
    for start in range(0,repeats,100):
        size=min(100,repeats-start)
        s=rng.integers(n_seeds,size=(size,n_seeds,1));i=rng.integers(n,size=(size,1,n))
        draws.extend(estimate(base[s,i],changed[s,i],weights[s,i]))
    return dict(change=float(estimate(base,changed,weights)),ci_low=float(np.quantile(draws,.025)),
        ci_high=float(np.quantile(draws,.975)),specimens=n,seeds=n_seeds,bootstrap_repeats=repeats)


def ut_safeguards(root,out):
    values={};weights=[];metric_weights={};ids=None
    for seed in (42,43,44):
        for variant,name in (("control",f"independent-inputs-261015-mixed-s{seed}"),
                             ("spatial",f"independent-losses-261022-mixed-spatial-s{seed}")):
            run=root/name;r=run/"results";cfg=json.loads((run/"model_data.json").read_text())
            field=pd.read_csv(r/"UT_val_field_sample_metrics.csv",dtype={"sample_id":str})
            curve=pd.read_csv(r/"UT_val_curve_sample_metrics.csv",dtype={"sample_id":str})
            actual=curve.sample_id.tolist()
            if ids is None:ids=actual
            if actual!=ids or field.sample_id.tolist()!=ids or list(map(str,cfg["sample_ids"]["val"]))!=ids:
                raise ValueError("Paired UT specimen IDs differ")
            span=np.ptp(cfg["metadata"]["curve_x_values"]["UT"])
            if np.any(np.abs(curve.true_peak)<1e-12) or np.any(np.abs(curve.true_energy)<1e-12):
                raise ValueError("Undefined peak/work percentage needs explicit masking")
            data={"Field RMSE":field.sample_mse.to_numpy(),"Curve RMSE":curve.sample_mse.to_numpy(),
                "Peak-height MAE / mean true peak (%)":100*np.abs(curve.pred_peak-curve.true_peak).to_numpy()/np.abs(curve.true_peak),
                "Peak-location MAE (% loading span)":100*np.abs(curve.pred_peak_x-curve.true_peak_x).to_numpy()/span,
                "Work MAE / mean true integral (%)":100*np.abs(curve.pred_energy-curve.true_energy).to_numpy()/np.abs(curve.true_energy)}
            for metric,v in data.items():values.setdefault((variant,metric),[]).append(np.asarray(v))
            w=field.valid_fraction.to_numpy()
            if variant=="control":
                weights.append(w)
                for metric,column in (("Peak-height MAE / mean true peak (%)","true_peak"),("Work MAE / mean true integral (%)","true_energy")):
                    metric_weights.setdefault(metric,[]).append(np.abs(curve[column].to_numpy()))
            elif not np.array_equal(w,weights[-1]):raise ValueError("Paired UT validity fractions differ")
    rows=[]
    for metric in data:
        base=np.stack(values["control",metric]);changed=np.stack(values["spatial",metric]);is_rmse=metric.endswith("RMSE")
        w=np.stack(weights) if metric=="Field RMSE" else np.stack(metric_weights[metric]) if metric in metric_weights else np.ones_like(base)
        result=paired_interval(base,changed,w,rmse=is_rmse)
        # RMSE aggregate is the mean of per-seed RMSEs, matching report tables.
        means=lambda a:np.sqrt((a*w).sum(1)/w.sum(1)).mean() if is_rmse else ((a*w).sum(1)/w.sum(1)).mean()
        result.update(metric=metric,control=float(means(base)),spatial=float(means(changed)),
            units="relative RMSE change (%)" if is_rmse else "MAE change (percentage points)")
        rows.append(result)
    table=pd.DataFrame(rows);table.to_csv(out/"ut-spatial-paired-bootstrap.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    for ax,part,title in ((axes[0],table.iloc[:2],"Global errors"),(axes[1],table.iloc[2:],"Curve fracture-sensitive safeguards")):
        x=np.arange(len(part));ax.errorbar(x,part.change,yerr=np.vstack([part.change-part.ci_low,part.ci_high-part.change]),fmt="o",capsize=5,color="#407f87")
        ax.axhline(0,color="black",lw=.8);ax.set(title=title,xticks=x,xticklabels=part.metric,ylabel=part.units.iloc[0]+" — lower is better")
        ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.2)
    fig.suptitle("UT spatial0.1: paired validation evidence; intervals resample seeds and specimen IDs")
    fig.tight_layout();fig.savefig(out/"ut-spatial-paired-bootstrap.png",dpi=170);plt.close(fig)
    return table


def selected_histories(fields,out):
    fig,ax=plt.subplots(figsize=(12,5.5))
    for row in fields[fields.variant.isin(("control","0.001","0.1"))].itertuples():
        h=pd.read_csv(Path(row.path)/"loss_history.csv")
        color={"control":"#59686f","0.001":"#238d83","0.1":"#b16a48"}[row.variant]
        for split,style in (("train","--"),("val","-")):
            values=h[f"{split}_field_components_displacement_FT"]
            ax.plot(h.epoch,values.where(values>0),style,color=color,label=f"{row.variant} {split}")
        ax.axvline(row.best_epoch,color=color,alpha=.3,linestyle=":")
    ax.set(yscale="log",xlabel="Epoch",ylabel="Displacement MSE (log)",title="Selected train/validation displacement errors; dotted line = chosen checkpoint")
    ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/"spatial-balance-train-validation.png",dpi=170);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--suite",default="spatial-balance-261028")
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/loss-study")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    receipt=verify_download(a.root,a.out/"spatial-balance-download-sha256.txt",5)
    (a.out/"spatial-balance-download-verification.json").write_text(json.dumps(receipt,indent=2))
    cases=[]
    for value,label in ((0.,"0"),(.001,"0p001"),(.003,"0p003"),(.01,"0p01"),(.1,"0p1")):
        cases.append(dict(mode="FT",seed=42,variant="control" if not value else str(value),
            source_variant="spatial_"+str(value).replace(".","p"),path=a.root/f"{a.suite}-spatial-{label}-FT-s42"))
    api=runpy.run_path(str(Path(__file__).with_name("review_fracture_regions.py")))
    fields,regions,frames=collect_fields(cases,a.data_root,api)
    fields["seconds_per_epoch"]=fields.hours*3600/fields.epochs
    fields.to_csv(a.out/"spatial-balance-metrics.csv",index=False);regions.to_csv(a.out/"spatial-balance-regions.csv",index=False)
    frames.to_csv(a.out/"spatial-balance-frames.csv",index=False)
    repeated_field_figures(fields,regions,a.out,"spatial-balance")
    spread=prediction_variation_section(cases,a.out,prefix="spatial-balance-variation").replace("## Prediction","### Prediction")
    fig,axes=plt.subplots(1,2,figsize=(15,5.5));components=[]
    for row in fields.itertuples():
        h=pd.read_csv(Path(row.path)/"loss_history.csv");coefficient=0. if row.variant=="control" else float(row.variant)
        axes[0].plot(h.epoch,h.val_field_components_displacement_FT.where(h.val_field_components_displacement_FT>0),label=row.variant)
        best=h.loc[h.epoch==row.best_epoch].iloc[0]
        weighted=coefficient*best.val_field_components_spatial_FT
        components.append(dict(coefficient=coefficient,displacement=best.val_field_components_displacement_FT,
            spatial_raw=best.val_field_components_spatial_FT,spatial_weighted=weighted,
            weighted_to_displacement=weighted/best.val_field_components_displacement_FT))
    for ax in axes:ax.grid(alpha=.2)
    axes[0].set(yscale="log",xlabel="Epoch",ylabel="Validation displacement MSE (log)",title="Fixed displacement error, not differing total objectives");axes[0].legend()
    balance=pd.DataFrame(components);balance.to_csv(a.out/"spatial-balance-loss-components.csv",index=False)
    axes[1].plot(balance.coefficient[1:],balance.weighted_to_displacement[1:],"o-")
    axes[1].set(xscale="log",yscale="log",xlabel="Spatial coefficient",ylabel="Weighted spatial / displacement loss (log)",title="Selected-checkpoint loss values, not gradient ratios")
    fig.tight_layout();fig.savefig(a.out/"spatial-balance-validation.png",dpi=170);plt.close(fig)
    selected_histories(fields,a.out)
    paired=ut_safeguards(a.root,a.out)
    selected=fields[fields.variant!="control"].sort_values("rmse").iloc[0]
    descriptor=json.loads((Path(selected.path)/"model_data.json").read_text())
    field_examples(Path(selected.path),"FT",a.out,f"Spatial {selected.variant}, seed42",a.data_root,
        api["training_mean"](a.data_root,descriptor,"FT"),prefix="spatial-balance-")
    img=lambda name,caption:figure(a.out,name+".png",caption)
    table=fields[["variant","change_percent","skill_percent","hours"]]
    text=["## Completed spatial-balance follow-up —29 October\n\n",
        "All five FT final-generator fits completed0:0 and passed reload; downloads match SHA256. Only the spatial coefficient changes:0/0.001/0.003/0.01/0.1. Fresh0/0.1 controls use the same trainer; do not compare the new coefficients directly with an old-trainer control. One seed42: no automatic adoption. These fits do not retrain curves.\n\n",
        "**Verdict:**0.001 is a modest FT balance candidate: global RMSE+0.07%, confirmed-fracture endpoints−0.91%, failing-strut motion−1.14%, broader box−0.12%. The0.003 arm trades a smaller endpoint gain for−1.60% strut-motion error. Neither is confirmed across seeds. Keep FT MSE as the accepted loss; reject the tested0.1 setting (global+119.75%) despite its narrow-event gains. Do not combine these with CNN2 without a later isolated comparison.\n\n",
        "### Global performance versus matched control\n\n",img("spatial-balance-global-control","FT spatial coefficients versus the fresh zero control"),
        "### Global performance versus TRAIN mean\n\n",img("spatial-balance-global-mean","FT accuracy beyond the training-only mean"),spread,
        "### STATUS fracture regions versus control\n\n",img("spatial-balance-fracture-control","Narrow actual failure events versus the broader FT box"),
        "### STATUS fracture regions versus TRAIN mean\n\n",img("spatial-balance-fracture-mean","Identical failure regions and training-mean references"),
        "### Best/worst FT examples\n\n",
        f"Display coefficient{selected.variant}, chosen solely as the lowest-global-error nonzero coefficient in this screen—not an accepted configuration. Whole-field RMSE ranks specimens; U2 at the target's strongest regional increment.\n\n",
        img("spatial-balance-FT-field-examples","True / predicted / signed error / percent error in the FT crack-tip region"),
        "### Training time\n\n",img("spatial-balance-runtime","Final-generator fit hours; no queue, OOF or downstream curve training"),
        "### Validation and loss balance\n\n",img("spatial-balance-validation","Fixed validation MSE and selected-checkpoint weighted component ratios"),
        img("spatial-balance-train-validation","Selected train/validation displacement errors, separate from weighted total objectives"),
        "Smaller coefficients do not necessarily improve every fracture metric monotonically. Loss-value ratios are not measured gradient competition. The geometric FT-box-only spatial study remains separate and queued; no STATUS path defines its training region.\n\n",
        "### UT spatial0.1: field/fracture gains versus curve peaks\n\n",
        img("ut-spatial-paired-bootstrap","Paired specimens and training seeds expose the curve-peak trade-off"),
        "These use the completed independent Mixed pipeline, not the new FT-only fits. Preserve the existing ratios of mean absolute peak-height/work error to mean absolute true peak/integral, and hard-location MAE divided by loading span. Bootstrap resampling retains these target denominators through paired weights. Intervals are descriptive95% hierarchical bootstrap intervals over three seeds and793 paired validation IDs; validation also selected checkpoints, so these are not an untouched-test or confirmatory significance claim. Lower is better in both panels.\n\n",
        "Keep UT spatial as a conditional candidate if repeated peak deterioration remains: a global/fracture RMSE gain alone does not settle the curve safeguard. No new curve metrics, loss or acceptance threshold is introduced.\n\n",
        "[FT metrics/cost](spatial-balance-metrics.csv) · [Event counts and denominators](spatial-balance-regions.csv) · [UT paired results/intervals](ut-spatial-paired-bootstrap.csv) · [Download verification](spatial-balance-download-verification.json). Below: the original three-seed custom-loss comparison, retained separately from this new FT coefficient screen.\n"]
    replace_study_followup(a.out/"README.md","spatial-balance","".join(text))
    print(table.to_string(index=False),flush=True);print(paired.to_string(index=False),flush=True)


if __name__=="__main__":main()
