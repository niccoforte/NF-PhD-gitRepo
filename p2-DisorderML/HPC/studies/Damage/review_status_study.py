"""Readable STATUS study protocol and saved-result review; no training."""
import argparse
import json
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np
from resources.MLstudy_reports import (collect_fields, comparison_figures, loss_figures,
    architecture_figure, field_examples, figure)


def damage_figures(table,root,suite,out):
    labels={"damage":"STATUS only","weak":"Auxiliary0.1","strong":"Auxiliary1"}
    variants=list(labels)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,mode in enumerate(("UT","FT")):
        part=table[(table["mode"]==mode)&(table.threshold_policy=="fixed_05")].set_index("variant").loc[variants]
        bars=axes[0,col].bar(range(3),part.average_precision,color="#387c9b")
        axes[0,col].bar_label(bars,fmt="%.3f",padding=3)
        axes[0,col].axhline(part.actual_positive_fraction.iloc[0],color="#bd765b",linestyle="--",label="No-information prevalence reference")
        axes[0,col].set(title=mode,ylabel="Average precision (0–1, higher is better)",ylim=(0,1.05),xticks=range(3),xticklabels=list(labels.values()));axes[0,col].legend(fontsize=7)
        for v in variants:
            curve=pd.read_csv(root/f"{suite}-{v}-{mode}-s42/results/damage-pr-curve.csv")
            axes[1,col].plot(curve.recall,curve.precision,label=labels[v])
        axes[1,col].set(xlabel="Recall: fraction of actual fractures found",ylabel="Precision: fraction of predicted fractures correct",xlim=(0,1),ylim=(0,1));axes[1,col].legend(fontsize=8);axes[1,col].grid(alpha=.2)
    fig.suptitle("Can geometry identify fracture? Separate from whether damage improves displacement\nValidation AP ranks all thresholds; curves are compact saved displays")
    fig.tight_layout();fig.savefig(out/"damage-discrimination.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,mode in enumerate(("UT","FT")):
        for policy,marker in (("fixed_05","o"),("validation_f1","x")):
            part=table[(table["mode"]==mode)&(table.threshold_policy==policy)].set_index("variant").loc[variants]
            for metric,color in (("precision","#397aa2"),("recall","#ba7653")):
                axes[0,col].plot(range(3),part[metric],marker+"-",color=color,label=metric+" "+("threshold 0.5" if policy=="fixed_05" else "validation-selected"))
        part=table[(table["mode"]==mode)&(table.threshold_policy=="fixed_05")].set_index("variant").loc[variants]
        axes[0,col].set(title=mode,ylabel="Precision / recall",ylim=(0,1.04),xticks=range(3),xticklabels=list(labels.values()));axes[0,col].legend(fontsize=7);axes[0,col].grid(alpha=.2)
        axes[1,col].bar(np.arange(3)-.16,part.predicted_positive_fraction*100,width=.32,label="Predicted broken entries")
        axes[1,col].bar(np.arange(3)+.16,part.actual_positive_fraction*100,width=.32,label="Actual broken entries")
        axes[1,col].set(ylabel="Fraction of known strut/frame entries (%)",xticks=range(3),xticklabels=list(labels.values()));axes[1,col].legend(fontsize=7)
    fig.suptitle("Class balancing avoids all-intact predictions, but threshold 0.5 overcalls fractures\nValidation-selected thresholds are optimistic diagnostics, not locked-test results")
    fig.tight_layout();fig.savefig(out/"damage-thresholds.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,("UT","FT")):
        part=table[(table["mode"]==mode)&(table.threshold_policy=="fixed_05")].set_index("variant").loc[variants]
        ax.plot(range(3),part.onset_interval_mae,"o-",label="Onset error, matched failed struts")
        ax.set(title=mode,ylabel="Onset MAE (retained frame intervals)",xticks=range(3),xticklabels=list(labels.values()));ax.grid(alpha=.2)
        for x,row in enumerate(part.itertuples()):ax.annotate(f"missed{row.missed_failed_struts:,}\nfalse{row.false_failed_struts:,}",(x,row.onset_interval_mae),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
    fig.suptitle("Failure timing is conditional on detecting the strut\nFalse/missed strut counts must accompany onset error; no irreversible repair")
    fig.tight_layout();fig.savefig(out/"damage-timing.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3));balance_rows=[]
    for ax,mode in zip(axes,("UT","FT")):
        for x,variant in enumerate(("weak","strong")):
            path=root/f"{suite}-{variant}-{mode}-s42";cfg=json.loads((path/"model.json").read_text())
            h=pd.read_csv(path/"loss_history.csv");row=h[h.epoch==cfg["best_epoch"]].iloc[0]
            coefficient=.1 if variant=="weak" else 1.
            ratio=coefficient*row.val_damage/row.val_field
            ax.bar(x,ratio,color="#a87650");ax.text(x,ratio*1.1,f"{ratio:.2f}×",ha="center")
            balance_rows.append(dict(mode=mode,variant=variant,best_epoch=cfg["best_epoch"],field_loss=row.val_field,
                raw_damage_loss=row.val_damage,coefficient=coefficient,weighted_damage_to_field=ratio))
        ax.set(title=mode,yscale="log",ylabel="Weighted STATUS loss / field loss (log)",xticks=[0,1],xticklabels=["Auxiliary0.1","Auxiliary1"]);ax.axhline(1,color="black",linestyle="--");ax.grid(axis="y",alpha=.2);ax.margins(y=.3)
    fig.suptitle("The same auxiliary coefficient does not give the same loss balance\nSelected validation checkpoint; loss-value ratios are not measured gradient ratios")
    fig.tight_layout();fig.savefig(out/"auxiliary-balance.png",dpi=170);plt.close(fig)
    pd.DataFrame(balance_rows).to_csv(out/"auxiliary-balance.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,mode in enumerate(("UT","FT")):
        for variant in variants:
            h=pd.read_csv(root/f"{suite}-{variant}-{mode}-s42/loss_history.csv")
            axes[0,col].plot(h.epoch,h.average_precision,label=labels[variant])
            axes[1,col].plot(h.epoch,h.val_damage.where(h.val_damage>0),label=labels[variant])
        axes[0,col].set(title=mode,ylabel="Validation average precision (higher is better)",xlabel="Epoch");axes[0,col].legend(fontsize=7);axes[0,col].grid(alpha=.2)
        axes[1,col].set(yscale="log",ylabel="Raw balanced validation BCE (log)",xlabel="Epoch");axes[1,col].legend(fontsize=7);axes[1,col].grid(alpha=.2)
    fig.suptitle("Damage-only and auxiliary learning histories\nSTATUS-only selects by AP; auxiliary fits select by field error, not best damage score")
    fig.tight_layout();fig.savefig(out/"damage-learning.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,mode in enumerate(("UT","FT")):
        for col,variant in enumerate(("weak","strong")):
            coefficient=.1 if variant=="weak" else 1.
            h=pd.read_csv(root/f"{suite}-{variant}-{mode}-s42/loss_history.csv");ax=axes[row,col]
            if not np.allclose(h.val_field+coefficient*h.val_damage,h.val_loss,rtol=1e-5,atol=1e-8):raise ValueError("STATUS loss components do not reconstruct total")
            for values,label in ((h.val_loss,"Total validation loss"),(h.val_field,"Field MSE"),(coefficient*h.val_damage,f"{coefficient:g} × balanced STATUS BCE")):
                ax.plot(h.epoch,values.where(values>0),label=label)
            ax.set(title=f"{mode} {labels[variant]}",yscale="log",xlabel="Epoch",ylabel="Validation loss contribution (log)");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("What each auxiliary objective actually optimizes\nWeighted terms reconstruct total; FT field MSE can be much smaller than the damage term")
    fig.tight_layout();fig.savefig(out/"status-loss-components.png",dpi=170);plt.close(fig)


def damage_path_examples(root,suite,data_root,out,table):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap,BoundaryNorm
    for mode in ("UT","FT"):
        best=table[(table["mode"]==mode)&(table.threshold_policy=="fixed_05")].sort_values("average_precision").iloc[-1].variant
        path=root/f"{suite}-{best}-{mode}-s42";data=json.loads((path/"model_data.json").read_text())
        metrics=json.loads((path/"results/metrics.json").read_text());threshold=metrics["damage"]["validation_f1"]["threshold"]
        with np.load(path/"results/predictions.npz",allow_pickle=False) as z:
            score,target,known=z["damage_probability"],z["damage_truth"].astype(bool),z["damage_mask"]
            edges=z["edge_index"];ids=list(map(str,z["sample_ids"]));xy=z["canonical_coords"]
        ranking=[];frames=[]
        for index in range(len(ids)):
            observed=known[index].any(axis=0);frame=int(np.flatnonzero(observed)[-1]);frames.append(frame)
            valid=known[index,:,frame];actual=target[index,:,frame]&valid;predicted=(score[index,:,frame]>=threshold)&valid
            union=(actual|predicted).sum();ranking.append((actual&predicted).sum()/union if union else np.nan)
        picks=[int(np.nanargmax(ranking)),int(np.nanargmin(ranking))]
        with np.load(data_root/f"MULTI-{mode}-disNodes-allFIELD.npz",allow_pickle=False) as bank:
            lookup={str(v):i for i,v in enumerate(bank["sample_ids"])};positions=bank["sample_node_coords"][[lookup[ids[i]] for i in picks]]
        fig,axes=plt.subplots(2,3,figsize=(13,9))
        for row,(sample,native) in enumerate(zip(picks,positions)):
            frame=frames[sample];valid=known[sample,:,frame];actual=target[sample,:,frame];predicted=score[sample,:,frame]>=threshold
            loc=xy.copy();loc[np.asarray(data["metadata"]["field_to_canonical"][mode])]=native
            segments=loc[edges];decision=np.zeros(len(edges),int)
            decision[valid&~actual&predicted]=1;decision[valid&actual&~predicted]=2;decision[valid&actual&predicted]=3;decision[~valid]=4
            values=[np.where(valid,actual.astype(float),np.nan),np.where(valid,score[sample,:,frame],np.nan),decision]
            cmaps=[ListedColormap(["#6c51a0","#b7293b"]),"magma",ListedColormap(["#b4b4b4","#d46f34","#2786a9","#87449a","#222222"])]
            norms=[BoundaryNorm([-.5,.5,1.5],2),None,BoundaryNorm(np.arange(6)-.5,5)]
            for col,(title,value,cmap,norm) in enumerate(zip(("True strut fracture (0 intact,1 broken)","Predicted fracture score","Decision: false / missed / found"),values,cmaps,norms)):
                emphasize=(actual if col==0 else predicted if col==1 else actual|predicted)&valid
                ax=axes[row,col];collection=LineCollection(segments,cmap=cmap,norm=norm,linewidths=np.where(emphasize,1.7,.55))
                collection.set_array(value)
                if col==1:collection.set_clim(0,1)
                row_label=f"{'Best' if row==0 else 'Worst'} ID{ids[sample]}, frame{frame+1}, path Jaccard={ranking[sample]:.2f}\n" if col==0 else ""
                ax.add_collection(collection);ax.autoscale();ax.set(title=row_label+title,aspect="equal",xlabel="Initial x (mm)",ylabel="Initial y (mm)")
                ax.title.set_fontsize(10)
                if mode=="FT":ax.set_xlim(96.6,173.6);ax.set_ylim(54,136)
                bar=fig.colorbar(collection,ax=ax,shrink=.75)
                if col==2:bar.set_ticks([0,1,2,3,4],labels=["Intact correct","False fracture","Missed fracture","Fracture found","Unknown"])
                elif col==0:bar.set_ticks([0,1])
        fig.suptitle(f"{mode} {best}: damage paths at each specimen's last observed frame\nRank by snapshot path Jaccard at validation-selected threshold{threshold:.4f}; optimistic validation examples, initial geometry",fontsize=11)
        fig.tight_layout(rect=(0,0,1,.91),h_pad=3);fig.savefig(out/f"{mode}-damage-path-examples.png",dpi=170);plt.close(fig)


def review_results(root,suite,data_root,out):
    region_api=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Loss/review_fracture_regions.py"))
    labels={"control":"Displacement MSE","weak":"MSE +0.1 STATUS","strong":"MSE +1 STATUS"}
    entries=[dict(mode=m,variant=v,label=labels[v],path=root/f"{suite}-{v}-{m}-s42") for m in ("UT","FT") for v in labels]
    fields,regions,frames=collect_fields(entries,data_root,region_api)
    for name,values in (("field-metrics",fields),("fracture-metrics",regions),("frame-metrics",frames)):values.to_csv(out/f"{name}.csv",index=False)
    damage=[];cost=[]
    for mode in ("UT","FT"):
        for variant in ("control","damage","weak","strong"):
            path=root/f"{suite}-{variant}-{mode}-s42";meta=json.loads((path/"run_metadata.json").read_text())
            cfg=json.loads((path/"model.json").read_text());m=json.loads((path/"results/metrics.json").read_text())
            if meta.get("status")!="complete" or meta.get("checkpoint_reload")!="passed":raise ValueError("Incomplete STATUS run")
            h=pd.read_csv(path/"loss_history.csv")
            cost.append(dict(mode=mode,variant=variant,hours=meta["training_seconds"]/3600,epochs=len(h),best_epoch=cfg["best_epoch"],job=meta["slurm_job_id"]))
            if variant=="control":continue
            with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
                probabilities,targets,masks=a["damage_probability"],a["damage_truth"],a["damage_mask"]
                tp=fp=fn=0
                for start in range(0,len(a["sample_ids"]),32):
                    p=probabilities[start:start+32];y=targets[start:start+32].astype(bool);v=masks[start:start+32]
                    tp+=int(((p>=.5)&y&v).sum());fp+=int(((p>=.5)&~y&v).sum());fn+=int(((p<.5)&y&v).sum())
                saved=m["damage"]["fixed_05"]
                if (tp,fp,fn)!=(saved["true_positive"],saved["false_positive"],saved["missed_fractures"]):raise ValueError("Damage confusion counts differ")
            for policy in ("fixed_05","validation_f1"):
                damage.append(dict(mode=mode,variant=variant,threshold_policy=policy,**m["damage"][policy]))
    damage=pd.DataFrame(damage);damage.to_csv(out/"damage-metrics.csv",index=False);pd.DataFrame(cost).to_csv(out/"all-training-costs.csv",index=False)
    render_results(root,suite,data_root,out,fields,regions,damage,cost)


def render_results(root,suite,data_root,out,fields,regions,damage,cost):
    """Render a verified numerical review without refitting or retraining."""
    region_api=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Loss/review_fracture_regions.py"))
    labels={"control":"Displacement MSE","weak":"MSE +0.1 STATUS","strong":"MSE +1 STATUS"}
    comparison_figures(fields,regions,out);damage_figures(damage,root,suite,out)
    damage_path_examples(root,suite,data_root,out,damage)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,mode in zip(axes,("UT","FT")):
        part=pd.DataFrame(cost);part=part[part["mode"]==mode]
        bars=ax.bar(range(len(part)),part.hours,color="#477a9b");ax.bar_label(bars,fmt="%.2f",padding=3)
        ax.set(title=mode,ylabel="Training hours (queue excluded)",xticks=range(len(part)),xticklabels=part.variant);ax.grid(axis="y",alpha=.2)
    fig.suptitle("All eight STATUS-study fits: different selection/stopping objectives affect cost")
    fig.tight_layout();fig.savefig(out/"runtime.png",dpi=170);plt.close(fig)
    selected={m:fields[fields["mode"]==m].sort_values("rmse").iloc[0].variant for m in ("UT","FT")}
    loss_figures(fields,out,selected);architecture_figure(out,"Auxiliary STATUS head",damage=True)
    examples={}
    for mode in ("UT","FT"):
        row=fields[(fields["mode"]==mode)&(fields.variant==selected[mode])].iloc[0]
        data=json.loads((Path(row.path)/"model_data.json").read_text())
        examples[mode]=field_examples(row.path,mode,out,labels[row.variant],data_root,region_api["training_mean"](data_root,data,mode))
    text=["# STATUS learning study\n\n## Recap, findings and verdict\n\n",
        "All eight production runs completed successfully: displacement-only, STATUS-only, and displacement+STATUS at coefficients 0.1 and 1, independently for UT/FT. STATUS-only trains BCE alone and intentionally has no displacement score. Control trains displacement MSE alone; both auxiliary arms train displacement MSE + weighted BCE. STATUS-only AP/PR, path/timing, costs and losses are included below. No curve training or explicit damage feedback. **Damage can be predicted; adding its loss did not clearly improve displacement at these presets. Do not promote auxiliary STATUS into WINNER.md.** One seed 42, validation only.\n\n",
        "UT auxiliary global field changes are below 0.1%, too small to establish a general benefit, while confirmed fracture endpoints improve up to 3.17%. FT auxiliary fields worsen globally and at confirmed fractures, despite strong fracture discrimination. Predicting which strut breaks and predicting its displacement are separate outcomes. The same loss coefficient is not equally balanced in UT/FT.\n\n",
        figure(out,"architecture-comparison.png","Original displacement control versus candidate strut-head architecture"),
        "The control remains the practical displacement recommendation from this study. The candidate diagram explains the added head; it is not a claim that it won. Both branches update the same encoder during auxiliary training; no true damage is an input and predicted damage does not yet feed back into a refinement pass.\n\n",
        "## Global field performance versus the study control\n\nUse the freshly trained STATUS-study control, not another study's slightly different training realization. Curves were not fitted.\n\n",
        figure(out,"global-control.png","Displacement change from auxiliary STATUS"),
        "## Global field performance versus the training mean\n\nRMSE skill=100×(1−model RMSE/TRAIN-mean RMSE). Mean is fitted per node/frame/component using training IDs only; zero equals mean-predictor error.\n\n",
        figure(out,"global-mean.png","Displacement accuracy beyond the average field"),
        figure(out,"diversity.png","Across-specimen field variation"),
        figure(out,"UT-mean-field-examples.png","UT actual training-mean field comparison"),
        figure(out,"FT-mean-field-examples.png","FT actual training-mean field comparison"),
        "## STATUS fracture-region performance versus control\n\nBoth endpoints and signed relative motion of observed intact→failed struts at onset ±1 retained frame. Unknown frames stay masked. The larger FT crack-tip box is a broader, overlapping diagnostic. True labels define evaluation regions only.\n\n",
        figure(out,"fracture-control.png","Displacement changes at confirmed fractures"),
        "## STATUS fracture-region performance versus training mean\n\nModel and mean are evaluated on identical event entries; reconstructed training-mean global error must reproduce saved baseline metrics.\n\n",
        figure(out,"fracture-mean.png","Confirmed-fracture displacement skill"),
        "## Best and worst field examples\n\nUse each task's lowest global-error arm, including control. Rank samples by full valid field RMSE; show U2 at the largest true regional increment, UT full and FT zoom. Percent error is 100×|prediction−truth|/|truth|; tiny truth below 1% of training-mean RMS is grey and colours cap at P95. This display rule does not change metrics.\n\n",
        figure(out,"UT-field-examples.png","UT four-panel field review"),figure(out,"FT-field-examples.png","FT four-panel field review"),
        "## Training time\n\nFinal generators only; no OOF or downstream curves. Queue time is excluded. STATUS-only selection uses average precision and can train longer than field-selected arms. Detailed all-eight runtimes/epochs are in the CSV.\n\n",
        figure(out,"runtime.png","Field and auxiliary training hours"),
        "## Loss histories\n\nValidation-only compares arms; selected train/validation diagnoses field overfitting; component plots show the MSE/BCE balance; damage-learning plots include STATUS-only AP and BCE. These are distinct diagnostics. Total losses differ by auxiliary coefficient and cannot rank formulations. Field checkpoints use the unchanged physical field-MSE/TRAIN-mean-MSE score; STATUS-only uses average precision. All plotted losses are positive saved values on log axes.\n\n",
        figure(out,"loss-validation.png","Validation losses"),figure(out,"loss-train-validation.png","Selected training validation histories"),
        figure(out,"status-loss-components.png","Field and weighted STATUS loss contributions"),
        figure(out,"damage-learning.png","STATUS-only and auxiliary learning histories"),
        "## What the damage head learned\n\nAverage precision is threshold-independent; the no-information reference is the broken-entry prevalence, approximately 1.80% UT and 0.119% FT in validation. High ordinary intact/broken accuracy would hide rare missed fractures.\n\n",
        figure(out,"damage-discrimination.png","Fracture discrimination and precision recall"),
        figure(out,"damage-thresholds.png","Class balance and threshold effects"),
        figure(out,"damage-timing.png","Fracture timing with false and missed struts"),
        "The following saved strut maps use each task's highest-AP arm. Best/worst rank snapshot path overlap at the last observed frame, using the validation-selected threshold; this is an optimistic diagnostic, not a test score. Initial disordered geometry is shown, not deformed beams. A percentage error relative to a binary intact value 0 is undefined, so classification errors are shown explicitly instead.\n\n",
        figure(out,"UT-damage-path-examples.png","UT actual and predicted fracture paths with missed and false struts"),
        figure(out,"FT-damage-path-examples.png","FT crack-tip actual and predicted fracture paths"),
        "## Relative auxiliary-loss influence\n\nAt the chosen checkpoint, weighted STATUS/field loss is 0.13× and 1.33× in UT for coefficients 0.1 and 1, but 1.98× and 43.82× in FT. These are measured validation loss-value ratios, not gradient magnitudes. The supposedly weak 0.1 auxiliary objective can therefore outweigh FT displacement supervision. This supports a smaller-coefficient field-only screen before adding explicit damage feedback; it does not prove the entire cause of deterioration.\n\n",
        figure(out,"auxiliary-balance.png","Measured auxiliary loss balance at selected checkpoints"),
        "The MLP combines both endpoint latents and initial strut length/direction into 20 raw fracture scores, one per retained frame. Sigmoid gives scores between 0 and 1. TRAIN-derived positive weights 54.3 UT/839.8 FT emphasize rare failed entries; balancing changes scores, so threshold 0.5 is not a calibrated failure-probability decision. Validation-selected F1 thresholds illustrate the trade-off and are optimistic, not independent test findings. No cumulative-maximum repair is applied.\n\n",
        "**Next:** retain the separate damage predictor as evidence and a possible future output, but do not assume feeding it into displacement will fix the field. FT auxiliary 0.1/1 needs a substantially smaller loss-budget comparison, such as 0.001/0.01, before stronger coupling is justified. UT strong improves confirmed endpoint error 3.17% and relative-motion error 1.10%, despite only 0.04% global gain: a local candidate to confirm, not a null result or an adopted winner. FT weak/strong worsen confirmed endpoints 2.82%/8.53%. Damage-to-curve inputs remain a separate deferred experiment. Separate balance/confirmation follow-ups are recorded below; they do not change these completed results.\n\n",
        "[Field metrics and provenance](field-metrics.csv) · [Actual fracture errors/counts](fracture-metrics.csv) · [Damage thresholds, AP, timing and reversals](damage-metrics.csv) · [All-eight training costs](all-training-costs.csv) · [Download verification](download-verification.json). No locked-test evaluation or ODB re-export.\n"]
    from resources.MLstudy_reports import submission_note
    text.append(submission_note(out,"balance-jobs.tsv","STATUS balance and UT confirmation"))
    (out/"README.md").write_text("".join(text),encoding="utf-8")
    from resources.MLstudy_reports import insert_variation_section
    insert_variation_section(out/"README.md",fields[["variant","path"]].to_dict("records"))
    (out/"example-selection.json").write_text(json.dumps(examples,indent=2))
    print(fields[["mode","variant","change_percent","skill_percent","hours"]].to_string(index=False))
    print(damage[damage.threshold_policy=="fixed_05"][["mode","variant","average_precision","precision","recall"]].to_string(index=False))


def schematic(out):
    fig,ax=plt.subplots(figsize=(13,7));ax.set(xlim=(0,13),ylim=(0,7));ax.axis("off")
    def box(x,y,w,h,text,color="#dfeaf3"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",facecolor=color,edgecolor="#3a586b"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)
    def arrow(a,b,label=None,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,color="#40596a",linestyle="--" if dashed else "-",lw=1.5))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,label,fontsize=9,ha="center")
    box(.3,3,2.1,1.4,"Initial disorder\n+ saved context\n+ presence mask")
    box(3,3,2.4,1.4,"One field Transformer\n384 channels,8 heads\n3 encoder layers")
    box(6.1,4.5,2.5,1.1,"Nodal displacement head\nU1/U2 ×20 frames")
    box(6.1,1.8,2.5,1.6,"Native strut readout\nTake latent nodes i,j\nmean + absolute difference\n+ initial edge geometry")
    box(9.2,2,3.3,1.2,"One MLP shared across struts\n20 fracture logits per strut\nD=1 means broken")
    box(9.2,4.5,3.3,1.1,"Masked displacement MSE\nAll present valid nodes",color="#e8efdc")
    box(9.2,.3,3.3,1.1,"Masked class-balanced BCE\nTRAIN-derived weights",color="#f5e6d7")
    arrow((2.4,3.7),(3,3.7));arrow((5.4,4.1),(6.1,5));arrow((5.4,3.3),(6.1,2.6));arrow((8.6,5.05),(9.2,5.05));arrow((8.6,2.6),(9.2,2.6));arrow((10.85,2),(10.85,1.4))
    ax.text(6.4,.55,"L = Ldisplacement + λ LSTATUS\nλ=0.1 or 1; both update the same encoder",ha="center",fontsize=12,color="#384d60")
    ax.text(6.5,6.5,"Repeat independently for UT and FT: no shared parameters or losses between tasks",ha="center",fontsize=14,weight="bold")
    ax.text(3.4,1.75,"Control: displacement head only\nSTATUS-only: no displacement objective\nAuxiliary: both branches; no damage feedback",ha="center",fontsize=11)
    ax.text(6.5,6.05,"Truth is used only in losses. At inference: disorder/context → predicted displacement and/or damage.",ha="center",fontsize=11)
    fig.tight_layout();fig.savefig(out/"status-architecture.png",dpi=180);fig.savefig(out/"status-architecture.svg");plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/status-study")
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Damage/Transformer"))
    p.add_argument("--suite",default="status-261027")
    p.add_argument("--jobs",type=Path,help="Optional actual jobs.tsv; no inferred scheduler state")
    p.add_argument("--execution-note",help="Explicit dated scheduler observation; not an automatic live check")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--protocol-only",action="store_true")
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True);schematic(a.out)
    if not a.protocol_only:
        review_results(a.root,a.suite,a.data_root,a.out)
        print("REPORT",a.out/"README.md")
        return
    text=["# STATUS learning: preliminary study report\n\n**This is the experiment protocol, not an accuracy result.** Dataset extraction is complete. Learning outcomes will be added from completed production archives; queued jobs and a successful preflight do not establish predictive quality.\n\n",
        f"![STATUS and displacement architecture](<{(a.out/'status-architecture.png').resolve().as_posix()}>)\n\n",
        "## Exactly what is compared\n\nEight seed 42 fits: four arms × two fully independent tasks. All retain the saved7137/793/881 specimen partition, geometry/context contract,20 retained frames, field architecture384/8/3, AdamW settings, batch2,450 maximum epochs and patience52. No OOF bank, curve fit, SDEG export or explicit damage feedback is included.\n\n",
        "| Arm | Trained outputs | Objective | Checkpoint selection |\n|---|---|---|---|\n| Control | Displacement | Masked MSE | Fixed physical field MSE / training-mean MSE |\n| STATUS-only | Strut fracture histories | Balanced masked BCE | Validation average precision; BCE tie-break |\n| Weak auxiliary | Displacement + fracture | MSE +0.1 BCE | Same field criterion as control |\n| Strong auxiliary | Displacement + fracture | MSE +1 BCE | Same field criterion as control |\n\n",
        "The STATUS-only arm retains the same nodal encoder but does not train or report displacement predictions. Controls are freshly fitted in the same consumer so trainer/selection differences cannot silently favour an intervention. Common encoder initialization is identical per task; adding the strut head uses a separate RNG stream.\n\n",
        "## How one strut is predicted\n\nFor native strut i–j, gather both final nodal vectors. Concatenate their mean, absolute difference, and three initial-geometry quantities: length/specimen span and sin(2θ), cos(2θ). Reversing endpoints gives the same representation. The same small MLP reads every strut and returns20 logits. It creates no new attention tokens. Geometry comes from the actual initial disordered specimen; true future damage is never provided.\n\n",
        "In the auxiliary arms, displacement and STATUS losses both update the nodal encoder. This is the initial linkage being tested. Predicted STATUS is not yet fed back into a second refinement pass and does not remove connections. UT and FT have separate encoders, heads, optimizers and training losses.\n\n",
        "## Labels and rare fractures\n\nThe predicted binary label is D=1 for broken, derived from Abaqus STATUS; it is not raw STATUS=1 (intact). A strut is broken if at least one constituent element has verified STATUS=0. Unknown entries remain masked, initial FT gaps are absent geometry, and frame0 is dropped consistently. IDs, native-to-canonical endpoints, retained frames and sample initial coordinates must match before training. Short UT histories remain unknown after their observed prefix.\n\n",
        "From TRAIN only, let p be the fraction of known broken entries. Set r=min((1−p)/p,1000). Use BCEWithLogits(pos_weight=r), divided by (1−p)+r p. This gives rare missed fractures more influence while keeping the expected weight near one; λ then controls auxiliary influence. The cap and coefficients are fixed presets. No validation/test statistics fit the balancing.\n\n",
        "## How success will be judged\n\nReport average precision against the prevalence reference, precision/recall, missed and false fractures, predicted positive fraction, strut-path overlap, onset interval error and broken→intact reversals. Report fixed threshold 0.5 plus the F1-maximizing validation threshold. The latter is an optimistic validation diagnostic, frozen before any future test evaluation; weighted logits are not calibrated failure probabilities. Never silently apply cumulative maximum.\n\n",
        "Auxiliary benefit requires improved displacement, especially fracture-event/jump metrics, with global RMSE safeguards. Accurate STATUS and better displacement are distinct outcomes. Confirm useful arms on seeds 43/44 before expensive OOF generation or curve training. Loss histories are logarithmic.\n\n",
        "## Execution and results\n\nOne disposable8h all-arm GPU preflight, then eight 240h/1GPU/12CPU/90000MB jobs through home→scratch→archive. Successful preflight keeps only its home log. Run outputs use `MULTI/Damage/Transformer/<suite>-<arm>-<UT|FT>-s42`; STATUS checkpoints have their own explicit versioned reader, while exported field arrays follow the existing diagnostic contract.\n\n"]
    if a.jobs:
        jobs=pd.read_csv(a.jobs,sep="\t");text += ["Submitted job IDs below are a receipt, not a live status check.\n\n",jobs.to_markdown(index=False),"\n\n"]
    if a.execution_note:text += [a.execution_note+"\n\n"]
    rows=[]
    for mode in ("UT","FT"):
        for variant in ("control","damage","weak","strong"):
            path=a.root/f"{a.suite}-{variant}-{mode}-s42"
            if (path/"run_metadata.json").exists() and json.loads((path/"run_metadata.json").read_text()).get("status")=="complete":
                m=json.loads((path/"results/metrics.json").read_text());d=m.get("damage",{}).get("fixed_05",{})
                field=m["diagnostics"]["field"].get(mode,{})
                skill=100*field["skill_vs_mean_field_rmse"] if field.get("mean_field_baseline_source")=="train_mean_field" else None
                rows.append(dict(task=mode,arm=variant,field_skill_vs_train_mean_percent=skill,
                    field_rmse=field.get("rmse"),average_precision=d.get("average_precision"),recall=d.get("recall"),precision=d.get("precision")))
    if rows:text += [pd.DataFrame(rows).to_markdown(index=False),"\n"]
    else:text += ["**No completed production learning results are available in this local report yet.**\n"]
    (a.out/"README.md").write_text("".join(text),encoding="utf-8");print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
