"""Read-only scientific study figures and fixed physical field comparisons.

No model inference, fitting, scientific defaults or archive mutation occurs here.
Study entry points supply explicit runs and the reviewed regional calculations.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


def figure(out, name, caption):
    return f"\n![{caption}](<{(Path(out)/name).resolve().as_posix()}>)\n\n"


def verify_download(root, manifest, expected_runs):
    root=Path(root).resolve();runs=set();count=size=0
    for line in Path(manifest).read_text(encoding="utf-8-sig").splitlines():
        expected,name=line.split("  ",1);path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError("Checksum path escapes result root")
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b""):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError("Downloaded checksum differs: "+name)
        runs.add(Path(name).parts[0]);count+=1;size+=path.stat().st_size
    if len(runs)!=expected_runs:raise ValueError("Unexpected downloaded run count")
    return dict(runs=len(runs),files=count,bytes=size,sha256="all matched")


def collect_fields(entries,data_root,region_api):
    """Compare physical validation fields, exact IDs/masks and TRAIN-only means."""
    rows=[];regional=[];frames=[];contracts={};cache={}
    for entry in entries:
        mode,variant,path=entry["mode"],entry["variant"],Path(entry["path"])
        meta=json.loads((path/"run_metadata.json").read_text())
        metrics=json.loads((path/"results/metrics.json").read_text())
        data=json.loads((path/"model_data.json").read_text())
        if meta.get("status")!="complete" or meta.get("checkpoint_reload")!="passed" or metrics.get("evaluation_split")!="val":
            raise ValueError("Require completed/reloaded validation artifacts: "+str(path))
        if meta.get("mode")!=mode or meta.get("variant")!=variant:raise ValueError("Run identity disagrees with study entry")
        ids=list(map(str,data["sample_ids"]["val"]))
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate validation IDs")
        for key,value in (("splits",data["sample_ids"]),("geometry_scale",data["normalizers"]["geometry"]),
            ("presence",data["node_masks"]),("reference",data["metadata"]["canonical_coords"]),
            ("frames",data["metadata"]["field_frame_values"]),("components",data["metadata"]["field_components"])):
            digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            if contracts.setdefault(key,digest)!=digest:raise ValueError("Changed comparison contract: "+key)
        if mode not in cache:
            nodes,events,edges,counts=region_api["region_masks"](Path(data_root)/f"MULTI-{mode}-disNodes-allD.npz",data,mode,ids)
            mean=region_api["training_mean"](Path(data_root),data,mode)
            cache[mode]=nodes,events,edges,counts,mean
        nodes,events,edges,counts,mean=cache[mode]
        with np.load(path/"results/predictions.npz",allow_pickle=False) as z:
            if list(map(str,z["sample_ids"]))!=ids:raise ValueError("Prediction IDs differ")
            pred,truth=z[f"{mode}_val_field_outputs"],z[f"{mode}_val_field_truth"]
            valid=z[f"{mode}_val_field_mask"].astype(bool)&z[f"{mode}_node_mask"][None,:,None]&np.isfinite(truth)
            if not np.isfinite(pred[valid]).all():raise ValueError("Nonfinite valid predictions")
            # Residual inverse reconstruction can round physical targets slightly;
            # do not demand byte equality after changing affine coordinates.
            if mode not in contracts:contracts[mode]=(truth.copy(),valid.copy())
            else:
                old,mask=contracts[mode]
                if not np.array_equal(valid,mask) or not np.allclose(truth[valid],old[valid],rtol=2e-5,atol=3e-6):raise ValueError("Physical truths or masks differ")
            measured=region_api["measure_regions"](pred,truth,valid,nodes,events,edges,mean,mode,np.asarray(data["metadata"]["canonical_coords"]))
            diag=metrics["diagnostics"]["field"][mode]
            if diag.get("mean_field_baseline_source")!="train_mean_field":raise ValueError("Unverified mean baseline")
            if not np.isclose(measured[0]["rmse"],diag["rmse"],rtol=3e-6):raise ValueError("Physical RMSE does not reproduce")
            if not np.isclose(measured[0]["mean_baseline_rmse"],diag["mean_field_baseline_rmse"],rtol=3e-6):raise ValueError("Reconstructed training mean does not reproduce")
            for row in measured:row.update(mode=mode,variant=variant)
            regional.extend(measured)
            p,y,v=pred.reshape(len(ids),800,20,2),truth.reshape(len(ids),800,20,2),valid.reshape(len(ids),800,20,2)
            for frame in range(20):
                keep=v[:,:,frame];err=(p[:,:,frame].astype(float)-y[:,:,frame])[keep]
                avg=np.broadcast_to(mean[:,frame],y[:,:,frame].shape)
                base=(avg-y[:,:,frame])[keep]
                rmse=np.sqrt(np.mean(err**2));den=np.sqrt(np.mean(base**2))
                frames.append(dict(mode=mode,variant=variant,frame=frame+1,rmse=rmse,skill_percent=100*(1-rmse/den),valid_entries=int(keep.sum())))
        history=pd.read_csv(path/"loss_history.csv")
        cfg=json.loads((path/"model.json").read_text())
        rows.append(dict(mode=mode,variant=variant,label=entry.get("label",variant),rmse=diag["rmse"],
            baseline_rmse=diag["mean_field_baseline_rmse"],skill_percent=100*diag["skill_vs_mean_field_rmse"],
            collapse_ratio=diag["collapse_ratio"],hours=meta["training_seconds"]/3600,epochs=len(history),
            best_epoch=cfg.get("best_epoch",cfg.get("training",{}).get("best_epoch")),job=meta.get("slurm_job_id"),
            revision=meta.get("source_revision"),path=str(path),**counts))
        print("FIELD_REVIEW",mode,variant,flush=True)
    table=pd.DataFrame(rows);regions=pd.DataFrame(regional)
    controls=table[table.variant=="control"].set_index("mode").rmse
    table["change_percent"]=100*(table.rmse/table["mode"].map(controls)-1)
    base=regions[regions.variant=="control"].set_index(["mode","region"]).rmse
    regions["change_percent"]=[100*(r.rmse/base.loc[r.mode,r.region]-1) for r in regions.itertuples()]
    return table,regions,pd.DataFrame(frames)


def comparison_figures(table,regions,out):
    """Global/control, global/mean, fracture/control and fracture/mean in that order."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    variants=list(dict.fromkeys(table.variant));labels=dict(zip(table.variant,table.label))
    for name,metric,title,ylabel in (("global-control","change_percent","Global performance versus the study control","RMSE change (%) — lower is better"),
        ("global-mean","skill_percent","Global performance beyond the average field","RMSE reduction vs TRAIN mean (%) — higher is better")):
        fig,axes=plt.subplots(1,2,figsize=(13,4.7))
        for ax,mode in zip(axes,("UT","FT")):
            part=table[table["mode"]==mode].set_index("variant");available=[v for v in variants if v in part.index]
            values=part.loc[available,metric]
            bars=ax.bar(range(len(available)),values,color=["#398d83" if (v<0 if metric=="change_percent" else v>0) else "#bd765b" for v in values])
            ax.bar_label(bars,fmt="%.2f",fontsize=8,padding=3);ax.axhline(0,color="black",lw=.8)
            ax.set(title=mode,ylabel=ylabel,xticks=range(len(available)),xticklabels=[labels[v] for v in available]);ax.tick_params(axis="x",rotation=40,labelsize=8);ax.grid(axis="y",alpha=.2);ax.margins(y=.2)
        fig.suptitle(title+"\nValidation only; one seed, no uncertainty estimate")
        fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=170);plt.close(fig)
    selected=("STATUS onset endpoints ±1 frame","STATUS onset strut relative motion ±1 frame","FT crack-tip box")
    short=("Failure endpoints","Failing-strut motion","Larger FT tip box")
    for name,metric,title in (("fracture-control","change_percent","Verified fracture regions versus the matched control"),
        ("fracture-mean","skill_percent","Verified fracture regions versus a TRAIN-mean field")):
        fig,axes=plt.subplots(1,2,figsize=(13,4.7))
        for ax,mode in zip(axes,("UT","FT")):
            available=[v for v in variants if ((regions["mode"]==mode)&(regions.variant==v)).any()]
            for offset,(region,label) in enumerate(zip(selected,short)):
                sub=regions[(regions["mode"]==mode)&(regions.region==region)].set_index("variant")
                if sub.empty:continue
                values=[sub.loc[v,metric] for v in available]
                ax.plot(np.arange(len(available))+(offset-1)*.13,values,"o",label=label)
            ax.axhline(0,color="black",lw=.8);ax.set(title=mode,ylabel="RMSE change (%) — lower is better" if metric=="change_percent" else "RMSE reduction vs TRAIN mean (%) — higher is better",
                xticks=range(len(available)),xticklabels=[labels[v] for v in available]);ax.tick_params(axis="x",rotation=40,labelsize=8);ax.grid(alpha=.2)
            ax.legend(fontsize=8,loc="upper left")
        fig.suptitle(title+"\nObserved intact→failed struts, onset ±1 retained frame; labels used for evaluation only")
        fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=170);plt.close(fig)
    for name,metric in (("runtime","hours"),("diversity","collapse_ratio")):
        fig,axes=plt.subplots(1,2,figsize=(13,4.4))
        for ax,mode in zip(axes,("UT","FT")):
            part=table[table["mode"]==mode];bars=ax.bar(range(len(part)),part[metric],color="#477a9b")
            ax.bar_label(bars,fmt="%.2f",fontsize=8,padding=2)
            ax.set(title=mode,ylabel="Training hours (queue excluded)" if metric=="hours" else "Mean predicted / true specimen variation",
                xticks=range(len(part)),xticklabels=part.label);ax.tick_params(axis="x",rotation=40,labelsize=8);ax.grid(axis="y",alpha=.2)
            if metric=="collapse_ratio":ax.axhline(1,color="black",linestyle="--",label="Matched spread");ax.legend()
        fig.suptitle("Training cost: final generators only" if metric=="hours" else "Prediction diversity: zero means constant output; matched spread is not accuracy")
        fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=170);plt.close(fig)


def loss_figures(table,out,selected=None):
    for name,columns in (("loss-validation",("val_loss",)),("loss-train-validation",("train_loss","val_loss"))):
        fig,axes=plt.subplots(1,2,figsize=(13,4.8))
        for ax,mode in zip(axes,("UT","FT")):
            for index,row in enumerate(table[table["mode"]==mode].itertuples()):
                if len(columns)>1 and selected and row.variant not in ("control",selected.get(mode)):continue
                h=pd.read_csv(Path(row.path)/"loss_history.csv");color=plt.get_cmap("tab10")(index%10)
                for column in columns:
                    values=h[column].where(h[column]>0)
                    ax.plot(h.epoch,values,"--" if column.startswith("train") else "-",color=color,label=row.label+" "+column.replace("_loss",""))
                if len(columns)>1:ax.axvline(row.best_epoch,color=color,alpha=.3,linestyle=":")
            ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Saved training objective (log)");ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle("Validation histories" if len(columns)==1 else "Selected training and validation histories; vertical line = chosen checkpoint")
        fig.tight_layout();fig.savefig(Path(out)/f"{name}.png",dpi=170);plt.close(fig)


def architecture_figure(out,intervention,damage=False,curve_stage=False):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for index,ax in enumerate(axes):
        ax.set(xlim=(0,10),ylim=(0,7));ax.axis("off")
        def box(x,y,w,h,text,color):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.1",facecolor=color,edgecolor="#456075"));ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
        def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color="#456075"))
        box(.3,4.3,2.1,1.1,"Disorder\n+ task context","#dbe8f3")
        box(3.1,4.3,2.7,1.1,"Independent field\nTransformer","#dceee8")
        box(6.5,4.3,2.9,1.1,"U1/U2 histories\nDisplacement head","#dceee8")
        arrow((2.4,4.85),(3.1,4.85));arrow((5.8,4.85),(6.5,4.85))
        if curve_stage:
            box(2.4,1.6,6.6,1.5,"Independent curve Transformer →201-point curve\nTRAIN: predicted + true fields; deploy: predicted only\nCurve gradients stop before the field model","#f3e6d5")
            arrow((7.95,4.3),(7.95,3.1))
            ax.text(5,6.2,"Custom losses tested; architecture unchanged" if index else "Masked field MSE + curve MSE",ha="center",fontsize=10)
        elif index and damage:
            box(3.1,1.7,2.7,1.3,"Gather each strut's\nendpoint latents\n+ initial edge geometry","#f3e6d5")
            box(6.5,1.7,2.9,1.3,"Shared strut MLP\n20 fracture logits\nMasked balanced BCE","#f3e6d5")
            arrow((4.45,4.3),(4.45,3));arrow((5.8,2.35),(6.5,2.35))
            ax.text(5,6.15,"TRAIN: displacement MSE + λ STATUS BCE\nλ=0.1 or1; both losses update the field encoder",ha="center",fontsize=10)
            ax.text(5,.95,"INFERENCE: predicted U and fracture scores; no true damage input",ha="center",fontsize=8)
        else:
            box(2.4,1.6,6.6,1.5,intervention if index else "Masked displacement MSE\nNo additional feature or auxiliary head","#f3e6d5")
            if not index:arrow((7.95,4.3),(7.95,3.1))
        ax.set_title("Tested candidate (not adopted)" if index else "Pre-study control",fontsize=12)
        ax.text(5,.45,"Four independent models; losses and stopping remain separate" if curve_stage else "UT and FT remain separate; no curve training in this screen",ha="center",fontsize=9)
    fig.suptitle("STATUS study: displacement control versus auxiliary fracture head" if damage else "Pre-study control versus the isolated interventions; no replacement adopted",fontsize=12)
    fig.tight_layout();fig.savefig(Path(out)/"architecture-comparison.png",dpi=170);plt.close(fig)


def percent_error(pred,truth,valid,training_scale):
    """Pointwise absolute error; mask tiny denominators rather than fabricate ratios."""
    keep=np.asarray(valid,bool)&np.isfinite(truth)&(np.abs(truth)>=.01*max(float(training_scale),1e-12))
    values=np.full(np.shape(truth),np.nan,dtype=float)
    np.divide(100*np.abs(pred-truth),np.abs(truth),out=values,where=keep)
    return values


def field_examples(path,mode,out,tag,data_root,mean):
    path=Path(path);data=json.loads((path/"model_data.json").read_text())
    with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
        y,p=a[f"{mode}_val_field_truth"],a[f"{mode}_val_field_outputs"]
        valid=a[f"{mode}_val_field_mask"].astype(bool)&a[f"{mode}_node_mask"][None,:,None]&np.isfinite(y)
        ids=list(map(str,a["sample_ids"]));xy=a["canonical_coords"];present=a[f"{mode}_node_mask"].astype(bool)
        error=np.where(valid,(p.astype(float)-y)**2,0)
        scores=np.sqrt(error.sum((1,2))/valid.sum((1,2)));picks=[int(scores.argmin()),int(scores.argmax())]
        # Field source coordinates are matched by IDs, not inherited array order.
        with np.load(Path(data_root)/f"MULTI-{mode}-disNodes-allFIELD.npz",allow_pickle=False) as bank:
            lookup={str(v):i for i,v in enumerate(bank["sample_ids"])}
            mapping=np.asarray(data["metadata"]["field_to_canonical"][mode]);positions=bank["sample_node_coords"][[lookup[ids[i]] for i in picks]]
            shapes=[]
            for row in positions:
                loc=xy.copy();loc[mapping]=row;shapes.append(loc)
        if mode=="FT":region=(xy[:,0]>=96.6)&(xy[:,0]<=173.6)&(xy[:,1]>=54)&(xy[:,1]<=136)&present
        else:region=present
        # Show U2 at the largest observed increment within the displayed region
        # for each selected specimen; report selection explicitly, no fixed late frame.
        chosen=[]
        for sample in picks:
            history=y[sample,:,1::2];keep=valid[sample,:,1::2]
            motion=np.diff(history,axis=1);ok=keep[:,1:]&keep[:,:-1]&region[:,None]
            energy=np.where(ok,motion**2,0).sum(0)/np.maximum(ok.sum(0),1)
            chosen.append(int(energy.argmax()+1))
        colour_values=[];error_values=[];percent_values=[]
        for sample,frame in zip(picks,chosen):
            channel=2*frame+1;keep=valid[sample,:,channel]&region
            colour_values.extend((y[sample,keep,channel],p[sample,keep,channel]))
            error_values.append(np.abs(p[sample,keep,channel]-y[sample,keep,channel]))
            scale=float(np.sqrt(np.mean(mean[region,frame,1]**2)))
            percent_values.append(percent_error(p[sample,:,channel],y[sample,:,channel],keep,scale))
        common_lo=float(min(v.min() for v in colour_values));common_hi=float(max(v.max() for v in colour_values))
        common_error=max(float(max(v.max() for v in error_values)),1e-12)
        finite_percent=np.concatenate([v[np.isfinite(v)] for v in percent_values])
        common_percent=max(float(np.percentile(finite_percent,95)),1.) if len(finite_percent) else 1.
        fig,axes=plt.subplots(2,4,figsize=(15,7.6));details=[]
        for row,(sample,frame,loc) in enumerate(zip(picks,chosen,shapes)):
            channel=2*frame+1;v=valid[sample,:,channel]&region
            truth,pred=y[sample,:,channel],p[sample,:,channel]
            # RMS of the training mean is a documented display-only scale.
            display_scale=float(np.sqrt(np.mean(mean[region,frame,1]**2)))
            ape=percent_error(pred,truth,v,display_scale)
            lo,hi,limit,cap=common_lo,common_hi,common_error,common_percent
            for col,(label,values) in enumerate((("Truth",truth),("Prediction",pred),("Signed error",pred-truth),("Absolute error / |truth| (%)",ape))):
                ax=axes[row,col];cmap=plt.get_cmap("magma" if col==3 else "coolwarm" if col==2 else "viridis").copy();cmap.set_bad("#b5b5b5")
                view=np.ma.masked_invalid(values[region]);image=ax.scatter(loc[region,0],loc[region,1],c=view,s=17,cmap=cmap,
                    vmin=0 if col==3 else -limit if col==2 else lo,vmax=cap if col==3 else limit if col==2 else hi,plotnonfinite=True)
                ax.set(title=label,aspect="equal",xlabel="Initial x (mm)",ylabel="Initial y (mm)")
                if mode=="FT":ax.set_xlim(96.6,173.6);ax.set_ylim(54,136)
                fig.colorbar(image,ax=ax,label="% (colour capped at P95)" if col==3 else "U2 / error (mm)",shrink=.8,extend="max" if col==3 else "neither")
            axes[row,0].text(0,1.13,f"{'Best' if row==0 else 'Worst'} ID{ids[sample]} · frame{frame+1}/20 · sample RMSE{scores[sample]:.3g}mm",transform=axes[row,0].transAxes,fontsize=10)
            details.append(dict(rank="best" if row==0 else "worst",sample_id=ids[sample],retained_frame=frame+1,component="U2",sample_rmse=scores[sample],percent_colour_cap=cap,tiny_truth_threshold=.01*display_scale))
        fig.suptitle(f"{mode} {tag}: best/worst by complete valid field RMSE\nU2 at largest true regional increment; percent-error grey = tiny/invalid truth, no metric filtering",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(Path(out)/f"{mode}-field-examples.png",dpi=170);plt.close(fig)
        # Direct mean-field comparison at the same displayed states; no fitting
        # to these validation specimens and no change to model predictions.
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for row,(sample,frame,loc) in enumerate(zip(picks,chosen,shapes)):
            channel=2*frame+1
            for col,(label,values) in enumerate((("Truth",y[sample,:,channel]),("TRAIN mean",mean[:,frame,1]),("Model",p[sample,:,channel]))):
                ax=axes[row,col];im=ax.scatter(loc[region,0],loc[region,1],c=values[region],s=17,cmap="viridis",vmin=common_lo,vmax=common_hi)
                ax.set(title=f"{label}: {'best' if row==0 else 'worst'} ID{ids[sample]}, frame{frame+1}",aspect="equal",xlabel="Initial x (mm)",ylabel="Initial y (mm)")
                if mode=="FT":ax.set_xlim(96.6,173.6);ax.set_ylim(54,136)
                fig.colorbar(im,ax=ax,label="U2 (mm)",shrink=.8)
        fig.suptitle(f"{mode}: does the model reproduce more than the average field?\nSame specimens/frames as the error gallery; mean fit on TRAIN only")
        fig.tight_layout();fig.savefig(Path(out)/f"{mode}-mean-field-examples.png",dpi=170);plt.close(fig)
        return details


def regional_mean_figures(table,out,variants=None):
    variants=list(dict.fromkeys(table.variant)) if variants is None else variants
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,mode in zip(axes,("UT","FT")):
        for index,(region,label) in enumerate((("STATUS onset endpoints ±1 frame","Fracture endpoints"),
            ("STATUS onset strut relative motion ±1 frame","Failing-strut motion"),("FT crack-tip box","Larger FT box"))):
            for x,variant in enumerate(variants):
                values=table[(table["mode"]==mode)&(table.variant==variant)&(table.region==region)].skill_percent
                if values.empty:continue
                ax.errorbar(x+(index-1)*.13,values.mean(),yerr=values.std() if len(values)>1 else None,
                    marker="o",color=plt.get_cmap("tab10")(index),capsize=3,label=label if x==0 else None)
        ax.axhline(0,color="black",lw=.8);ax.set(title=mode,ylabel="RMSE reduction vs TRAIN mean (%) — higher is better",xticks=range(len(variants)),xticklabels=variants);ax.tick_params(axis="x",rotation=30,labelsize=8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Confirmed-fracture performance beyond the mean field\nIdentical evaluation regions; means fit training specimens only")
    fig.tight_layout();fig.savefig(Path(out)/"fracture-mean.png",dpi=170);plt.close(fig)
