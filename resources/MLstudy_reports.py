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


def prediction_variation_section(entries,out,prefix="prediction-variation"):
    """Saved predictions only; return a dedicated, accuracy-independent section.

    Match IDs/truths for each output within a study. Average pointwise ratios
    of across-specimen standard deviations, excluding nearly constant truth
    locations only for this diagnostic. Never change archived collapse metrics.
    """
    rows=[];contracts={};out=Path(out)
    for entry in entries:
        path=Path(entry["path"])
        with np.load(path/"results/predictions.npz",allow_pickle=False) as z:
            for mode in ("UT","FT"):
                for kind in ("field","curve"):
                    if kind not in entry.get("kinds",("field","curve")):continue
                    key=f"{mode}_val_{kind}_outputs"
                    if key not in z:continue
                    y=z[f"{mode}_val_{kind}_truth"].astype(float);p=z[key].astype(float)
                    valid=np.isfinite(y)
                    if kind=="field":
                        valid &= z[f"{mode}_val_field_mask"].astype(bool)
                        valid &= z[f"{mode}_node_mask"].astype(bool)[None,:,None]
                    if p.shape!=y.shape or np.any(valid & ~np.isfinite(p)):
                        raise ValueError("Invalid variation inputs: "+str(path))
                    digest=hashlib.sha256()
                    for value in (np.asarray(z["sample_ids"]).astype(str),valid):
                        digest.update(np.ascontiguousarray(value).tobytes())
                    contract=contracts.setdefault((mode,kind),(digest.hexdigest(),np.where(valid,y,0.).astype(np.float32)))
                    tolerance=max(float(np.max(np.abs(contract[1])))*8*np.finfo(np.float32).eps,1e-7)
                    if contract[0]!=digest.hexdigest() or not np.allclose(np.where(valid,y,0.),contract[1],rtol=0,atol=tolerance):
                        raise ValueError("Variation comparisons require aligned IDs, truths and masks: "+str(path))
                    # Residual-target reconstruction can differ by float32
                    # rounding; use one verified common truth for all spreads.
                    y=contract[1]
                    count=valid.sum(0)
                    def std(a):
                        mean=np.where(valid,a,0.).sum(0)/np.maximum(count,1)
                        return np.sqrt(np.where(valid,(a-mean)**2,0.).sum(0)/np.maximum(count,1))
                    sy,sp=std(y),std(p)
                    eligible=(count>=2)&(sy>max(float(sy.max())*1e-6,1e-12))
                    rows.append(dict(mode=mode,kind=kind,variant=entry["variant"],
                        seed=entry.get("seed",42),ratio=float(np.mean(sp[eligible]/sy[eligible])) if eligible.any() else np.nan,
                        eligible_locations=int(eligible.sum()),total_locations=int(sy.size),
                        mean_prediction_truth_bias=float(np.mean((p-y)[valid])),path=str(path)))
    if not rows:
        return "## Prediction variation and collapse\n\nNot applicable: no saved regression predictions in this report.\n\n"
    table=pd.DataFrame(rows);table.to_csv(out/f"{prefix}.csv",index=False)
    kinds=[k for k in ("field","curve") if (table.kind==k).any()]
    fig,axes=plt.subplots(len(kinds),2,figsize=(15,5*len(kinds)),squeeze=False)
    for row,kind in enumerate(kinds):
        for ax,mode in zip(axes[row],("UT","FT")):
            part=table[(table.kind==kind)&(table["mode"]==mode)]
            if part.empty:ax.set_axis_off();continue
            for index,(variant,group) in enumerate(part.groupby("variant",sort=False)):
                values=group.ratio.to_numpy();ax.scatter(index+np.linspace(-.09,.09,len(values)),values,s=36,color="#477a9b")
                ax.plot([index-.2,index+.2],[np.nanmean(values)]*2,color="#215b64",lw=3)
            names=list(part.variant.drop_duplicates())
            ax.axhline(1,color="black",linestyle="--",label="Equal specimen spread")
            ax.set(title=f"{mode} {kind}",xticks=range(len(names)),xticklabels=names,
                ylabel="Mean predicted / true specimen standard deviation")
            ax.tick_params(axis="x",rotation=35,labelsize=10);ax.tick_params(axis="y",labelsize=11)
            ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.2);ax.legend(fontsize=10)
    fig.suptitle("Prediction variation: zero means constant across specimens; one does not establish accuracy",fontsize=16)
    fig.tight_layout();fig.savefig(out/f"{prefix}.png",dpi=170);plt.close(fig)
    return ("## Prediction variation and collapse\n\n"
        "This measures specimen-to-specimen output spread separately from TRAIN-mean accuracy. "
        "Zero indicates the same output for every specimen; values below one indicate compressed variation. "
        "A value near one can still describe inaccurate predictions, misplaced fracture paths or incorrect specimen ordering. "
        "Dots are seeds; short bars show their mean, not a confidence interval. "
        "Nearly constant truth locations are excluded only from this diagnostic (relative1e-6/absolute1e-12); "
        "all accuracy metrics keep their original masks. Unknown field entries remain masked. "
        "STATUS-only models have no displacement output and are evaluated separately by precision–recall.\n\n"
        +figure(out,f"{prefix}.png","Across-specimen predicted versus true variation")
        +f"[Values, eligible locations and provenance]({prefix}.csv).\n\n")


def insert_variation_section(report,entries):
    """Place the diagnostic after the global TRAIN-mean block; avoid repeats."""
    import re
    report=Path(report);text=report.read_text(encoding="utf-8")
    text=re.sub(r"## Prediction variation and collapse\n.*?(?=\n## |\Z)","",text,flags=re.S)
    # Superseded inline spread plots are replaced by the dedicated common view.
    text=re.sub(r"!\[[^\]]*\]\(<[^>]*(?:/diversity|/output-variation)\.png>\)\s*","",text)
    headings=list(re.finditer(r"^## .+$",text,re.M))
    mean=next((i for i,h in enumerate(headings) if "global" in h.group().lower() and "mean" in h.group().lower()),None)
    if mean is None:raise ValueError("Report needs a global TRAIN-mean section before collapse diagnostics")
    end=headings[mean+1].start() if mean+1<len(headings) else len(text)
    section=prediction_variation_section(entries,report.parent)
    report.write_text(text[:end].rstrip()+"\n\n"+section+text[end:],encoding="utf-8")


def submission_note(out,filename,title):
    path=Path(out)/filename
    if not path.exists():return ""
    jobs=pd.read_csv(path,sep="\t",dtype=str)
    gate=jobs[jobs.role=="preflight"];production=jobs[jobs.role!="preflight"]
    if len(gate)!=1 or production.empty:raise ValueError("Malformed study submission receipt")
    verification=Path(out)/(filename.removesuffix("-jobs.tsv")+"-download-verification.json")
    if verification.is_file():
        return (f"\n## Completed follow-up receipt: {title}\n\n{len(production)} production fits; "
            f"execution gate{gate.iloc[0].job_id}, source `{gate.iloc[0].source_revision}`. "
            f"[Jobs/dependencies]({filename}) · [Verified downloads]({verification.name}). "
            "The completed-evidence section above owns the findings; this is not a fresh scheduler check.\n")
    return (f"\n## Submitted follow-up: {title}\n\n{len(production)} production fits, execution gate "
            f"{gate.iloc[0].job_id}, immutable source `{gate.iloc[0].source_revision}`. "
            "These are separate follow-ups, not additional completed results in the figures above. "
            f"[Exact jobs and dependencies]({filename}); PROJECT_STATUS owns the latest execution check.\n")


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


def verify_study_inventory(root,inventory,expected_runs):
    """Verify explicit archive-relative multi-layout inventories without a transfer."""
    root=Path(root).resolve();record=json.loads(Path(inventory).read_text())
    runs=record["runs"];seen=set();names=set();size=0
    if len(runs)!=expected_runs:raise ValueError("Unexpected downloaded study run count")
    for run in runs:
        name=run["path"];folder=(root/name).resolve()
        if name in names or not folder.is_relative_to(root):raise ValueError("Invalid/duplicate archive run")
        names.add(name)
        meta=json.loads((folder/"run_metadata.json").read_text())
        if meta!=run["metadata"] or meta.get("status")!="complete" or meta.get("checkpoint_reload")!="passed":
            raise ValueError("Incomplete or changed run metadata")
        for row in run["files"]:
            path=(root/row["path"]).resolve()
            if row["path"] in seen or not path.is_relative_to(folder):raise ValueError("Invalid inventory file")
            seen.add(row["path"]);h=hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b""):h.update(chunk)
            if path.stat().st_size!=row["size"] or h.hexdigest()!=row["sha256"]:raise ValueError("Downloaded checksum differs: "+row["path"])
            size+=row["size"]
    return dict(runs=len(runs),files=len(seen),bytes=size,sha256="all matched")


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
        if meta.get("mode")!=mode or meta.get("variant")!=entry.get("source_variant",variant):raise ValueError("Run identity disagrees with study entry")
        if "seed" in entry and meta.get("run_config",{}).get("seed")!=entry["seed"]:raise ValueError("Training seed differs from study entry")
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
            for row in measured:row.update(mode=mode,variant=variant,seed=entry.get("seed",42))
            regional.extend(measured)
            p,y,v=pred.reshape(len(ids),800,20,2),truth.reshape(len(ids),800,20,2),valid.reshape(len(ids),800,20,2)
            for frame in range(20):
                keep=v[:,:,frame];err=(p[:,:,frame].astype(float)-y[:,:,frame])[keep]
                avg=np.broadcast_to(mean[:,frame],y[:,:,frame].shape)
                base=(avg-y[:,:,frame])[keep]
                rmse=np.sqrt(np.mean(err**2));den=np.sqrt(np.mean(base**2))
                frames.append(dict(mode=mode,variant=variant,seed=entry.get("seed",42),frame=frame+1,rmse=rmse,skill_percent=100*(1-rmse/den),valid_entries=int(keep.sum())))
        history=pd.read_csv(path/"loss_history.csv")
        cfg=json.loads((path/"model.json").read_text())
        rows.append(dict(mode=mode,variant=variant,seed=entry.get("seed",42),label=entry.get("label",variant),rmse=diag["rmse"],
            baseline_rmse=diag["mean_field_baseline_rmse"],skill_percent=100*diag["skill_vs_mean_field_rmse"],
            collapse_ratio=diag["collapse_ratio"],hours=meta["training_seconds"]/3600,epochs=len(history),
            best_epoch=cfg.get("best_epoch",cfg.get("training",{}).get("best_epoch")),job=meta.get("slurm_job_id"),
            revision=meta.get("source_revision"),path=str(path),parameters=meta.get("active_parameters"),gpu=meta.get("gpu_name"),**counts))
        print("FIELD_REVIEW",mode,variant,flush=True)
    table=pd.DataFrame(rows);regions=pd.DataFrame(regional)
    controls=table[table.variant=="control"].set_index(["mode","seed"]).rmse
    if not controls.index.is_unique:raise ValueError("Require one field control per task/seed")
    table["change_percent"]=[100*(r.rmse/controls.loc[r.mode,r.seed]-1) for r in table.itertuples()]
    base=regions[regions.variant=="control"].set_index(["mode","seed","region"]).rmse
    regions["change_percent"]=[100*(r.rmse/base.loc[r.mode,r.seed,r.region]-1) for r in regions.itertuples()]
    return table,regions,pd.DataFrame(frames)


def write_study_report(path,text):
    """Preserve completed follow-up sections when the initial screen is rebuilt."""
    import re
    path=Path(path);previous=path.read_text(encoding="utf-8") if path.exists() else ""
    blocks=re.findall(r"<!-- completed-followup:[\w-]+ -->.*?<!-- end-completed-followup -->",previous,re.S)
    if blocks:
        title_end=text.find("\n")+1
        text=text[:title_end]+"\n"+"\n\n".join(blocks)+"\n\n"+text[title_end:].lstrip()
    path.write_text(text.rstrip()+"\n",encoding="utf-8")


def replace_study_followup(path,name,text):
    """One canonical report; newest completed evidence follows its title."""
    import re
    if not re.fullmatch(r"[\w-]+",name):raise ValueError("Invalid follow-up identity")
    path=Path(path);previous=path.read_text(encoding="utf-8")
    pattern=rf"<!-- completed-followup:{name} -->.*?<!-- end-completed-followup -->\s*"
    previous=re.sub(pattern,"",previous,flags=re.S)
    title_end=previous.find("\n")+1
    block=f"\n<!-- completed-followup:{name} -->\n{text.rstrip()}\n<!-- end-completed-followup -->\n\n"
    path.write_text(previous[:title_end]+block+previous[title_end:].lstrip(),encoding="utf-8")


def repeated_field_figures(table,regions,out,prefix):
    """Matched-seed changes and TRAIN references; one-seed points have no CI."""
    modes=list(dict.fromkeys(table["mode"]))
    for filename,values,column,regional in (("global-control",table,"change_percent",False),
        ("global-mean",table,"skill_percent",False),("fracture-control",regions,"change_percent",True),
        ("fracture-mean",regions,"skill_percent",True),("runtime",table,"hours",False)):
        fig,axes=plt.subplots(1,len(modes),figsize=(8*len(modes),5.8),squeeze=False)
        for ax,mode in zip(axes[0],modes):
            part=values[values["mode"]==mode]
            selected=list(part.variant.drop_duplicates())
            groups=("STATUS onset endpoints ±1 frame","STATUS onset strut relative motion ±1 frame","FT crack-tip box") if regional else ("",)
            for j,region in enumerate(groups):
                sub=part[part.region==region] if regional else part
                if sub.empty:continue
                for index,variant in enumerate(selected):
                    y=sub[sub.variant==variant][column].dropna().to_numpy();x=index+(j-(len(groups)-1)/2)*.16
                    if not len(y):continue
                    color=plt.get_cmap("tab10")(j)
                    ax.errorbar(x,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt="o",capsize=4,color=color,
                        label=region if regional and index==0 else None)
                    ax.scatter(x+np.linspace(-.035,.035,len(y)),y,s=16,color=color,alpha=.7)
                    if not regional:
                        ax.annotate(f"{y.mean():.2f}",(x,y.mean()),xytext=(3,8),textcoords="offset points",fontsize=10)
            ylabel="Training hours (queue excluded)" if column=="hours" else "RMSE change vs matched control (%) — lower is better" if column=="change_percent" else "RMSE reduction vs TRAIN mean (%) — higher is better"
            ax.axhline(0,color="black",lw=.8);ax.set(title=mode,ylabel=ylabel,xticks=range(len(selected)),xticklabels=selected)
            ax.tick_params(axis="x",rotation=28,labelsize=11);ax.grid(axis="y",alpha=.2);ax.margins(y=.18)
            if column=="change_percent" and part[column].abs().max()>30:
                ax.set_yscale("symlog",linthresh=5);ax.set_ylabel(ylabel+"; symlog beyond ±5%")
            if regional:ax.legend(fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.18))
        fig.suptitle("Saved validation: large markers = means; small points = seeds; whiskers = seed SD, not confidence intervals",fontsize=13)
        fig.tight_layout();fig.savefig(Path(out)/f"{prefix}-{filename}.png",dpi=170);plt.close(fig)


def completed_field_review(entries,data_root,region_api,out,prefix,title,question,change,verdict):
    """Render a saved final-generator study; never train or select a default.

    Callers verify downloads and supply explicit controls, identities and prose.
    Shared physical metrics retain same-seed controls and TRAIN-only means.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fields,regions,frames=collect_fields(entries,Path(data_root),region_api)
    for name,table in (("metrics",fields),("regions",regions),("frames",frames)):
        table.to_csv(out/f"{prefix}-{name}.csv",index=False)
    repeated_field_figures(fields,regions,out,prefix)
    spread=prediction_variation_section(entries,out,prefix=prefix+"-variation").replace("## Prediction","### Prediction")
    serial_schematic(out,prefix+"-architecture.png",[
        ("Matched pre-study control",["Disorder + context","Independent field\nTransformer","20-frame U1/U2"],"Masked displacement MSE; fixed physical validation selection"),
        ("Isolated candidate",["Same disorder/context","Same field\nTransformer","20-frame U1/U2"],change)])
    # Fixed physical selection score answers accuracy; saved objective totals
    # answer optimization. Both are shown, without comparing unlike totals.
    modes=list(fields["mode"].drop_duplicates());selected={}
    for mode in modes:
        candidates=fields[fields["mode"]==mode].groupby("variant").change_percent.mean()
        selected[mode]=candidates.idxmin()
    for filename,columns in (("validation",("selection_score",)),("objectives",("val_loss",)),("train-validation",("train_loss","val_loss"))):
        fig,axes=plt.subplots(1,len(modes),figsize=(8*len(modes),5.8),squeeze=False)
        for ax,mode in zip(axes[0],modes):
            part=fields[fields["mode"]==mode];variants=list(part.variant.drop_duplicates())
            for row in part.itertuples():
                if len(columns)>1 and (row.variant not in ("control",selected[mode]) or row.seed!=42):continue
                h=pd.read_csv(Path(row.path)/"loss_history.csv").rename(columns={"val_selection_score":"selection_score"})
                color=plt.get_cmap("tab10")(variants.index(row.variant)%10)
                for column in columns:
                    values=h[column].where(h[column]>0)
                    ax.plot(h.epoch,values,"--" if column.startswith("train") else "-",alpha=.65 if len(part)>8 else .9,
                        color=color,label=f"{row.variant} s{row.seed} {column}")
                if len(columns)>1:ax.axvline(row.best_epoch,color=color,alpha=.3,linestyle=":")
            ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Physical MSE / TRAIN-mean MSE (log)" if filename=="validation" else "Saved optimization objective (log)")
            ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f"{prefix}-{filename}.png",dpi=170);plt.close(fig)
    examples=[];selection=[]
    for mode in modes:
        row=fields[(fields["mode"]==mode)&(fields.variant==selected[mode])&(fields.seed==42)].iloc[0]
        data=json.loads((Path(row.path)/"model_data.json").read_text())
        details=field_examples(row.path,mode,out,f"{row.variant}, seed42",data_root,
            region_api["training_mean"](Path(data_root),data,mode),prefix=prefix+"-")
        selection.append(dict(mode=mode,variant=row.variant,seed=42,path=row.path,examples=details))
        examples.extend([figure(out,f"{prefix}-{mode}-field-examples.png",f"{mode}: truth, prediction, signed error and percentage error"),
            figure(out,f"{prefix}-{mode}-mean-field-examples.png",f"{mode}: actual TRAIN-mean field across loading")])
    (out/f"{prefix}-examples.json").write_text(json.dumps(selection,indent=2))
    image=lambda suffix,caption:figure(out,f"{prefix}-{suffix}.png",caption)
    text=[f"## {title}\n\n{question}\n\n{verdict}\n\n",
        image("architecture","Pre-study control and isolated intervention; no winner automatically promoted"),
        "### Global performance versus the matched study control\n\nNegative RMSE changes are better. Large markers are means; small points are individual seeds; whiskers are seed standard deviation, not confidence intervals.\n\n",
        image("global-control","Global RMSE change with same-task/same-seed controls"),
        "### Global performance versus TRAIN mean\n\nSkill = 100 × (1 − model RMSE / TRAIN-mean RMSE), on the same validation observations and masks. Zero matches the average-field predictor. The mean uses TRAIN IDs only and reproduces the saved reference error.\n\n",
        image("global-mean","Field accuracy beyond the mean-field predictor"),spread,
        "### STATUS fracture regions versus control\n\nObserved intact→failed struts define onset ±1 retained-frame windows. Endpoint displacement and signed connected-pair relative motion are distinct metrics. The larger FT crack-tip box is shown separately. Unknown labels remain masked. True STATUS is used retrospectively for evaluation, never to supply a fracture path as an input. Counts and denominators are in the regional CSV.\n\n",
        image("fracture-control","Confirmed-event and broader crack-tip RMSE changes"),
        "### STATUS fracture regions versus TRAIN mean\n\nIdentical event masks evaluate the model and TRAIN-mean field.\n\n",image("fracture-mean","Accuracy beyond mean displacement at confirmed fracture events"),
        "### Best and worst examples\n\nDisplay each task's lowest mean-global-error arm, including its control, at seed42. This is illustrative selection, not promotion. Rank specimens by full valid field RMSE; show U2 at their largest true regional increment, whole UT body and FT crack-tip zoom. Shared colour scales apply to both ranks. Grey percentage pixels have |truth| below 1% of TRAIN-mean RMS; the P95 colour cap and tiny-denominator mask affect display only.\n\n",*examples,
        "### Training time\n\nFinal-generator training only, excluding queue, export, OOF banks and downstream curves. Different stopping epochs can dominate total cost. No inference-latency result is implied.\n\n",image("runtime","Measured training hours by task and seed"),
        "### Loss histories\n\nFirst: the fixed physical validation MSE/TRAIN-mean-MSE selection score, comparable across objectives. Second: saved total validation objectives, which diagnose optimization but do not rank unlike losses. Third: seed42 control and displayed candidate train/validation objectives; dotted lines mark chosen checkpoints. These are distinct diagnostics, not duplicate plots. Positive values use log axes.\n\n",
        image("validation","Comparable fixed validation error"),image("objectives","Saved validation training objectives"),image("train-validation","Selected training/validation histories"),
        f"### Scope and numerical evidence\n\nNo downstream curve training, new inference, locked-test evaluation or model promotion occurs here. [Global metrics/provenance]({prefix}-metrics.csv), [regional metrics/counts]({prefix}-regions.csv), [frame metrics]({prefix}-frames.csv), [example identities]({prefix}-examples.json).\n\n"]
    return fields,regions,"".join(text)


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


def serial_schematic(out,name,panels):
    """Show two explicitly described serial workflows without inferring sharing."""
    fig,axes=plt.subplots(1,2,figsize=(15,4.5))
    for ax,(title,nodes,note) in zip(axes,panels):
        ax.set(xlim=(0,10),ylim=(0,5));ax.axis("off");ax.set_title(title,fontsize=15)
        for index,label in enumerate(nodes):
            x=.2+index*3.3
            ax.add_patch(FancyBboxPatch((x,2),2.8,1.5,boxstyle="round,pad=.08",facecolor="#dcebe9",edgecolor="#456075"))
            ax.text(x+1.4,2.75,label,ha="center",va="center",fontsize=12)
            if index<2:ax.add_patch(FancyArrowPatch((x+2.8,2.75),(x+3.3,2.75),arrowstyle="-|>",mutation_scale=14))
        ax.text(5,.9,note,ha="center",va="center",fontsize=11)
    fig.tight_layout();fig.savefig(Path(out)/name,dpi=170);plt.close(fig)


def percent_error(pred,truth,valid,training_scale):
    """Pointwise absolute error; mask tiny denominators rather than fabricate ratios."""
    keep=np.asarray(valid,bool)&np.isfinite(truth)&(np.abs(truth)>=.01*max(float(training_scale),1e-12))
    values=np.full(np.shape(truth),np.nan,dtype=float)
    np.divide(100*np.abs(pred-truth),np.abs(truth),out=values,where=keep)
    return values


def training_curve_mean(data_root, descriptor, mode, truth, ids, x):
    """Reproduce the existing absolute full-curve target and TRAIN-only reference."""
    raw = pd.read_csv(Path(data_root)/f"MULTI-{mode}-disNodes-allOUT.csv", index_col=0)
    recorded_x = raw.iloc[0].drop("0").to_numpy(float)
    values = raw.iloc[1:].drop(columns="0"); values.index = values.index.map(str)
    if not values.index.is_unique or not np.allclose(recorded_x, x, rtol=0, atol=1e-7):
        raise ValueError("Curve IDs/grid differ from delivered source")
    observed = values.loc[list(map(str, ids))].to_numpy(float)
    stats = descriptor["normalizers"]["curve"][mode]
    # Saved float32 affine inverse can leave a small residual even at true zero.
    tolerance = 4*np.finfo(np.float32).eps*(np.max(np.abs(stats["mean"]))+np.max(stats["scale"]))
    if not np.allclose(observed, truth, rtol=2e-5, atol=max(tolerance, 1e-6)):
        raise ValueError("CSV physical curves differ from saved validation truths")
    train_ids = list(map(str, descriptor["sample_ids"]["train"]))
    if set(train_ids) & set(map(str, ids)):
        raise ValueError("TRAIN mean overlaps reviewed validation IDs")
    return values.loc[train_ids].mean().to_numpy(float)


def measure_curve_regions(pred, truth, x, mean):
    """Fixed target-peak windows; curve regions are not spatial fracture labels."""
    pred, truth, x = np.asarray(pred, float), np.asarray(truth, float), np.asarray(x, float)
    if pred.shape != truth.shape or truth.ndim != 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Require aligned full curves on a strictly increasing grid")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite reviewed curve values")
    peak = truth.argmax(1); peak_x = x[peak, None]
    masks = {"Whole curve": np.ones_like(truth, bool),
             "Peak window ±5% span": np.abs(x[None]-peak_x) <= .05*np.ptp(x),
             "Pre-peak": x[None] < peak_x, "Post-peak to recorded end": x[None] > peak_x}
    rows = []
    for region, keep in masks.items():
        keep = keep & np.isfinite(truth) & np.isfinite(pred)
        error = np.sqrt(np.mean((pred[keep]-truth[keep])**2))
        baseline = np.sqrt(np.mean((np.broadcast_to(mean, truth.shape)[keep]-truth[keep])**2))
        target = np.sqrt(np.mean(truth[keep]**2))
        rows.append(dict(region=region, rmse=error, baseline_rmse=baseline,
            skill_percent=100*(1-error/baseline) if baseline > 1e-12 else np.nan,
            target_nrmse_percent=100*error/target if target > 1e-12 else np.nan,
            valid_points=int(keep.sum()), eligible_specimens=int(keep.any(1).sum())))
    return rows


def curve_examples(path,mode,out,tag,data_root,prefix=""):
    """One large four-column curve gallery, including the percentage diagnostic."""
    path=Path(path);data=json.loads((path/"model_data.json").read_text())
    with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
        y,p,x,ids=(a[f"{mode}_val_curve_truth"],a[f"{mode}_val_curve_outputs"],a[f"{mode}_curve_x_values"],a["sample_ids"])
        mean=training_curve_mean(data_root,data,mode,y,ids,x)
        score=np.sqrt(np.mean((p.astype(float)-y)**2,axis=1));picks=[score.argmin(),score.argmax()]
        unit=1000. if mode=="FT" else 1.;ylabel="Force (kN)" if mode=="FT" else "Stress (MPa)"
        limit=max(float(np.max(np.abs(p[picks]-y[picks])))/unit,1e-12)
        lo=min(float(y[picks].min()/unit),float(p[picks].min()/unit),float(mean.min()/unit),-limit)
        hi=max(float(y[picks].max()/unit),float(p[picks].max()/unit),float(mean.max()/unit))
        percent=[percent_error(p[i],y[i],np.isfinite(y[i]),np.sqrt(np.mean(mean**2))) for i in picks]
        fig,axes=plt.subplots(2,4,figsize=(21,10.5))
        for row,index in enumerate(picks):
            for col,ax in enumerate(axes[row]):
                if col in (0,1):ax.plot(x,y[index]/unit,color="black",label="Truth")
                if col==1:
                    ax.plot(x,p[index]/unit,"--",color="#367da0",label="Prediction")
                    ax.plot(x,mean/unit,":",color="#a77448",label="TRAIN mean")
                    ax.bar(x,(p[index]-y[index])/unit,width=np.median(np.diff(x))*.95,alpha=.18,color="#ad6158",label="Signed error")
                if col in (0,1):ax.set_ylim(lo*1.05,hi*1.05)
                if col==2:
                    ax.bar(x,(p[index]-y[index])/unit,width=np.median(np.diff(x))*.95,alpha=.3,label="Signed error")
                    ax.set_ylim(-limit*1.05,limit*1.05);ax.axhline(0,color="black",lw=.8)
                if col==3:ax.plot(x,percent[row],color="#ad6158",label="100 × |error| / |truth|")
                ax.set(title=("Truth","Prediction + reference","Signed error","Percent error")[col],
                    xlabel="Strain" if mode=="UT" else "Load-line displacement (mm)",ylabel="Percent" if col==3 else ylabel)
                ax.title.set_fontsize(17);ax.tick_params(labelsize=13);ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
                ax.set_xlim(float(x[0]),float(x[-1]));ax.legend(fontsize=11);ax.grid(alpha=.15)
            axes[row,0].text(0,1.13,f"{'Best' if row==0 else 'Worst'} ID {ids[index]}",transform=axes[row,0].transAxes,fontsize=15)
        fig.suptitle(f"{mode} {tag}: best/worst by full-curve RMSE\nPercent gaps = tiny/zero truth below 1% TRAIN-mean RMS; aggregate metrics unfiltered",fontsize=18)
        fig.tight_layout(rect=(0,0,1,.93),h_pad=3)
        fig.savefig(Path(out)/f"{prefix}{mode}-curve-examples.png",dpi=170);plt.close(fig)


def field_examples(path,mode,out,tag,data_root,mean,prefix=""):
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
        fig=plt.figure(figsize=(22,12),layout="constrained")
        rowfigs=fig.subfigures(2,1,hspace=.04)
        axes=np.asarray([rowfig.subplots(1,4) for rowfig in rowfigs]);details=[]
        for row,(sample,frame,loc) in enumerate(zip(picks,chosen,shapes)):
            channel=2*frame+1;v=valid[sample,:,channel]&region
            truth,pred=y[sample,:,channel],p[sample,:,channel]
            # RMS of the training mean is a documented display-only scale.
            display_scale=float(np.sqrt(np.mean(mean[region,frame,1]**2)))
            ape=percent_error(pred,truth,v,display_scale)
            lo,hi,limit,cap=common_lo,common_hi,common_error,common_percent
            for col,(label,values) in enumerate((("Truth",truth),("Prediction",pred),("Signed error",pred-truth),("Absolute % error",ape))):
                ax=axes[row,col];cmap=plt.get_cmap("magma" if col==3 else "coolwarm" if col==2 else "viridis").copy();cmap.set_bad("#b5b5b5")
                view=np.ma.masked_invalid(values[region]);image=ax.scatter(loc[region,0],loc[region,1],c=view,s=27,cmap=cmap,
                    vmin=0 if col==3 else -limit if col==2 else lo,vmax=cap if col==3 else limit if col==2 else hi,plotnonfinite=True)
                ax.set(title=label,aspect="equal",xlabel="Initial x (mm)",ylabel="Initial y (mm)")
                ax.title.set_fontsize(17);ax.xaxis.label.set_size(15);ax.yaxis.label.set_size(15);ax.tick_params(labelsize=13)
                if mode=="FT":ax.set_xlim(96.6,173.6);ax.set_ylim(54,136)
                bar=rowfigs[row].colorbar(image,ax=ax,label="% error (P95 cap)" if col==3 else "Error (mm)" if col==2 else "U2 (mm)",shrink=.85,fraction=.05,pad=.03,extend="max" if col==3 else "neither")
                bar.ax.tick_params(labelsize=12);bar.ax.yaxis.label.set_size(14)
            rowfigs[row].suptitle(f"{'Best' if row==0 else 'Worst'} specimen {ids[sample]} · frame {frame+1}/20 · full-field RMSE {scores[sample]:.3g} mm",fontsize=17)
            details.append(dict(rank="best" if row==0 else "worst",sample_id=ids[sample],retained_frame=frame+1,component="U2",sample_rmse=scores[sample],percent_colour_cap=cap,tiny_truth_threshold=.01*display_scale))
        fig.suptitle(f"{mode} {tag}: best/worst by complete valid field RMSE\nU2 at largest true regional increment; percent-error grey = tiny/invalid truth, no metric filtering",fontsize=18)
        fig.savefig(Path(out)/f"{prefix}{mode}-field-examples.png",dpi=170);plt.close(fig)
        mean_field_evolution(mean,xy,present,mode,out,prefix=prefix)
        return details


def mean_field_evolution(mean,coords,present,mode,out,prefix=""):
    """Display the TRAIN reference alone, without duplicating model galleries."""
    mean,coords,present=np.asarray(mean),np.asarray(coords),np.asarray(present,bool)
    region=present.copy()
    if mode=="FT":region&=(coords[:,0]>=96.6)&(coords[:,0]<=173.6)&(coords[:,1]>=54)&(coords[:,1]<=136)
    frames=[max(0,int(np.ceil(q*mean.shape[1]))-1) for q in (.25,.5,.75,1)]
    values=mean[region][:,frames,1];lo,hi=float(values.min()),float(values.max())
    fig,axes=plt.subplots(1,4,figsize=(20,5.5),layout="constrained")
    for ax,frame in zip(axes,frames):
        image=ax.scatter(coords[region,0],coords[region,1],c=mean[region,frame,1],cmap="viridis",s=24,vmin=lo,vmax=hi)
        ax.set(title=f"TRAIN mean · frame {frame+1}",aspect="equal",xlabel="Reference x (mm)",ylabel="Reference y (mm)")
        ax.title.set_fontsize(15);ax.tick_params(labelsize=12);ax.xaxis.label.set_size(13);ax.yaxis.label.set_size(13)
        if mode=="FT":ax.set_xlim(96.6,173.6);ax.set_ylim(54,136)
    bar=fig.colorbar(image,ax=axes,label="Mean U2 (mm)",shrink=.8,pad=.02);bar.ax.tick_params(labelsize=12)
    fig.suptitle(f"{mode}: TRAIN-only mean-field evolution\nCanonical reference positions; no individual specimen or model prediction",fontsize=17)
    fig.savefig(Path(out)/f"{prefix}{mode}-mean-field-examples.png",dpi=170);plt.close(fig)


def pipeline_regions(runs, control, data_root, region_api, out):
    """Retrospective field regions for completed serial pipeline comparisons."""
    rows=[];cache={}
    for run in runs:
        path=Path(run["path"]);data=json.loads((path/"model_data.json").read_text())
        metrics=json.loads((path/"results/metrics.json").read_text())
        meta=json.loads((path/"run_metadata.json").read_text())
        if metrics.get("evaluation_split")!="val" or meta.get("status")!="complete":
            raise ValueError("Require completed validation pipeline: "+str(path))
        with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
            ids=list(map(str,a["sample_ids"]))
            if ids!=list(map(str,data["sample_ids"]["val"])) or len(ids)!=len(set(ids)):
                raise ValueError("Pipeline specimen IDs disagree")
            for mode in ("UT","FT"):
                fingerprint=hashlib.sha256(json.dumps(dict(ids=data["sample_ids"],
                    frames=data["metadata"]["field_frame_values"][mode],
                    coords=data["metadata"]["canonical_coords"],presence=data["node_masks"][mode],
                    mapping=data["metadata"]["field_to_canonical"][mode]),sort_keys=True).encode()).hexdigest()
                key=mode,fingerprint
                if key not in cache:
                    nodes,event,edges,counts=region_api["region_masks"](Path(data_root)/f"MULTI-{mode}-disNodes-allD.npz",data,mode,ids)
                    mean=region_api["training_mean"](Path(data_root),data,mode)
                    cache[key]=nodes,event,edges,counts,mean
                nodes,event,edges,counts,mean=cache[key]
                y,p=a[f"{mode}_val_field_truth"],a[f"{mode}_val_field_outputs"]
                valid=a[f"{mode}_val_field_mask"].astype(bool)&a[f"{mode}_node_mask"][None,:,None]&np.isfinite(y)
                if not np.isfinite(p[valid]).all():raise ValueError("Nonfinite pipeline predictions")
                measured=region_api["measure_regions"](p,y,valid,nodes,event,edges,mean,mode,a["canonical_coords"])
                saved=metrics["diagnostics"]["field"][mode]
                if saved.get("mean_field_baseline_source")!="train_mean_field":raise ValueError("Unverified pipeline mean")
                if not np.isclose(measured[0]["rmse"],saved["rmse"],rtol=3e-6) or not np.isclose(measured[0]["mean_baseline_rmse"],saved["mean_field_baseline_rmse"],rtol=3e-6):
                    raise ValueError("Pipeline physical/mean field error differs")
                rows.extend(dict(variant=run["variant"],seed=run["seed"],mode=mode,**r,**counts) for r in measured)
        print("PIPELINE_REGION",run["variant"],run["seed"],flush=True)
    table=pd.DataFrame(rows);base=table[table.variant==control].set_index(["seed","mode","region"]).rmse
    table["change_percent"]=[100*(r.rmse/base.loc[r.seed,r.mode,r.region]-1) for r in table.itertuples()]
    table.to_csv(Path(out)/"confirmed-fracture-regions.csv",index=False)
    region_api["region_figures"](table,Path(out));regional_mean_figures(table,Path(out))
    pipeline_curve_regions(runs,control,data_root,out)
    return table


def pipeline_curve_regions(runs,control,data_root,out):
    """Peak/post-peak curve regions, distinct from spatial STATUS evaluation."""
    curve_rows=[];cache={}
    for run in runs:
        path=Path(run["path"]);data=json.loads((path/"model_data.json").read_text())
        metrics=json.loads((path/"results/metrics.json").read_text())
        if metrics.get("evaluation_split")!="val":raise ValueError("Require validation-only curves")
        with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
            ids=list(map(str,a["sample_ids"]))
            if ids!=list(map(str,data["sample_ids"]["val"])):raise ValueError("Curve IDs differ")
            for mode in ("UT","FT"):
                y,p,x=a[f"{mode}_val_curve_truth"],a[f"{mode}_val_curve_outputs"],a[f"{mode}_curve_x_values"]
                key=mode,tuple(map(str,data["sample_ids"]["train"])),tuple(ids),tuple(x)
                if key not in cache:cache[key]=training_curve_mean(data_root,data,mode,y,ids,x)
                mean=cache[key];measured=measure_curve_regions(p,y,x,mean)
                saved=metrics["diagnostics"]["curve"][mode]
                if not np.isclose(measured[0]["rmse"],saved["rmse"],rtol=3e-6) or not np.isclose(measured[0]["baseline_rmse"],saved["mean_curve_baseline_rmse"],rtol=3e-6):
                    raise ValueError("Curve global/baseline error does not reproduce")
                curve_rows.extend(dict(variant=run["variant"],seed=run["seed"],mode=mode,**r) for r in measured)
    curves=pd.DataFrame(curve_rows);base=curves[curves.variant==control].set_index(["seed","mode","region"]).rmse
    curves["change_percent"]=[100*(r.rmse/base.loc[r.seed,r.mode,r.region]-1) for r in curves.itertuples()]
    curves.to_csv(Path(out)/"curve-region-metrics.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(15,9));variants=list(dict.fromkeys(curves.variant))
    for row,column in enumerate(("change_percent","skill_percent")):
        for col,mode in enumerate(("UT","FT")):
            ax=axes[row,col]
            for j,region in enumerate(("Peak window ±5% span","Post-peak to recorded end")):
                means=[];sds=[]
                for variant in variants:
                    values=curves[(curves.variant==variant)&(curves["mode"]==mode)&(curves.region==region)][column]
                    means.append(values.mean());sds.append(values.std() if len(values)>1 else 0)
                ax.errorbar(np.arange(len(variants))+(j-.5)*.13,means,yerr=sds,marker="o",capsize=3,label=region)
            ax.set(title=f"{mode}: {'vs control, lower better' if row==0 else 'vs TRAIN mean, higher better'}",
                   ylabel="Percent",xticks=range(len(variants)),xticklabels=variants)
            ax.tick_params(axis="x",rotation=25,labelsize=9);ax.axhline(0,color="black",lw=.8);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Mechanically relevant curve windows, not spatial STATUS regions; whiskers = seed SD")
    fig.tight_layout();fig.savefig(Path(out)/"curve-regions.png",dpi=170);plt.close(fig)
    return curves


def pipeline_global_figures(table,out):
    """Four fixed outputs, paired controls and TRAIN-mean reference across seeds."""
    variants=list(dict.fromkeys(table.variant))
    for name,column,title in (("global-control","change_percent","Global RMSE change versus the study control — lower is better"),
                               ("global-mean","skill_percent","RMSE reduction versus TRAIN mean — higher is better"),
                               ("output-variation","collapse_ratio","Predicted / true specimen variation — zero is constant output, one matches spread")):
        fig,axes=plt.subplots(2,2,figsize=(15,9))
        for row,kind in enumerate(("field","curve")):
            for col,mode in enumerate(("UT","FT")):
                ax=axes[row,col]
                values=[table[(table.variant==v)&(table.kind==kind)&(table["mode"]==mode)][column] for v in variants]
                heights=[v.mean() for v in values];sd=[v.std() if len(v)>1 else 0 for v in values]
                bars=ax.bar(range(len(variants)),heights,yerr=sd,capsize=3,color="#477f9f")
                ax.bar_label(bars,fmt="%.2f",fontsize=9,padding=4)
                ax.set(title=f"{mode} {kind}",ylabel="Variation ratio" if column=="collapse_ratio" else "Percent",xticks=range(len(variants)),xticklabels=variants)
                if column=="collapse_ratio":ax.axhline(1,color="black",linestyle=":",lw=1)
                ax.tick_params(axis="x",rotation=25,labelsize=9);ax.axhline(0,color="black",lw=.8);ax.grid(axis="y",alpha=.2);ax.margins(y=.2)
        fig.suptitle(title+"\nWhiskers = seed SD, not confidence intervals")
        fig.tight_layout();fig.savefig(Path(out)/f"{name}.png",dpi=170);plt.close(fig)


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
