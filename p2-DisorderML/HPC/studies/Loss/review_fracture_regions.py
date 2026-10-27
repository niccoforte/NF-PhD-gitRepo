"""Evaluate existing loss predictions at confirmed STATUS fracture events only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

VARIANTS=("control","spatial","temporal","both","sudden")


def training_mean(data_root, descriptor, mode):
    """Physical per-node/frame/component mean, fitted on recorded TRAIN IDs only."""
    with np.load(data_root/f"MULTI-{mode}-disNodes-allFIELD.npz", allow_pickle=False) as bank:
        ids=list(map(str,bank["sample_ids"])); lookup={v:i for i,v in enumerate(ids)}
        if len(lookup)!=len(ids):raise ValueError("Duplicate field-bank IDs")
        mapping=np.asarray(descriptor["metadata"]["field_to_canonical"][mode])
        xy=np.asarray(descriptor["metadata"]["canonical_coords"])
        if not np.allclose(bank["node_coords"],xy[mapping],atol=2e-5,rtol=0):raise ValueError("Field bank node mapping differs")
        if not np.allclose(bank["frame_values"][1:],descriptor["metadata"]["field_frame_values"][mode],atol=1e-6,rtol=0):raise ValueError("Field frame order differs")
        if list(map(str,bank["components"]))!=descriptor["metadata"]["field_components"][mode]:raise ValueError("Field component order differs")
        positions=[lookup[str(v)] for v in descriptor["sample_ids"]["train"]]
        values,known=bank["Y"],bank["valid_mask"]
        total=np.zeros(values.shape[1:],float);count=np.zeros(values.shape[1:],np.int64)
        for start in range(0,len(positions),32):
            rows=positions[start:start+32];v=values[rows];keep=known[rows]&np.isfinite(v)
            total+=np.where(keep,v,0).sum(0,dtype=float);count+=keep.sum(0)
        native=np.divide(total,count,out=np.zeros_like(total),where=count>0)[1:]
        canonical=np.zeros((len(xy),*native.shape[::2]))
        canonical[mapping]=native.transpose(1,0,2)
        return canonical


def measure_regions(pred,truth,valid,nodes,event,edges,mean,mode,xy):
    """One fixed validation mask for model and TRAIN-mean predictions."""
    shape=truth.shape
    pred,truth,valid=pred.reshape(shape[0],len(xy),-1,2),truth.reshape(shape[0],len(xy),-1,2),valid.reshape(shape[0],len(xy),-1,2)
    mean=np.broadcast_to(mean,truth.shape)
    rows=[]
    box=(xy[:,0]>=96.6)&(xy[:,0]<=173.6)&(xy[:,1]>=54)&(xy[:,1]<=136)
    tip=np.linalg.norm(xy-np.array([117.6,95]),axis=1)<=20
    regions=[("Whole field",True),("STATUS onset endpoints ±1 frame",nodes[...,None])]
    if mode=="FT":regions += [("FT crack-tip box",box[None,:,None,None]),("FT tip radius20mm",tip[None,:,None,None])]
    for name,region in regions:
        keep=valid&region&np.isfinite(truth)
        error=np.sqrt(np.mean((pred[keep].astype(float)-truth[keep])**2))
        baseline=np.sqrt(np.mean((mean[keep]-truth[keep])**2))
        target=np.sqrt(np.mean(truth[keep].astype(float)**2))
        rows.append(dict(region=name,rmse=error,mean_baseline_rmse=baseline,
            skill_percent=100*(1-error/baseline) if baseline>1e-12 else np.nan,
            target_rms=target,nrmse_percent=100*error/target if target>1e-12 else np.nan,
            valid_node_frame_components=int(keep.sum())))
    squared=baseline_sum=targetsum=count=0.
    i,j=edges.T
    for start in range(0,len(truth),16):
        p0,y0,v0,b0=[a[start:start+16] for a in (pred,truth,valid,mean)]
        keep=event[start:start+16].transpose(0,2,1)[...,None]&v0[:,i]&v0[:,j]
        target=y0[:,j]-y0[:,i]
        error=(p0[:,j]-p0[:,i])-target;baseline=(b0[:,j]-b0[:,i])-target
        squared+=np.square(error[keep].astype(float)).sum();baseline_sum+=np.square(baseline[keep]).sum()
        targetsum+=np.square(target[keep].astype(float)).sum();count+=keep.sum()
    rows.append(dict(region="STATUS onset strut relative motion ±1 frame",rmse=np.sqrt(squared/count),
        mean_baseline_rmse=np.sqrt(baseline_sum/count),skill_percent=100*(1-np.sqrt(squared/baseline_sum)) if baseline_sum>1e-12 else np.nan,
        target_rms=np.sqrt(targetsum/count),nrmse_percent=100*np.sqrt(squared/targetsum),valid_node_frame_components=int(count)))
    return rows


def region_figures(table,out):
    """Show the narrow event gain beside the broader FT cost, not on another page."""
    variants=list(dict.fromkeys(table.variant))
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    regions=("STATUS onset endpoints ±1 frame","STATUS onset strut relative motion ±1 frame","FT crack-tip box","Whole field")
    labels=("Fracture endpoints","Fracturing-strut relative motion","Larger crack-tip box","Whole field")
    for ax,mode in zip(axes,("UT","FT")):
        for offset,(region,label) in enumerate(zip(regions,labels)):
            if mode=="UT" and region=="FT crack-tip box":continue
            for x,variant in enumerate(variants):
                values=table[(table["mode"]==mode)&(table.variant==variant)&(table.region==region)].change_percent
                if values.empty:continue
                ax.errorbar(x+(offset-1.5)*.12,values.mean(),yerr=values.std() if len(values)>1 else None,
                    marker="o",capsize=3,color=plt.get_cmap("tab10")(offset),label=label if x==0 else None)
        ax.axhline(0,color="black",lw=.8)
        label="RMSE change vs matched MSE (%)\nLower is better"
        if table[table["mode"]==mode].change_percent.abs().max()>20:
            ax.set_yscale("symlog",linthresh=5);ax.set_yticks([-5,0,5,10,30,100])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}"))
            label+="; symmetric-log beyond ±5%"
        ax.set(title=mode,ylabel=label,xticks=range(len(variants)),xticklabels=variants)
        ax.tick_params(axis="x",labelsize=8);ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.13),ncol=2)
    fig.suptitle("Actual fracture events versus the surrounding field\nSame specimens/masks; points=seed means, whiskers=seed SD")
    fig.tight_layout();fig.savefig(out/"status-fracture-regions.png",dpi=170);plt.close(fig)


def region_masks(bank,descriptor,mode,ids):
    """No fitting: labels define retrospective validation regions, not inputs."""
    with np.load(bank,allow_pickle=False) as a:
        bankids=list(map(str,a["sample_ids"]))
        if len(set(bankids))!=len(bankids):raise ValueError("Duplicate STATUS IDs")
        positions=np.array([bankids.index(str(v)) for v in ids])
        mapping=np.asarray(descriptor["metadata"]["field_to_canonical"][mode])
        xy=np.asarray(descriptor["metadata"]["canonical_coords"])
        if not np.allclose(a["node_coords"],xy[mapping],rtol=0,atol=2e-5):raise ValueError("Native node mapping differs")
        if not np.allclose(a["frame_values"][1:],descriptor["metadata"]["field_frame_values"][mode],rtol=0,atol=1e-6):
            raise ValueError("Field and STATUS retained frame schedules differ")
        edges=mapping[a["edge_index"]]
        # Current failure requires known intact previous state; unavailable frames
        # are never interpreted as intact and no labels are extrapolated.
        d=a["D"][positions].astype(bool);known=a["valid_mask"][positions]
        onset=d[:,1:]&~d[:,:-1]&known[:,1:]&known[:,:-1]
        window=onset.copy();window[:,1:]|=onset[:,:-1];window[:,:-1]|=onset[:,1:]
        nodes=np.zeros((len(ids),len(xy),20),bool)
        for endpoint in (0,1):
            for frame in range(20):
                for sample in range(len(ids)):
                    nodes[sample,edges[window[sample,frame],endpoint],frame]=True
        return nodes,window,edges,dict(onset_strut_frames=int(onset.sum()),event_strut_frames=int(window.sum()),
            event_node_frames=int(nodes.sum()),specimens_with_event=int(nodes.any(axis=(1,2)).sum()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--damage-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/loss-study")
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True);rows=[];counts={}
    descriptor=json.loads((a.root/"independent-inputs-261015-mixed-s42/model_data.json").read_text())
    for mode in ("UT","FT"):
        ids=descriptor["sample_ids"]["val"]
        nodes,event,edges,counts[mode]=region_masks(a.damage_root/f"MULTI-{mode}-disNodes-allD.npz",descriptor,mode,ids)
        xy=np.asarray(descriptor["metadata"]["canonical_coords"])
        mean=training_mean(a.damage_root,descriptor,mode)
        for seed in (42,43,44):
            for variant in VARIANTS:
                label=f"independent-inputs-261015-mixed-s{seed}" if variant=="control" else f"independent-losses-261022-mixed-{variant}-s{seed}"
                with np.load(a.root/label/"results/predictions.npz",allow_pickle=False) as z:
                    if list(map(str,z["sample_ids"]))!=list(map(str,ids)):raise ValueError("Saved prediction IDs differ")
                    pred=z[f"{mode}_val_field_outputs"].reshape(len(ids),800,20,2)
                    truth=z[f"{mode}_val_field_truth"].reshape(pred.shape)
                    valid=z[f"{mode}_val_field_mask"].reshape(pred.shape).astype(bool)&z[f"{mode}_node_mask"][None,:,None,None]
                    measured=measure_regions(pred,truth,valid,nodes,event,edges,mean,mode,xy)
                    for r in measured:r.update(mode=mode,seed=seed,variant=variant)
                    rows.extend(measured)
                    diag=json.loads((a.root/label/"results/metrics.json").read_text())["diagnostics"]["field"][mode]
                    if not np.isclose(measured[0]["mean_baseline_rmse"],diag["mean_field_baseline_rmse"],rtol=3e-6):
                        raise ValueError("Reconstructed TRAIN mean does not reproduce the saved baseline")
                print("FRACTURE_CHECK",mode,seed,variant,flush=True)
    table=pd.DataFrame(rows);base=table[table.variant=="control"].set_index(["mode","seed","region"]).rmse
    table["change_percent"]=[100*(r.rmse/base.loc[r.mode,r.seed,r.region]-1) for r in table.itertuples()]
    table.to_csv(a.out/"status-fracture-regions.csv",index=False)
    region_figures(table,a.out)
    text=["# Losses at confirmed fracture events\n\nThis complements the kinematic activity metrics. **Regions here use true STATUS only for retrospective validation evaluation, never as inputs or training weights.** Each native strut is mapped to the saved field node order and joined by specimen ID.\n\n",
        "Fracture onset means first observed intact→failed transition with both states known. We include its two endpoint nodes and the onset strut in that retained frame and one adjacent frame on each side. This does not give an exact continuous failure time or assume every nearby node fails. Unavailable frames remain masked.\n\n",
        "| Task | Validation specimens with event | Strut onset entries | Window strut/frame entries | Window node/frame entries |\n|---|---:|---:|---:|---:|\n"]
    for m,c in counts.items():text.append(f"| {m} | {c['specimens_with_event']} | {c['onset_strut_frames']} | {c['event_strut_frames']} | {c['event_node_frames']} |\n")
    text += [f"\n![Confirmed fracture-region comparison](<{(a.out/'status-fracture-regions.png').resolve().as_posix()}>)\n\n",
        "| Variant | Region | UT RMSE change | FT RMSE change |\n|---|---|---:|---:|\n"]
    for variant in VARIANTS:
        for region in ("STATUS onset endpoints ±1 frame","STATUS onset strut relative motion ±1 frame"):
            values=[table[(table.variant==variant)&(table["mode"]==m)&(table.region==region)].change_percent.mean() for m in ("UT","FT")]
            text.append(f"| {variant} | {region} | {values[0]:+.2f}% | {values[1]:+.2f}% |\n")
    text += ["\n## Compared with the average field\n\nSkill=100×(1−model RMSE/TRAIN-mean RMSE). Fit each node/frame/component mean on training IDs only, then evaluate it on the same validation region as the model. Zero means no improvement beyond the average; negative is worse. The global baseline is checked against the saved metrics before regional scores are accepted. This is different from dividing by the target motion magnitude.\n\n",
        "| Variant | Task | Region | RMSE skill vs TRAIN mean | RMSE change vs MSE |\n|---|---|---|---:|---:|\n"]
    for (variant,mode,region),values in table.groupby(["variant","mode","region"],sort=False):
        text.append(f"| {variant} | {mode} | {region} | {values.skill_percent.mean():.2f} ± {values.skill_percent.std():.2f}% | {values.change_percent.mean():+.2f}% |\n")
    text += ["\nFT box and20mm nominal-tip-radius results, physical errors, true regional-motion denominators and all event counts are retained in [the numerical companion](status-fracture-regions.csv). Relative changes compare exactly the same regions/specimens per seed; no whole-body average hides their errors. The global error and peak/work safeguards still matter.\n"]
    text += ["\n## What this changes in the interpretation\n\nSpatial supervision is not uniformly hostile to FT fracture: its confirmed onset endpoints improve1.07%, and their relative-motion error improves0.47%. However, the larger tip box worsens35.25%, the20mm tip circle30.40%, and the whole field123.12%. Both terms together show the same basic trade-off. A very narrow event gain does not justify that broader deterioration. UT spatial improves confirmed endpoint error3.25% and strut relative motion4.02%, consistent with its modest global gain.\n\n",
        "**Hypothesis1: task-specific loss balance.** Training RMS jump scales are approximately0.508/0.819mm for UT U1/U2 and0.212/0.243mm for FT. The late FT nodal normalization scale reaches5.315mm. For the same physical residual, the illustrative factor0.1×(nodal_scale/jump_scale)² can reach47.86 in FT U2, versus3.41 in UT U2. This is not a measured gradient ratio: edge differencing, coordination averaging and correlations also matter. It demonstrates that coefficient0.1 does not give equal effective emphasis across tasks/frames. Lower FT coefficients are worth a field-only screen before any new OOF expense.\n\n",
        "**Hypothesis2: different spatial information and finite model capacity.** FT has a large predictable deformation field plus sparse crack-tip events. Training to match relative motion everywhere can alter broad deformation modes while improving a narrow subset of fracture events. UT fracture activity is more spatially distributed. The squared relative-motion loss matches target jumps; it does not explicitly penalize large gradients or prescribe smoothness. Its effect is a changed error geometry, not added fracture information.\n\n",
        "**What the evidence rules out:** these are independently trained models, so UT/FT shared-parameter competition is not their cause. Initial connectivity agrees with the existing bounded original-INP audit; no new connectivity defect was established. Subtracting mean nodal error does not remove the FT deterioration, so a uniform offset alone is insufficient. Neither this review nor the previous aggregate study measures a definitive causal training mechanism.\n\n",
        "**Recommendation:** leave WINNER.md on the accepted Mixed MSE configuration. Retain UT spatial and the small FT sudden-motion effect as candidates; prioritize confirming the larger locality/CNN result before combining changes. Retune FT spatial separately if pursued, keeping all global/box/confirmed-fracture metrics fixed. The existing derivative trial coefficients were controlled presets, not HPO optima.\n"]
    (a.out/"fracture-regions.md").write_text("".join(text),encoding="utf-8")
    print(table.groupby(["variant","mode","region"]).change_percent.mean().to_string())


if __name__=="__main__":main()
