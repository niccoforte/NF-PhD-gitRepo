"""Evaluate existing loss predictions at confirmed STATUS fracture events only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS=("control","spatial","temporal","both","sudden")


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
        box=(xy[:,0]>=96.6)&(xy[:,0]<=173.6)&(xy[:,1]>=54)&(xy[:,1]<=136)
        tip=np.linalg.norm(xy-np.array([117.6,95]),axis=1)<=20
        for seed in (42,43,44):
            for variant in VARIANTS:
                label=f"independent-inputs-261015-mixed-s{seed}" if variant=="control" else f"independent-losses-261022-mixed-{variant}-s{seed}"
                with np.load(a.root/label/"results/predictions.npz",allow_pickle=False) as z:
                    if list(map(str,z["sample_ids"]))!=list(map(str,ids)):raise ValueError("Saved prediction IDs differ")
                    pred=z[f"{mode}_val_field_outputs"].reshape(len(ids),800,20,2)
                    truth=z[f"{mode}_val_field_truth"].reshape(pred.shape)
                    valid=z[f"{mode}_val_field_mask"].reshape(pred.shape).astype(bool)&z[f"{mode}_node_mask"][None,:,None,None]
                    residual=(pred.astype(float)-truth)**2
                    for name,region in [("STATUS onset endpoints ±1 frame",nodes[...,None]),
                        *(([("FT crack-tip box",box[None,:,None,None]),("FT tip radius20mm",tip[None,:,None,None])]) if mode=="FT" else [])]:
                        keep=valid&region&np.isfinite(truth)
                        rmse=float(np.sqrt(residual[keep].mean()))
                        scale=float(np.sqrt(np.mean(truth[keep].astype(float)**2)))
                        rows.append(dict(mode=mode,seed=seed,variant=variant,region=name,rmse=rmse,
                            target_rms=scale,nrmse_percent=100*rmse/scale,valid_node_frame_components=int(keep.sum())))
                    squared=targetsum=count=0.
                    for start in range(0,len(ids),16):
                        i,j=edges.T
                        p0=pred[start:start+16];y0=truth[start:start+16];v0=valid[start:start+16]
                        keep=event[start:start+16].transpose(0,2,1)[...,None]&v0[:,i]&v0[:,j]
                        err=(p0[:,j]-p0[:,i])-(y0[:,j]-y0[:,i]);target=y0[:,j]-y0[:,i]
                        squared+=np.square(err[keep].astype(float)).sum();targetsum+=np.square(target[keep].astype(float)).sum();count+=keep.sum()
                    rows.append(dict(mode=mode,seed=seed,variant=variant,region="STATUS onset strut relative motion ±1 frame",
                        rmse=np.sqrt(squared/count),target_rms=np.sqrt(targetsum/count),nrmse_percent=100*np.sqrt(squared/targetsum),valid_node_frame_components=int(count)))
                print("FRACTURE_CHECK",mode,seed,variant,flush=True)
    table=pd.DataFrame(rows);base=table[table.variant=="control"].set_index(["mode","seed","region"]).rmse
    table["change_percent"]=[100*(r.rmse/base.loc[r.mode,r.seed,r.region]-1) for r in table.itertuples()]
    table.to_csv(a.out/"status-fracture-regions.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,("UT","FT")):
        for index,region in enumerate(("STATUS onset endpoints ±1 frame","STATUS onset strut relative motion ±1 frame")):
            for x,variant in enumerate(VARIANTS):
                v=table[(table["mode"]==mode)&(table.variant==variant)&(table.region==region)].change_percent
                ax.errorbar(x+index*.14-.07,v.mean(),yerr=v.std(),marker="o",capsize=3,
                    color=("#386c9b","#c6754e")[index],label=region.replace("STATUS onset ","") if x==0 else None)
        ax.axhline(0,color="black",lw=.7);ax.set(title=mode,ylabel="RMSE change vs matched MSE (%)",xticks=range(5),xticklabels=VARIANTS);ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("Confirmed STATUS fracture events: endpoints and connected-pair motion\nRetrospective validation only; lower is better; whiskers are seed standard deviation")
    fig.tight_layout();fig.savefig(a.out/"status-fracture-regions.png",dpi=170);plt.close(fig)
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
    text += ["\nFT box and20mm nominal-tip-radius results, physical errors, true regional-motion denominators and all event counts are retained in [the numerical companion](status-fracture-regions.csv). Relative changes compare exactly the same regions/specimens per seed; no whole-body average hides their errors. The global error and peak/work safeguards still matter.\n"]
    (a.out/"fracture-regions.md").write_text("".join(text),encoding="utf-8")
    print(table.groupby(["variant","mode","region"]).change_percent.mean().to_string())


if __name__=="__main__":main()
