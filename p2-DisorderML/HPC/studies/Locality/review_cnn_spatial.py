"""Review three-seed CNN3 and task-specific CNN2/spatial combinations."""
import argparse
import json
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resources.MLstudy_reports import (completed_field_review,verify_study_inventory,
    replace_study_followup,figure,serial_schematic)


def entries(root):
    rows=[]
    for mode in ("UT","FT"):
        for seed in (42,43,44):
            old="locality-261026-r1" if seed==42 else "cnn-confirmation-261028"
            rows.append(dict(mode=mode,seed=seed,variant="control",source_variant="cnn2",path=root/f"{old}-cnn2-{mode}-s{seed}"))
            old="cnn-confirmation-261028" if seed==42 else "cnn-spatial-depth-261102"
            rows.append(dict(mode=mode,seed=seed,variant="cnn3",path=root/f"{old}-cnn3-{mode}-s{seed}"))
            rows.append(dict(mode=mode,seed=seed,variant="cnn2_spatial",path=root/f"cnn-spatial-depth-261102-cnn2_spatial-{mode}-s{seed}"))
    return rows


def bootstrap_fields(cases,out):
    """Paired seed/specimen global error; descriptive validation intervals."""
    bootstrap=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_spatial_balance.py"))["paired_interval"]
    values={};counts={};ids={};rows=[]
    for e in cases:
        with np.load(e["path"]/"results/predictions.npz",allow_pickle=False) as z:
            mode=e["mode"];sample=z["sample_ids"].astype(str).tolist()
            if ids.setdefault(mode,sample)!=sample:raise ValueError("Paired specimen IDs differ")
            y=z[f"{mode}_val_field_truth"].astype(float);p=z[f"{mode}_val_field_outputs"].astype(float)
            v=z[f"{mode}_val_field_mask"].astype(bool)&z[f"{mode}_node_mask"][None,:,None]&np.isfinite(y)
            n=v.sum((1,2));mse=np.where(v,(p-y)**2,0).sum((1,2))/n
            values[mode,e["variant"],e["seed"]]=mse;counts[mode,e["seed"]]=n
    for mode in ("UT","FT"):
        base=np.stack([values[mode,"control",s] for s in (42,43,44)])
        weight=np.stack([counts[mode,s] for s in (42,43,44)])
        for variant in ("cnn3","cnn2_spatial"):
            changed=np.stack([values[mode,variant,s] for s in (42,43,44)])
            rows.append(dict(mode=mode,variant=variant,**bootstrap(base,changed,weight,rmse=True)))
    table=pd.DataFrame(rows);table.to_csv(out/"spatial-depth-bootstrap.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    for ax,mode in zip(axes,("UT","FT")):
        part=table[table["mode"]==mode]
        for i,r in enumerate(part.itertuples()):
            ax.errorbar(r.change,i,xerr=[[r.change-r.ci_low],[r.ci_high-r.change]],fmt="o",capsize=5)
        ax.axvline(0,color="black",lw=.8);ax.set(title=mode,yticks=[0,1],yticklabels=part.variant,xlabel="Global RMSE change vs matched CNN2 (%)");ax.grid(alpha=.2)
    fig.suptitle("Paired seed/specimen bootstrap,95% descriptive interval\nThree seeds; validation-selected checkpoints, not locked-test significance",fontsize=14)
    fig.tight_layout();fig.savefig(out/"spatial-depth-bootstrap.png",dpi=170);plt.close(fig)
    return table


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/locality-study")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    a=p.parse_args(argv);out=a.out;root=a.root/"MULTI/Dual/Transformer"
    receipt=verify_study_inventory(a.root,out/"spatial-depth-inventory.json",10)
    (out/"spatial-depth-verification.json").write_text(json.dumps(receipt,indent=2))
    cases=entries(root)
    # Bridge fresh feature controls to reused CNN2: compare scalers, IDs and
    # exact predictions. Do not hide a source/hardware mismatch in seed effects.
    bridge=[]
    for mode in ("UT","FT"):
        old=next(e["path"] for e in cases if e["mode"]==mode and e["seed"]==42 and e["variant"]=="control")
        new=root/f"nodal-features-261102-control-{mode}-s42"
        d0=json.loads((old/"model_data.json").read_text());d1=json.loads((new/"model_data.json").read_text())
        for key in ("sample_ids","normalizers","node_masks","reference_task_features"):
            if d0[key]!=d1[key]:raise ValueError("Fresh CNN2 bridge changed "+key)
        with np.load(old/"results/predictions.npz",allow_pickle=False) as z0,np.load(new/"results/predictions.npz",allow_pickle=False) as z1:
            k=f"{mode}_val_field_outputs";delta=float(np.max(abs(z0[k]-z1[k])))
        m0=json.loads((old/"results/metrics.json").read_text())["diagnostics"]["field"][mode]["rmse"]
        m1=json.loads((new/"results/metrics.json").read_text())["diagnostics"]["field"][mode]["rmse"]
        bridge.append(dict(mode=mode,old=str(old),fresh=str(new),max_prediction_difference=delta,rmse_change_percent=100*(m1/m0-1)))
    pd.DataFrame(bridge).to_csv(out/"spatial-depth-control-bridge.csv",index=False)
    api=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    fields,regions,text=completed_field_review(cases,a.data_root,api,out,"spatial-depth","Completed CNN/spatial and CNN3 confirmation",
        "All ten new fits completed; reuse CNN2 controls at seeds42/43/44 and the existing CNN3 seed42 to compare three seeds for every arm. Control means CNN2/MSE, not the original plain Transformer. CNN3 adds only one masked grid block. CNN2+spatial uses UT coefficient0.1 on all initial connections, or FT0.001 on connections whose two endpoints lie inside the fixed tip box. Whole-field MSE remains active. No context removal, STATUS auxiliary, temporal/dynamic weights, OOF banks or curves are bundled.",
        "CNN2/MSE → CNN3/MSE OR CNN2 +\ntask-specific signed spatial loss",
        "**Field-only confirmation; no serial-curve promotion.** Judge global and actual STATUS-fracture metrics together. Three training seeds quantify variability; they do not establish a final locked-test result.")
    serial_schematic(out,"spatial-depth-architecture.png",[
        ("CNN2/MSE control",["Disorder + unchanged\nphysical context","2 masked CNN blocks\n→ field Transformer","20-frame\nU1/U2 fields"],"Whole-field masked MSE; independent task model"),
        ("Two separate candidates",["Same disorder\n+ physical context","CNN3/MSE OR\nCNN2 + spatial loss","Same20-frame\nU1/U2 fields"],"UT spatial0.1 all edges; FT0.001 fixed-tip-box edges")])
    summary=fields.groupby(["mode","variant"],sort=False).agg(global_change=("change_percent","mean"),seed_sd=("change_percent","std"),skill_percent=("skill_percent","mean"),hours=("hours","mean"))
    for region,label in (("STATUS onset endpoints ±1 frame","endpoint_change"),("STATUS onset strut relative motion ±1 frame","strut_change")):
        summary[label]=regions[regions.region==region].groupby(["mode","variant"]).change_percent.mean()
    summary.to_csv(out/"spatial-depth-decision.csv")
    interval=bootstrap_fields(cases,out)
    intro="\n\n".join(f"**{mode} {variant}:** mean global RMSE {r.global_change:+.2f}% (seed SD {r.seed_sd:.2f}%); confirmed endpoints {r.endpoint_change:+.2f}%; signed failing-strut motion {r.strut_change:+.2f}%." for (mode,variant),r in summary.iterrows() if variant!="control")
    intro+="\n\n**Recommended field shortlist:** CNN2/MSE remains the conservative replicated architecture for both tasks. Among incremental candidates, UT CNN2 + spatial0.1 has the clearest fracture gain: endpoint and signed failing-strut errors improve in all three seeds, with global error essentially unchanged in seed42 and lower in43/44. CNN3 offers similar UT global improvement but worsens both fracture metrics in seed44, so extra depth is not the preferred UT choice. FT CNN3/MSE lowers global error in all three seeds and failing-strut error in all three; endpoint error rises0.47% in seed42 but falls in43/44. Prefer it as the FT global-accuracy candidate. FT CNN2 + box-spatial0.001 is a fracture-endpoint alternative, not an additive proven improvement: its global and signed-strut errors worsen slightly in seed43. CNN3+spatial has not been tested. All four incremental global bootstrap intervals include zero: these modest gains are candidates, not decisive superiority over CNN2. Keep the distinction between a field shortlist and an accepted serial winner: new TRAIN-only OOF fields and Predicted+True curve fits must check curve/peak/work behaviour before an end-to-end promotion. The previous plain-UT spatial peak trade-off cannot be assumed either to persist or disappear with CNN2."
    text=text.replace("### Global performance",intro+"\n\n### Global performance",1)
    text+="\n### Robustness and control bridge\n\n"+figure(out,"spatial-depth-bootstrap.png","Paired bootstrap global changes against CNN2")
    text+="Each bootstrap resamples training seeds and paired validation specimen IDs (2000 replicates). Validation checkpoint selection and only three seeds limit inference; this is descriptive, not a confirmatory locked-test claim. These intervals cover global error only; local robustness is shown by the individual-seed regional points, not inferred from a global interval. Fresh seed42 CNN2 controls share IDs/scalers/masks with the reused controls but differ in global RMSE by −0.24% UT/+1.35% FT. Equality of seeds does not guarantee identical predictions; maximum nodal differences are also recorded. Small gains therefore deserve caution. No cause of that repeat difference has been established. [Control bridge](spatial-depth-control-bridge.csv), [bootstrap intervals](spatial-depth-bootstrap.csv), [decision metrics](spatial-depth-decision.csv), [download verification](spatial-depth-verification.json). Original completed screen/confirmation results follow below and retain their original comparators.\n"
    replace_study_followup(out/"README.md","cnn-spatial-depth",text)
    print(summary.to_string());print(interval.to_string(index=False));print("REPORT",out/"README.md")


if __name__=="__main__":main()
