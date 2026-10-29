"""Review completed UT STATUS confirmation and smaller FT auxiliary budgets."""
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
    replace_study_followup,figure)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2"))
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/status-study")
    a=p.parse_args(argv);out=a.out;root=a.root/"MULTI/Damage/Transformer"
    receipt=verify_study_inventory(a.root,out/"balance-inventory.json",6)
    (out/"balance-verification.json").write_text(json.dumps(receipt,indent=2))
    cases=[]
    for seed in (42,43,44):
        suite="status-261027" if seed==42 else "status-balance-261028"
        for variant in ("control","strong"):
            cases.append(dict(mode="UT",seed=seed,variant="control" if variant=="control" else "auxiliary1",source_variant=variant,path=root/f"{suite}-{variant}-UT-s{seed}"))
    for variant,label in (("control","control"),("weak","auxiliary0.1"),("strong","auxiliary1")):
        cases.append(dict(mode="FT",seed=42,variant=label,source_variant=variant,path=root/f"status-261027-{variant}-FT-s42"))
    for coefficient,token in ((.001,"0p001"),(.01,"0p01")):
        cases.append(dict(mode="FT",seed=42,variant=f"auxiliary{coefficient}",source_variant="weak",path=root/f"status-balance-261028-weak-{token}-FT-s42"))
    cases.sort(key=lambda r:(r["mode"]!="UT",r["seed"],0 if r["variant"]=="control" else float(r["variant"].removeprefix("auxiliary"))))
    api=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    fields,regions,text=completed_field_review(cases,a.data_root,api,out,"balance","Completed auxiliary-balance and UT confirmation",
        "Six new final fits: fresh UT control/auxiliary1 at seeds43/44 confirm the original seed42 screen, while FT auxiliary0.001/0.01 reuse its original same-trainer seed42 control. Earlier FT0.1/1 are included for context. Every auxiliary model trains displacement MSE + coefficient × balanced strut BCE, using the same encoder for both heads. No SDEG, predicted-damage feedback, OOF or curve training.",
        "Same nodal encoder and displacement head;\nSTATUS head adds λ × balanced BCE",
        "**Completed and SHA256 verified; no auxiliary model is automatically adopted.** UT has three-seed evidence; FT smaller coefficients are a one-seed loss-budget screen. Damage discrimination and displacement improvement are separate findings.")
    damage=[];balance=[]
    for row in fields.itertuples():
        if row.variant=="control":continue
        path=Path(row.path);cfg=json.loads((path/"model.json").read_text());metrics=json.loads((path/"results/metrics.json").read_text())
        coefficient=cfg["damage_coefficient"];h=pd.read_csv(path/"loss_history.csv");best=h[h.epoch==row.best_epoch].iloc[0]
        if not np.allclose(h.val_field+coefficient*h.val_damage,h.val_loss,rtol=1e-5,atol=1e-8):raise ValueError("Auxiliary terms do not reconstruct total")
        balance.append(dict(mode=row.mode,seed=row.seed,variant=row.variant,coefficient=coefficient,
            weighted_damage_to_field=coefficient*best.val_damage/best.val_field))
        with np.load(path/"results/predictions.npz",allow_pickle=False) as z:
            probabilities,targets,masks=z["damage_probability"],z["damage_truth"],z["damage_mask"]
            tp=fp=fn=0
            for start in range(0,len(z["sample_ids"]),32):
                pred=probabilities[start:start+32];truth=targets[start:start+32].astype(bool);valid=masks[start:start+32]
                tp+=int(((pred>=.5)&truth&valid).sum());fp+=int(((pred>=.5)&~truth&valid).sum());fn+=int(((pred<.5)&truth&valid).sum())
            m=metrics["damage"]["fixed_05"]
            if (tp,fp,fn)!=(m["true_positive"],m["false_positive"],m["missed_fractures"]):raise ValueError("Damage confusion counts disagree")
        for policy in ("fixed_05","validation_f1"):
            damage.append(dict(mode=row.mode,seed=row.seed,variant=row.variant,coefficient=coefficient,policy=policy,**metrics["damage"][policy]))
    damage=pd.DataFrame(damage);damage.to_csv(out/"balance-damage.csv",index=False)
    balance=pd.DataFrame(balance);balance.to_csv(out/"balance-loss-ratios.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(17,11))
    for col,mode in enumerate(("UT","FT")):
        part=damage[(damage["mode"]==mode)&(damage.policy=="fixed_05")];variants=list(part.variant.drop_duplicates())
        for j,metric in enumerate(("average_precision","precision","recall")):
            for i,variant in enumerate(variants):
                values=part[part.variant==variant][metric].to_numpy()
                axes[0,col].errorbar(i+(j-1)*.14,values.mean(),yerr=values.std(ddof=1) if len(values)>1 else None,fmt="o",capsize=4,label=metric if i==0 else None)
        axes[0,col].axhline(part.actual_positive_fraction.iloc[0],color="black",linestyle="--",label="AP prevalence reference")
        axes[0,col].set(title=mode,ylabel="AP / precision / recall (0–1)",xticks=range(len(variants)),xticklabels=variants,ylim=(0,1.05))
        axes[0,col].legend(fontsize=10);axes[0,col].tick_params(axis="x",rotation=25);axes[0,col].grid(alpha=.2)
        for i,variant in enumerate(variants):
            values=balance[(balance["mode"]==mode)&(balance.variant==variant)].weighted_damage_to_field
            axes[1,col].scatter(np.full(len(values),i),values)
        axes[1,col].axhline(1,color="black",linestyle="--");axes[1,col].set(yscale="log",title=mode,xticks=range(len(variants)),xticklabels=variants,ylabel="Weighted BCE / field MSE (log; loss values)")
        axes[1,col].tick_params(axis="x",rotation=25);axes[1,col].grid(alpha=.2)
    fig.suptitle("Fracture discrimination versus auxiliary-loss budget\nPrecision/recall at threshold0.5; AP across thresholds; UT points include three seeds",fontsize=16)
    fig.tight_layout();fig.savefig(out/"balance-damage-and-budget.png",dpi=170);plt.close(fig)
    summary=fields.groupby(["mode","variant"],sort=False).agg(seeds=("seed","count"),global_change=("change_percent","mean"),global_sd=("change_percent","std"))
    for region,short in (("STATUS onset endpoints ±1 frame","endpoint_change"),("STATUS onset strut relative motion ±1 frame","strut_change")):
        summary[short]=regions[regions.region==region].groupby(["mode","variant"]).change_percent.mean()
    summary.to_csv(out/"balance-decision.csv")
    intro="\n\n".join(f"**{mode} {variant}:** mean global RMSE {row.global_change:+.2f}%; confirmed endpoints {row.endpoint_change:+.2f}%; signed failing-strut motion {row.strut_change:+.2f}% ({int(row.seeds)} seed{'s' if row.seeds>1 else ''})." for (mode,variant),row in summary.iterrows() if variant!="control")
    ut=summary.loc["UT","auxiliary1"]
    intro=(f"**Updated verdict:** UT auxiliary1 now improves global RMSE in all three seeds (mean{ut.global_change:+.2f}%), "
        f"with confirmed endpoints{ut.endpoint_change:+.2f}% and failing-strut motion{ut.strut_change:+.2f}%. "
        "This supports a small task-specific auxiliary candidate, rather than the earlier nearly-null one-seed impression. "
        "FT still shows no displacement benefit at either smaller coefficient, including the actual fracture regions. "
        "Retain FT displacement MSE; keep predicting STATUS as a distinct demonstrated output rather than assuming that its discrimination fixes displacements. "
        "No explicit damage-conditioned feedback or CNN combination has been tested, and WINNER.md remains unchanged.\n\n"+intro)
    text=text.replace("### Global performance",intro+"\n\n### Global performance",1)
    text+="\n### Did smaller coefficients fix FT auxiliary-loss imbalance?\n\n"+figure(out,"balance-damage-and-budget.png","Damage discrimination and actual weighted damage/field loss at selected checkpoints")
    text+="These are loss-value ratios, not measured gradient ratios. AP measures ranking; fixed-threshold precision/recall expose overcalling from class balancing. The numerical companion retains both threshold0.5 and the optimistic validation-selected F1 threshold, missed/false struts, path overlap, onset errors and broken→intact reversals. No cumulative-maximum repair is applied. STATUS-only remains in the original completed screen below, with no displacement score. [Updated damage metrics](balance-damage.csv) · [Loss ratios](balance-loss-ratios.csv) · [Decision metrics](balance-decision.csv) · [Verification](balance-verification.json).\n"
    runpy.run_path(str(Path(__file__).with_name("review_status_study.py")))["schematic"](
        out,coefficient_label="1 UT; .001/.01/.1/1 FT",prefix="balance-architecture")
    replace_study_followup(out/"README.md","status-balance",text)
    print(summary.to_string());print("REPORT",out/"README.md")


if __name__=="__main__":main()
