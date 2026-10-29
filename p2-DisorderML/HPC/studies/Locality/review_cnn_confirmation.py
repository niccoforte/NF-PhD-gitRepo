"""Read completed CNN confirmation/depth results; no training or new selection."""
import argparse
import json
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from resources.MLstudy_reports import (collect_fields,figure,verify_download,
    repeated_field_figures,prediction_variation_section,replace_study_followup)


def entries(root,suite):
    result=[]
    for seed in (42,43,44):
        for mode in ("UT","FT"):
            for variant in ("control","cnn2"):
                source="locality-261026-r1" if seed==42 else suite
                result.append(dict(mode=mode,variant=variant,seed=seed,path=root/f"{source}-{variant}-{mode}-s{seed}"))
    result += [dict(mode=m,variant="cnn3",seed=42,path=root/f"{suite}-cnn3-{m}-s42") for m in ("UT","FT")]
    return result


def histories(table,out):
    for name,columns in (("validation",("val_loss",)),("train-validation",("train_loss","val_loss"))):
        fig,axes=plt.subplots(1,2,figsize=(15,5.5))
        for ax,mode in zip(axes,("UT","FT")):
            for row in table[table["mode"]==mode].itertuples():
                if len(columns)>1 and row.seed!=42:continue
                h=pd.read_csv(Path(row.path)/"loss_history.csv")
                color={"control":"#59686f","cnn2":"#238d83","cnn3":"#c07747"}[row.variant]
                for column in columns:
                    ax.plot(h.epoch,h[column].where(h[column]>0),color=color,
                        linestyle="--" if column.startswith("train") else "-",alpha=.8,
                        label=f"{row.variant} s{row.seed} {column.split('_')[0]}")
                if len(columns)>1:ax.axvline(row.best_epoch,color=color,alpha=.3,linestyle=":")
            ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Masked displacement MSE (log)");ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f"confirmation-loss-{name}.png",dpi=170);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--suite",default="cnn-confirmation-261028")
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/locality-study")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    receipt=verify_download(a.root,a.out/"confirmation-download-sha256.txt",10)
    (a.out/"confirmation-download-verification.json").write_text(json.dumps(receipt,indent=2))
    api=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    cases=entries(a.root,a.suite)
    # One collection checks shared IDs/targets/masks across all seeds and depth.
    table,regions,frames=collect_fields(cases,a.data_root,api)
    table["seconds_per_epoch"]=3600*table.hours/table.epochs
    table.to_csv(a.out/"confirmation-metrics.csv",index=False)
    regions.to_csv(a.out/"confirmation-regions.csv",index=False);frames.to_csv(a.out/"confirmation-frames.csv",index=False)
    repeated_field_figures(table,regions,a.out,"confirmation");histories(table,a.out)
    spread=prediction_variation_section(cases,a.out,prefix="confirmation-variation").replace("## Prediction","### Prediction")
    img=lambda n,caption:figure(a.out,"confirmation-"+n+".png",caption)
    means=table[table.variant=="cnn2"].groupby("mode").change_percent.agg(["mean","std"])
    local=regions[(regions.variant=="cnn2")&(regions.region=="STATUS onset endpoints ±1 frame")].groupby("mode").change_percent.mean()
    strut=regions[(regions.variant=="cnn2")&(regions.region=="STATUS onset strut relative motion ±1 frame")].groupby("mode").change_percent.mean()
    cost=table[table.variant.isin(("control","cnn2"))].groupby(["mode","variant"]).hours.mean().unstack()
    lines=["## Completed CNN confirmation and depth screen —29 October\n\n",
        f"**CNN2 improved both tasks in all three seeds.** Mean global RMSE change: UT{means.loc['UT','mean']:.2f}% ±{means.loc['UT','std']:.2f}% seed SD; FT{means.loc['FT','mean']:.2f}% ±{means.loc['FT','std']:.2f}%. "
        f"Confirmed-fracture endpoint changes average{local['UT']:.2f}% UT/{local['FT']:.2f}% FT; signed failing-strut motion changes{strut['UT']:.2f}%/{strut['FT']:.2f}%. "
        "Recommendation: CNN2 is supported as a field-architecture candidate for both tasks; do not claim a curve improvement from these field-only fits. Accepted WINNER.md remains unchanged pending acceptance.\n\n",
        "The10 newly completed production fits all exited0:0 and passed reload. Seed42 control/CNN2 are reused from the original screen; seeds43/44 have fresh matched controls. CNN3 is only seed42 and cannot displace a replicated CNN2 result on a marginal global numerical gain. Its stronger one-seed UT fracture-endpoint improvement warrants retaining it as an unconfirmed depth candidate. All use the same MSE/384-width baseline, inputs,793 validation IDs and original stopping. Locked test untouched.\n\n",
        "### Global performance versus matched control\n\n",img("global-control","CNN2 three-seed confirmation; CNN3 single-seed depth screen"),
        "### Global performance versus TRAIN mean\n\n",img("global-mean","Accuracy beyond training-only mean fields"),spread,
        "### STATUS fracture regions versus control\n\n",img("fracture-control","Actual failure endpoints and signed strut motion; STATUS is evaluation only"),
        "### STATUS fracture regions versus TRAIN mean\n\n",img("fracture-mean","Same confirmed-event entries and TRAIN reference"),
        "### Best/worst examples and architecture\n\nThe large four-column CNN2 seed42 galleries and architecture explanation remain in the original-screen section below; they are representative of the confirmed model, not additional evidence of its seed variability. They are not duplicated here.\n\n",
        "### Training cost\n\n",img("runtime","Final-generator training hours; no OOF/curve fitting or queue time"),
        f"Mean control→CNN2 fit hours: UT{cost.loc['UT','control']:.2f}→{cost.loc['UT','cnn2']:.2f}; FT{cost.loc['FT','control']:.2f}→{cost.loc['FT','cnn2']:.2f}. "
        "See epochs and seconds/epoch in the CSV: longer convergence and per-epoch work both affect cost. Saved GPU identity is retained; these are training times, not inference latency.\n\n",
        "### Loss histories\n\nValidation compares all seeds; train/validation uses seed42 to keep it readable. Dotted vertical lines mark chosen checkpoints; all positive objectives use log axes.\n\n",
        img("loss-validation","Validation MSE across confirmation seeds"),img("loss-train-validation","Seed42 training/validation and selected epochs"),
        "[Per-fit metrics/costs](confirmation-metrics.csv) · [Regional counts/denominators](confirmation-regions.csv) · [Download verification](confirmation-download-verification.json). "
        "Below: the initial one-seed locality screen, retained as feature/architecture attribution evidence rather than the latest confirmation verdict.\n"]
    replace_study_followup(a.out/"README.md","cnn-confirmation","".join(lines))
    print(table[["mode","variant","seed","change_percent","skill_percent","hours","epochs"]].to_string(index=False),flush=True)


if __name__=="__main__":main()
