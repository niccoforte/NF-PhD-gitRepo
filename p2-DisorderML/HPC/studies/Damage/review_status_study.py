"""Readable STATUS study protocol and saved-result review; no training."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd


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
    ax.text(6.4,.55,"L = Ldisplacement + λ LSTATUS\nλ=0.1 or1; both update the same encoder",ha="center",fontsize=12,color="#384d60")
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
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True);schematic(a.out)
    text=["# STATUS learning: preliminary study report\n\n**This is the experiment protocol, not an accuracy result.** Dataset extraction is complete. Learning outcomes will be added from completed production archives; queued jobs and a successful preflight do not establish predictive quality.\n\n",
        f"![STATUS and displacement architecture](<{(a.out/'status-architecture.png').resolve().as_posix()}>)\n\n",
        "## Exactly what is compared\n\nEight seed42 fits: four arms × two fully independent tasks. All retain the saved7137/793/881 specimen partition, geometry/context contract,20 retained frames, field architecture384/8/3, AdamW settings, batch2,450 maximum epochs and patience52. No OOF bank, curve fit, SDEG export or explicit damage feedback is included.\n\n",
        "| Arm | Trained outputs | Objective | Checkpoint selection |\n|---|---|---|---|\n| Control | Displacement | Masked MSE | Fixed physical field MSE / training-mean MSE |\n| STATUS-only | Strut fracture histories | Balanced masked BCE | Validation average precision; BCE tie-break |\n| Weak auxiliary | Displacement + fracture | MSE +0.1 BCE | Same field criterion as control |\n| Strong auxiliary | Displacement + fracture | MSE +1 BCE | Same field criterion as control |\n\n",
        "The STATUS-only arm retains the same nodal encoder but does not train or report displacement predictions. Controls are freshly fitted in the same consumer so trainer/selection differences cannot silently favour an intervention. Common encoder initialization is identical per task; adding the strut head uses a separate RNG stream.\n\n",
        "## How one strut is predicted\n\nFor native strut i–j, gather both final nodal vectors. Concatenate their mean, absolute difference, and three initial-geometry quantities: length/specimen span and sin(2θ), cos(2θ). Reversing endpoints gives the same representation. The same small MLP reads every strut and returns20 logits. It creates no new attention tokens. Geometry comes from the actual initial disordered specimen; true future damage is never provided.\n\n",
        "In the auxiliary arms, displacement and STATUS losses both update the nodal encoder. This is the initial linkage being tested. Predicted STATUS is not yet fed back into a second refinement pass and does not remove connections. UT and FT have separate encoders, heads, optimizers and training losses.\n\n",
        "## Labels and rare fractures\n\nThe predicted binary label is D=1 for broken, derived from Abaqus STATUS; it is not raw STATUS=1 (intact). A strut is broken if at least one constituent element has verified STATUS=0. Unknown entries remain masked, initial FT gaps are absent geometry, and frame0 is dropped consistently. IDs, native-to-canonical endpoints, retained frames and sample initial coordinates must match before training. Short UT histories remain unknown after their observed prefix.\n\n",
        "From TRAIN only, let p be the fraction of known broken entries. Set r=min((1−p)/p,1000). Use BCEWithLogits(pos_weight=r), divided by (1−p)+r p. This gives rare missed fractures more influence while keeping the expected weight near one; λ then controls auxiliary influence. The cap and coefficients are fixed presets. No validation/test statistics fit the balancing.\n\n",
        "## How success will be judged\n\nReport average precision against the prevalence reference, precision/recall, missed and false fractures, predicted positive fraction, strut-path overlap, onset interval error and broken→intact reversals. Report fixed threshold0.5 plus the F1-maximizing validation threshold. The latter is an optimistic validation diagnostic, frozen before any future test evaluation; weighted logits are not calibrated failure probabilities. Never silently apply cumulative maximum.\n\n",
        "Auxiliary benefit requires improved displacement, especially fracture-event/jump metrics, with global RMSE safeguards. Accurate STATUS and better displacement are distinct outcomes. Confirm useful arms on seeds43/44 before expensive OOF generation or curve training. Loss histories are logarithmic.\n\n",
        "## Execution and results\n\nOne disposable8h all-arm GPU preflight, then eight240h/1GPU/12CPU/90000MB jobs through home→scratch→archive. Successful preflight keeps only its home log. Run outputs use `MULTI/Damage/Transformer/<suite>-<arm>-<UT|FT>-s42`; STATUS checkpoints have their own explicit versioned reader, while exported field arrays follow the existing diagnostic contract.\n\n"]
    if a.jobs:
        jobs=pd.read_csv(a.jobs,sep="\t");text += ["Submitted job IDs below are a receipt, not a live status check.\n\n",jobs.to_markdown(index=False),"\n\n"]
    if a.execution_note:text += [a.execution_note+"\n\n"]
    rows=[]
    for mode in ("UT","FT"):
        for variant in ("control","damage","weak","strong"):
            path=a.root/f"{a.suite}-{variant}-{mode}-s42"
            if (path/"run_metadata.json").exists() and json.loads((path/"run_metadata.json").read_text()).get("status")=="complete":
                m=json.loads((path/"results/metrics.json").read_text());d=m.get("damage",{}).get("fixed_05",{})
                rows.append(dict(task=mode,arm=variant,field_rmse=m["diagnostics"]["field"].get(mode,{}).get("rmse"),average_precision=d.get("average_precision"),recall=d.get("recall"),precision=d.get("precision")))
    if rows:text += [pd.DataFrame(rows).to_markdown(index=False),"\n"]
    else:text += ["**No completed production learning results are available in this local report yet.**\n"]
    (a.out/"README.md").write_text("".join(text),encoding="utf-8");print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
