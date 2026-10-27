"""Readable isolated-study protocol and temporal weighting profiles; no training."""
import argparse
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/field-variations")
    p.add_argument("--jobs",type=Path)
    p.add_argument("--execution-note",default="Prepared; consult PROJECT_STATUS for actual submission.")
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    cases=runpy.run_path(str(Path(__file__).with_name("A0-FieldVariations.py")))["CASES"]
    frames=np.arange(1,21);q=(frames-1)/19
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for amplitude in (1,3):
        for width in (.15,.25):
            weights=1+amplitude*np.exp(-.5*((q-.5)/width)**2)
            for ax,values in zip(axes,(weights,weights/weights.mean())):
                ax.plot(frames,values,"o-",markersize=3,label=f"amplitude{amplitude}, width{width}")
    for ax,title in zip(axes,("Raw: baseline1 + middle-frame emphasis","Mean-one example: all20 frames valid")):
        ax.axhline(1,color="black",lw=.8,linestyle="--",label="Uniform MSE")
        ax.set(title=title,xlabel="Retained frame (time order)",ylabel="MSE weight",xticks=[1,5,10,15,20]);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Gaussian weights vary over time frames, not x/y or expected fracture paths")
    fig.tight_layout();fig.savefig(a.out/"gaussian-frame-profiles.png",dpi=170);plt.close(fig)
    text=["# Isolated field variations — preliminary report\n\n**Protocol and execution receipt, not completed accuracy results.**\n\n",
        a.execution_note+"\n\n",
        "Thirteen final-generator fits, seed42, unchanged plain independent384/8/3 architecture and Mixed study's data contract. Two completed Locality plain controls are reused. No OOF banks, curve training, CNN, damage or derivative penalties are added. Each fit isolates one intervention. Full budget240h/1GPU/12CPU/90000MB;450 epochs/patience52.\n\n",
        "| Run variant | Task | Only change |\n|---|---|---|\n"]
    for variant,mode in cases:
        meaning=("Append static initial crack-face flag" if variant=="crack_face" else
            "Train-mean residual target coordinates,0.1 pooled-scale floor" if variant=="residual" else
            f"FT crack-tip box raw weight{variant[-1]}, outside1" if variant.startswith("box") else
            "Gaussian temporal weights: "+variant.replace("gaussian_","").replace("_",", "))
        text.append(f"| {variant} | {mode} | {meaning} |\n")
    text += [f"\n![Gaussian temporal weights](<{(a.out/'gaussian-frame-profiles.png').resolve().as_posix()}>)\n\n",
        "The two middle frames have the highest weights; endpoints remain positive. Per-specimen normalization includes valid node/frame/component entries only, so missing histories can change the final normalized profile. These weights do not identify observed fracture and require no true outputs at inference. FT crack-face and box interventions run only FT.\n\n",
        "## Selection and reporting\n\nKeep the existing physical field-MSE/TRAIN-mean-MSE checkpoint criterion. Report global TRAIN-mean RMSE skill, actual STATUS-event endpoint/strut-motion errors, broader FT box, temporal/sign safeguards and costs. Positive skill means improvement over always predicting the mean;0 is the mean baseline. Keep normalized weighted training objectives separate from fixed physical selection and plot histories on log axes. Confirm promising effects across seeds43/44 before combining or retraining curves.\n\n"]
    if a.jobs:
        text += ["## Submitted IDs\n\nThis is a submission receipt, not a live scheduler check.\n\n",pd.read_csv(a.jobs,sep="\t").to_markdown(index=False),"\n"]
    (a.out/"README.md").write_text("".join(text),encoding="utf-8")
    print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
