"""Readable isolated-study protocol and temporal weighting profiles; no training."""
import argparse
import json
from pathlib import Path
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resources.MLstudy_reports import (collect_fields, comparison_figures, loss_figures,
    architecture_figure, field_examples, figure)

LABELS={"control":"MSE control","crack_face":"Crack-face flag","residual":"Residual targets",
    "box2":"FT box ×2","box4":"FT box ×4",**{f"gaussian_a{a}_w{w}":f"Gaussian a{a}, w{w}" for a in (1,3) for w in (.15,.25)}}


def training_event_timing(root,control_suite,data_root,out):
    """Audit the temporal prior using TRAIN labels only; no new weights are fitted."""
    fig,axes=plt.subplots(1,2,figsize=(12,4));rows=[]
    for ax,mode in zip(axes,("UT","FT")):
        data=json.loads((root/f"{control_suite}-control-{mode}-s42/model_data.json").read_text())
        with np.load(data_root/f"MULTI-{mode}-disNodes-allD.npz",allow_pickle=False) as bank:
            ids={str(v):i for i,v in enumerate(bank["sample_ids"])};positions=[ids[str(v)] for v in data["sample_ids"]["train"]]
            target,known=bank["D"],bank["valid_mask"]
            counts=np.zeros(20,np.int64);first=np.zeros(20,np.int64);observed=0
            for start in range(0,len(positions),32):
                idx=positions[start:start+32];d=target[idx].astype(bool);v=known[idx]
                onset=d[:,1:]&~d[:,:-1]&v[:,1:]&v[:,:-1]
                counts+=onset.sum(axis=(0,2));any_event=onset.any(axis=2);eligible=any_event.any(axis=1)
                first+=np.bincount(any_event[eligible].argmax(axis=1),minlength=20);observed+=int(eligible.sum())
        ax.bar(np.arange(1,21),100*counts/max(counts.sum(),1),color="#5e97af",alpha=.65,label="All observed strut onsets")
        ax.plot(np.arange(1,21),100*first/max(first.sum(),1),"o-",color="#b27650",label="First observed onset per specimen",markersize=3)
        ax.axvline(10.5,color="black",linestyle="--",alpha=.6,label="Fixed Gaussian centre")
        ax.set(title=f"{mode}: {observed:,}/{len(positions):,} TRAIN specimens with observed onset",xlabel="Retained frame (time order)",ylabel="Fraction of respective observed events (%)",xticks=[1,5,10,15,20]);ax.legend(fontsize=7);ax.grid(alpha=.2)
        median_first=int(np.searchsorted(np.cumsum(first),first.sum()/2)+1)
        median_strut=int(np.searchsorted(np.cumsum(counts),counts.sum()/2)+1)
        ax.text(.03,.92,f"Median first/strut onset: {median_first}/{median_strut}",transform=ax.transAxes,fontsize=9)
        rows.extend(dict(mode=mode,frame=i+1,strut_onsets=int(counts[i]),first_specimen_onsets=int(first[i]),
            train_specimens=len(positions),specimens_with_observed_onset=observed) for i in range(20))
    fig.suptitle("Does the mid-frame prior match recorded training fracture timing?\nKnown intact→failed transitions only; missing/truncated histories are not extrapolated")
    fig.tight_layout();fig.savefig(out/"training-fracture-timing.png",dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"training-fracture-timing.csv",index=False)


def review_results(root,control_suite,suite,data_root,out):
    region_api=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Loss/review_fracture_regions.py"))
    cases=runpy.run_path(str(Path(__file__).with_name("A0-FieldVariations.py")))["CASES"]
    entries=[dict(mode=m,variant="control",label=LABELS["control"],path=root/f"{control_suite}-control-{m}-s42") for m in ("UT","FT")]
    entries += [dict(mode=m,variant=v,label=LABELS[v],path=root/f"{suite}-{v}-{m}-s42") for v,m in cases]
    table,regions,frames=collect_fields(entries,data_root,region_api)
    for name,values in (("metrics",table),("fracture-metrics",regions),("frame-metrics",frames)):values.to_csv(out/f"{name}.csv",index=False)
    render_results(root,suite,data_root,out,table,regions)


def render_results(root,suite,data_root,out,table,regions):
    """Render the verified fixed metrics and saved examples; no fitting."""
    region_api=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Loss/review_fracture_regions.py"))
    comparison_figures(table,regions,out)
    winners={m:table[table["mode"]==m].sort_values("rmse").iloc[0].variant for m in ("UT","FT")}
    loss_figures(table,out,winners)
    architecture_figure(out,"Same Transformer, separate interventions:\nstatic FT feature / residual targets /\nregional or temporal MSE weighting\nNo combined candidate adopted")
    examples={}
    for mode in ("UT","FT"):
        row=table[(table["mode"]==mode)&(table.variant==winners[mode])].iloc[0]
        descriptor=json.loads((Path(row.path)/"model_data.json").read_text())
        mean=region_api["training_mean"](data_root,descriptor,mode)
        examples[mode]=field_examples(row.path,mode,out,LABELS[row.variant],data_root,mean)
    text=["# Independent field variations\n\n## Recap, findings and verdict\n\n",
        "All 13 production fits completed and reloaded successfully. This is a seed 42 field-only screen, compared with the two completed matched plain Locality controls. No OOF bank or downstream curve was trained. The accepted Case1(i)+Mixed configuration is unchanged; numerical leaders below are candidates, not promoted winners.\n\n"]
    for mode in ("UT","FT"):
        row=table[(table["mode"]==mode)&(table.variant==winners[mode])].iloc[0]
        text.append(f"**{mode} global candidate: {LABELS[row.variant]}**, RMSE change {row.change_percent:+.2f}% versus control; {row.skill_percent:.2f}% RMSE improvement over TRAIN mean.\n\n")
    text += ["The residual-target result is task-dependent; the independent FT run worsens overall error, so the older shared-model residual result is not transferable. Static FT crack-face and stronger FT-box weighting do not improve whole-field error at these presets. Evaluate local event gains separately before deciding. Gaussian effects are small compared with the prior CNN2 FT gain and require seed confirmation.\n\n",
        figure(out,"architecture-comparison.png","Unchanged control versus isolated candidate changes"),
        "## Global performance versus the study control\n\nNegative RMSE change is better. Every comparison uses identical validation IDs/masks and physical targets, checked with floating tolerance after residual inverse scaling.\n\n",
        figure(out,"global-control.png","Global field RMSE changes"),
        "## Global performance versus the training mean\n\nSkill=100×(1−model RMSE/TRAIN-mean RMSE). The mean is fitted per node/frame/component on training IDs only; 0 means baseline error, negative is worse. This measures improvement beyond the common deformation pattern.\n\n",
        figure(out,"global-mean.png","Field RMSE skill versus training-only mean"),
        figure(out,"diversity.png","Output diversity as a complementary collapse check"),
        figure(out,"UT-mean-field-examples.png","UT actual training-mean field comparison"),
        figure(out,"FT-mean-field-examples.png","FT actual training-mean field comparison"),
        "## STATUS fracture regions versus the study control\n\nUse both endpoints of verified intact→failed struts and their signed relative motion at onset ±1 retained frame. The larger FT box is a separate broader view across valid frames. STATUS defines retrospective evaluation only; it is not given to these models.\n\n",
        figure(out,"fracture-control.png","Confirmed-fracture changes versus control"),
        "## STATUS fracture regions versus the training mean\n\nApply the same validation regions to the model and TRAIN-mean field; no validation means are fitted. Baseline reconstruction must reproduce saved global baseline RMSE.\n\n",
        figure(out,"fracture-mean.png","Confirmed-fracture skill versus average field"),
        "## Best and worst examples\n\nUse each task's global-RMSE candidate, rank specimens by full valid field RMSE, then show U2 at the largest observed increment in the displayed region. UT shows the body; FT shows its crack-tip box. Signed errors retain mm. Percentage error is 100×|prediction−truth|/|truth|; grey excludes truth smaller than 1% of the displayed training-mean RMS, and the colour scale caps at P95 only for readability. All metrics retain every valid entry.\n\n",
        figure(out,"UT-field-examples.png","UT truth prediction signed error and percent error"),
        figure(out,"FT-field-examples.png","FT crack-tip truth prediction signed error and percent error"),
        "## Training time\n\nQueue/export time is excluded. Different early stopping and hardware can affect observed cost; this is not an inference-latency benchmark.\n\n",
        figure(out,"runtime.png","Final-generator training hours"),
        "## Loss histories\n\nThe validation-only figure compares all variants; selected train/validation compares the control and descriptive candidate to diagnose overfitting. Weighted/residual objectives differ and are not ranked against one another by their totals. Checkpoint selection always uses fixed physical field MSE relative to TRAIN mean. Only positive saved values are plotted on log axes.\n\n",
        figure(out,"loss-validation.png","Validation objectives for every variant"),
        figure(out,"loss-train-validation.png","Control and selected candidate training validation curves"),
        "## What changed and what did not\n\nCrack-face adds one static feature for 26 retained FT nodes, initially zero projected. Residual targets subtract the training mean and rescale specimen variation with a 0.1 pooled-scale floor; the physical inverse is retained. Box weights are 2/4 inside the established FT refinement region and 1 outside. Gaussian weights act over time-frame index only, peak in the middle and retain positive tails. All fixed weights normalize to valid-entry mean 1 per specimen. No interventions are combined.\n\n",
        figure(out,"gaussian-frame-profiles.png","Gaussian weights over retained time frames"),
        figure(out,"training-fracture-timing.png","Actual training fracture-onset timing compared with the fixed Gaussian centre"),
        "The timing audit uses TRAIN only. Median first specimen/individual strut onset is frame 7/10 for UT and 9/14 for FT; all 7,137 training specimens have an observed onset. Initiation and later propagation therefore have different timing, especially for FT. The fixed centre at 10.5 is a broad prior, not each specimen's actual failure frame. It was prescribed before these fits, not fitted to validation damage. This plot checks the prior rather than changing it retrospectively; truncated histories are never extrapolated.\n\n",
        "**Regional verdict:** UT residual targets improve confirmed endpoints 2.82% and failing-strut motion 3.02%, alongside 1.29% global improvement. FT Gaussian amplitude 3/width 0.25 improves endpoints 3.42%, strut motion 1.27%, and the larger box 0.75%, alongside 1.46% global improvement. FT box×2 slightly improves strut-motion error 2.43% but worsens endpoints 0.81%, the larger box 1.54% and global error 0.72%; stronger box×4 is less useful here. Static crack-face shows no independent benefit in this seed. FT residual targets worsen confirmed endpoints 28.02% and strut motion 24.50%, so do not adopt them from the historical shared-model result.\n\n",
        "**Next:** confirm promising candidates on seeds 43/44. The stronger prior CNN2 candidate remains the more useful architectural confirmation. Do not combine changes or regenerate expensive OOF banks until their individual effects are established.\n\n",
        "[Full metrics and provenance](metrics.csv) · [Fracture counts, physical errors and denominators](fracture-metrics.csv) · [Frame metrics](frame-metrics.csv) · [Download verification](download-verification.json). One seed, validation only; locked test untouched.\n"]
    (out/"README.md").write_text("".join(text),encoding="utf-8")
    from resources.MLstudy_reports import insert_variation_section
    insert_variation_section(out/"README.md",table[["variant","path"]].to_dict("records"))
    (out/"example-selection.json").write_text(json.dumps(examples,indent=2))
    print(table[["mode","variant","change_percent","skill_percent","hours"]].to_string(index=False))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/field-variations")
    p.add_argument("--jobs",type=Path)
    p.add_argument("--execution-note",default="Prepared; consult PROJECT_STATUS for actual submission.")
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    p.add_argument("--suite",default="field-variations-261027")
    p.add_argument("--control-suite",default="locality-261026-r1")
    p.add_argument("--protocol-only",action="store_true")
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
    if not a.protocol_only:
        training_event_timing(a.root,a.control_suite,a.data_root,a.out)
        review_results(a.root,a.control_suite,a.suite,a.data_root,a.out)
        return
    text=["# Isolated field variations — preliminary report\n\n**Protocol and execution receipt, not completed accuracy results.**\n\n",
        a.execution_note+"\n\n",
        "Thirteen final-generator fits, seed 42, unchanged plain independent384/8/3 architecture and Mixed study's data contract. Two completed Locality plain controls are reused. No OOF banks, curve training, CNN, damage or derivative penalties are added. Each fit isolates one intervention. Full budget240h/1GPU/12CPU/90000MB;450 epochs/patience52.\n\n",
        "| Run variant | Task | Only change |\n|---|---|---|\n"]
    for variant,mode in cases:
        meaning=("Append static initial crack-face flag" if variant=="crack_face" else
            "Train-mean residual target coordinates,0.1 pooled-scale floor" if variant=="residual" else
            f"FT crack-tip box raw weight{variant[-1]}, outside1" if variant.startswith("box") else
            "Gaussian temporal weights: "+variant.replace("gaussian_","").replace("_",", "))
        text.append(f"| {variant} | {mode} | {meaning} |\n")
    text += [f"\n![Gaussian temporal weights](<{(a.out/'gaussian-frame-profiles.png').resolve().as_posix()}>)\n\n",
        "The two middle frames have the highest weights; endpoints remain positive. Per-specimen normalization includes valid node/frame/component entries only, so missing histories can change the final normalized profile. These weights do not identify observed fracture and require no true outputs at inference. FT crack-face and box interventions run only FT.\n\n",
        "## Selection and reporting\n\nKeep the existing physical field-MSE/TRAIN-mean-MSE checkpoint criterion. Report global TRAIN-mean RMSE skill, actual STATUS-event endpoint/strut-motion errors, broader FT box, temporal/sign safeguards and costs. Positive skill means improvement over always predicting the mean;0 is the mean baseline. Keep normalized weighted training objectives separate from fixed physical selection and plot histories on log axes. Confirm promising effects across seeds 43/44 before combining or retraining curves.\n\n"]
    if a.jobs:
        text += ["## Submitted IDs\n\nThis is a submission receipt, not a live scheduler check.\n\n",pd.read_csv(a.jobs,sep="\t").to_markdown(index=False),"\n"]
    (a.out/"README.md").write_text("".join(text),encoding="utf-8")
    print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
