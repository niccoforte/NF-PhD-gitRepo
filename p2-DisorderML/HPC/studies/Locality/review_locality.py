"""Read completed single-seed locality fields; no training or curve inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANTS=("control","degree","graph","gated_graph","adjacency","geometry","cnn1","cnn2")
LABELS=("Control","Degree","GNN","Gated GNN","Adjacency","Edge geometry","CNN1","CNN2")


def verify_download(root,manifest):
    files,total,runs=0,0,set()
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        expected,name=line.split("  ",1);path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError("Download path escapes result root")
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(4194304),b""):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError("Checksum mismatch: "+name)
        files+=1;total+=path.stat().st_size;runs.add(Path(name).parts[0])
    if len(runs)!=16:raise ValueError("Expected all sixteen Locality runs")
    return dict(files=files,bytes=total,runs=len(runs),sha256="all matched")


def collect(root,suite):
    rows=[];contracts={}
    for mode in ("UT","FT"):
        for variant in VARIANTS:
            path=root/f"{suite}-{variant}-{mode}-s42"
            meta=json.loads((path/"run_metadata.json").read_text())
            metrics=json.loads((path/"results/metrics.json").read_text())
            data=json.loads((path/"model_data.json").read_text())
            if meta["status"]!="complete" or meta["checkpoint_reload"]!="passed" or metrics["evaluation_split"]!="val":
                raise ValueError("Require complete validation/reload artifacts")
            if meta["variant"]!=variant or meta["mode"]!=mode:raise ValueError("Run identity mismatch")
            for key in ("sample_ids","normalizers","node_masks"):
                value=hashlib.sha256(json.dumps(data[key],sort_keys=True).encode()).hexdigest()
                if contracts.setdefault(key,value)!=value:raise ValueError("Changed common data: "+key)
            with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
                if list(map(str,a["sample_ids"]))!=list(map(str,data["sample_ids"]["val"])):raise ValueError("IDs differ")
                pred,truth=a[f"{mode}_val_field_outputs"],a[f"{mode}_val_field_truth"]
                valid=a[f"{mode}_val_field_mask"].astype(bool)&a[f"{mode}_node_mask"][None,:,None].astype(bool)&np.isfinite(truth)
                for tag,array in (("truth",truth),("mask",valid)):
                    value=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    if contracts.setdefault((mode,tag),value)!=value:raise ValueError("Targets or masks differ")
                rmse=np.sqrt(np.mean((pred[valid].astype(float)-truth[valid])**2))
            diag=metrics["diagnostics"]["field"][mode]
            if not np.isclose(rmse,diag["rmse"],rtol=3e-6):raise ValueError("Saved RMSE does not reproduce")
            if diag["mean_field_baseline_source"]!="train_mean_field":raise ValueError("Require TRAIN-fitted mean")
            baseline=diag["mean_field_baseline_rmse"]
            if not np.isfinite(baseline) or baseline<=0 or not np.isclose(baseline,contracts.setdefault((mode,"baseline"),baseline),rtol=1e-8):raise ValueError("Mean-field denominator changed")
            if not np.isclose(diag["skill_vs_mean_field_rmse"],1-rmse/baseline,atol=1e-7):raise ValueError("Mean-relative skill differs")
            motion=pd.read_csv(path/f"results/{mode}_val_field_motion.csv")
            rows.append(dict(mode=mode,variant=variant,rmse=rmse,skill=100*diag["skill_vs_mean_field_rmse"],
                local_jump=motion.local_jump_nrmse_percent.mean(),timing=motion.jump_interval_mae.mean(),
                collapse_ratio=diag["collapse_ratio"],baseline_rmse=diag["mean_field_baseline_rmse"],
                hours=meta["training_seconds"]/3600,parameters=meta["active_parameters"],job=meta["slurm_job_id"],
                revision=meta["source_revision"],best_epoch=metrics["best_epoch"],run=str(path)))
    table=pd.DataFrame(rows)
    controls=table[table.variant=="control"].set_index("mode")
    table["change"]=100*(table.rmse/table["mode"].map(controls.rmse)-1)
    return table


def figures(table,root,suite,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,("UT","FT")):
        values=table[table["mode"]==mode].set_index("variant").loc[list(VARIANTS)]
        bars=ax.bar(range(8),values.skill,color="#42769b")
        ax.bar_label(bars,fmt="%.1f",fontsize=8,padding=2)
        ax.axhline(0,color="black",lw=.8)
        ax.set(title=mode,ylabel="RMSE reduction vs TRAIN mean (%)",xticks=range(8),xticklabels=LABELS)
        ax.tick_params(axis="x",rotation=30,labelsize=8);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Improvement beyond the average field\nZero = mean predictor; higher is better; seed42 only")
    fig.tight_layout();fig.savefig(out/"locality-mean-skill.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for col,mode in enumerate(("UT","FT")):
        values=table[table["mode"]==mode].set_index("variant").loc[list(VARIANTS)]
        axes[0,col].bar(range(8),values.change,color=["#24877c" if x<0 else "#c57860" for x in values.change])
        axes[0,col].set(title=f"{mode} field",ylabel="RMSE change versus control (%)")
        axes[1,col].plot(range(8),values.local_jump,"o-",label="Local-jump error")
        axes[1,col].set(ylabel="Local-jump RMSE / target jump RMS (%)")
        for ax in axes[:,col]:ax.set(xticks=range(8),xticklabels=[v.replace(" ","\n") for v in LABELS]);ax.grid(axis="y",alpha=.2);ax.tick_params(axis="x",labelsize=8)
    fig.suptitle("Independent field-only locality screen, seed42\nLower is better; single-seed differences require confirmation")
    fig.tight_layout();fig.savefig(out/"locality-accuracy.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,("UT","FT")):
        values=table[table["mode"]==mode].set_index("variant").loc[list(VARIANTS)]
        ax.bar(range(8),values.hours,color="#42769b");ax.set(title=mode,ylabel="Recorded training hours",xticks=range(8),xticklabels=[v.replace(" ","\n") for v in LABELS]);ax.tick_params(axis="x",labelsize=8)
    fig.tight_layout();fig.savefig(out/"locality-cost.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,("UT","FT")):
        for v in VARIANTS:
            h=pd.read_csv(root/f"{suite}-{v}-{mode}-s42/loss_history.csv")
            ax.plot(h.epoch,h.val_loss.clip(lower=1e-12),label=v)
        ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Validation field MSE (log)");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/"locality-loss.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,("UT","FT")):
        for variant,color in zip(("control","cnn1","cnn2"),("#6b6b6b","#4e9b8a","#316a9d")):
            h=pd.read_csv(root/f"{suite}-{variant}-{mode}-s42/loss_history.csv")
            for name,style in (("train_loss","--"),("val_loss","-")):
                ax.plot(h.epoch,h[name].clip(lower=1e-12),style,color=color,label=variant+" "+name)
            best=table[(table["mode"]==mode)&(table.variant==variant)].best_epoch.iloc[0]
            ax.axvline(best,color=color,alpha=.35,lw=.9,linestyle=":")
        ax.set(title=mode,yscale="log",xlabel="Epoch",ylabel="Normalized field MSE (log)");ax.legend(fontsize=7)
    fig.suptitle("Training versus validation: selected checkpoints precede the overfit tail\nDotted vertical markers = saved best epoch; validation selects, never last epoch")
    fig.tight_layout();fig.savefig(out/"locality-train-validation.png",dpi=170);plt.close(fig)
    for mode in ("UT","FT"):
        # Saved examples already contain matched truth/prediction/error, FT zoom,
        # common limits and sample/frame labels; preserve the producer's figure.
        source=root/f"{suite}-cnn2-{mode}-s42/results/field-examples.png"
        (out/f"{mode}-cnn2-examples.png").write_bytes(source.read_bytes())


def confirmed_regions(root,suite,damage_root,out):
    helper=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Loss/review_fracture_regions.py"))
    rows=[]
    for mode in ("UT","FT"):
        descriptor=json.loads((root/f"{suite}-control-{mode}-s42/model_data.json").read_text())
        ids=descriptor["sample_ids"]["val"];xy=np.asarray(descriptor["metadata"]["canonical_coords"])
        nodes,event,edges,_=helper["region_masks"](damage_root/f"MULTI-{mode}-disNodes-allD.npz",descriptor,mode,ids)
        mean=helper["training_mean"](damage_root,descriptor,mode)
        for variant in ("control","cnn1","cnn2"):
            path=root/f"{suite}-{variant}-{mode}-s42"
            with np.load(path/"results/predictions.npz",allow_pickle=False) as a:
                if list(map(str,a["sample_ids"]))!=list(map(str,ids)):raise ValueError("STATUS and Locality IDs differ")
                truth=a[f"{mode}_val_field_truth"];pred=a[f"{mode}_val_field_outputs"]
                valid=a[f"{mode}_val_field_mask"].astype(bool)&a[f"{mode}_node_mask"][None,:,None]
                measured=helper["measure_regions"](pred,truth,valid,nodes,event,edges,mean,mode,xy)
                diag=json.loads((path/"results/metrics.json").read_text())["diagnostics"]["field"][mode]
                if not np.isclose(measured[0]["mean_baseline_rmse"],diag["mean_field_baseline_rmse"],rtol=3e-6):raise ValueError("TRAIN mean differs")
                for r in measured:r.update(mode=mode,variant=variant)
                rows.extend(measured)
    table=pd.DataFrame(rows);base=table[table.variant=="control"].set_index(["mode","region"]).rmse
    table["change_percent"]=[100*(r.rmse/base.loc[r.mode,r.region]-1) for r in table.itertuples()]
    table.to_csv(out/"confirmed-fracture-regions.csv",index=False)
    helper["region_figures"](table,out)
    return table


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--suite",default="locality-261026-r1")
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/locality-study")
    p.add_argument("--download-manifest",type=Path)
    p.add_argument("--damage-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    if a.download_manifest:
        receipt=verify_download(a.root,a.download_manifest)
        (a.out/"download-verification.json").write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
    table=collect(a.root,a.suite);figures(table,a.root,a.suite,a.out)
    regional=confirmed_regions(a.root,a.suite,a.damage_root,a.out)
    text=["# Locality: independent field predictions\n\n**All16 production fits completed successfully. CNN2 is the strongest confirmation candidate for both tasks; it is not yet a multi-seed winner.** No curves or OOF banks were trained in this screen.\n\n",
        "The experiment keeps the saved793 validation specimens, train-fitted scaling, field architecture384/8/3, MSE, seed42 and stopping policy fixed. Each intervention is separate. Control is freshly trained in this same protocol; compare against it, not older independently timed runs.\n\n",
        "| Variant | Isolated change | UT RMSE change | FT RMSE change | UT / FT local-jump error (%) |\n|---|---|---:|---:|---:|\n"]
    meanings=("Original Transformer","Add degree/8 only","One pre-attention message block","Same block with small learned residual gate","Learned connected-pair attention bias","Adjacency plus initial strut direction/length bias","One masked reference-grid block","Two masked reference-grid blocks")
    for v,label,meaning in zip(VARIANTS,LABELS,meanings):
        ut=table[(table.variant==v)&(table["mode"]=="UT")].iloc[0];ft=table[(table.variant==v)&(table["mode"]=="FT")].iloc[0]
        text.append(f"| {label} | {meaning} | {ut.change:+.2f}% | {ft.change:+.2f}% | {ut.local_jump:.2f} / {ft.local_jump:.2f} |\n")
    def image(name,caption):return f"\n![{caption}](<{(a.out/name).resolve().as_posix()}>)\n\n"
    text += ["\n## Is it better than predicting the average field?\n\nSkill =100×(1−model RMSE/TRAIN-mean RMSE), evaluated on identical validation masks. Zero means the mean predictor; negative is worse. TRAIN mean is fitted separately at every node/frame/component. This tests specimen-specific accuracy, not simply a shared deformation pattern. Collapse ratio measures mean predicted/truth across-specimen standard deviation:0 indicates constant output,1 matched spread; spread alone does not prove correct predictions.\n\n",
        "| Variant | UT skill | FT skill | UT / FT variation ratio |\n|---|---:|---:|---:|\n"]
    for v,label in zip(VARIANTS,LABELS):
        ut=table[(table.variant==v)&(table["mode"]=="UT")].iloc[0];ft=table[(table.variant==v)&(table["mode"]=="FT")].iloc[0]
        text.append(f"| {label} | {ut.skill:.2f}% | {ft.skill:.2f}% | {ut.collapse_ratio:.2f} / {ft.collapse_ratio:.2f} |\n")
    text += [image("locality-mean-skill.png","Field accuracy beyond the training-mean predictor"),
        image("locality-accuracy.png","Accuracy and local-jump safeguards"),
        "## Actual fracture events\n\nSTATUS is used retrospectively for evaluation only. Evaluate both endpoints of each observed intact→failed strut at onset ±1 retained frame; relative-motion error compares the connected pair. TRAIN-mean skill below uses exactly those same events. The larger FT box is a separate, broader region.\n\n",
        "| Task | Region | Control skill | CNN2 skill | CNN2 RMSE change |\n|---|---|---:|---:|---:|\n"]
    for mode in ("UT","FT"):
        selected=regional[(regional["mode"]==mode)&(regional.variant=="cnn2")]
        for r in selected.itertuples():
            base=regional[(regional["mode"]==mode)&(regional.variant=="control")&(regional.region==r.region)].iloc[0]
            text.append(f"| {mode} | {r.region} | {base.skill_percent:.2f}% | {r.skill_percent:.2f}% | {r.change_percent:+.2f}% |\n")
    text += [image("status-fracture-regions.png","CNNs at verified fracture events and in the surrounding field"),
        "## Verdict\n\nConfirm CNN2 across seeds43/44 before combining it with loss changes or updating WINNER.md. CNN1 is a useful depth comparator. Gated graph is worth retaining for FT, while the tested ungated graph does not support adoption. Small degree/attention gains need seed confirmation. More local layers are not proven always better; the grid convolution is not physical connectivity, and global attention remains available.\n\n",
        "## Representative CNN2 predictions\n\nThe saved plots use initial disordered positions and show truth, prediction and signed error, with full UT and the FT crack-tip box. Best/worst use whole-field sample RMSE at one labelled frame; they do not summarize every fracture event.\n\n",
        image("UT-cnn2-examples.png","UT CNN2 examples"),image("FT-cnn2-examples.png","FT crack-tip CNN2 examples"),
        "## Runtime and training\n\nThese are final generators only, much cheaper than the loss study's three-fold banks plus final fits. Queue time is excluded; different stopping epochs and parameter counts contribute to runtime.\n\n",
        image("locality-cost.png","Recorded training cost"),image("locality-loss.png","Log-scale loss histories"),
        image("locality-train-validation.png","Training and validation curves for the control and CNN variants"),
        "A declining training loss with stalled/rising validation loss is overfitting; the saved best validation checkpoint is used, not the final epoch. UT shows that pattern strongly, including the original control; the CNN does not solve it. FT validation continues improving much longer. Training uses dropout and the extrema-aware training population, while validation disables dropout; a train/validation gap alone is not an overfitting test. UT has specimen-specific, spatially distributed competing localization paths, whereas FT has a fixed initial crack with concentrated responses. Local grid processing may therefore be a stronger prior for FT. These are hypotheses, not proof of the cause: target variability, normalization, optimization and one-seed noise also differ. The same weights/hyperparameters need not suit both tasks.\n\n",
        "## Verification and limits\n\nThe reader checks completed/reloaded status, exact IDs, scaler/mask agreement, identical targets and physical RMSE from saved arrays. The actual-fracture section aligns delivered STATUS and checks a reconstructed TRAIN mean against saved global baseline RMSE. This is one seed and validation only. Locked test remains untouched. Other local-jump activity metrics remain kinematic proxies. No downstream curve improvement is claimed.\n\n",
        "[Full metrics, parameter counts and run provenance](metrics.csv). Raw archives: `Z:/p2/MULTI/Dual/Transformer/locality-261026-r1-<variant>-<UT|FT>-s42`.\n"]
    table.to_csv(a.out/"metrics.csv",index=False);(a.out/"README.md").write_text("".join(text),encoding="utf-8")
    print(table[["variant","mode","change","local_jump","timing","hours"]].to_string(index=False))
    print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
