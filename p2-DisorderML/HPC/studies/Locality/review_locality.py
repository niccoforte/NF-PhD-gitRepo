"""Read completed single-seed locality fields; no training or curve inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
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
            motion=pd.read_csv(path/f"results/{mode}_val_field_motion.csv")
            rows.append(dict(mode=mode,variant=variant,rmse=rmse,skill=100*diag["skill_vs_mean_field_rmse"],
                local_jump=motion.local_jump_nrmse_percent.mean(),timing=motion.jump_interval_mae.mean(),
                hours=meta["training_seconds"]/3600,parameters=meta["active_parameters"],job=meta["slurm_job_id"],
                revision=meta["source_revision"],best_epoch=metrics["best_epoch"],run=str(path)))
    table=pd.DataFrame(rows)
    controls=table[table.variant=="control"].set_index("mode")
    table["change"]=100*(table.rmse/table["mode"].map(controls.rmse)-1)
    return table


def figures(table,root,suite,out):
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
    for mode in ("UT","FT"):
        # Saved examples already contain matched truth/prediction/error, FT zoom,
        # common limits and sample/frame labels; preserve the producer's figure.
        source=root/f"{suite}-cnn2-{mode}-s42/results/field-examples.png"
        (out/f"{mode}-cnn2-examples.png").write_bytes(source.read_bytes())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--suite",default="locality-261026-r1")
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[3]/"samples/locality-study")
    p.add_argument("--download-manifest",type=Path)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    if a.download_manifest:
        receipt=verify_download(a.root,a.download_manifest)
        (a.out/"download-verification.json").write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
    table=collect(a.root,a.suite);figures(table,a.root,a.suite,a.out)
    text=["# Locality: independent field predictions\n\n**All16 production fits completed successfully. CNN2 is the strongest confirmation candidate for both tasks; it is not yet a multi-seed winner.** No curves or OOF banks were trained in this screen.\n\n",
        "The experiment keeps the saved793 validation specimens, train-fitted scaling, field architecture384/8/3, MSE, seed42 and stopping policy fixed. Each intervention is separate. Control is freshly trained in this same protocol; compare against it, not older independently timed runs.\n\n",
        "| Variant | Isolated change | UT RMSE change | FT RMSE change | UT / FT local-jump error (%) |\n|---|---|---:|---:|---:|\n"]
    meanings=("Original Transformer","Add degree/8 only","One pre-attention message block","Same block with small learned residual gate","Learned connected-pair attention bias","Adjacency plus initial strut direction/length bias","One masked reference-grid block","Two masked reference-grid blocks")
    for v,label,meaning in zip(VARIANTS,LABELS,meanings):
        ut=table[(table.variant==v)&(table["mode"]=="UT")].iloc[0];ft=table[(table.variant==v)&(table["mode"]=="FT")].iloc[0]
        text.append(f"| {label} | {meaning} | {ut.change:+.2f}% | {ft.change:+.2f}% | {ut.local_jump:.2f} / {ft.local_jump:.2f} |\n")
    def image(name,caption):return f"\n![{caption}](<{(a.out/name).resolve().as_posix()}>)\n\n"
    text += [image("locality-accuracy.png","Accuracy and local-jump safeguards"),
        "## Verdict\n\nConfirm CNN2 across seeds43/44 before combining it with loss changes or updating WINNER.md. CNN1 is a useful depth comparator. Gated graph is worth retaining for FT, while the tested ungated graph does not support adoption. Small degree/attention gains need seed confirmation. More local layers are not proven always better; the grid convolution is not physical connectivity, and global attention remains available.\n\n",
        "## Representative CNN2 predictions\n\nThe saved plots use initial disordered positions and show truth, prediction and signed error, with full UT and the FT crack-tip box. Best/worst use whole-field sample RMSE at one labelled frame; they do not summarize every fracture event.\n\n",
        image("UT-cnn2-examples.png","UT CNN2 examples"),image("FT-cnn2-examples.png","FT crack-tip CNN2 examples"),
        "## Runtime and training\n\nThese are final generators only, much cheaper than the loss study's three-fold banks plus final fits. Queue time is excluded; different stopping epochs and parameter counts contribute to runtime.\n\n",
        image("locality-cost.png","Recorded training cost"),image("locality-loss.png","Log-scale loss histories"),
        "## Verification and limits\n\nThe reader checks completed/reloaded status, exact IDs, scaler/mask agreement, identical targets and physical RMSE from saved arrays. This is one seed and validation only. Locked test remains untouched. Local-jump activity is a kinematic fracture proxy, not confirmed STATUS. No downstream curve improvement is claimed.\n\n",
        "[Full metrics, parameter counts and run provenance](metrics.csv). Raw archives: `Z:/p2/MULTI/Dual/Transformer/locality-261026-r1-<variant>-<UT|FT>-s42`.\n"]
    table.to_csv(a.out/"metrics.csv",index=False);(a.out/"README.md").write_text("".join(text),encoding="utf-8")
    print(table[["variant","mode","change","local_jump","timing","hours"]].to_string(index=False))
    print("REPORT",a.out/"README.md")


if __name__=="__main__":main()
