"""Review saved independent Mixed loss trials; no training or locked-test access."""
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

VARIANTS = ("control", "spatial", "temporal", "both", "sudden",
            "combined_no_location", "combined_soft")
LABELS = ("MSE control", "+ spatial", "+ temporal", "+ both", "Sudden weighting",
          "Combined curve", "+ soft peak location")
TASKS = [(kind, mode) for kind in ("field", "curve") for mode in ("UT", "FT")]


def digest(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_download(root, manifest):
    count, size, runs = 0, 0, set()
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        expected, name = line.split("  ", 1)
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Checksum path escapes the result root")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b""): hasher.update(chunk)
        if hasher.hexdigest() != expected: raise ValueError("Download checksum mismatch: "+name)
        count += 1; size += path.stat().st_size; runs.add(Path(name).parts[0])
    if len(runs) != 30: raise ValueError("Expected all twelve banks and eighteen comparisons")
    return dict(files=count, bytes=size, runs=len(runs), sha256="all matched", root=root.as_posix())


def verify_ids(values, expected):
    ids = list(map(str, values))
    if ids != list(map(str, expected)) or len(ids) != len(set(ids)):
        raise ValueError("Validation IDs, order or uniqueness differ")


def error_metrics(prediction, truth, mask):
    mask = np.asarray(mask, bool) & np.isfinite(truth)
    if prediction.shape != truth.shape or not mask.any() or not np.isfinite(prediction[mask]).all():
        raise ValueError("Invalid physical prediction/target/mask")
    error = prediction[mask].astype(float)-truth[mask]
    return float(np.sqrt(np.mean(error**2)))


def field_error_decomposition(prediction, truth, mask):
    """Separate each sample/channel's mean nodal error; diagnostic, not a repair."""
    valid = np.asarray(mask, bool) & np.isfinite(truth)
    residual = np.where(valid, prediction-truth, 0.).astype(float)
    count = valid.sum(axis=1)
    mean = np.divide(residual.sum(axis=1), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    total = np.square(residual).sum()/valid.sum()
    offset = (count*mean**2).sum()/valid.sum()
    return dict(mean_nodal_error_rmse=np.sqrt(offset), centered_field_rmse=np.sqrt(max(total-offset, 0.)),
                mean_nodal_error_mse_percent=100*offset/total if total else 0.)


def relative_changes(rows):
    control = rows[rows.variant == "control"].set_index(["seed", "kind", "mode"]).rmse
    result = rows.copy()
    result["rmse_change_percent"] = [100*(r.rmse/control.loc[r.seed, r.kind, r.mode]-1)
                                     for r in rows.itertuples()]
    return result


def run_path(root, variant, seed):
    label = (f"independent-inputs-261015-mixed-s{seed}" if variant == "control" else
             f"independent-losses-261022-mixed-{variant}-s{seed}")
    return root/label


def collect_banks(root):
    rows = []
    reference = None
    for seed in (42, 43, 44):
        for variant in VARIANTS[:5]:
            label = (f"independence-261013-r1-bank-1i-s{seed}" if variant == "control" else
                     f"independent-losses-261018-bank-{variant}-s{seed}")
            path = root/label
            manifest = json.loads((path/"field_bank.json").read_text())
            status = json.loads((path/"run_metadata.json").read_text())
            if status["status"] != "complete": raise ValueError("Incomplete field bank")
            contract = (manifest["data_fingerprint"], digest(manifest["field_config"]))
            reference = contract if reference is None else reference
            if reference != contract: raise ValueError("Field-bank data/architecture differ")
            train = set(map(str, manifest["sample_ids"]["train"]))
            held = []
            for fold in manifest["folds"]:
                groups = [set(map(str, fold[k])) for k in ("fit_ids", "selection_ids", "heldout_train_ids")]
                if set.union(*groups) != train or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                    raise ValueError("Invalid training-only cross-fit partition")
                held.extend(groups[2])
            if len(held) != len(train) or set(held) != train:
                raise ValueError("OOF coverage is incomplete or duplicated")
            rows.append(dict(variant=variant, seed=seed, run=label, hours=status["training_seconds"]/3600))
    return pd.DataFrame(rows)


def collect(root):
    rows, extras, sources, checks, contracts, control_fields = [], [], [], [], {}, {}
    for seed in (42, 43, 44):
        for variant in VARIANTS:
            run = run_path(root, variant, seed)
            metrics = json.loads((run/"results/metrics.json").read_text())
            descriptor = json.loads((run/"model_data.json").read_text())
            status = json.loads((run/"run_metadata.json").read_text())
            if metrics["metadata"].get("curve_training_source") != "mixed":
                raise ValueError("All compared curve models must use the accepted Mixed policy")
            if metrics["evaluation_split"] != "val" or metrics["prediction_space"] != "physical" or metrics["run_config"]["seed"] != seed:
                raise ValueError("Expected matched saved validation evaluation")
            if status.get("status") != "complete":
                raise ValueError("Incomplete run: "+run.name)
            if variant != "control":
                config = metrics["run_config"]
                if variant.startswith("combined"):
                    valid_variant = config["curve_loss_ablation"] == variant and config["loss"] == "combined" and config["field_loss_variant"] is None
                else:
                    valid_variant = config["field_loss_variant"] == variant and config["loss"] == "mse" and config["curve_loss_ablation"] is None
                if not valid_variant: raise ValueError("Saved losses do not match the study label")
            for name in ("sample_ids", "normalizers", "node_masks", "reference_task_features"):
                value = digest(descriptor[name])
                if contracts.setdefault(name, value) != value:
                    raise ValueError("Changed data contract: "+name)
            with np.load(run/"results/predictions.npz", allow_pickle=False) as arrays:
                ids = arrays["sample_ids"].astype(str)
                verify_ids(ids, descriptor["sample_ids"]["val"])
                for kind, mode in TASKS:
                    pred, truth = (arrays[f"{mode}_val_{kind}_{key}"] for key in ("outputs", "truth"))
                    mask = np.isfinite(truth)
                    if kind == "field":
                        mask &= arrays[f"{mode}_val_field_mask"].astype(bool)
                        mask &= arrays[f"{mode}_node_mask"][None, :, None].astype(bool)
                        key = seed, mode
                        if variant == "control":
                            control_fields[key] = digest(pred)
                        if variant.startswith("combined") and digest(pred) != control_fields[key]:
                            raise ValueError("Curve-loss ablation changed frozen fields")
                    for tag, value in (("truth", truth), ("mask", mask)):
                        key = kind, mode, tag
                        if contracts.setdefault(key, digest(value)) != digest(value):
                            raise ValueError("Targets or validity changed")
                    rmse = error_metrics(pred, truth, mask)
                    diag = metrics["diagnostics"][kind][mode]
                    if not np.isclose(rmse, diag["rmse"], rtol=3e-6):
                        raise ValueError("Physical RMSE does not reproduce saved metrics")
                    baseline = diag[f"mean_{kind}_baseline_rmse"]
                    if diag[f"mean_{kind}_baseline_source"] != f"train_mean_{kind}" or baseline <= 0:
                        raise ValueError("Unverified training-mean denominator")
                    key = kind, mode, "denominator"
                    if not np.isclose(contracts.setdefault(key, baseline), baseline, rtol=1e-8):
                        raise ValueError("Comparison denominator changed")
                    rows.append(dict(variant=variant, seed=seed, kind=kind, mode=mode,
                        rmse=rmse, mean_relative_percent=100*rmse/baseline,
                        skill_percent=100*(1-rmse/baseline), collapse_ratio=diag["collapse_ratio"]))
                    suffix = "field_motion" if kind == "field" else "curve_sample_metrics"
                    table = pd.read_csv(run/f"results/{mode}_val_{suffix}.csv")
                    verify_ids(table.sample_id, ids)
                    if kind == "field":
                        for name, value in field_error_decomposition(pred, truth, mask).items():
                            extras.append(dict(variant=variant, seed=seed, kind=kind, mode=mode,
                                               metric=name, value=value, n=len(ids)))
                        for name in ("spatial_rmse", "local_jump_nrmse_percent", "temporal_nrmse_percent",
                                     "jump_interval_mae", "u2_jump_sign_accuracy", "tip_u2_sign_accuracy", "activity_node_jaccard"):
                            if name in table:
                                extras.append(dict(variant=variant, seed=seed, kind=kind, mode=mode,
                                    metric=name, value=table[name].mean(), n=table[name].notna().sum()))
                    else:
                        x = arrays[f"{mode}_curve_x_values"].reshape(-1)
                        for name, value in dict(
                            peak_height_percent=100*abs(table.peak_error).mean()/abs(table.true_peak).mean(),
                            peak_location_percent=100*abs(table.peak_x_error).mean()/np.ptp(x),
                            work_percent=100*abs(table.energy_error).mean()/abs(table.true_energy).mean()).items():
                            extras.append(dict(variant=variant, seed=seed, kind=kind, mode=mode,
                                               metric=name, value=value, n=len(table)))
                if variant in ("spatial", "temporal", "both", "sudden"):
                    with np.load(run/"fixed_readout_curves.npz", allow_pickle=False) as frozen:
                        verify_ids(frozen["sample_ids"], ids)
                        for mode in ("UT", "FT"):
                            truth = arrays[f"{mode}_val_curve_truth"]
                            if not np.array_equal(frozen[f"{mode}_truth"], truth):
                                raise ValueError("Frozen curve probe targets differ")
                            value = error_metrics(frozen[f"{mode}_outputs"], truth, np.isfinite(truth))
                            extras.append(dict(variant=variant, seed=seed, kind="curve", mode=mode,
                                               metric="frozen_readout_rmse", value=value, n=len(ids)))
            meta = metrics["metadata"]
            sources.append(dict(variant=variant, seed=seed, run=run.name, job=meta.get("slurm_job_id"),
                revision=meta.get("source_revision"), hours=meta["training_seconds"]/3600,
                field_bank_sha256=meta.get("field_bank_sha256")))
            checks.append(run.name+": complete; validation IDs/scales/masks/truth/RMSE checked")
            print(checks[-1], flush=True)
    return relative_changes(pd.DataFrame(rows)), pd.DataFrame(extras), pd.DataFrame(sources), checks


def comparison_plots(rows, extra, sources, banks, out):
    mean_predictor_plots(rows,out)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (kind, mode) in zip(axes.flat, TASKS):
        for x, variant in enumerate(VARIANTS[1:]):
            v = rows.query("variant == @variant and kind == @kind and mode == @mode").rmse_change_percent
            ax.bar(x, v.mean(), color="#258675" if v.mean() < 0 else "#c96f56", alpha=.75)
            ax.errorbar(x, v.mean(), yerr=v.std(), color="black", capsize=3)
            ax.scatter(np.full(3, x)+np.linspace(-.13, .13, 3), v, color="black", s=15)
        ax.axhline(0, color="black", lw=.7)
        ax.set(title=f"{mode} {kind}", ylabel="RMSE change versus matched MSE (%)",
               xticks=range(6), xticklabels=[s.replace(" ", "\n") for s in LABELS[1:]])
        ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Independent models, Predicted + True curves: negative means better\nDots: seeds 42/43/44; bars: mean; whiskers: seed standard deviation")
    fig.tight_layout(); fig.savefig(out/"accuracy-changes.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, mode in enumerate(("UT", "FT")):
        for ax, name, label in zip(axes[row],
            ("local_jump_nrmse_percent", "jump_interval_mae", "peak_location_percent"),
            ("Local jump error / target jump RMS (%)", "Jump timing error (frame intervals)", "Peak location error / loading span (%)")):
            for x, variant in enumerate(VARIANTS):
                v = extra.query("variant == @variant and mode == @mode and metric == @name").value
                ax.errorbar(x, v.mean(), yerr=v.std(), marker="o", capsize=3, color="#336c97")
            ax.set(title=mode, ylabel=label, xticks=range(7),
                   xticklabels=[s.replace(" ", "\n") for s in LABELS]); ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fracture-sensitive safeguards: lower is better; motion events are proxies, not damage labels")
    fig.tight_layout(); fig.savefig(out/"local-and-peak-errors.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("UT", "FT")):
        for index, (metric, label) in enumerate((("mean_nodal_error_rmse", "Mean nodal error"), ("centered_field_rmse", "Error after removing that mean"))):
            for x, variant in enumerate(VARIANTS[:5]):
                v = extra.query("variant == @variant and mode == @mode and metric == @metric").value
                ax.errorbar(x+index*.15-.075, v.mean(), yerr=v.std(), marker="o", capsize=3,
                            color=("#c96f56", "#336c97")[index], label=label if x == 0 else None)
        ax.set(title=mode, ylabel="Diagnostic error component (mm)", xticks=range(5), xticklabels=LABELS[:5]); ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Why can local motion and absolute displacement disagree?\nPer-specimen/frame/component mean removed only for this diagnostic, never for reported predictions")
    fig.tight_layout(); fig.savefig(out/"field-error-decomposition.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("UT", "FT")):
        for seed in (42, 43, 44):
            reference = rows.query("variant == 'control' and kind == 'curve' and mode == @mode and seed == @seed").rmse.iloc[0]
            adapted = rows.query("variant in ['spatial','temporal','both','sudden'] and kind == 'curve' and mode == @mode and seed == @seed").set_index("variant")
            frozen = extra.query("metric == 'frozen_readout_rmse' and mode == @mode and seed == @seed").set_index("variant")
            for x, variant in enumerate(VARIANTS[1:5]):
                ax.scatter(x-.1, 100*(frozen.loc[variant, "value"]/reference-1), color="#c96f56", label="Frozen curve model" if seed == 42 and x == 0 else None)
                ax.scatter(x+.1, adapted.loc[variant, "rmse_change_percent"], color="#258675", label="Retrained curve model" if seed == 42 and x == 0 else None)
        ax.axhline(0, color="black", lw=.7); ax.set(title=mode, ylabel="Curve RMSE change versus MSE (%)", xticks=range(4), xticklabels=LABELS[1:5]); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/"frozen-versus-adapted.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, table, variants, labels, title in ((axes[0], banks, VARIANTS[:5], LABELS[:5], "Field bank: 3 OOF folds + final generators"),
            (axes[1], sources, VARIANTS, LABELS, "Independent UT + FT curve fits")):
        for x, variant in enumerate(variants):
            v = table.query("variant == @variant").hours
            ax.bar(x, v.mean(), color="#336c97", alpha=.7); ax.scatter(np.full(3, x), v, color="black", s=14)
        ax.set(ylabel="Recorded training runtime (hours)", xticks=range(len(variants)), xticklabels=[s.replace(" ", "\n") for s in labels], title=title)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(out/"curve-fit-cost.png", dpi=170); plt.close(fig)


def mean_predictor_plots(rows,out):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(kind,mode) in zip(axes.flat,TASKS):
        for x,variant in enumerate(VARIANTS):
            values=rows[(rows.variant==variant)&(rows.kind==kind)&(rows["mode"]==mode)].skill_percent
            ax.bar(x,values.mean(),color="#377f95" if values.mean()>0 else "#c57860",alpha=.8)
            ax.errorbar(x,values.mean(),yerr=values.std(),color="black",capsize=3)
            ax.scatter(x+np.linspace(-.12,.12,len(values)),values,color="black",s=10)
        ax.axhline(0,color="black",lw=.8);ax.set(title=f"{mode} {kind}",ylabel="RMSE reduction vs TRAIN mean (%)",
            xticks=range(len(VARIANTS)),xticklabels=[v.replace(" ","\n") for v in LABELS]);ax.tick_params(axis="x",labelsize=7);ax.grid(axis="y",alpha=.2)
    fig.suptitle("How much is learned beyond the average output?\nHigher is better; zero=mean predictor, negative=worse; whiskers=seed SD")
    fig.tight_layout();fig.savefig(out/"mean-predictor-skill.png",dpi=170);plt.close(fig)


def galleries(root, rows, out, data_root):
    from resources.MLstudy_reports import curve_examples
    for mode in ("UT","FT"):
        best=rows[(rows.kind=="curve")&(rows["mode"]==mode)].groupby("variant").rmse.mean().idxmin()
        curve_examples(run_path(root,best,42),mode,out,best,data_root)


def loss_histories(root, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, group in zip(axes.flat, ("field_UT", "field_FT", "curve_UT", "curve_FT")):
        for variant in VARIANTS:
            if group.startswith("field"):
                if variant.startswith("combined"): continue
                bank = (f"independence-261013-r1-bank-1i-s42" if variant == "control" else
                        f"independent-losses-261018-bank-{variant}-s42")
                path = root/bank/"groups"/group/"loss_history.csv"
            else:
                path = run_path(root, variant, 42)/"loss_history.csv"
            table = pd.read_csv(path)
            if "training_group" in table: table = table[table.training_group == group]
            ax.plot(table.epoch, table.val_loss.clip(lower=1e-12), label=variant)
        ax.set(title=group, yscale="log", xlabel="Epoch", ylabel="Validation training objective (log)")
        ax.legend(fontsize=6); ax.grid(alpha=.2)
    fig.suptitle("Loss histories, fixed seed 42: shapes diagnose training; differently weighted totals do not rank methods")
    fig.tight_layout(); fig.savefig(out/"loss-histories.png", dpi=170); plt.close(fig)


def extended_examples(root,rows,out,data_root):
    from resources.MLstudy_reports import field_examples,percent_error
    helper=runpy.run_path(str(Path(__file__).with_name("review_fracture_regions.py")))
    selections={}
    for mode in ("UT","FT"):
        best=rows[(rows.kind=="field")&(rows["mode"]==mode)&rows.variant.isin(VARIANTS[:5])].groupby("variant").rmse.mean().idxmin()
        path=run_path(root,best,42);descriptor=json.loads((path/"model_data.json").read_text())
        mean=helper["training_mean"](data_root,descriptor,mode)
        selections[mode]=field_examples(path,mode,out,best,data_root,mean)
    (out/"example-selection.json").write_text(json.dumps(selections,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,group in zip(axes.flat,("field_UT","field_FT","curve_UT","curve_FT")):
        kind,mode=group.split("_")
        eligible=rows[(rows.kind==kind)&(rows["mode"]==mode)]
        if kind=="field":eligible=eligible[eligible.variant.isin(VARIANTS[:5])]
        best=eligible.groupby("variant").rmse.mean().idxmin()
        for index,variant in enumerate(dict.fromkeys(("control",best))):
            if kind=="field":
                bank=f"independence-261013-r1-bank-1i-s42" if variant=="control" else f"independent-losses-261018-bank-{variant}-s42"
                path=root/bank/"groups"/group/"loss_history.csv"
            else:path=run_path(root,variant,42)/"loss_history.csv"
            h=pd.read_csv(path)
            if "training_group" in h:h=h[h.training_group==group]
            color=plt.get_cmap("tab10")(index)
            for column,style in (("train_loss","--"),("val_loss","-")):
                ax.plot(h.epoch,h[column].where(h[column]>0),style,color=color,label=variant+" "+column)
        ax.set(title=group,yscale="log",xlabel="Epoch",ylabel="Saved objective (log)");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("Control and descriptive global candidates: seed 42 train/validation histories\nTraining totals with different weights cannot rank methods")
    fig.tight_layout();fig.savefig(out/"loss-train-validation.png",dpi=170);plt.close(fig)


def component_figures(root,out):
    fig,axes=plt.subplots(2,3,figsize=(14,7.5));rows=[]
    for row,mode in enumerate(("UT","FT")):
        for col,variant in enumerate(("spatial","temporal","both")):
            path=root/f"independent-losses-261018-bank-{variant}-s42/groups/field_{mode}/loss_history.csv"
            history=pd.read_csv(path);ax=axes[row,col]
            total=history[f"val_field_components_displacement_{mode}"].to_numpy().copy()
            ax.plot(history.epoch,history.val_loss.where(history.val_loss>0),color="black",label="Total validation objective")
            ax.plot(history.epoch,total,color="#367996",label="Displacement MSE")
            for term,color in (("spatial","#ae754d"),("temporal","#4f9878")):
                if variant not in (term,"both"):continue
                raw=history[f"val_field_components_{term}_{mode}"].to_numpy();weighted=.1*raw;total+=weighted
                ax.plot(history.epoch,np.where(weighted>0,weighted,np.nan),color=color,label=f"0.1 × {term} term")
                rows.extend(dict(mode=mode,variant=variant,epoch=int(e),term=term,raw_loss=r,coefficient=.1,weighted_loss=w)
                    for e,r,w in zip(history.epoch,raw,weighted))
            if not np.allclose(total,history.val_loss,rtol=1e-5,atol=1e-7):raise ValueError("Saved component losses do not reconstruct total")
            ax.set(title=f"{mode}: {variant}",yscale="log",xlabel="Epoch",ylabel="Validation objective contribution (log)");ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle("How custom field losses contribute to training\nWeighted components reconstruct the total; values do not rank different formulations or measure gradients")
    fig.tight_layout();fig.savefig(out/"field-loss-components.png",dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/"loss-component-values.csv",index=False)


def write_report(root, out, rows, extra, sources, banks, checks):
    from resources.MLstudy_reports import figure as embed, architecture_figure, regional_mean_figures
    def image(name,caption): return embed(out,name,caption)
    rows=rows.copy();rows["mse_skill_percent"]=100*(1-(1-rows.skill_percent/100)**2)
    architecture_figure(out,"Same independent field/curve models\nChanged training losses only\nNo universal replacement adopted",curve_stage=True)
    regional=out/"status-fracture-regions.csv"
    if regional.exists():regional_mean_figures(pd.read_csv(regional),out)
    text=["# Independent custom-loss study\n\n## Recap, principal findings and verdict\n\n",
        "Case1(i): four separately fitted Transformers, Predicted+True curve training and predicted-field deployment. Three seeds 42/43/44, identical 793 validation specimens. **Keep the accepted Mixed MSE baseline.** UT spatial is a modest task-specific candidate; FT spatial at 0.1 strongly worsens global error despite tiny confirmed-fracture gains. Neither full combined curve preset consistently improves accuracy. No locked-test evaluation.\n\n",
        image("architecture-comparison.png","Pre-study control and unchanged post-study architecture recommendation"),
        "## Global performance versus the study control\n\nNegative RMSE change is better. Compare each variant with its same-seed MSE control. Whiskers are seed standard deviations, not confidence intervals. Curve-only custom losses leave field predictions identical.\n\n",
        image("accuracy-changes.png","Changes in global field and curve accuracy"),
        "## Global performance versus the training mean\n\nRMSE skill=100×(1−model RMSE/TRAIN-mean RMSE). TRAIN means preserve each field node/frame/component or curve point; both predictions use identical validation entries. Zero equals mean-predictor error; negative is worse. This is not percent of maximum displacement/load. MSE skill is a different quantity, retained in the numerical companion.\n\n",
        image("mean-predictor-skill.png","Accuracy beyond average field and curve outputs"),
        image("UT-mean-field-examples.png","UT TRAIN-mean field evolution"),
        image("FT-mean-field-examples.png","FT TRAIN-mean field evolution")]
    if regional.exists():
        text += ["## STATUS fracture regions versus the study control\n\nActual verified intact→failed strut labels define retrospective evaluation only. Evaluate both endpoints and signed connected-pair motion at onset ±1 retained frame. Require known adjacent STATUS states; retain unknown masks. All 793 validation specimens have observed events:67,949 UT and 6,306 FT strut-onset entries; windows contain 369,304 UT and 30,308 FT node/frame entries. The larger FT box includes its present nodes across valid frames; these are overlapping scales, not disjoint partitions.\n\n",
            image("status-fracture-regions.png","Confirmed failure endpoints versus the broader field"),
            "**The FT spatial trade-off:** endpoint error −1.07%, failing-strut relative-motion error −0.47%, but larger-box error +35.25% and global error +123.12%. UT spatial improves endpoints 3.25% and relative motion 4.02%. These are mean paired changes across three seeds. A tiny narrow-region gain does not justify broad deterioration.\n\n",
            "## STATUS fracture regions versus the training mean\n\nApply exactly those same regions to a training-only mean field. Its reconstructed global baseline must reproduce saved metrics before regional skill is accepted. Positive is better than the mean; this is distinct from target-motion-normalized error.\n\n",
            image("fracture-mean.png","Confirmed-fracture improvement beyond the average field")]
    text += ["## Best and worst predictions\n\nEach output independently uses its lowest mean-RMSE variant, fixed seed 42, as a descriptive example. Field and curve galleries can therefore show different variants, labelled in their titles; they are not one assembled deployable model or an adopted default. Rank specimens by complete valid output RMSE. Field panels show U2 at the largest observed increment in the displayed region, using actual initial disordered coordinates. UT shows the body; FT zooms at the crack tip. Percent error is 100×|prediction−truth|/|truth|: grey masks tiny truth below 1% of training-mean RMS, P95 only caps the colour scale; aggregate metrics remain unfiltered. Curves retain signed residual bars and a separate percentage-error view.\n\n",
        image("UT-field-examples.png","UT truth prediction signed error and percentage error"),image("FT-field-examples.png","FT crack-tip truth prediction signed error and percentage error"),
        image("UT-curve-examples.png","UT truth/prediction/error/percentage curves"),image("FT-curve-examples.png","FT truth/prediction/error/percentage curves")]
    text += ["## Training time\n\nField banks include three training-only OOF folds plus final generators. Curve costs are separate; queue time is excluded. These are observed training costs, not controlled inference benchmarks.\n\n",
        image("curve-fit-cost.png","Training cost of field banks and downstream curve fits"),
        "## Loss histories\n\nEach loss figure has a different role: validation-only compares all interventions; selected train/validation diagnoses overfitting; weighted-component plots show which terms drive the objective. They are complementary, not repeat exports. Different weighted totals do not rank different formulations. The checkpoint criterion remains fixed physical error; plots use saved positive entries on logarithmic axes.\n\n",
        image("loss-histories.png","Validation-only loss histories")]
    if (out/"loss-train-validation.png").exists():text.append(image("loss-train-validation.png","Selected training and validation histories"))
    if (out/"field-loss-components.png").exists():text.append(image("field-loss-components.png","Weighted custom field-loss components and total objectives"))
    text += ["## What the objectives measure\n\nBaseline is masked normalized field MSE and ordinary curve MSE. Spatial adds 0.1 times node-degree-balanced signed neighbour-motion error; temporal adds 0.1 times adjacent-frame displacement-increment error; both adds each. Differences are calculated in physical displacement using fixed training-derived scales. Sudden weighting uses target temporal abruptness plus local non-affine motion, gain 1, with no derivative penalties. It is not displacement-magnitude weighting.\n\n",
        "Changes are not reported only as global RMSE: fixed local-jump magnitude, timing/sign, actual fracture endpoints/strut motion, hard curve peak height/location and work remain in the safeguards. Training custom-loss components are diagnostics; differently weighted totals are not comparable scores.\n\n",
        image("local-and-peak-errors.png","Fixed local motion and curve peak safeguards"),
        "**Why smaller FT spatial coefficients?** The tested spatial coefficient is 0.1. FT jump scales are about 0.212/0.243mm for U1/U2 while a late nodal scale reaches 5.315mm. The illustrative 0.1×(nodal_scale/jump_scale)² reaches 47.86 in FT U2 versus 3.41 in UT U2. This is not a measured gradient ratio; it shows that equal coefficients need not mean equal physical emphasis. Propose five FT-only final-generator fits at coefficients 0, 0.001, 0.003, 0.01 and 0.1, using seed 42 and the same trainer, plain architecture, split, stopping and global/event/box metrics. Fresh 0 and 0.1 fits bridge the earlier field-bank experiment, avoiding a cross-trainer comparison. Confirm useful settings across seeds 43/44 before OOF/Mixed curve training. No larger coefficient is justified by the current severe broad-field cost. See the submission receipt below for execution state.\n\n",
        "**Temporal verdict:** deprioritise the unchanged0.1 term; it is not a universal improvement. It matches increments, whereas fracture weighting targets abrupt changes of increments; those answer different questions. Do not add second differences or divide by uncertain specimen-exact times automatically. If revisited, first check a smaller task-specific coefficient on final generators, independently of spatial/CNN; redesign only when a controlled failure identifies what is missing.\n\n",
        "**Curve verdict:** keep MSE for the current workflow and stop repeating the unchanged full combined preset. FT combined-without-location improves peak height/location at a cost to global/work error; the corrected soft term does not improve hard location over that preset. Historical true-field/shared-encoder tests also worsen global RMSE 17–18% UT / 30–31% FT, so noisy predicted inputs are not the sole explanation. If peaks still need work after field improvement, test MSE plus one modest train-scaled peak-height OR soft-location term, disabling zone/derivative/work additions. Isolate it with independent true-field inputs before Mixed/predicted-field deployment. This is deferred, not newly submitted.\n\n",
        "## Further diagnostics and evidence\n\nThe FT spatial error persists after removing each sample/frame/component's mean nodal error. This rules out a uniform offset as the entire explanation; it does not establish a causal optimization mechanism. Connectivity agrees with the bounded INP audit. These models are independent, so UT/FT competing shared gradients cannot explain this result.\n\n",
        image("field-error-decomposition.png","Mean nodal and centred error decomposition"),
        image("frozen-versus-adapted.png","Frozen curve readout versus freshly adapted curves"),
        "[Historical true-field curve-loss check](true-field-curves.md) is a separate input-source diagnostic, not a Case1(i) oracle result. [Full metrics](metrics.csv), [fracture-region errors/denominators/counts](status-fracture-regions.csv), [motion and curve safeguards](safeguards.csv), [runtime and provenance](runs.csv), [field banks](field-banks.csv), [verification](verification.json). No model is retrained by this report.\n"]
    from resources.MLstudy_reports import submission_note
    text.append(submission_note(out,"spatial-balance-jobs.tsv","FT spatial coefficients"))
    (out/"README.md").write_text("".join(text),encoding="utf-8")
    rows.to_csv(out/"metrics.csv",index=False);extra.to_csv(out/"safeguards.csv",index=False)
    sources.to_csv(out/"runs.csv",index=False);banks.to_csv(out/"field-banks.csv",index=False)
    (out/"verification.json").write_text(json.dumps(dict(checks=checks,evaluation="val",run_count=len(sources)),indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3]/"samples/loss-study")
    p.add_argument("--verify-download-manifest", type=Path,
                   help="Optional full remote sha256sum manifest for the thirty downloads")
    p.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    args = p.parse_args(argv); args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_download_manifest:
        receipt = verify_download(args.run_root, args.verify_download_manifest)
        (args.output/"download-verification.json").write_text(json.dumps(receipt, indent=2))
        print("DOWNLOAD_VERIFIED", receipt, flush=True)
    rows, extra, sources, checks = collect(args.run_root)
    banks = collect_banks(args.run_root)
    # Keep the verified numerical results if a plotting/backend problem occurs.
    rows.to_csv(args.output/"metrics.csv", index=False)
    extra.to_csv(args.output/"safeguards.csv", index=False)
    sources.to_csv(args.output/"runs.csv", index=False)
    banks.to_csv(args.output/"field-banks.csv", index=False)
    comparison_plots(rows, extra, sources, banks, args.output)
    galleries(args.run_root, rows, args.output,args.data_root); loss_histories(args.run_root, args.output)
    runpy.run_path(str(Path(__file__).with_name("review_fracture_regions.py")))["main"](
        ["--root",str(args.run_root),"--damage-root",str(args.data_root),"--out",str(args.output)])
    extended_examples(args.run_root,rows,args.output,args.data_root)
    component_figures(args.run_root,args.output)
    write_report(args.run_root, args.output, rows, extra, sources, banks, checks)
    print(rows.groupby(["variant", "kind", "mode"]).rmse_change_percent.mean().to_string())
    print("REPORT", args.output/"README.md")


if __name__ == "__main__":
    main()
