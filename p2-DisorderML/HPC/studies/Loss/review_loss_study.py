"""Review saved independent Mixed loss trials; no training or locked-test access."""
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


def galleries(root, rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, mode in enumerate(("UT", "FT")):
        means = rows.query("kind == 'curve' and mode == @mode").groupby("variant").rmse.mean()
        best = means.idxmin()
        with np.load(run_path(root, best, 42)/"results/predictions.npz", allow_pickle=False) as arrays:
            y, p = arrays[f"{mode}_val_curve_truth"], arrays[f"{mode}_val_curve_outputs"]
            x = arrays[f"{mode}_curve_x_values"].reshape(-1)
            errors = np.sqrt(np.mean((p-y)**2, axis=1)); picks = (errors.argmin(), errors.argmax())
            for col, sample in enumerate(picks):
                ax = axes[row, col]; err = ax.twinx()
                err.bar(x, p[sample]-y[sample], width=np.ptp(x)/len(x), alpha=.18, color="#b15036", label="Signed error")
                err.set_ylabel("Signed error (MPa)" if mode == "UT" else "Signed error (N)", color="#b15036")
                limit = max(float(np.abs(p[sample]-y[sample]).max()), 1e-9)
                err.set_ylim(-1.1*limit, 1.1*limit)
                ax.plot(x, y[sample], color="black", label="Truth"); ax.plot(x, p[sample], "--", color="#247b9e", label="Prediction")
                ax.set(title=f"{mode} {best}; {'best' if col == 0 else 'worst'} ID {arrays['sample_ids'][sample]}",
                    xlabel="Strain" if mode == "UT" else "Applied displacement (mm)", ylabel="Stress (MPa)" if mode == "UT" else "Force (N)")
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Curve examples: lowest mean-RMSE variant per task, fixed seed42; rank by sample RMSE")
    fig.tight_layout(); fig.savefig(out/"curve-examples.png", dpi=170); plt.close(fig)
    for mode in ("UT", "FT"):
        best = rows.query("kind == 'field' and mode == @mode and variant in ['control','spatial','temporal','both','sudden']").groupby("variant").rmse.mean().idxmin()
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        with np.load(run_path(root, best, 42)/"results/predictions.npz", allow_pickle=False) as a:
            y, p, xy = a[f"{mode}_val_field_truth"], a[f"{mode}_val_field_outputs"], a["canonical_coords"]
            present = a[f"{mode}_node_mask"].astype(bool)
            valid = a[f"{mode}_val_field_mask"].astype(bool) & present[None, :, None]
            e = np.sqrt(np.nanmean(np.where(valid, (p-y)**2, np.nan), axis=(1, 2)))
            picks = (e.argmin(), e.argmax()); channel = 21  # U2, retained frame index10, same across examples.
            fields = y[list(picks), :, channel][:, present]
            lo, hi = fields.min(), fields.max()
            limit = max(float(np.abs((p-y)[list(picks), :, channel][:, present]).max()), 1e-9)
            for row, sample in enumerate(picks):
                for col, (name, value) in enumerate((("Truth", y[sample, :, channel]), ("Prediction", p[sample, :, channel]), ("Signed error", p[sample, :, channel]-y[sample, :, channel]))):
                    ax = axes[row, col]
                    image = ax.scatter(xy[present, 0], xy[present, 1], c=value[present], s=16,
                        cmap="coolwarm" if col == 2 else "viridis", vmin=-limit if col == 2 else lo, vmax=limit if col == 2 else hi)
                    ax.set(title=f"{name}; {'best' if row == 0 else 'worst'} ID {a['sample_ids'][sample]}", aspect="equal", xlabel="Reference x (mm)", ylabel="Reference y (mm)")
                    if mode == "FT":
                        origin, scale = xy.min(axis=0), np.ptp(xy[:, 0])/200
                        box = np.array([[96.6, 54], [173.6, 136]])*scale+origin
                        ax.set_xlim(*box[:, 0]); ax.set_ylim(*box[:, 1])
                    fig.colorbar(image, ax=ax, label="U2 (mm)")
            frame = a[f"{mode}_frame_values"].reshape(-1)[10]
        fig.suptitle(f"{mode} field: {best}, seed42; U2 at recorded coordinate {frame:g}\nReference-position view; shared truth/prediction and symmetric error scales")
        fig.tight_layout(); fig.savefig(out/f"{mode}-field-examples.png", dpi=170); plt.close(fig)


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
    fig.suptitle("Loss histories, fixed seed42: shapes diagnose training; differently weighted totals do not rank methods")
    fig.tight_layout(); fig.savefig(out/"loss-histories.png", dpi=170); plt.close(fig)


def write_report(root, out, rows, extra, sources, banks, checks):
    def figure(name, caption): return f"\n![{caption}](<{(out/name).resolve().as_posix()}>)\n"
    text = ["# Independent custom-loss study\n", "Case1(i): four separately trained Transformers; **Predicted + True** curve training, predicted-field inference. Seeds42/43/44, identical793 validation specimens. No locked-test evaluation.\n",
        "## What changed\n", "| Variant | Field objective | Curve objective |\n|---|---|---|\n| Control | Masked MSE | MSE |\n| Spatial | MSE +0.1 signed neighbour-motion loss | MSE |\n| Temporal | MSE +0.1 adjacent-frame increment loss | MSE |\n| Both | MSE +0.1 each | MSE |\n| Sudden weighting | Specimen/node/frame target weighting, gain1; no added derivative terms | MSE |\n| Combined curve | Frozen control fields | Combined loss, location disabled |\n| Soft location | Same frozen control fields | Same combined loss +0.02 soft/soft location, beta20 |\n",
        "The derivative terms reconstruct physical displacements before taking differences. Sudden weighting uses abrupt changes in increments and local non-affine motion; it is not displacement-magnitude or a fixed crack-region weighting. Coefficients are controlled presets, not tuned optima. Each new field source has three training-only OOF folds and final independent generators; its curve fits are retrained. Curves never update fields.\n",
        "## Main findings\n", "Values below are mean paired percentage changes in RMSE versus each seed's MSE control; **negative is better**. Parentheses show seed standard deviation, not confidence intervals.\n",
        "| Change | UT field | FT field | UT curve | FT curve |\n|---|---:|---:|---:|---:|\n"]
    for variant, label in zip(VARIANTS[1:], LABELS[1:]):
        values = []
        for kind, mode in TASKS:
            v = rows.query("variant == @variant and kind == @kind and mode == @mode").rmse_change_percent
            values.append(f"{v.mean():+.2f}% ({v.std():.2f})")
        text.append("| "+label+" | "+" | ".join(values)+" |\n")
    text.append(figure("accuracy-changes.png", "Changes in global field and curve accuracy"))
    skills = []
    for kind, mode in TASKS:
        v = rows.query("variant == 'control' and kind == @kind and mode == @mode").skill_percent
        skills.append(f"{mode} {kind}: {v.mean():.2f}%")
    text.append("MSE-control RMSE reductions versus the training-mean predictor: **"+"; ".join(skills)+"**. Higher is better;100% would mean zero error,0% no improvement over the mean, and a negative value worse than that reference. These are not percentages of maximum displacement or force.\n")
    text += ["**Verdict:** retain the existing Mixed MSE baseline. Spatial is a modest UT-specific candidate, not a common UT/FT improvement. Spatial and both strongly deteriorate FT absolute fields at the tested0.1 coefficient. Temporal is inconsistent for UT and worsens FT; sudden weighting is close to the baseline with mixed seed-level effects. Neither combined curve formulation consistently improves global curve accuracy. Inspect the peak/work table before deciding whether a specialized curve objective warrants retuning. No new loss is a universal winner.\n",
        "The lower FT curve error for spatial/both does not rescue their poor FT field accuracy. Curve models were freshly adapted, so this is not curve-gradient interference with fields: all four models train independently. A curve reads particular patterns rather than the unweighted field average. The frozen-readout comparison tests sensitivity to the changed field source; these results alone do not identify a causal mechanism.\n"]
    text += ["## Absolute position versus local motion\n", "A spatial-difference loss is insensitive to a common displacement offset. The diagnostic below splits squared error into each specimen/frame/component's mean nodal error and the residual after removing that mean. Their **squared** values add to whole-field MSE. This is an explanation tool, not a correction applied to predictions, and does not establish a causal training mechanism.\n",
             figure("field-error-decomposition.png", "Mean nodal and centered field error")]
    text += ["## Fracture-sensitive checks\n", "These fixed quantities make different training objectives comparable. Local-jump error is normalized by the target jump RMS. Jump timing is in recorded frame intervals, not continuous time or confirmed damage onset. Peak location uses the hard reported maximum; training uses the selected smooth surrogate.\n",
        figure("local-and-peak-errors.png", "Local-motion and peak-location safeguards"),
        "| Variant | UT local jump error (%) | FT local jump error (%) | UT peak height / location / work (%) | FT peak height / location / work (%) |\n|---|---:|---:|---|---|\n"]
    for variant, label in zip(VARIANTS, LABELS):
        values = []
        for mode in ("UT", "FT"):
            selected = extra.query("variant == @variant and mode == @mode and metric == 'local_jump_nrmse_percent'")
            values.append(f"{selected.value.mean():.2f}")
        for mode in ("UT", "FT"):
            part = extra[(extra.variant == variant) & (extra["mode"] == mode)]
            values.append(" / ".join(f"{part.loc[part.metric == metric, 'value'].mean():.2f}"
                                     for metric in ("peak_height_percent", "peak_location_percent", "work_percent")))
        text.append("| "+label+" | "+" | ".join(values)+" |\n")
    text += ["\n**Candidates worth retaining:** UT spatial reduces the local-jump ratio from69.97% to67.87%, alongside its1.47% global-RMSE reduction, but does not improve jump timing. FT sudden weighting reduces local-jump error from20.19% to19.84% and timing from3.60 to3.53 frame intervals, with global RMSE0.82% worse on average (all three seeds within2%). These are modest, task-specific effects, not solved fracture paths.\n",
        "**Confirmed fracture-event follow-up:** [STATUS fracture-region review](fracture-regions.md) uses the actual delivered labels to evaluate endpoints and relative motion around first observed strut failure. It is a retrospective validation diagnostic, not new training or a presumed crack path. This finer region can behave differently from the larger FT box or the whole specimen.\n",
        "[True-field curve-loss check](true-field-curves.md) reviews the already completed historical shared-curve runs. It isolates input quality from the loss comparison but is not a new independent Case1(i) experiment.\n",
        "**Curve-loss verdict:** FT combined loss without location improves mean peak-height and hard-location errors (3.11→2.67% of mean true peak;3.61→3.21% of loading span), but work error worsens3.77→4.23% and global curve RMSE rises4.53%. Adding the corrected soft-location term does not improve hard peak location over that no-location loss in either task at this preset. UT combined-loss peak/work outcomes do not support adoption. This tests the saved coefficients/beta and unchanged RMSE checkpoint selection, not every possible custom loss or a fracture-selected checkpoint policy.\n",
        "The FT spatial deterioration persists after subtracting per-frame mean nodal error, so a common translation offset alone does not explain it. The inspected spatial graph has2259 FT edges, matching the validated initial topology; these results do not establish a new connectivity defect. Loss scaling, optimization and representation remain possible causes requiring a controlled follow-up, rather than a post-hoc claim.\n"]
    text += ["\n## Does a better field help a frozen curve model?\n", "Orange keeps the original Mixed curve weights; green retrains curves on the new field bank. A field-source change can shift the input distribution. Neither comparison uses true fields at deployment.\n",
        figure("frozen-versus-adapted.png", "Frozen and retrained downstream curve comparison"),
        "## Representative predictions\n", "Each task uses its lowest mean-RMSE variant, fixed seed42. These are descriptive validation leaders, not automatically adopted defaults. Best is left/top and worst right/bottom as labelled; samples are ranked by whole-output RMSE. Field views show truth, prediction and signed error, with the FT production crack-tip box. Curve bars show signed prediction-minus-truth error on a separate labelled axis.\n",
        figure("UT-field-examples.png", "UT field truth prediction and signed error"), figure("FT-field-examples.png", "FT crack-tip field truth prediction and signed error"), figure("curve-examples.png", "UT and FT curves with signed error bars"),
        "## Training behaviour and cost\n", figure("loss-histories.png", "Log-scale independent training histories"), figure("curve-fit-cost.png", "Recorded curve-fit runtime"),
        "Curve runtimes exclude queue time; field-bank costs are shown separately. The twelve retained banks include fold generators as well as final fields; they are reusable evidence, not disposable preflights. These are observed training costs, not controlled inference benchmarks.\n",
        "## Interpretation and selection\n", "Do not rank methods by their weighted training-loss totals. Inspect global error, jump magnitude/timing/sign and curve peak/work safeguards together, including differences across seeds. A failure at coefficient0.1 does not disprove derivative supervision at other scales. Curve-only improvements cannot establish improved fracture fields. Keep the accepted Mixed MSE configuration unchanged until these trade-offs are reviewed; promising individual-task variants can be confirmed or tuned separately.\n",
        "## Evidence and reproducibility\n", f"Raw downloaded archives: [{root.as_posix()}](<{root.as_posix()}>). This reader checks complete status, unique/ordered validation IDs, train-fitted normalization, masks, targets and independently reproduced physical RMSEs. Curve-only variants additionally require byte-identical frozen fields. No model is loaded for training.\n",
        "[Full fixed metrics](metrics.csv) · [Motion and curve safeguards](safeguards.csv) · [Run IDs, revisions and runtime](runs.csv) · [Checks](verification.json).\n"]
    markdown = re.sub(r"(?<=\|)\n\n(?=\|)", "\n", "\n".join(text))
    (out/"README.md").write_text(markdown, encoding="utf-8")
    rows.to_csv(out/"metrics.csv", index=False); extra.to_csv(out/"safeguards.csv", index=False)
    sources.to_csv(out/"runs.csv", index=False); banks.to_csv(out/"field-banks.csv", index=False)
    (out/"verification.json").write_text(json.dumps(dict(checks=checks, evaluation="val", run_count=len(sources)), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=Path("Z:/p2/MULTI/Dual/Transformer"))
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3]/"samples/loss-study")
    p.add_argument("--verify-download-manifest", type=Path,
                   help="Optional full remote sha256sum manifest for the thirty downloads")
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
    galleries(args.run_root, rows, args.output); loss_histories(args.run_root, args.output)
    write_report(args.run_root, args.output, rows, extra, sources, banks, checks)
    print(rows.groupby(["variant", "kind", "mode"]).rmse_change_percent.mean().to_string())
    print("REPORT", args.output/"README.md")


if __name__ == "__main__":
    main()
