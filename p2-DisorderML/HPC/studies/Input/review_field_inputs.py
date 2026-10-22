"""Input-study report: saved Case1(i) field-source results, no training or test access."""
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


POLICIES = ("predicted", "true", "mixed", "corrected", "corrected_mixed")
LABELS = {"predicted": "Predicted only", "true": "True only", "mixed": "Predicted + true",
          "corrected": "Corrected only", "corrected_mixed": "Corrected + true",
          "correction": "Correction / frozen curve"}
TASKS = [(kind, mode) for kind in ("field", "curve") for mode in ("UT", "FT")]


def absolute_figure_links(text, directory):
    """Local app previews need absolute paths; keep regeneration consistent."""
    return re.sub(r"(!\[[^\]]*\])\(([^)]+\.png)\)",
                  lambda match: f"{match[1]}(<{(directory / match[2]).resolve().as_posix()}>)", text)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def run_path(root, suite, control, policy, seed):
    name = f"{control}-1ip-s{seed}" if policy == "predicted" else f"{suite}-{policy}-s{seed}"
    return root / name


def verify_ids(values, expected):
    ids = [str(x) for x in values]
    if ids != [str(x) for x in expected] or len(set(ids)) != len(ids):
        raise ValueError("Validation specimen IDs/order/uniqueness differ.")


def select_winner(rows):
    """Same fixed four-output physical-MSE ratio; never rank oracle substitutions."""
    deploy = rows[rows.policy.isin(POLICIES)].copy()
    if not np.isfinite(deploy.rmse).all() or not (deploy.baseline > 0).all():
        raise ValueError("Finite errors and positive training-mean baselines required.")
    if deploy.duplicated(["policy", "seed", "kind", "mode"]).any():
        raise ValueError("Duplicate evaluation cell.")
    counts = deploy.groupby(["policy", "seed"]).size()
    if not (counts == 4).all() or set(deploy.policy) != set(POLICIES):
        raise ValueError("Each candidate/seed requires all four deployable outputs.")
    if any(set(g.seed) != set(deploy.seed) for _, g in deploy.groupby("policy")):
        raise ValueError("Candidates must use identical seeds.")
    deploy["score"] = (deploy.rmse / deploy.baseline) ** 2
    scores = deploy.groupby(["policy", "seed"]).score.mean().unstack("seed")
    return scores.mean(axis=1).idxmin(), scores


def collect(root, suite, control):
    rows, extra, checks, sources = [], [], [], []
    reference, frozen = {}, {}
    for policy in (*POLICIES, "correction"):
        for seed in (42, 43, 44):
            run = run_path(root, suite, control, policy, seed)
            metrics = json.loads((run / "results/metrics.json").read_text())
            descriptor = json.loads((run / "model_data.json").read_text())
            if metrics["evaluation_split"] != "val":
                raise ValueError("Only saved validation results are permitted.")
            if metrics["run_config"]["seed"] != seed:
                raise ValueError("Run seed does not match its label.")
            for key in ("sample_ids", "normalizers", "node_masks", "reference_task_features"):
                fingerprint = hashlib.sha256(json.dumps(descriptor[key], sort_keys=True).encode()).hexdigest()
                reference.setdefault(key, fingerprint)
                if reference[key] != fingerprint:
                    raise ValueError(f"Data contract changed: {key}.")
            with np.load(run / "results/predictions.npz", allow_pickle=False) as arrays:
                ids = arrays["sample_ids"].astype(str)
                verify_ids(ids, descriptor["sample_ids"]["val"])
                for kind, mode in TASKS:
                    prediction = arrays[f"{mode}_val_{kind}_outputs"]
                    truth = arrays[f"{mode}_val_{kind}_truth"]
                    valid = np.isfinite(truth)
                    if kind == "field":
                        valid &= arrays[f"{mode}_val_field_mask"].astype(bool)
                        valid &= arrays[f"{mode}_node_mask"][None, :, None].astype(bool)
                        group = "raw" if policy in ("predicted", "true", "mixed") else "corrected"
                        key = (seed, mode, group)
                        frozen.setdefault(key, array_hash(prediction))
                        if frozen[key] != array_hash(prediction):
                            raise ValueError("Frozen field pipeline changed across curve-source fits.")
                    if prediction.shape != truth.shape or not np.isfinite(prediction[valid]).all():
                        raise ValueError("Invalid prediction shape/values.")
                    for tag, value in (("truth", truth), ("mask", valid)):
                        key = (kind, mode, tag)
                        reference.setdefault(key, array_hash(value))
                        if reference[key] != array_hash(value):
                            raise ValueError("Targets or validity masks changed between candidates.")
                    delta = prediction[valid].astype(np.float64) - truth[valid]
                    rmse = float(np.sqrt(np.mean(delta * delta)))
                    d = metrics["diagnostics"][kind][mode]
                    if not np.isclose(rmse, d["rmse"], rtol=2e-6):
                        raise ValueError("Saved physical RMSE does not match arrays.")
                    baseline = d[f"mean_{kind}_baseline_rmse"]
                    if d[f"mean_{kind}_baseline_source"] != f"train_mean_{kind}":
                        raise ValueError("Unverified mean baseline.")
                    reference.setdefault((kind, mode, "baseline"), baseline)
                    if not np.isclose(baseline, reference[(kind, mode, "baseline")], rtol=1e-8):
                        raise ValueError("Training baseline changed.")
                    rows.append(dict(policy=policy, seed=seed, kind=kind, mode=mode,
                                     rmse=rmse, baseline=baseline, skill=100 * (1 - rmse / baseline),
                                     collapse_ratio=d["collapse_ratio"]))
                    if kind == "field":
                        table = pd.read_csv(run / f"results/{mode}_val_field_motion.csv")
                        verify_ids(table.sample_id, ids)
                        for metric in ("local_jump_nrmse_percent", "temporal_nrmse_percent",
                                       "tip_u2_sign_accuracy", "u2_jump_sign_accuracy", "activity_node_jaccard"):
                            if metric in table:
                                extra.append(dict(policy=policy, seed=seed, mode=mode, metric=metric,
                                                  value=table[metric].mean(), valid_samples=int(table[metric].notna().sum())))
                    else:
                        table = pd.read_csv(run / f"results/{mode}_val_curve_sample_metrics.csv")
                        verify_ids(table.sample_id, ids)
                        x = arrays[f"{mode}_curve_x_values"].reshape(-1)
                        for metric, value in {
                            "peak_height_error_percent": 100 * abs(table.peak_error).mean() / abs(table.true_peak).mean(),
                            "peak_location_error_percent": 100 * abs(table.peak_x_error).mean() / np.ptp(x),
                            "work_error_percent": 100 * abs(table.energy_error).mean() / abs(table.true_energy).mean(),
                        }.items():
                            extra.append(dict(policy=policy, seed=seed, mode=mode, metric=metric,
                                              value=value, valid_samples=len(table)))
                        with np.load(run / "results/true_field_curves.npz", allow_pickle=False) as oracle:
                            verify_ids(oracle["sample_ids"], ids)
                            y, p = truth, oracle[f"{mode}_outputs"]
                            error = np.sqrt(np.mean((p.astype(float) - y) ** 2))
                            extra.append(dict(policy=policy, seed=seed, mode=mode, metric="true_field_curve_skill",
                                              value=100 * (1 - error / baseline), valid_samples=len(ids)))
            meta = metrics["metadata"]
            sources.append(dict(policy=policy, seed=seed, run=run.name,
                                job=meta.get("slurm_job_id"), source_revision=meta.get("source_revision"),
                                training_seconds=meta.get("training_seconds"), split_hash=meta.get("split_hash")))
            checks.append(f"{run.name}: validation IDs, train scaling, masks, truth, frozen fields and four physical RMSEs verified")
            print(checks[-1], flush=True)
    for seed in (42, 43, 44):
        bank = run_path(root, suite, control, "correction", seed)
        manifest = json.loads((bank / "correction_bank.json").read_text())
        train = set(map(str, manifest["sample_ids"]["train"]))
        heldouts = []
        for fold in manifest["folds"]:
            groups = [set(map(str, fold[k])) for k in ("fit_ids", "selection_ids", "heldout_train_ids")]
            if set.union(*groups) != train or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("Nested outer train partition is invalid.")
            heldouts.extend(groups[2])
            inner = json.loads((bank / "nested" / str(fold["fold"]) / "field_bank.json").read_text())
            for item in inner["folds"]:
                members = set(map(str, item["fit_ids"] + item["selection_ids"] + item["heldout_train_ids"]))
                if not members <= groups[0] or members & groups[2]:
                    raise ValueError("Nested generator sees an outer held-out specimen.")
        if len(heldouts) != len(train) or set(heldouts) != train:
            raise ValueError("Nested holdouts must cover each training specimen exactly once.")
        checks.append(f"Correction seed{seed}: disjoint TRAIN-only nested fold provenance verified")
    return pd.DataFrame(rows), pd.DataFrame(extra), checks, sources


def plot_comparisons(rows, extra, scores, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (kind, mode) in zip(axes.flat, TASKS):
        for index, policy in enumerate(POLICIES):
            values = rows.query("policy == @policy and kind == @kind and mode == @mode").skill
            ax.bar(index, values.mean(), color="#26746c" if policy == "corrected_mixed" else "#8298af", alpha=.8)
            ax.errorbar(index, values.mean(), yerr=values.std(), color="#263238", capsize=4)
            ax.scatter(index + np.linspace(-.12, .12, len(values)), values, color="#263238", s=15)
        ax.set(title=f"{mode} {kind}", ylabel="RMSE reduction vs training mean (%)",
               xticks=range(5), xticklabels=[LABELS[p].replace(" + ", "\n+ ") for p in POLICIES])
        ax.axhline(0, color="grey", lw=.7); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Field-input study: higher skill is better; dots = three seeds, whiskers = seed SD")
    fig.tight_layout(); fig.savefig(out / "accuracy-overview.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in zip(axes, ("UT", "FT")):
        for index, policy in enumerate(POLICIES):
            deploy = rows.query("policy == @policy and kind == 'curve' and mode == @mode").skill
            true = extra.query("policy == @policy and mode == @mode and metric == 'true_field_curve_skill'").value
            for offset, values, label, color in ((-.18, deploy, "Predicted-field deployment", "#c36b42"),
                                                 (.18, true, "True-field substitution", "#26746c")):
                ax.bar(index + offset, values.mean(), width=.34, color=color, label=label if index == 0 else None)
        ax.set(title=f"{mode} curves: source mismatch", ylabel="RMSE reduction vs training mean (%)",
               xticks=range(5), xticklabels=[LABELS[p].replace(" + ", "\n+ ") for p in POLICIES])
        ax.axhline(0, color="grey", lw=.7); ax.legend(fontsize=8); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out / "true-versus-deployment.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for row, mode in enumerate(("UT", "FT")):
        for ax, metric, title in zip(axes[row],
                ("local_jump_nrmse_percent", "peak_location_error_percent", "work_error_percent"),
                ("Local jump error / target jump RMS (%)", "Peak-location error / loading span (%)", "Work error / mean true work (%)")):
            for index, policy in enumerate(POLICIES):
                v = extra.query("policy == @policy and mode == @mode and metric == @metric").value
                ax.errorbar(index, v.mean(), yerr=v.std(), marker="o", capsize=4, color="#26746c")
            ax.set(title=f"{mode}: {title}", xticks=range(5),
                   xticklabels=[LABELS[p].replace(" + ", "\n+ ") for p in POLICIES])
            ax.tick_params(axis="x", labelsize=7); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Mechanism and curve safeguards: lower is better; kinematic events are not confirmed fracture labels")
    fig.tight_layout(); fig.savefig(out / "local-and-peak-errors.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for seed in scores.columns:
        ax.plot(range(5), scores.loc[list(POLICIES), seed], "o-", label=f"Seed {seed}", alpha=.75)
    ax.set(xticks=range(5), xticklabels=[LABELS[p].replace(" + ", "\n+ ") for p in POLICIES],
           ylabel="Mean of four physical MSE / training-mean MSE ratios", title="Fixed comparison score: lower is better")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(out / "selection-score.png", dpi=170); plt.close(fig)


def galleries(root, suite, control, winner, out):
    from resources.MLmetrics import load_dual_diagnostics, plot_field_sample

    chosen = run_path(root, suite, control, winner, 42)
    baseline = run_path(root, suite, control, "predicted", 42)
    for mode in ("UT", "FT"):
        a, b = [load_dual_diagnostics(run, "field", mode) for run in (baseline, chosen)]
        errors = b["sample_metrics"]["sample_rmse" if "sample_rmse" in b["sample_metrics"] else "rmse"]
        order = np.argsort(errors.to_numpy())
        for label, index in (("best", int(order[0])), ("worst", int(order[-1]))):
            frame = int(np.nanargmax(np.nanmean(np.diff(b["y_true"][index], axis=0) ** 2, axis=(1, 2))) + 1)
            xy = b["node_coords"]
            region = np.ones(len(xy), dtype=bool) if mode == "UT" else (
                (xy[:, 0] >= 90) & (xy[:, 0] <= 150) & (xy[:, 1] >= 65) & (xy[:, 1] <= 125))
            valid = region & b["valid_mask"][index, frame, :, 1]
            values = np.concatenate([d[k][index, frame, valid, 1] for d in (a, b) for k in ("y_true", "y_pred")])
            errors = np.concatenate([d["y_pred"][index, frame, valid, 1] - d["y_true"][index, frame, valid, 1] for d in (a, b)])
            limit = max(float(np.nanpercentile(abs(values), 98)), 1e-12)
            error_limit = max(float(np.nanpercentile(abs(errors), 98)), 1e-12)
            for tag, diag in (("baseline", a), ("winner", b)):
                fig, axes = plot_field_sample(diag, index, frame, 1, show=False, point_size=30)
                for position, ax in enumerate(axes):
                    bound = error_limit if position == 2 else limit
                    for collection in ax.collections:
                        collection.set_clim(-bound, bound)
                    ax.set(xlabel="Reference x (mm)", ylabel="Reference y (mm)")
                    if mode == "FT": ax.set(xlim=(90, 150), ylim=(65, 125))
                fig.suptitle(f"{mode} {label} winner-ranked specimen {b['sample_ids'][index]}: {tag}, U2 (mm), frame {frame + 1}/20\n"
                             "Truth and prediction share a scale; signed error has a separate scale", fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, .88)); fig.savefig(out / f"{mode}-{label}-{tag}-field.png", dpi=160); plt.close(fig)
        del a, b
        a, b = [load_dual_diagnostics(run, "curve", mode) for run in (baseline, chosen)]
        order = np.argsort(b["sample_metrics"].sample_rmse.to_numpy())
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        for row, (tag, diag) in enumerate((("Predicted-only baseline", a), ("Selected pipeline", b))):
            for ax, label, index in zip(axes[row], ("Best", "Worst"), (order[0], order[-1])):
                scale = 1000 if mode == "FT" else 1
                x, y, p = diag["x"], diag["y_true"][index] / scale, diag["y_pred"][index] / scale
                ax.bar(x, p - y, width=np.median(np.diff(x)) * .95, alpha=.3, color="steelblue", label="Signed error")
                ax.plot(x, y, color="#26746c", label="Truth"); ax.plot(x, p, "--", color="#c36b42", label="Prediction")
                ax.set(title=f"{tag}: {label.lower()} specimen {diag['sample_ids'][index]}",
                       xlabel="Strain" if mode == "UT" else "Load-line displacement (mm)",
                       ylabel="Stress / error (MPa)" if mode == "UT" else "Force / error (kN)")
                ax.legend(fontsize=8)
        fig.suptitle(f"{mode}: seed42, same winner-ranked specimens; all curves start from predicted fields")
        fig.tight_layout(); fig.savefig(out / f"{mode}-curve-examples.png", dpi=170); plt.close(fig)
    history = pd.read_csv(chosen / "results/loss_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("UT", "FT")):
        group = history[history.training_group == f"curve_{mode}"]
        ax.plot(group.epoch, group.train_loss, label="Train"); ax.plot(group.epoch, group.val_loss, label="Validation")
        ax.set(title=f"Selected {mode} curve fit, seed42", yscale="log", xlabel="Epoch", ylabel="Normalized loss")
        ax.legend()
    fig.tight_layout(); fig.savefig(out / "winner-loss-history.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("UT", "FT")):
        for policy in POLICIES:
            run = run_path(root, suite, control, policy, 42)
            table = pd.read_csv(run / f"results/{mode}_val_curve_point_metrics.csv")
            scale = 1000 if mode == "FT" else 1
            ax.plot(table.x, table.pred_std / scale, label=LABELS[policy], lw=1.3)
        ax.plot(table.x, table.true_std / scale, "k--", lw=2, label="Truth variability")
        ax.set(title=f"{mode}: specimen-to-specimen curve diversity, seed42", xlabel="Loading coordinate",
               ylabel="Curve standard deviation (MPa)" if mode == "UT" else "Curve standard deviation (kN)")
        ax.legend(fontsize=7); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out / "curve-diversity.png", dpi=170); plt.close(fig)


def followup_section(out):
    """Include the submitted study when its local scheduler receipt is present."""
    manifest = out / "loss-study-jobs.tsv"
    if not manifest.exists():
        return []
    jobs = pd.read_csv(manifest, sep="\t")
    gate = jobs[jobs.role == "preflight"].iloc[0]
    full = jobs[jobs.role != "preflight"]
    lines = ["## Submitted custom-loss study", "",
             f"Suite **independent-losses-261018**: gate **{gate.job_id}**, then **{len(full)} full jobs ({full.job_id.min()}–{full.job_id.max()})**. Immutable training source: `{gate.source_revision}`.", "",
             "Each full job requests **240 hours, one GPU, 12 CPUs and 90,000 MB RAM**. Seeds 42/43/44 retain the selected corrected+true policy. Dependencies require successful upstream completion; submitted is not the same as completed.", "",
             "| Comparison | Generator loss change | Full jobs across three seeds |",
             "|---|---|---:|",
             "| Spatial | MSE + 0.1 signed neighbour-motion loss | 9 |",
             "| Temporal | MSE + 0.1 adjacent-frame increment loss | 9 |",
             "| Both | MSE + 0.1 spatial + 0.1 temporal | 9 |",
             "| Sudden motion | Capped specimen/node/frame weighting, gain 1 | 9 |",
             "| Curve combined | Frozen original fields/correctors; no peak location versus soft/soft location | 6 |", "",
             "The nine jobs per field variant are three generator banks, three new nested correction banks and three adapted curve fits. Correctors retain MSE. Curve-loss fits reuse the same original winning frozen field pipeline; the existing MSE winner is their control. Combined curve coefficients are MSE 1, zone 0.5, derivative 0.05, peak height 0.25 and work 0.1; peak-location coefficient is 0 versus 0.02 with soft-peak sharpness 20. These are starting ablation settings, not HPO-selected values.", "",
             "Sudden-motion weights combine changes in a node's displacement increments with unusual local neighbour motion after removing a fitted affine contribution. They are capped and normalized per specimen, not based on displacement magnitude or a prescribed fracture region. They weight training errors only: no true fields or target-derived weights are required at inference. Spatial/temporal differences are formed in physical displacement units with fixed train-fitted normalization; adjacent recorded-frame increments are not velocities.", "",
             "[Exact jobs, dependencies and source](loss-study-jobs.tsv). Home: `/data/home/exy053/p2/MULTI/Dual/Transformer/independent-losses-261018/`. Successful full-run archives: `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/independent-losses-261018-<role>-s<seed>/`. The disposable preflight is not archived.", ""]
    receipt = out / "loss-study-status.json"
    if receipt.exists():
        state = json.loads(receipt.read_text(encoding="utf-8"))
        lines += [f"Last verified **{state['checked_at']}**: {state['summary']}", ""]
    return lines


def corrector_changes(rows):
    """Paired per-seed RMSE changes; negative means the corrector improves error."""
    selected = rows[rows.policy.isin(("mixed", "corrected_mixed"))]
    keys = ["seed", "kind", "mode"]
    if selected.duplicated(["policy", *keys]).any():
        raise ValueError("Duplicate correction comparison cell.")
    pairs = selected.pivot(index=keys, columns="policy", values="rmse")
    if set(pairs.columns) != {"mixed", "corrected_mixed"} or pairs.isna().any().any():
        raise ValueError("Both policies need every paired seed/output.")
    if not np.isfinite(pairs.to_numpy()).all() or (pairs <= 0).any().any():
        raise ValueError("Comparison errors must be finite and positive.")
    pairs["rmse_change_percent"] = 100 * (pairs.corrected_mixed / pairs.mixed - 1)
    return pairs.reset_index()


def corrector_costs(evidence, suite, control):
    """Measured one-GPU allocation hours, including common field-bank preparation."""
    jobs = {item["name"]: item for item in evidence["accounting"]}
    if len(jobs) != len(evidence["accounting"]):
        raise ValueError("Scheduler snapshot has duplicate job names.")
    result = []
    for seed in (42, 43, 44):
        row = {"seed": seed}
        for key, name in {
            "field_bank_h": f"{control}-bank-1i-s{seed}",
            "mixed_curve_h": f"{suite}-mixed-s{seed}",
            "correction_h": f"{suite}-correction-s{seed}",
            "corrected_curve_h": f"{suite}-corrected_mixed-s{seed}",
        }.items():
            job = jobs[name]
            if job["state"] != "COMPLETED" or "gres/gpu=1" not in job["allocated_tres"].split(","):
                raise ValueError("Cost comparison requires completed one-GPU jobs.")
            if job["elapsed_seconds"] <= 0:
                raise ValueError("Completed job elapsed time must be positive.")
            row[key] = job["elapsed_seconds"] / 3600
        row["mixed_total_h"] = row["field_bank_h"] + row["mixed_curve_h"]
        row["corrected_followup_h"] = row["correction_h"] + row["corrected_curve_h"]
        row["corrected_total_h"] = row["field_bank_h"] + row["corrected_followup_h"]
        result.append(row)
    return pd.DataFrame(result)


def review_correction_cost(root, suite, control, out, evidence_path):
    """Recheck six saved validation fits; no field forwards, training or scheduler writes."""
    rows = pd.read_csv(out / "per-run-metrics.csv")
    extra = pd.read_csv(out / "additional-metrics.csv")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    paired = corrector_changes(rows)
    if set(paired.seed) != {42, 43, 44} or len(paired) != 12:
        raise ValueError("The cost review requires three seeds and all four outputs.")
    costs = corrector_costs(evidence, suite, control)
    reference, parameters, source_hashes = {}, {}, {}
    for policy in ("mixed", "corrected_mixed"):
        for seed in (42, 43, 44):
            path = run_path(root, suite, control, policy, seed)
            metrics = json.loads((path / "results/metrics.json").read_text())
            data = json.loads((path / "model_data.json").read_text())
            model = json.loads((path / "model.json").read_text())
            if metrics["evaluation_split"] != "val" or metrics["run_config"]["seed"] != seed:
                raise ValueError("Only matching saved validation results are allowed.")
            parameters[policy] = model["metadata"]["parameter_counts"]
            for key in ("normalizers", "node_masks", "reference_task_features"):
                value = hashlib.sha256(json.dumps(data[key], sort_keys=True).encode()).hexdigest()
                reference.setdefault(key, value)
                if reference[key] != value:
                    raise ValueError("Input/normalization contracts differ.")
            expected_ids = data["sample_ids"]["val"]
            reference.setdefault("ids", expected_ids)
            verify_ids(expected_ids, reference["ids"])
            for kind, mode in TASKS:
                cached = rows.query("policy == @policy and seed == @seed and kind == @kind and mode == @mode").iloc[0]
                diag = metrics["diagnostics"][kind][mode]
                if not np.isclose(cached.rmse, diag["rmse"], rtol=2e-6):
                    raise ValueError("Cached RMSE differs from its original run.")
                if not np.isclose(cached.baseline, diag[f"mean_{kind}_baseline_rmse"], rtol=1e-8):
                    raise ValueError("Training-mean baseline differs.")
                table = pd.read_csv(path / f"results/{mode}_val_{kind}_{'motion' if kind == 'field' else 'sample_metrics'}.csv")
                verify_ids(table.sample_id, expected_ids)
                if kind == "field":
                    values = {k: table[k].mean() for k in ("local_jump_nrmse_percent", "temporal_nrmse_percent", "activity_node_jaccard")}
                else:
                    with np.load(path / "results/predictions.npz", allow_pickle=False) as a:
                        verify_ids(a["sample_ids"], expected_ids)
                        y, p = a[f"{mode}_val_curve_truth"], a[f"{mode}_val_curve_outputs"]
                        if not np.isclose(np.sqrt(np.mean((p.astype(float)-y)**2)), cached.rmse, rtol=2e-6):
                            raise ValueError("Curve arrays disagree with saved RMSE.")
                        span = np.ptp(a[f"{mode}_curve_x_values"])
                    values = {"peak_height_error_percent": 100*abs(table.peak_error).mean()/abs(table.true_peak).mean(),
                              "peak_location_error_percent": 100*abs(table.peak_x_error).mean()/span,
                              "work_error_percent": 100*abs(table.energy_error).mean()/abs(table.true_energy).mean()}
                for metric, value in values.items():
                    saved = extra.query("policy == @policy and seed == @seed and mode == @mode and metric == @metric").value.iloc[0]
                    if not np.isclose(value, saved, rtol=2e-6):
                        raise ValueError("Auxiliary diagnostic differs from saved source tables.")
            source_hashes[path.name] = hashlib.sha256((path / "results/metrics.json").read_bytes()).hexdigest()

    paired.to_csv(out / "correction-paired-changes.csv", index=False)
    costs.to_csv(out / "correction-training-costs.csv", index=False)
    colors = ("#277f81", "#c67538")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), constrained_layout=True)
    for ax, (kind, mode) in zip(axes.flat, TASKS):
        values = paired.query("kind == @kind and mode == @mode")
        ax.barh(0, values.rmse_change_percent.mean(), color=colors[0] if values.rmse_change_percent.mean() < 0 else colors[1], alpha=.55, height=.35)
        for i, item in enumerate(values.itertuples()):
            ax.scatter(item.rmse_change_percent, (i-1)*.10, color="#243746", s=38)
            ax.annotate(str(item.seed), (item.rmse_change_percent, (i-1)*.10), xytext=(4, 5), textcoords="offset points", fontsize=9)
        ax.axvline(0, color="grey", lw=1)
        ax.set(title=f"{mode} {kind}: mean {values.rmse_change_percent.mean():+.2f}%", yticks=[], ylim=(-.35,.35),
               xlim=(-3.1,1.3), xlabel="RMSE change from adding corrector (%)")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Corrected + true versus predicted + true\nNegative = lower error; labels = training seeds, not confidence intervals")
    fig.savefig(out / "correction-performance.png", dpi=170); plt.close(fig)

    mean = costs.drop(columns="seed").mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), constrained_layout=True)
    for ax, include_bank in zip(axes, (False, True)):
        for x, policy in enumerate(("mixed", "corrected_mixed")):
            segments = [("Shared field bank", mean.field_bank_h if include_bank else 0., "#8fa6bd"),
                        ("Nested correction", mean.correction_h if policy == "corrected_mixed" else 0., colors[1]),
                        ("Curve fit", mean.mixed_curve_h if policy == "mixed" else mean.corrected_curve_h, colors[0])]
            bottom = 0.
            for label, height, color in segments:
                ax.bar(x, height, bottom=bottom, color=color, width=.55, label=label if x == 1 else None)
                if height: ax.text(x,bottom+height/2,f"{height:.1f} h",ha="center",va="center",fontsize=10)
                bottom += height
            ax.text(x,bottom+1,f"{bottom:.1f} h",ha="center",fontweight="bold")
        ax.set(xticks=[0,1],xticklabels=["Predicted\n+ true","Corrected\n+ true"],ylabel="Mean allocated GPU-hours per seed",
               title="Follow-up after shared field bank" if not include_bank else "Complete field + curve training",ylim=(0,95))
        ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    axes[1].legend(loc="upper left",fontsize=8)
    fig.suptitle("Measured training cost, not inference latency or queue wait")
    fig.savefig(out / "correction-training-cost.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), constrained_layout=True)
    metrics = ("local_jump_nrmse_percent","peak_location_error_percent","work_error_percent")
    titles = ("Jump RMSE / target jump RMS (%)","Peak-location MAE / loading span (%)","Work MAE / mean true work (%)")
    for row, mode in enumerate(("UT","FT")):
        for ax, metric, title in zip(axes[row], metrics, titles):
            for i, policy in enumerate(("mixed","corrected_mixed")):
                values=extra.query("policy == @policy and mode == @mode and metric == @metric").value
                ax.bar(i,values.mean(),color=colors[i],alpha=.7)
                ax.scatter(i+np.linspace(-.09,.09,3),values,color="#243746",s=18,zorder=3)
                ax.text(i,values.max()*1.05,f"{values.mean():.2f}%",ha="center",fontsize=9)
            ax.set(title=f"{mode}: {title}",xticks=[0,1],xticklabels=["Predicted\n+ true","Corrected\n+ true"])
            ax.set_ylim(0,ax.get_ylim()[1]*1.16); ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Local-motion and curve safeguards: lower is better; dots = three seeds")
    fig.savefig(out / "correction-safeguards.png",dpi=170); plt.close(fig)

    allocation = {item["job_id"]: item for item in evidence["accounting"]}
    queue = {item["job_id"]: item for item in evidence["queue"]}
    jobs = pd.read_csv(out / "loss-study-jobs.tsv",sep="\t")
    counts = {}
    used_h = {}
    for item in jobs[jobs.role != "preflight"].itertuples():
        record=queue.get(item.job_id,allocation.get(item.job_id))
        if record is None: raise ValueError("Scheduler evidence omits a study job.")
        state=record["state"]
        category="field banks" if item.role.startswith("bank-") else "correction banks" if item.role.startswith("correction-") else "curve-loss fits" if item.role.startswith("combined_") else "adapted curve fits"
        counts.setdefault(category,{"COMPLETED":0,"RUNNING":0,"PENDING":0})
        counts[category][state]+=1
        used_h[category]=used_h.get(category,0.)+allocation.get(item.job_id,{}).get("elapsed_seconds",0)/3600
    summary=dict(checked_at=evidence["queue_checked_at"],parameters=parameters,source_metrics_sha256=source_hashes,
                 scheduler_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                 source_summary_sha256={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ("per-run-metrics.csv","additional-metrics.csv","selection.json")},
                 mean_paired_rmse_change=paired.groupby(["kind","mode"]).rmse_change_percent.mean().to_dict(),
                 full_training_ratio=mean.corrected_total_h/mean.mixed_total_h,
                 followup_training_ratio=mean.corrected_followup_h/mean.mixed_curve_h,
                 extra_gpu_hours_all_seeds=float((costs.corrected_total_h-costs.mixed_total_h).sum()),
                 study_counts=counts,study_allocated_gpu_hours=used_h)
    summary["mean_paired_rmse_change"]={"_".join(k):v for k,v in summary["mean_paired_rmse_change"].items()}
    (out/"correction-cost-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


def review(root, suite, control, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, extra, checks, sources = collect(root, suite, control)
    winner, scores = select_winner(rows)
    rows.to_csv(out / "per-run-metrics.csv", index=False); extra.to_csv(out / "additional-metrics.csv", index=False)
    scores.to_csv(out / "selection-scores.csv")
    (out / "validation.txt").write_text("\n".join(checks), encoding="utf-8")
    decision = dict(winner=winner, criterion="Mean across three seeds of equal four-output physical MSE / training-mean MSE ratios",
                    score=float(scores.loc[winner].mean()), seeds=[42, 43, 44], split="val",
                    input_suite=suite, control_suite=control, runs=sources)
    (out / "selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    plot_comparisons(rows, extra, scores, out); galleries(root, suite, control, winner, out)
    summary = rows.groupby(["policy", "kind", "mode"]).skill.agg(["mean", "std"])
    baseline = rows.query("policy == 'predicted'").set_index(["seed", "kind", "mode"]).rmse
    selected = rows.query("policy == @winner").copy()
    selected["relative_change"] = [100 * (r.rmse / baseline.loc[(r.seed, r.kind, r.mode)] - 1) for r in selected.itertuples()]
    changes = selected.groupby(["kind", "mode"]).relative_change.mean()
    true_skills = extra.query("policy == 'true' and metric == 'true_field_curve_skill'").groupby("mode").value.mean()
    cost_review = out / "correction-cost.md"
    selection_title = "Original fixed-score winner" if cost_review.exists() else "Selected next-study pipeline"
    text = ["# Field-input study: completed validation comparison", "",
            f"**{selection_title}: {LABELS[winner]}.** It has the lowest fixed four-output score ({decision['score']:.6f}) across seeds 42/43/44.", "",
            "Case 1(i) remains four independently trained UT/FT field/curve models. The selected correction option adds two independently trained residual field refiners; it does not restore joint UT/FT training.", "",
            "## Main result", "", "Skill = 100 × (1 − model RMSE / training-mean baseline RMSE). Higher is better; zero means the mean predictor. Entries are mean ± sample SD across three seeds, not confidence intervals.", "",
            "| Curve training source | UT field skill | FT field skill | UT curve skill | FT curve skill |",
            "|---|---:|---:|---:|---:|"]
    for policy in POLICIES:
        values = [summary.loc[(policy, kind, mode)] for kind, mode in TASKS]
        text.append("| " + LABELS[policy] + " | " + " | ".join(f"{v['mean']:.2f} ± {v['std']:.2f}%" for v in values) + " |")
    text += ["", "![Accuracy overview](accuracy-overview.png)", "", "## What wins, and how much?", ""]
    text += [f"- {mode} {kind}: mean RMSE change **{changes.loc[(kind, mode)]:+.2f}%** versus its same-seed predicted-only control (negative is better)." for kind, mode in TASKS]
    text += ["", "These are modest improvements, not a solution to fracture-path prediction. Predicted+true training without correction is a cheaper alternative and has slightly better mean UT curve accuracy here. Selection uses the existing equal four-output physical-MSE/baseline-MSE score, not a new field-first ranking; the winner is not best on every individual metric/seed.", "",
             "![Fixed selection score](selection-score.png)", "", "## What each experiment changes", "",
             "- **Predicted only:** original OOF predicted-field curve training, then final-generator predicted-field validation.",
             "- **True only:** curve training uses true fields, but the main evaluation still uses predicted fields. Poor deployment results measure source mismatch, not inability to learn curves from true fields.",
             "- **Predicted + true:** each training specimen/update samples one complete source with probability 0.5; validation uses predicted fields.",
             "- **Corrected only:** a residual Transformer corrects predicted fields; curves train on nested-OOF corrected fields and deploy on corrected predictions.",
             "- **Corrected + true:** the same corrector, with 50/50 corrected/true curve training. No true fields are available at deployment.", "",
             "The corrector is field-supervised and frozen during curve training. Its nested cross-fitting excludes each held-out training specimen from both upstream and corrector fitting. Reference curves held fixed after correction are saved as a separate diagnostic, not a fresh curve fit.", "",
             "## True fields versus deployment", "",
             "The green bars below bypass the field generator using true validation fields in the same checkpoint. They are diagnostic and are excluded from winner selection. Checkpoints were selected on predicted/corrected-field validation, not independently selected as true-field oracles.", "",
             f"For the **true-only-trained** curve models, true-field evaluation gives **{true_skills['UT']:.2f}% UT and {true_skills['FT']:.2f}% FT RMSE skill** averaged across seeds, versus {summary.loc[('true', 'curve', 'UT'), 'mean']:.2f}% and {summary.loc[('true', 'curve', 'FT'), 'mean']:.2f}% when those same models receive predicted fields. This isolates a substantial source-distribution problem; it does not establish a deployable improvement.", "",
             "![True-field source substitution](true-versus-deployment.png)", "",
             "The UT deployment curve skill remains low, despite the winning overall score. The next plot shows whether across-specimen variation is recovered rather than merely the shared average curve shape; high global curve R² alone can conceal this collapse.", "",
             "![Curve diversity](curve-diversity.png)", "", "## Local-motion and curve safeguards", "",
             "Local jump error is normalized by target jump RMS on the same valid events, not by maximum displacement. Peak-location error uses the loading span; work error uses mean absolute true work. Kinematic jump events are not verified broken-strut labels. Full per-seed values, sign accuracy and eligible sample counts are in additional-metrics.csv; no new local acceptance threshold is silently imposed.", "",
             "![Local and peak errors](local-and-peak-errors.png)", "", "## Representative predictions", "",
             "Seed 42 is fixed, not selected for its score. Best/worst IDs are ranked by the selected pipeline's whole-field/full-curve RMSE; baseline and selected predictions use the same IDs. U2 field frames maximize the specimen's true adjacent-frame motion; this is not a confirmed fracture time. UT shows the full body; FT uses a 90–150 by 65–125 mm crack-tip window. Point plots avoid interpolation across cracks. Each baseline/selected pair uses common 98th-percentile truth/prediction limits and common signed-error limits, calculated over the displayed region. Error has a separately labelled scale. Numerical metrics retain all valid values.", ""]
    for mode in ("UT", "FT"):
        text += [f"### {mode}", "", f"![{mode} curve examples]({mode}-curve-examples.png)", ""]
        for rank in ("best", "worst"):
            for tag in ("baseline", "winner"):
                text += [f"![{mode} {rank} {tag} field]({mode}-{rank}-{tag}-field.png)", ""]
    text += ["## Training and follow-up", "", "![Selected curve losses](winner-loss-history.png)", "",
             "The correction-bank production jobs took approximately 49–53 hours per seed, including nested upstream fits; new curve fits took approximately 3–8 hours. These are training costs, not inference latency measurements. Correction adds field-refinement inference passes; latency was not benchmarked.", "",
             "The custom-loss comparison must preserve the selected source policy. Each changed field generator needs a newly cross-fitted corrector; the original corrector must not be reused on a different error distribution. Hold corrector MSE fixed while changing generator losses. Curve-loss comparisons instead reuse the same baseline field/correction pipeline. Loss coefficients remain controlled starting presets, not optimized optima.", "",
             *followup_section(out),
             "## Verification and limits", "", "All 18 saved validation evaluations (five candidates plus frozen-readout correction, three seeds) are checked for identical specimen IDs/order, training-fitted scalers, masks, truths, baseline denominators and recomputed physical RMSE. Raw/corrected fields remain identical across corresponding curve-source fits. [Validation checks](validation.txt) and [selection receipt](selection.json) record the rule and source runs.", "",
             "The split is 7,137 train / 793 validation / 881 locked test. No locked-test targets, inference, new training, FEA or ODB exports are used by this review. Selection on these same validation results is exploratory; three seeds are not a final generalization guarantee. Raw archives are under " + f"`{root}`.", ""]
    if cost_review.exists():
        text[2:2] = [f"**Cost-aware follow-up:** the [performance/runtime review]({cost_review.resolve().as_posix()}) recommends Predicted + True for future work. Original accuracy selection and the submitted corrected-source study below are preserved; no job policy is changed by that recommendation.", ""]
    receipt = out / "transfer-verification.json"
    if receipt.exists():
        transfer = json.loads(receipt.read_text(encoding="utf-8"))
        text += [f"Download receipt: **{transfer['verified_files']:,} files, {transfer['bytes']:,} bytes, {transfer['hash_mismatches']} SHA256 mismatches**. See the [transfer receipt](transfer-verification.json).", ""]
    (out / "README.md").write_text(absolute_figure_links("\n".join(text), out), encoding="utf-8")
    print(json.dumps({"winner": winner, "score": decision["score"], "report": str(out / "README.md")}), flush=True)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p2/MULTI/Dual/Transformer"))
    parser.add_argument("--suite", default="independent-inputs-261015")
    parser.add_argument("--control", default="independence-261013-r1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-evidence", type=Path,
                        help="Build only the correction-cost figures/summary from supplied scheduler evidence and existing verified review tables.")
    options = parser.parse_args()
    if options.cost_evidence:
        review_correction_cost(options.root, options.suite, options.control, options.output_dir, options.cost_evidence)
    else:
        review(options.root, options.suite, options.control, options.output_dir)
