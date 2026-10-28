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


def galleries(root, suite, control, winner, out, data_root=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata")):
    import runpy
    from resources.MLstudy_reports import field_examples,curve_examples
    helper=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    # Practical accepted route; numerical Corrected+True selection is unchanged.
    chosen=run_path(root,suite,control,"mixed",42)
    descriptor=json.loads((chosen/"model_data.json").read_text())
    for mode in ("UT","FT"):
        mean=helper["training_mean"](data_root,descriptor,mode)
        field_examples(chosen,mode,out,"Predicted + True (raw fields)",data_root,mean)
        curve_examples(chosen,mode,out,"Predicted + True deployment",data_root)




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


def review(root, suite, control, out, data_root=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata")):
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
    plot_comparisons(rows, extra, scores, out)
    refresh_visual_report(root,suite,control,out,data_root)
    return decision


def refresh_visual_report(root,suite,control,out,data_root=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata")):
    import runpy
    from resources.MLstudy_reports import pipeline_regions,pipeline_global_figures,figure
    rows=pd.read_csv(out/"per-run-metrics.csv");rows=rows[rows.policy.isin(POLICIES)].copy()
    baseline=rows[rows.policy=="predicted"].set_index(["seed","kind","mode"]).rmse
    rows["change_percent"]=[100*(r.rmse/baseline.loc[r.seed,r.kind,r.mode]-1) for r in rows.itertuples()]
    pipeline_global_figures(rows.rename(columns={"policy":"variant","skill":"skill_percent"}),out)
    helper=runpy.run_path(str(Path(__file__).parents[1]/"Loss/review_fracture_regions.py"))
    runs=[dict(variant=policy,seed=seed,path=run_path(root,suite,control,policy,seed)) for policy in POLICIES for seed in (42,43,44)]
    pipeline_regions(runs,"predicted",data_root,helper,out)
    galleries(root,suite,control,"mixed",out,data_root)
    for name,columns in (("validation",("val_loss",)),("train-validation",("train_loss","val_loss"))):
        fig,axes=plt.subplots(1,2,figsize=(15,5.5))
        for ax,mode in zip(axes,("UT","FT")):
            for index,policy in enumerate(POLICIES):
                if len(columns)>1 and policy not in ("predicted","mixed"):continue
                h=pd.read_csv(run_path(root,suite,control,policy,42)/"results/loss_history.csv")
                h=h[h.training_group==f"curve_{mode}"]
                for column in columns:ax.plot(h.epoch,h[column].where(h[column]>0),"--" if column.startswith("train") else "-",color=plt.get_cmap("tab10")(index),label=LABELS[policy]+" "+column.replace("_loss",""))
            ax.set(title=mode,yscale="log",xlabel="Curve training epoch",ylabel="Normalized curve MSE (log)");ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/f"loss-{name}.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    for ax,policy,title in zip(axes,("predicted","mixed"),("Pre-study control","Accepted source policy")):
        ax.axis("off")
        ax.text(.5,.85,title,ha="center",fontsize=16)
        ax.text(.5,.60,"Independent disorder → field models (unchanged)",ha="center",fontsize=13,bbox=dict(boxstyle="round",facecolor="#dceee8"))
        ax.annotate("",(.5,.36),(.5,.52),arrowprops=dict(arrowstyle="->"))
        ax.text(.5,.28,"Independent field → curve models\nTRAIN: "+("OOF predicted fields" if policy=="predicted" else "50/50 OOF predicted + true fields")+"\nDEPLOY: predicted fields only",ha="center",fontsize=13,bbox=dict(boxstyle="round",facecolor="#dbe8f3"))
    fig.tight_layout();fig.savefig(out/"source-architecture.png",dpi=170);plt.close(fig)
    image=lambda name,caption:figure(out,name,caption)
    text=["# Field-input study — completed validation comparison\n\n## Recap and verdict\n\n",
        "**Accepted practical policy: Predicted + True, no corrector.** Case1(i) remains four independently trained models. Corrected+True retains its original numerical fixed-score win (0.743791), but its small field gain does not materially improve curves and its extra training is costly. Three seeds, 793 validation specimens. All main predictions use predicted/corrected fields, never true deployment inputs.\n\n",
        image("source-architecture.png","Pre-study source versus accepted Mixed source; model architecture unchanged"),
        "## Global performance versus predicted-only control\n\nSame-seed physical RMSE changes; lower is better. Raw field generators are unchanged for predicted/true/mixed curve policies; corrected policies add supervised refiners.\n\n",image("global-control.png","Four-output changes versus predicted-only"),
        "## Global performance versus TRAIN mean\n\nSkill = 100×(1−model RMSE/TRAIN-mean RMSE). Zero matches the mean; this does not measure literal distance to the mean field.\n\n",image("global-mean.png","Accuracy beyond mean outputs"),
        image("output-variation.png","Specimen variation is separate from RMSE skill; matching spread does not establish accuracy"),
        image("UT-mean-field-examples.png","UT TRAIN-mean reference comparison"),image("FT-mean-field-examples.png","FT TRAIN-mean reference comparison"),
        "## Actual STATUS fracture regions versus control\n\nDelivered intact→failed struts define endpoint and relative-motion windows at onset ±1 retained frame. STATUS is evaluation only. Raw fields repeat across the three uncorrected curve policies by design.\n\n",image("status-fracture-regions.png","Actual fracture errors versus predicted-only"),
        "## Actual STATUS fracture regions versus TRAIN mean\n\nIdentical event masks apply to predictions and TRAIN-only means. The broader FT box is an overlapping diagnostic, not a damage label.\n\n",image("fracture-mean.png","Actual fracture accuracy beyond mean fields"),
        "## Best and worst predictions\n\nThese show the accepted Mixed pipeline at seed42, not the discarded-for-cost numerical corrector leader. Complete valid output RMSE ranks specimens. Large four-column galleries contain truth, prediction, error and percent error; tiny/zero truth is excluded from percentage displays only. FT fields zoom at the crack tip.\n\n"]
    for mode in ("UT","FT"):text += [image(f"{mode}-field-examples.png",f"{mode} Mixed fields"),image(f"{mode}-curve-examples.png",f"{mode} Mixed deployment curves")]
    text += ["## Training time\n\nThe measured cost/benefit review compares complete workflows and the incremental correction step. It is not inference latency.\n\n",image("correction-training-cost.png","Mixed versus corrected-Mixed measured training cost"),
        "[Detailed correction cost/benefit](correction-cost.md). The accepted policy avoids nested corrector fitting; existing compatible generator banks remain reusable.\n\n",
        "## Loss histories\n\nValidation-only compares all source policies; selected train/validation compares predicted-only and accepted Mixed. These are distinct views of curve training, not repeated field-loss plots.\n\n",image("loss-validation.png","All source-policy validation curves"),image("loss-train-validation.png","Selected curve training and validation histories"),
        "## Source mismatch, fracture-relevant curve windows and limits\n\nTrue-only-trained models reach 50.69% UT / 43.53% FT RMSE reduction versus TRAIN mean when given true fields, but −30.40% / 9.39% with predicted fields. This is a source-distribution diagnostic, not deployable performance.\n\n",image("true-versus-deployment.png","True-field diagnostic versus predicted-field deployment"),
        image("curve-regions.png","Target peak/post-peak windows versus control and TRAIN mean"),
        "Curve windows use each true hard peak (±5% x span) and post-peak through recorded end; they are not spatial STATUS nodes or inferred physical initiation times. The original hard peak/work and kinematic safeguards remain below.\n\n",image("local-and-peak-errors.png","Fixed motion and hard curve metrics"),
        "Predicted-only uses training-only OOF fields; true-only uses true fields for curve training; Mixed samples one entire field source with probability0.5. Corrected-only and corrected-Mixed use nested OOF refiners and remain comparison evidence, not current workflow. No in-training predictions substitute for OOF inputs.\n\n",
        "[Global metrics](per-run-metrics.csv) · [Actual fracture metrics](confirmed-fracture-regions.csv) · [Curve regions](curve-region-metrics.csv) · [Original selection](selection.json) · [Validation checks](validation.txt). Locked test unchanged; no new training/inference/export in this review.\n"]
    (out/"README.md").write_text("".join(text),encoding="utf-8")
    print("REFRESHED_REPORT",out/"README.md",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p2/MULTI/Dual/Transformer"))
    parser.add_argument("--suite", default="independent-inputs-261015")
    parser.add_argument("--control", default="independence-261013-r1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root",type=Path,default=Path("Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata"))
    parser.add_argument("--cost-evidence", type=Path,
                        help="Build only the correction-cost figures/summary from supplied scheduler evidence and existing verified review tables.")
    options = parser.parse_args()
    if options.cost_evidence:
        review_correction_cost(options.root, options.suite, options.control, options.output_dir, options.cost_evidence)
    else:
        review(options.root, options.suite, options.control, options.output_dir, options.data_root)
