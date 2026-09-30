"""Single-worker dual HPO: fixed validation ranking, durable study-level resume."""

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import time

import numpy as np
import optuna
import torch
from torch import nn

from resources.MLdual import DUAL_MODEL, DualLoss, DualStageTransformer, _json_safe
from resources.MLfunc import CombinedCurveLoss, curve_default_zone_boundaries, curve_default_zone_weights


class DualValidationScore:
    """Equal-task physical MSE / validation MSE of a TRAIN-fitted mean.

    Fixed before HPO and independent of all tunable training-loss weights.
    Validation is allowed for selection, never for fitting the mean/normalizers.
    """

    def __init__(self, data, kinds=("field", "curve")):
        self.kinds = tuple(kinds)
        if self.kinds not in (("field", "curve"), ("curve",)):
            raise ValueError("Selection must use all four tasks or the two frozen-source curve tasks.")
        self.scales, self.baseline_mse = {}, {}
        for kind in self.kinds:
            for mode in ("UT", "FT"):
                key = f"{kind}_{mode}"
                train = data.splits["train"][kind][mode]
                val = data.splits["val"][kind][mode]
                scale = np.asarray(data.normalizers[kind][mode]["scale"])
                if kind == "field":
                    mask = data.splits["train"]["field_mask"][mode]
                    mean = train.sum(axis=0, dtype=np.float64) / np.maximum(mask.sum(axis=0), 1)
                    valid = data.splits["val"]["field_mask"][mode]
                else:
                    mean = train.mean(axis=0, dtype=np.float64)
                    valid = np.isfinite(val)
                squared, count = 0.0, 0
                for start in range(0, len(val), 32):
                    keep = valid[start:start+32]
                    err = (val[start:start+32] - mean) * scale
                    per_sample_count = keep.reshape(len(keep), -1).sum(axis=1)
                    if np.any(per_sample_count == 0):
                        raise ValueError(f"Validation specimen without valid {key} targets.")
                    squared += (np.where(keep, err ** 2, 0).reshape(len(keep), -1).sum(axis=1) / per_sample_count).sum()
                    count += len(keep)
                baseline = float(squared / max(int(count), 1))
                if not np.isfinite(baseline) or baseline <= 1e-12:
                    raise ValueError(f"Degenerate validation baseline for {key}: {baseline}")
                self.scales[key] = scale
                self.baseline_mse[key] = baseline

    def __call__(self, predictions, targets, masks):
        scores = {}
        for kind in self.kinds:
            for mode in ("UT", "FT"):
                key = f"{kind}_{mode}"
                pred, true = predictions[kind][mode], targets[kind][mode]
                scale = torch.as_tensor(self.scales[key], dtype=pred.dtype, device=pred.device)
                valid = masks[mode].bool() if kind == "field" else torch.isfinite(true)
                error = ((pred - true) * scale).square().masked_fill(~valid, 0).flatten(1)
                mse = (error.sum(dim=1) / valid.flatten(1).sum(dim=1).clamp_min(1)).mean()
                scores[f"selection_{key}"] = float(mse.detach()) / self.baseline_mse[key]
        scores["selection_score"] = float(np.mean(list(scores.values())))
        return scores


def suggest_config(trial):
    """Conditional space around Trial 1; physical definitions/data stay fixed."""
    stages = {}
    for stage in ("field", "curve"):
        stages[stage] = {
            "d_model": trial.suggest_categorical(f"{stage}_d_model", [96, 128, 256, 384]),
            "n_heads": trial.suggest_categorical(f"{stage}_n_heads", [2, 4, 8]),
            "n_layers": trial.suggest_int(f"{stage}_n_layers", 2 if stage == "field" else 1, 6 if stage == "field" else 4),
            "ff_mult": trial.suggest_categorical(f"{stage}_ff_mult", [2, 4, 6]),
            "dropout": trial.suggest_float(f"{stage}_dropout", 0.0, 0.35),
            "head_dropout": trial.suggest_float(f"{stage}_head_dropout", 0.0, 0.30),
            "attention_dropout": trial.suggest_float(f"{stage}_attention_dropout", 0.0, 0.30),
            "head_hidden_mult": trial.suggest_categorical(f"{stage}_head_hidden_mult", [0, 1, 2]),
            "activation": trial.suggest_categorical(f"{stage}_activation", ["relu", "gelu"]),
            "pos_encoding": trial.suggest_categorical(f"{stage}_position", ["learned", "none", "sinusoidal"]),
            "norm_first": trial.suggest_categorical(f"{stage}_norm_first", [False, True]),
        }
    stages["curve"]["pool"] = trial.suggest_categorical("curve_pool", ["mean", "cls"])
    stages["curve"]["use_cls_token"] = True if stages["curve"]["pool"] == "cls" else trial.suggest_categorical("mean_cls_token", [True, False])
    # Three independent ratios span all four positive weights; sum=4 removes
    # redundant global rescaling, which would otherwise confound learning rate.
    ratios = [1.0] + [trial.suggest_float(f"relative_{key}", 0.125, 8.0, log=True)
                     for key in ("field_FT", "curve_UT", "curve_FT")]
    weights = np.asarray(ratios) * 4 / sum(ratios)
    loss_kind = trial.suggest_categorical("curve_loss", ["mse", "combined"])
    curve_terms = {}
    if loss_kind == "combined":
        curve_terms = {key: trial.suggest_float(key, low, high, log=True) for key, low, high in [
            ("weighted_mse_weight", 0.01, 1.0), ("derivative_weight", 0.001, 0.3),
            ("peak_weight", 0.01, 1.0), ("energy_weight", 0.005, 0.5),
            ("peak_location_weight", 0.001, 0.1),
        ]}
        curve_terms["derivative_order"] = trial.suggest_categorical("derivative_order", [1, 2])
        curve_terms["SoftPeak_beta"] = trial.suggest_float("soft_peak_beta", 5.0, 40.0, log=True)
    return dict(
        stages=stages, weights={"field": dict(zip(("UT", "FT"), weights[:2])),
                                "curve": dict(zip(("UT", "FT"), weights[2:]))},
        curve_loss=loss_kind, curve_terms=curve_terms,
        batch=trial.suggest_categorical("batch", [1, 2, 4, 8]),
        lr=trial.suggest_float("lr", 1e-5, 8e-4, log=True),
        curve_lr_factor=trial.suggest_float("curve_lr_factor", 0.25, 4.0, log=True),
        optimizer=trial.suggest_categorical("optimizer", ["adamw", "adam"]),
        weight_decay=trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True),
        grad_clip=trial.suggest_categorical("grad_clip", [None, 0.5, 1.0, 5.0]),
        scheduler_factor=trial.suggest_float("scheduler_factor", 0.3, 0.8),
        scheduler_patience=trial.suggest_int("scheduler_patience", 8, 25),
        early_stop_patience=trial.suggest_int("early_stop_patience", 50, 100),
    )


def trial1_parameters():
    params = {"curve_pool": "mean", "mean_cls_token": True, "curve_loss": "mse",
              "relative_field_FT": 1., "relative_curve_UT": 1., "relative_curve_FT": 1.,
              "batch": 2, "lr": 1e-4, "curve_lr_factor": 1., "optimizer": "adamw",
              "weight_decay": 1e-8, "grad_clip": None, "scheduler_factor": 0.7,
              "scheduler_patience": 12, "early_stop_patience": 75}
    for stage, depth, dropout in (("field", 4, .20), ("curve", 2, .15)):
        params.update({f"{stage}_{k}": v for k, v in dict(d_model=256, n_heads=4, n_layers=depth,
                       ff_mult=4, dropout=dropout, head_dropout=dropout, attention_dropout=dropout,
                       head_hidden_mult=0, activation="relu",
                       position="learned", norm_first=False).items()})
    return params


def run_dual_hpo(data, study_dir, *, study_name, target_trials=200, timeout_hours=230,
                 epochs=450, seed=42, device="cuda", archive_dir=None, resume=False):
    """Continue a single-worker study, not the optimizer state of a cut trial.

    SQLite lives in scratch. Periodic SQLite backups and completed-trial copies
    protect the archive; B1 also archives on exit. One archive-side lock prevents
    competing workers from overwriting the study. A hard kill may leave a lock;
    inspect the recorded job before manually removing it.
    """
    study_dir = Path(study_dir)
    archive_dir = Path(archive_dir) if archive_dir else study_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    lock = archive_dir / "study.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"job": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid()}, stream)
    try:
        return _run_locked(data, study_dir, archive_dir, study_name, target_trials,
                           timeout_hours, epochs, seed, device, resume)
    finally:
        lock.unlink()


def _run_locked(data, root, archive, name, target, hours, epochs, seed, device, resume):
    if root != archive and (archive / "full_study.db").exists():
        if not resume:
            raise FileExistsError("Study exists; use --resume with the same study configuration.")
        shutil.copytree(archive, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns("study.lock"))
    root.mkdir(parents=True, exist_ok=True)
    db = root / "full_study.db"
    if db.exists() and not resume:
        raise FileExistsError("Study exists; use --resume.")
    if resume and not db.exists():
        raise FileNotFoundError("No archived study to resume.")
    score = DualValidationScore(data)
    digest = hashlib.sha256()
    # Hash actual normalized arrays and masks, not just paths or sample counts.
    for split in ("train", "val", "test"):
        for kind in ("geometry", "field", "curve", "field_mask"):
            values = data.splits[split][kind]
            for array in (values.values() if isinstance(values, dict) else [values]):
                digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    contract = {"version": 1, "data_sha256": digest.hexdigest(), "data": data.to_metadata(),
                "seed": seed, "epochs": epochs, "score": "mean physical MSE / train-mean validation MSE",
                "baseline_mse": score.baseline_mse,
                "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "optuna": optuna.__version__},
                "code_sha256": hashlib.sha256(b"".join(Path(__file__).with_name(file).read_bytes()
                    for file in ("MLdualHPO.py", "MLdual.py", "MLfunc.py", "MLmodels.py", "MLdata.py"))).hexdigest()}
    contract = _json_safe(contract)
    manifest = root / "study_contract.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Data, split, model/HPO code, seed or epoch budget changed; create a new study.")
    manifest.write_text(json.dumps(contract, indent=2))
    data.to_json(root / "model_data.json")
    storage = optuna.storages.RDBStorage(f"sqlite:///{db}")
    study = optuna.create_study(study_name=name, storage=storage, direction="minimize", load_if_exists=resume,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=25),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=75, interval_steps=10))
    if resume:
        study.sampler = optuna.samplers.TPESampler(seed=seed + len(study.trials), n_startup_trials=25)
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            study.enqueue_trial(trial.params, user_attrs={"retry_of": trial.number})
    if not study.trials:
        study.enqueue_trial(trial1_parameters(), user_attrs={"reference": "Trial 1 configuration; new fixed ranking"})
    deadline = time.monotonic() + hours * 3600

    def sync(full=False):
        if root == archive:
            return
        # No concurrent worker; backup yields a consistent DB unlike copying live SQLite.
        with sqlite3.connect(db) as src, sqlite3.connect(archive / "full_study.db") as dst:
            src.backup(dst)
        if full:
            shutil.copytree(root, archive, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("full_study.db*", "study.lock"))
        else:
            shutil.copy2(manifest, archive / manifest.name)

    def objective(trial):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cfg = suggest_config(trial)
        trial.set_user_attr("effective_config", cfg)
        model = None
        trial_dir = root / "trials" / f"trial-{trial.number:04d}"
        metadata = {"run_layout": {"task": "MULTI", "output_kind": "Dual", "model": "Transformer", "run_name": name},
                    "study_name": name, "trial_number": trial.number, "selection_metric": contract["score"],
                    "source_revision": os.environ.get("ML_SOURCE_REVISION"), "run_config": cfg}
        try:
            network = DualStageTransformer.from_data(data, field_kwargs=cfg["stages"]["field"], curve_kwargs=cfg["stages"]["curve"])
            curve_loss = nn.MSELoss()
            if cfg["curve_loss"] == "combined":
                curve_loss = {mode: CombinedCurveLoss(mse_weight=1., **cfg["curve_terms"],
                    x_values=data.metadata["curve_x_values"][mode], zone_boundaries=curve_default_zone_boundaries(mode),
                    zone_weights=curve_default_zone_weights()) for mode in ("UT", "FT")}
            loss = DualLoss(curve_loss=curve_loss, weights=cfg["weights"],
                            curve_normalizers=data.normalizers["curve"] if cfg["curve_loss"] == "combined" else None)
            model = DUAL_MODEL(network, loss, data=data, device=device,
                batch=cfg["batch"], lr=cfg["lr"], opt=(cfg["optimizer"], cfg["weight_decay"]),
                curve_lr_factor=cfg["curve_lr_factor"], grad_clip=cfg["grad_clip"], selection_metric=score,
                scheduler=("plateau", "min", cfg["scheduler_factor"], cfg["scheduler_patience"], 1e-4))

            def epoch_end(trainer, row):
                trial.report(row["val_selection_score"], step=int(row["epoch"]))
                if row["epoch"] % 5 == 0:
                    sync()
                    if root != archive:
                        shutil.copytree(trial_dir, archive / "trials" / trial_dir.name, dirs_exist_ok=True)
                if time.monotonic() >= deadline:
                    trial.set_user_attr("stopped_for_walltime", True)
                    study.enqueue_trial(trial.params, user_attrs={"retry_of": trial.number})
                    study.stop()
                    raise optuna.TrialPruned("Time budget reached; retry this configuration in the next job.")
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return False

            model.train(epochs, verbose=10, early_stop_patience=cfg["early_stop_patience"],
                        early_stop_delta=1e-5, checkpoint_path=trial_dir, metadata=metadata, epoch_callback=epoch_end)
            best = model.history[model.best_epoch - 1]
            trial.set_user_attr("best_epoch", model.best_epoch)
            trial.set_user_attr("epochs_run", len(model.history))
            trial.set_user_attr("validation_task_ratios", {k: v for k, v in best.items() if k.startswith("val_selection_")})
            completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
            if not completed or model.best_loss < min(t.value for t in completed):
                model.save(root / "best", metadata=metadata)
                model.save_results(eval_split="val", run_config=cfg, metadata=metadata)
                (root / "best_params.json").write_text(json.dumps(trial.params, indent=2))
                (root / "best_trial_user_attrs.json").write_text(json.dumps(trial.user_attrs, indent=2))
            return model.best_loss
        except torch.cuda.OutOfMemoryError:
            trial.set_user_attr("failure_reason", "CUDA out of memory; no automatic batch-size change")
            raise
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def trial_end(study, trial):
        study.trials_dataframe().to_csv(root / "trials.csv", index=False)
        sync(full=True)

    sync(full=True)
    # Target counts completed and pruned evaluations; failed/OOM trials do not
    # masquerade as useful evaluations. A timeout still bounds repeated failures.
    while time.monotonic() < deadline:
        finished = study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))
        if len(finished) >= target:
            break
        study.optimize(objective, n_trials=1, callbacks=[trial_end],
                       catch=(torch.cuda.OutOfMemoryError, FloatingPointError), gc_after_trial=True)
    sync(full=True)
    return study
