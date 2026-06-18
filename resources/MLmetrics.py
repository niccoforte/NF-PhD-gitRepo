from resources.imports import *

import json
from pathlib import Path


# =============================================================================
# Curve Metrics
# =============================================================================

def _curve_2d_array(data, name):
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D curve array, got shape {arr.shape}.")
    return arr

def _safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))

def _safe_corr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.nanstd(a) <= eps or np.nanstd(b) <= eps:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _curve_x_array(x_values, n_points):
    x = _coerce_curve_x_values(x_values)
    if x is None:
        return np.arange(n_points, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != n_points:
        raise ValueError(f"x_values length ({x.size}) does not match curve length ({n_points}).")
    return x

def _curve_zone_slices(n_points, zone_boundaries=None, zone_names=None):
    if zone_boundaries is None:
        b1 = n_points // 3
        b2 = (2 * n_points) // 3
    else:
        if len(zone_boundaries) != 2:
            raise ValueError("zone_boundaries must contain exactly two point indices.")
        b1, b2 = int(zone_boundaries[0]), int(zone_boundaries[1])
    if not (0 < b1 < b2 < n_points):
        raise ValueError(f"Invalid zone boundaries ({b1}, {b2}) for n_points={n_points}.")
    if zone_names is None:
        zone_names = ("elastic", "peak_region", "post_peak")
    if len(zone_names) != 3:
        raise ValueError("zone_names must contain exactly three names.")
    return [
        (str(zone_names[0]), slice(0, b1)),
        (str(zone_names[1]), slice(b1, b2)),
        (str(zone_names[2]), slice(b2, n_points)),
    ]

def _curve_zone_boundaries_from_slices(zones):
    return [zone.stop for _, zone in zones[:-1]]

def _curve_integral(y, x):
    if y.shape[-1] <= 1:
        return np.zeros(y.shape[0], dtype=float)
    return np.trapz(y, x=x, axis=1)

def _coerce_curve_x_values(x_values):
    if x_values is None:
        return None
    if hasattr(x_values, "iloc") and getattr(x_values, "ndim", None) == 2:
        row = x_values.iloc[0]
        if hasattr(row, "iloc"):
            row = row.iloc[1:]
        return row.to_numpy(dtype=float) if hasattr(row, "to_numpy") else np.asarray(row, dtype=float)
    if hasattr(x_values, "to_numpy"):
        return x_values.to_numpy(dtype=float)
    return x_values

def curve_performance_diagnostics(
    y_pred,
    y_true,
    x_values=None,
    train_truth=None,
    zone_boundaries=None,
    zone_names=None,
    eps=1e-12,
):
    """
    Compute curve-level diagnostics for stress-strain or force-displacement outputs.

    The collapse ratio is mean pointwise std(predictions) divided by mean pointwise
    std(targets). Values near zero indicate average-curve collapse.
    """
    y_pred = _curve_2d_array(y_pred, "y_pred")
    y_true = _curve_2d_array(y_true, "y_true")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred shape {y_pred.shape} does not match y_true shape {y_true.shape}.")

    n_samples, n_points = y_true.shape
    x = _curve_x_array(x_values, n_points)
    zones = _curve_zone_slices(n_points, zone_boundaries=zone_boundaries, zone_names=zone_names)
    resolved_zone_boundaries = tuple(_curve_zone_boundaries_from_slices(zones))

    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    ddof = 1 if n_samples > 1 else 0
    true_std = np.nanstd(y_true, axis=0, ddof=ddof)
    pred_std = np.nanstd(y_pred, axis=0, ddof=ddof)
    std_ratio_curve = pred_std / np.maximum(true_std, eps)
    collapse_ratio = float(np.nanmean(pred_std) / max(float(np.nanmean(true_std)), eps))

    point_metrics = pd.DataFrame(
        {
            "x": x,
            "true_mean": np.nanmean(y_true, axis=0),
            "pred_mean": np.nanmean(y_pred, axis=0),
            "bias": np.nanmean(err, axis=0),
            "mae": np.nanmean(abs_err, axis=0),
            "rmse": np.sqrt(np.nanmean(sq_err, axis=0)),
            "true_std": true_std,
            "pred_std": pred_std,
            "std_ratio": std_ratio_curve,
        }
    )

    sample_mae = np.nanmean(abs_err, axis=1)
    sample_mse = np.nanmean(sq_err, axis=1)
    sample_rmse = np.sqrt(sample_mse)
    sample_bias = np.nanmean(err, axis=1)
    sample_max_abs_error = np.nanmax(abs_err, axis=1)
    sample_curve_corr = np.array([_safe_corr(p, t, eps=eps) for p, t in zip(y_pred, y_true)], dtype=float)

    true_peak_idx = np.nanargmax(y_true, axis=1)
    pred_peak_idx = np.nanargmax(y_pred, axis=1)
    true_peak = y_true[np.arange(n_samples), true_peak_idx]
    pred_peak = y_pred[np.arange(n_samples), pred_peak_idx]
    true_peak_x = x[true_peak_idx]
    pred_peak_x = x[pred_peak_idx]
    true_energy = _curve_integral(y_true, x)
    pred_energy = _curve_integral(y_pred, x)

    sample_metrics = pd.DataFrame(
        {
            "sample": np.arange(n_samples),
            "sample_mae": sample_mae,
            "sample_mse": sample_mse,
            "sample_rmse": sample_rmse,
            "sample_bias": sample_bias,
            "sample_max_abs_error": sample_max_abs_error,
            "sample_curve_corr": sample_curve_corr,
            "true_peak": true_peak,
            "pred_peak": pred_peak,
            "peak_error": pred_peak - true_peak,
            "true_peak_x": true_peak_x,
            "pred_peak_x": pred_peak_x,
            "peak_x_error": pred_peak_x - true_peak_x,
            "true_energy": true_energy,
            "pred_energy": pred_energy,
            "energy_error": pred_energy - true_energy,
        }
    )

    zone_rows = []
    for zone_name, zone in zones:
        zone_err = err[:, zone]
        zone_abs = abs_err[:, zone]
        zone_sq = sq_err[:, zone]
        zone_true_std = np.nanstd(y_true[:, zone], axis=0, ddof=ddof)
        zone_pred_std = np.nanstd(y_pred[:, zone], axis=0, ddof=ddof)
        zone_rows.append(
            {
                "zone": zone_name,
                "start_idx": zone.start,
                "end_idx": zone.stop,
                "x_start": float(x[zone.start]),
                "x_end": float(x[zone.stop - 1]),
                "mae": _safe_nanmean(zone_abs),
                "mse": _safe_nanmean(zone_sq),
                "rmse": float(np.sqrt(_safe_nanmean(zone_sq))),
                "bias": _safe_nanmean(zone_err),
                "true_std_mean": _safe_nanmean(zone_true_std),
                "pred_std_mean": _safe_nanmean(zone_pred_std),
                "collapse_ratio": float(
                    np.nanmean(zone_pred_std) / max(float(np.nanmean(zone_true_std)), eps)
                ),
            }
        )
    zone_metrics = pd.DataFrame(zone_rows)

    mse_value = _safe_nanmean(sq_err)
    rmse_value = float(np.sqrt(mse_value))
    true_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    true_std_global = float(np.nanstd(y_true, ddof=ddof))
    sse = float(np.nansum(sq_err))
    sst = float(np.nansum((y_true - np.nanmean(y_true)) ** 2))
    summary = {
        "n_samples": int(n_samples),
        "n_points": int(n_points),
        "mae": _safe_nanmean(abs_err),
        "mse": mse_value,
        "rmse": rmse_value,
        "bias": _safe_nanmean(err),
        "r2_global": float(1.0 - sse / sst) if sst > eps else np.nan,
        "nrmse_range": float(rmse_value / true_range) if true_range > eps else np.nan,
        "nrmse_std": float(rmse_value / true_std_global) if true_std_global > eps else np.nan,
        "collapse_ratio": collapse_ratio,
        "mean_sample_curve_corr": _safe_nanmean(sample_curve_corr),
        "median_sample_curve_corr": float(np.nanmedian(sample_curve_corr)) if not np.all(np.isnan(sample_curve_corr)) else np.nan,
        "peak_corr": _safe_corr(true_peak, pred_peak, eps=eps),
        "peak_x_corr": _safe_corr(true_peak_x, pred_peak_x, eps=eps),
        "energy_corr": _safe_corr(true_energy, pred_energy, eps=eps),
        "zone_boundaries": resolved_zone_boundaries,
    }

    baseline_source = None
    if train_truth is not None:
        train_truth = _curve_2d_array(train_truth, "train_truth")
        if train_truth.shape[1] != n_points:
            raise ValueError(
                f"train_truth curve length ({train_truth.shape[1]}) does not match y_true length ({n_points})."
            )
        baseline_curve = np.nanmean(train_truth, axis=0)
        baseline_source = "train_mean_curve"
    else:
        baseline_curve = np.nanmean(y_true, axis=0)
        baseline_source = "truth_mean_curve"

    baseline_err = baseline_curve.reshape(1, -1) - y_true
    baseline_mse = _safe_nanmean(baseline_err ** 2)
    baseline_rmse = float(np.sqrt(baseline_mse))
    baseline_mae = _safe_nanmean(np.abs(baseline_err))
    summary.update(
        {
            "mean_curve_baseline_source": baseline_source,
            "mean_curve_baseline_mae": baseline_mae,
            "mean_curve_baseline_mse": baseline_mse,
            "mean_curve_baseline_rmse": baseline_rmse,
            "skill_vs_mean_curve_rmse": float(1.0 - rmse_value / baseline_rmse) if baseline_rmse > eps else np.nan,
            "skill_vs_mean_curve_mae": float(1.0 - summary["mae"] / baseline_mae) if baseline_mae > eps else np.nan,
        }
    )

    return {
        "summary": summary,
        "sample_metrics": sample_metrics,
        "point_metrics": point_metrics,
        "zone_metrics": zone_metrics,
        "x": x,
        "y_pred": y_pred,
        "y_true": y_true,
        "baseline_curve": baseline_curve,
    }


# =============================================================================
# Curve Post-Processing
# =============================================================================

def _fmt_metric(value, digits=4):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"

def print_curve_diagnostics(diagnostics, label="Curve"):
    summary = diagnostics["summary"] if isinstance(diagnostics, dict) else diagnostics
    label = str(label).upper()
    print(
        f"{label} prediction diagnostics | "
        f"collapse ratio: {_fmt_metric(summary.get('collapse_ratio'), 3)} | "
        f"RMSE: {_fmt_metric(summary.get('rmse'))} | "
        f"mean-curve RMSE: {_fmt_metric(summary.get('mean_curve_baseline_rmse'))} | "
        f"skill vs mean curve: {_fmt_metric(summary.get('skill_vs_mean_curve_rmse'), 3)} | "
        f"peak corr: {_fmt_metric(summary.get('peak_corr'), 3)} | "
        f"energy corr: {_fmt_metric(summary.get('energy_corr'), 3)} | "
        f"sample curve corr: {_fmt_metric(summary.get('mean_sample_curve_corr'), 3)}"
    )
    if isinstance(diagnostics, dict) and "zone_metrics" in diagnostics:
        zone_text = []
        for _, row in diagnostics["zone_metrics"].iterrows():
            zone_text.append(
                f"{row['zone']}: collapse {_fmt_metric(row['collapse_ratio'], 3)}, "
                f"RMSE {_fmt_metric(row['rmse'])}"
            )
        print(f"{label} zones | " + " | ".join(zone_text))

def _diagnostic_order(sample_metrics, sort_by="rmse"):
    sort_by = "sample_rmse" if sort_by is None else str(sort_by)
    aliases = {
        "rmse": "sample_rmse",
        "mae": "sample_mae",
        "bias": "sample_bias",
        "max": "sample_max_abs_error",
        "max_abs_error": "sample_max_abs_error",
        "corr": "sample_curve_corr",
        "curve_corr": "sample_curve_corr",
        "sample": "sample",
        "index": "sample",
    }
    col = aliases.get(sort_by.lower(), sort_by)
    if col not in sample_metrics.columns:
        col = "sample_rmse"
    ascending = col in ["sample", "sample_curve_corr"]
    ordered = sample_metrics.sort_values(col, ascending=ascending)
    return ordered["sample"].astype(int).to_numpy()

def _add_zone_lines(ax, x, zone_boundaries=None, **kwargs):
    if zone_boundaries is None:
        return
    n_points = len(x)
    for boundary in zone_boundaries:
        boundary = int(boundary)
        if 0 < boundary < n_points:
            ax.axvline(x[boundary], **kwargs)

def plot_prediction_error_curves(
    OUT_df,
    test_outputs,
    truth,
    diagnostics=None,
    mode="ut",
    max_samples=50,
    sort_by="rmse",
    zone_boundaries=None,
):
    if diagnostics is None:
        diagnostics = curve_performance_diagnostics(
            test_outputs,
            truth,
            x_values=OUT_df,
            zone_boundaries=zone_boundaries,
        )
    x = diagnostics["x"]
    err = diagnostics["y_pred"] - diagnostics["y_true"]
    order = _diagnostic_order(diagnostics["sample_metrics"], sort_by=sort_by)
    if max_samples is not None:
        order = order[: int(max_samples)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for sample_idx in order:
        ax.plot(x, err[sample_idx], color="tab:blue", alpha=0.18, linewidth=1.0)
    ax.plot(x, np.nanmean(err[order], axis=0), color="black", linewidth=2.0, label="Mean error")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    _add_zone_lines(ax, x, diagnostics["summary"].get("zone_boundaries"), color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"{str(mode).upper()} Prediction Error Curves")
    ax.set_xlabel("Macroscopic Strain ($\\epsilon$)" if str(mode).lower() == "ut" else "Load-line Displacement ($d$)")
    ax.set_ylabel("Prediction - Truth")
    ax.legend()
    plt.show()
    return fig, ax

def plot_curve_correlation_matrix(diagnostics, columns=None, method="pearson", figsize=(8, 7)):
    sample_metrics = diagnostics["sample_metrics"] if isinstance(diagnostics, dict) else diagnostics
    if columns is None:
        columns = [
            "sample_rmse",
            "sample_bias",
            "sample_curve_corr",
            "true_peak",
            "pred_peak",
            "peak_error",
            "true_peak_x",
            "pred_peak_x",
            "true_energy",
            "pred_energy",
            "energy_error",
        ]
    columns = [col for col in columns if col in sample_metrics.columns]
    if len(columns) == 0:
        raise ValueError("No requested columns are present in the diagnostics sample metrics.")
    corr = sample_metrics[columns].corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title("Prediction Diagnostic Correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if len(columns) <= 10:
        for i in range(len(columns)):
            for j in range(len(columns)):
                val = corr.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    plt.show()
    return corr, fig, ax

def plot_curve_diagnostics(
    OUT_df,
    test_outputs,
    truth,
    diagnostics=None,
    mode="ut",
    max_samples=64,
    sort_by="rmse",
    zone_boundaries=None,
):
    if diagnostics is None:
        diagnostics = curve_performance_diagnostics(
            test_outputs,
            truth,
            x_values=OUT_df,
            zone_boundaries=zone_boundaries,
        )

    x = diagnostics["x"]
    y_pred = diagnostics["y_pred"]
    y_true = diagnostics["y_true"]
    point = diagnostics["point_metrics"]
    samples = diagnostics["sample_metrics"]
    summary = diagnostics["summary"]
    order = _diagnostic_order(samples, sort_by=sort_by)
    if max_samples is not None:
        order = order[: int(max_samples)]
    err = y_pred - y_true

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    ax.plot(x, point["true_mean"], color="darkgreen", label="Truth mean")
    ax.fill_between(
        x,
        point["true_mean"] - point["true_std"],
        point["true_mean"] + point["true_std"],
        color="darkgreen",
        alpha=0.18,
        label="Truth +/- std",
    )
    ax.plot(x, point["pred_mean"], color="orangered", label="Prediction mean")
    ax.fill_between(
        x,
        point["pred_mean"] - point["pred_std"],
        point["pred_mean"] + point["pred_std"],
        color="orangered",
        alpha=0.18,
        label="Prediction +/- std",
    )
    if "baseline_curve" in diagnostics:
        ax.plot(x, diagnostics["baseline_curve"], color="gray", linestyle="--", linewidth=1.5, label="Mean-curve baseline")
    _add_zone_lines(ax, x, summary.get("zone_boundaries"), color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Mean Curve And Diversity")
    ax.set_xlabel("Macroscopic Strain ($\\epsilon$)" if str(mode).lower() == "ut" else "Load-line Displacement ($d$)")
    ax.set_ylabel("Stress / force")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, point["mae"], color="tab:blue", label="MAE")
    ax.plot(x, point["rmse"], color="tab:orange", label="RMSE")
    ax.plot(x, point["bias"], color="black", linestyle="--", label="Bias")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    _add_zone_lines(ax, x, summary.get("zone_boundaries"), color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Pointwise Error")
    ax.set_xlabel("Macroscopic Strain ($\\epsilon$)" if str(mode).lower() == "ut" else "Load-line Displacement ($d$)")
    ax.set_ylabel("Error")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(x, point["true_std"], color="darkgreen", label="Truth std")
    ax.plot(x, point["pred_std"], color="orangered", label="Prediction std")
    ax2 = ax.twinx()
    ax2.plot(x, point["std_ratio"], color="tab:purple", alpha=0.65, label="Pred/true std")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    _add_zone_lines(ax, x, summary.get("zone_boundaries"), color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Diversity Collapse Ratio = {_fmt_metric(summary.get('collapse_ratio'), 3)}")
    ax.set_xlabel("Macroscopic Strain ($\\epsilon$)" if str(mode).lower() == "ut" else "Load-line Displacement ($d$)")
    ax.set_ylabel("Across-sample std")
    ax2.set_ylabel("Std ratio")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[1, 0]
    heat = err[order]
    vmax = np.nanpercentile(np.abs(heat), 98) if heat.size else 1.0
    vmax = max(float(vmax), 1e-12)
    im = ax.imshow(
        heat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[float(x[0]), float(x[-1]), len(order), 0],
    )
    _add_zone_lines(ax, x, summary.get("zone_boundaries"), color="black", linestyle=":", alpha=0.45)
    ax.set_title(f"Residual Heatmap ({len(order)} samples)")
    ax.set_xlabel("Macroscopic Strain ($\\epsilon$)" if str(mode).lower() == "ut" else "Load-line Displacement ($d$)")
    ax.set_ylabel(f"Samples sorted by {sort_by}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.scatter(samples["true_peak"], samples["pred_peak"], c=samples["sample_rmse"], cmap="viridis", s=28, alpha=0.8)
    lo = np.nanmin([samples["true_peak"].min(), samples["pred_peak"].min()])
    hi = np.nanmax([samples["true_peak"].max(), samples["pred_peak"].max()])
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.0)
    ax.set_title(f"Peak Correlation = {_fmt_metric(summary.get('peak_corr'), 3)}")
    ax.set_xlabel("True peak")
    ax.set_ylabel("Predicted peak")

    ax = axes[1, 2]
    ax.scatter(samples["true_energy"], samples["pred_energy"], c=samples["sample_rmse"], cmap="viridis", s=28, alpha=0.8)
    lo = np.nanmin([samples["true_energy"].min(), samples["pred_energy"].min()])
    hi = np.nanmax([samples["true_energy"].max(), samples["pred_energy"].max()])
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.0)
    ax.set_title(f"Energy Correlation = {_fmt_metric(summary.get('energy_corr'), 3)}")
    ax.set_xlabel("True integrated curve")
    ax.set_ylabel("Predicted integrated curve")

    fig.tight_layout()
    plt.show()
    return fig, axes

def _postprocess_data_kind_label(data):
    input_kind = str(getattr(data, "input_kind", "geometry") or "geometry").lower()
    output_kind = str(getattr(data, "output_kind", "curve") or "curve").lower()
    if input_kind == "field" and output_kind == "curve":
        return "FieldToCurve"
    return output_kind


def _postprocess_load_run(
    run_path,
    run_root=None,
    prefer_hpo_best=True,
    load_data=True,
    load_model=True,
    data_path_override=None,
    device="cpu",
    verbose=True,
):
    artifacts = postprocess_resolve_artifacts(run_path, run_root=run_root, prefer_hpo_best=prefer_hpo_best)
    loaded = postprocess_load_artifacts(artifacts)

    data = None
    if load_data and artifacts.get("data_json") is not None:
        try:
            data = postprocess_load_data(
                artifacts["data_json"],
                data_path_override=data_path_override,
                auto_path_root=run_root,
            )
            if verbose:
                print("Loaded DATA from:", artifacts["data_json"])
                print("DATA input_kind:", getattr(data, "input_kind", "geometry"))
                print("DATA output_kind:", getattr(data, "output_kind", "curve"))
                print("DATA kind:", _postprocess_data_kind_label(data))
        except Exception as exc:
            if verbose:
                print("DATA load failed:", repr(exc))

    model = None
    if load_model and artifacts.get("model_json") is not None and data is not None:
        try:
            from resources.MLmodels import MODEL
            model = MODEL.from_json(
                artifacts["model_json"],
                data=data,
                load_weights=artifacts.get("model_mdl") is not None,
                model_path=str(artifacts["model_mdl"]) if artifacts.get("model_mdl") is not None else None,
                device=device,
                scan_matches_on_init=False,
            )
            model = postprocess_attach_results(model, loaded)
            if verbose:
                print("Loaded MODEL from:", artifacts["model_json"])
        except Exception as exc:
            if verbose:
                print("MODEL load failed:", repr(exc))
    elif load_model and verbose:
        print("MODEL was not loaded because a model JSON or DATA object is missing.")

    return artifacts, loaded, data, model


def postprocess_load_curve_run(
    run_path,
    run_root=None,
    prefer_hpo_best=True,
    load_data=True,
    load_model=True,
    data_path_override=None,
    device="cpu",
    verbose=True,
):
    return _postprocess_load_run(
        run_path,
        run_root=run_root,
        prefer_hpo_best=prefer_hpo_best,
        load_data=load_data,
        load_model=load_model,
        data_path_override=data_path_override,
        device=device,
        verbose=verbose,
    )


def postprocess_curve_run_overview(
    artifacts,
    loaded,
    data=None,
    run_name=None,
    run_type=None,
    mech_mode=None,
    view_mode="UT",
    model_name=None,
    device=None,
    active_split=None,
):
    descriptor = loaded.get("descriptor") or {}
    metrics = loaded.get("metrics") or {}
    hpo = loaded.get("hpo") or {}

    saved_output_kind = _postprocess_saved_output_kind(loaded)
    loaded_output_kind = getattr(data, "output_kind", None) if data is not None else saved_output_kind

    summary_values = {
        "run": run_name,
        "run_type": run_type,
        "mechMode": mech_mode,
        "VIEW_MODE": view_mode,
        "model": model_name,
        "output_kind": loaded_output_kind,
        "device": device,
        "evaluation_split": metrics.get("evaluation_split"),
        "is_hpo": artifacts.get("is_hpo"),
        "model_json": artifacts.get("model_json"),
        "results_dir": artifacts.get("results_dir"),
    }
    summary_fields = [
        "run",
        "run_type",
        "mechMode",
        "VIEW_MODE",
        "model",
        "output_kind",
        "device",
        "evaluation_split",
        "is_hpo",
        "model_json",
        "results_dir",
    ]
    summary_table = pd.DataFrame([(key, summary_values.get(key)) for key in summary_fields], columns=["item", "value"])

    run_descriptor = descriptor.get("run_descriptor") if isinstance(descriptor, dict) else None
    model_setup = pd.DataFrame(columns=["item", "value"])
    if isinstance(run_descriptor, dict):
        compact_descriptor = {
            "model_type": descriptor.get("model_type"),
            "name": descriptor.get("name"),
            "in_size": run_descriptor.get("in_size"),
            "out_size": run_descriptor.get("out_size"),
            "hidden_size": run_descriptor.get("hidden_size"),
            "n_layers": run_descriptor.get("n_layers"),
            "n_heads": run_descriptor.get("n_heads"),
            "dropout": run_descriptor.get("dropout"),
        }
        compact_descriptor = {key: value for key, value in compact_descriptor.items() if value is not None}
        if compact_descriptor:
            model_setup = pd.DataFrame(compact_descriptor.items(), columns=["item", "value"])

    active_metric_keys = ["best_epoch", "best_loss", "best_mse", "best_rmse", "mae", "mse", "rmse", "best", "worst"]
    metric_rows = []
    for key in [f"{str(view_mode).upper()}_{metric}" for metric in active_metric_keys]:
        if key in metrics:
            metric_rows.append((key, metrics[key]))
    active_metrics = pd.DataFrame(metric_rows, columns=["metric", "value"])

    available_evals = postprocess_available_evaluations(loaded)
    curve_eval_columns = ["mode", "split", "outputs", "truth", "sample_metrics", "point_metrics", "zone_metrics"]
    available_curve_evals = available_evals[[col for col in curve_eval_columns if col in available_evals.columns]] if not available_evals.empty else pd.DataFrame(columns=curve_eval_columns)

    resolved_split = active_split
    if resolved_split is None:
        resolved_split = metrics.get("evaluation_split", None)
    if resolved_split is None and not available_evals.empty:
        matching = available_evals[available_evals["mode"].astype(str).str.upper() == str(view_mode).upper()]
        resolved_split = (matching.iloc[0] if not matching.empty else available_evals.iloc[0])["split"]

    warnings = []
    if loaded_output_kind is not None and str(loaded_output_kind).lower() == "field":
        warnings.append("This notebook is intended for curve-output runs, but the selected run looks like output_kind='field'.")

    return {
        "summary": summary_table,
        "model_setup": model_setup,
        "active_metrics": active_metrics,
        "available_evals": available_evals,
        "available_curve_evals": available_curve_evals,
        "active_split": resolved_split,
        "hpo": hpo,
        "warnings": warnings,
    }

def display_curve_run_overview(overview):
    from IPython.display import Markdown, display

    for warning in overview.get("warnings", []):
        print("WARNING:", warning)

    display(overview["summary"])
    if not overview["model_setup"].empty:
        display(Markdown("### Model Setup"))
        display(overview["model_setup"])
    if not overview["active_metrics"].empty:
        display(Markdown("### Saved Metrics For Active View"))
        display(overview["active_metrics"])

    display(Markdown("### Available Curve Predictions / Diagnostic Tables"))
    display(overview["available_curve_evals"])

    hpo = overview.get("hpo", {})
    if hpo:
        display(Markdown("### HPO Summary"))
        for key, value in hpo.items():
            if isinstance(value, dict):
                display(Markdown(f"#### {key}"))
                display(pd.DataFrame(value.items(), columns=["parameter", "value"]))
    else:
        print("No HPO files were found for this run.")

def postprocess_build_active_curve_diagnostics(
    data,
    loaded,
    available_evals,
    view_mode="UT",
    active_split=None,
    model=None,
    zone_boundaries=None,
):
    diagnostics = {}
    for _, row in available_evals.iterrows():
        mode = str(row["mode"]).upper()
        split = str(row["split"]).lower()
        saved_summary = (loaded.get("diagnostics_summary") or {}).get(mode, {})
        zone_cfg = zone_boundaries
        if zone_cfg is None and isinstance(saved_summary, dict):
            zone_cfg = saved_summary.get("zone_boundaries", None)

        diag = postprocess_build_diagnostics(
            data,
            loaded,
            mode=mode,
            split=split,
            zone_boundaries=zone_cfg,
            prefer_saved_tables=True,
            recompute_from_predictions=False,
        )
        if diag is None:
            print(f"No saved diagnostics are available for {mode} {split}.")
            continue
        if "field_shape" in diag:
            print(f"Skipping {mode} {split}: diagnostics are field diagnostics.")
            continue

        diagnostics[(mode, split)] = diag
        if model is not None:
            setattr(model, f"{mode}_{split}_diagnostics", diag)
            if split == "test":
                setattr(model, f"{mode}_diagnostics", diag)
                setattr(model, f"{mode}_prediction_summary", diag.get("summary"))

    active_key = (str(view_mode).upper(), str(active_split).lower()) if view_mode and active_split else None
    active_diag = diagnostics.get(active_key) if active_key is not None else None
    return diagnostics, active_key, active_diag

def curve_summary_table(diagnostics, metrics=None):
    if diagnostics is None:
        return pd.DataFrame(columns=["metric", "value"])
    metrics = metrics or [
        "rmse",
        "mae",
        "mse",
        "bias",
        "r2_global",
        "collapse_ratio",
        "mean_curve_baseline_rmse",
        "skill_vs_mean_curve_rmse",
        "mean_sample_curve_corr",
        "peak_corr",
        "peak_x_corr",
        "energy_corr",
        "n_samples",
        "n_points",
    ]
    summary = diagnostics.get("summary", {})
    return pd.DataFrame([(key, summary.get(key)) for key in metrics if key in summary], columns=["metric", "value"])

def _curve_out_df(data, mode):
    return getattr(data, f"{str(mode).upper()}_OUT_df", None) if data is not None else None

def display_curve_main_dashboard(
    diagnostics,
    data=None,
    mode="UT",
    max_samples=64,
    sort_by="rmse",
):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None
    return plot_curve_diagnostics(
        _curve_out_df(data, mode),
        diagnostics["y_pred"],
        diagnostics["y_true"],
        diagnostics=diagnostics,
        mode=str(mode).lower(),
        max_samples=max_samples,
        sort_by=sort_by,
    )

def display_curve_pointwise_summary(
    diagnostics,
    data=None,
    mode="UT",
    max_error_curves=100,
    sort_by="rmse",
    show_correlation=True,
):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None

    plot_prediction_error_curves(
        _curve_out_df(data, mode),
        diagnostics["y_pred"],
        diagnostics["y_true"],
        diagnostics=diagnostics,
        mode=str(mode).lower(),
        max_samples=max_error_curves,
        sort_by=sort_by,
    )

    if show_correlation:
        plot_curve_correlation_matrix(diagnostics, method="pearson")

def plot_curve_all_point_parity(
    diagnostics,
    gridsize=75,
    cmap="viridis",
    mincnt=1,
    figsize=(6, 5),
):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None, None

    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    true_values = y_true[mask]
    pred_values = y_pred[mask]
    if true_values.size == 0:
        print("No valid curve points are available for parity plotting.")
        return None, None

    lo = np.nanpercentile(np.concatenate([true_values, pred_values]), 1)
    hi = np.nanpercentile(np.concatenate([true_values, pred_values]), 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(np.concatenate([true_values, pred_values]))
        hi = np.nanmax(np.concatenate([true_values, pred_values]))
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(true_values, pred_values, gridsize=gridsize, mincnt=mincnt, cmap=cmap)
    ax.plot([lo, hi], [lo, hi], color="white", linewidth=2.0)
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    bias = float(np.nanmean(pred_values - true_values))
    rmse = float(np.sqrt(np.nanmean((pred_values - true_values) ** 2)))
    ax.set_title(f"All-Point Curve Parity\nRMSE={rmse:.4g}, bias={bias:.4g}, n={true_values.size:,}")
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    plt.show()
    return fig, ax

def curve_initial_slope_diagnostics(
    diagnostics,
    window_fraction=0.08,
    x_threshold=None,
    min_points=3,
    eps=1e-12,
):
    if diagnostics is None:
        return pd.DataFrame(), {}

    x = np.asarray(diagnostics["x"], dtype=float).reshape(-1)
    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    n_points = x.size

    if x_threshold is not None:
        window_mask = x <= float(x_threshold)
        if int(np.sum(window_mask)) < int(min_points):
            order = np.argsort(x)
            window_indices = order[: int(min_points)]
        else:
            window_indices = np.where(window_mask)[0]
    else:
        window_size = max(int(min_points), int(np.ceil(float(window_fraction) * n_points)))
        window_size = min(window_size, n_points)
        window_indices = np.arange(window_size)

    window_indices = np.asarray(window_indices, dtype=int)
    x_window = x[window_indices]

    def _slope(y_values):
        values = np.asarray(y_values, dtype=float)[window_indices]
        mask = np.isfinite(x_window) & np.isfinite(values)
        if int(np.sum(mask)) < int(min_points):
            return np.nan
        try:
            return float(np.polyfit(x_window[mask], values[mask], deg=1)[0])
        except Exception:
            return np.nan

    true_slope = np.array([_slope(row) for row in y_true], dtype=float)
    pred_slope = np.array([_slope(row) for row in y_pred], dtype=float)
    slope_error = pred_slope - true_slope
    slope_percent_error = np.full_like(slope_error, np.nan, dtype=float)
    denom_ok = np.abs(true_slope) > eps
    slope_percent_error[denom_ok] = 100.0 * slope_error[denom_ok] / true_slope[denom_ok]

    slopes = pd.DataFrame({
        "sample": np.arange(y_true.shape[0]),
        "true_slope": true_slope,
        "pred_slope": pred_slope,
        "slope_error": slope_error,
        "slope_percent_error": slope_percent_error,
    })
    finite = np.isfinite(true_slope) & np.isfinite(pred_slope)
    slope_corr = _safe_corr(true_slope[finite], pred_slope[finite], eps=eps) if np.any(finite) else np.nan
    summary = {
        "window_start_idx": int(window_indices.min()) if window_indices.size else None,
        "window_end_idx": int(window_indices.max()) if window_indices.size else None,
        "window_start_x": float(np.nanmin(x_window)) if x_window.size else np.nan,
        "window_end_x": float(np.nanmax(x_window)) if x_window.size else np.nan,
        "window_points": int(window_indices.size),
        "slope_correlation": slope_corr,
        "slope_mae": _safe_nanmean(np.abs(slope_error)),
        "slope_rmse": float(np.sqrt(_safe_nanmean(slope_error ** 2))),
        "slope_bias": _safe_nanmean(slope_error),
        "slope_median_abs_percent_error": float(np.nanmedian(np.abs(slope_percent_error)))
        if np.any(np.isfinite(slope_percent_error)) else np.nan,
    }
    return slopes, summary

def display_curve_initial_slope_summary(
    diagnostics,
    window_fraction=0.08,
    x_threshold=None,
    min_points=3,
    top_n=5,
    figsize=(12, 4.5),
):
    from IPython.display import Markdown, display

    slopes, summary = curve_initial_slope_diagnostics(
        diagnostics,
        window_fraction=window_fraction,
        x_threshold=x_threshold,
        min_points=min_points,
    )
    if slopes.empty:
        print("Initial slope diagnostics are unavailable.")
        return slopes, summary

    display(Markdown("### Initial Slope Summary"))
    display(pd.DataFrame(summary.items(), columns=["metric", "value"]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    finite = np.isfinite(slopes["true_slope"]) & np.isfinite(slopes["pred_slope"])
    if np.any(finite):
        axes[0].scatter(
            slopes.loc[finite, "true_slope"],
            slopes.loc[finite, "pred_slope"],
            c=np.abs(slopes.loc[finite, "slope_percent_error"]),
            cmap="viridis",
            alpha=0.8,
        )
        lo = np.nanmin([slopes["true_slope"].min(), slopes["pred_slope"].min()])
        hi = np.nanmax([slopes["true_slope"].max(), slopes["pred_slope"].max()])
        axes[0].plot([lo, hi], [lo, hi], color="gray", linestyle="--")
    axes[0].set_title(f"Initial slope parity\ncorr={_fmt_metric(summary.get('slope_correlation'), 3)}")
    axes[0].set_xlabel("True slope")
    axes[0].set_ylabel("Predicted slope")

    percent_error = slopes["slope_percent_error"].replace([np.inf, -np.inf], np.nan).dropna()
    if not percent_error.empty:
        axes[1].hist(percent_error, bins=40, color="tab:blue", alpha=0.8)
        axes[1].axvline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Initial slope percent error")
    axes[1].set_xlabel("Percent error")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    plt.show()

    if top_n is not None and int(top_n) > 0:
        display(Markdown("### Worst Initial Slope Samples"))
        worst = slopes.assign(abs_percent_error=np.abs(slopes["slope_percent_error"]))
        display(worst.sort_values("abs_percent_error", ascending=False).head(int(top_n)))
    return slopes, summary

def display_curve_sample_error_summary(
    diagnostics,
    bins=40,
    ncols=3,
    top_n=5,
    columns=None,
):
    from IPython.display import Markdown, display

    if diagnostics is None:
        print("No active diagnostics are available.")
        return
    samples = diagnostics.get("sample_metrics")
    if samples is None or not hasattr(samples, "copy"):
        print("Sample metrics are unavailable.")
        return

    samples = samples.copy()
    if "sample_rmse" in samples.columns:
        display(Markdown("### Best And Worst Samples By RMSE"))
        best_worst = pd.concat([
            samples.sort_values("sample_rmse").head(int(top_n)),
            samples.sort_values("sample_rmse").tail(int(top_n)),
        ]).drop_duplicates()
        display(best_worst)

    columns = columns or [
        "sample_mae",
        "sample_mse",
        "sample_rmse",
        "sample_bias",
        "sample_max_abs_error",
        "sample_curve_corr",
        "peak_error",
        "peak_x_error",
        "energy_error",
    ]
    plot_cols = [col for col in columns if col in samples.columns]
    if plot_cols:
        nrows = int(np.ceil(len(plot_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, col in zip(axes, plot_cols):
            ax.hist(samples[col].dropna(), bins=bins, color="tab:blue", alpha=0.8)
            ax.set_title(col)
            ax.set_ylabel("Count")
        for ax in axes[len(plot_cols):]:
            ax.axis("off")
        fig.tight_layout()
        plt.show()

        display(Markdown("### Sample Metric Summary"))
        display(samples[plot_cols].describe().T)

def display_curve_peak_energy_summary(diagnostics):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None
    samples = diagnostics.get("sample_metrics")
    if samples is None or not hasattr(samples, "copy"):
        print("Sample metrics are unavailable.")
        return None

    samples = samples.copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("true_peak", "pred_peak", "Peak value", "True peak", "Predicted peak"),
        ("true_peak_x", "pred_peak_x", "Peak location", "True peak x", "Predicted peak x"),
        ("true_energy", "pred_energy", "Integrated curve energy", "True energy", "Predicted energy"),
    ]
    for ax, (true_col, pred_col, title, xlabel, ylabel) in zip(axes, specs):
        if true_col not in samples.columns or pred_col not in samples.columns:
            ax.axis("off")
            continue
        ax.scatter(samples[true_col], samples[pred_col], c=samples.get("sample_rmse"), cmap="viridis", alpha=0.8)
        lo = np.nanmin([samples[true_col].min(), samples[pred_col].min()])
        hi = np.nanmax([samples[true_col].max(), samples[pred_col].max()])
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()
    return fig, axes

def display_curve_zone_summary(diagnostics):
    from IPython.display import Markdown, display

    if diagnostics is None:
        print("No active diagnostics are available.")
        return None
    zone = diagnostics.get("zone_metrics")
    if zone is None or not hasattr(zone, "copy"):
        print("Zone metrics are unavailable.")
        return None

    zone = zone.copy()
    display(Markdown("### Zone Metrics"))
    display(zone)

    zone_cols = ["mae", "mse", "rmse", "bias", "true_std_mean", "pred_std_mean", "collapse_ratio"]
    zone_cols = [col for col in zone_cols if col in zone.columns]
    if not zone_cols:
        return None
    nrows = 2
    ncols = int(np.ceil(len(zone_cols) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.7 * nrows), sharex=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, zone_cols):
        ax.bar(zone["zone"], zone[col], color="tab:blue", alpha=0.85)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=35)
    for ax in axes[len(zone_cols):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig, axes

def display_curve_sample_examples(
    diagnostics,
    selected_sample=0,
    random_count=8,
    random_seed=42,
    rank_by="sample_rmse",
    ncols=2,
):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None
    samples = diagnostics.get("sample_metrics")
    if samples is None or not hasattr(samples, "copy"):
        print("Sample metrics are unavailable.")
        return None

    samples = samples.copy()
    y_pred = diagnostics["y_pred"]
    y_true = diagnostics["y_true"]
    x = diagnostics["x"]

    rank_col = rank_by if rank_by in samples.columns else "sample_rmse"
    candidate_indices = []
    if rank_col in samples.columns:
        candidate_indices.extend(samples.sort_values(rank_col).head(1)["sample"].astype(int).tolist())
        candidate_indices.extend(samples.sort_values(rank_col).tail(1)["sample"].astype(int).tolist())
    candidate_indices.append(int(selected_sample))

    rng = np.random.default_rng(random_seed)
    if len(samples) > 0:
        candidate_indices.extend(
            rng.choice(samples["sample"].astype(int), size=min(int(random_count), len(samples)), replace=False).tolist()
        )
    candidate_indices = [idx for idx in dict.fromkeys(candidate_indices) if 0 <= idx < len(y_pred)]

    nrows = int(np.ceil(len(candidate_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, candidate_indices):
        ax.plot(x, y_true[idx], color="darkgreen", label="Truth")
        ax.plot(x, y_pred[idx], color="orangered", label="Prediction")
        row = samples.loc[samples["sample"].astype(int) == int(idx)]
        rmse_text = f", RMSE={row['sample_rmse'].iloc[0]:.4g}" if not row.empty and "sample_rmse" in row else ""
        ax.set_title(f"Sample {idx}{rmse_text}")
        ax.legend(fontsize=8)
    for ax in axes[len(candidate_indices):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig, axes

def display_curve_loss_component_breakdown(model, diagnostics, active_key=None):
    from IPython.display import display
    import torch

    if model is None:
        print("MODEL is not loaded, so loss components cannot be evaluated.")
        return pd.DataFrame()
    if diagnostics is None:
        print("No active diagnostics are available.")
        return pd.DataFrame()

    mode = active_key[0] if active_key is not None else "UT"
    loss_rows = []
    y_pred_t = torch.as_tensor(diagnostics["y_pred"], dtype=torch.float32, device=model.device)
    y_true_t = torch.as_tensor(diagnostics["y_true"], dtype=torch.float32, device=model.device)
    for loss_idx, loss_obj in enumerate(getattr(model, f"{mode}_losses", [])):
        if not hasattr(loss_obj, "component_losses"):
            loss_rows.append({
                "loss_index": loss_idx,
                "loss_class": loss_obj.__class__.__name__,
                "component": "total",
                "value": np.nan,
                "note": "component_losses unavailable",
            })
            continue
        try:
            with torch.no_grad():
                components = loss_obj.component_losses(y_pred_t, y_true_t, weighted=False)
                weighted_components = loss_obj.component_losses(y_pred_t, y_true_t, weighted=True)
            for comp_name, comp_value in components.items():
                value = comp_value.detach().cpu().item() if torch.is_tensor(comp_value) else float(comp_value)
                weighted_value = weighted_components.get(comp_name, np.nan)
                weighted_value = weighted_value.detach().cpu().item() if torch.is_tensor(weighted_value) else float(weighted_value)
                loss_rows.append({
                    "loss_index": loss_idx,
                    "loss_class": loss_obj.__class__.__name__,
                    "component": comp_name,
                    "value": value,
                    "weighted_value": weighted_value,
                })
        except Exception as exc:
            loss_rows.append({
                "loss_index": loss_idx,
                "loss_class": loss_obj.__class__.__name__,
                "component": "error",
                "value": np.nan,
                "note": repr(exc),
            })

    loss_components_df = pd.DataFrame(loss_rows)
    display(loss_components_df)

    if not loss_components_df.empty and "weighted_value" in loss_components_df.columns:
        plot_df = loss_components_df.dropna(subset=["weighted_value"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(plot_df["component"], plot_df["weighted_value"], color="tab:blue")
            ax.set_title("Weighted Loss Components")
            ax.set_yscale("log")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            plt.show()
    return loss_components_df

# =============================================================================
# Field Metrics
# =============================================================================

def _field_to_frame_node_component(data, field_shape=None, name="field"):
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 4:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape [samples, nodes, node_outputs] or [samples, frames, nodes, components].")
    if field_shape is None:
        raise ValueError(f"{name} needs field_shape=(frames, nodes, components) to reconstruct field tensors.")

    n_frames, n_nodes, n_components = [int(v) for v in field_shape]
    expected = n_frames * n_components
    if arr.shape[1] != n_nodes or arr.shape[2] != expected:
        raise ValueError(
            f"{name} shape {arr.shape} is incompatible with field_shape={tuple(field_shape)}. "
            f"Expected [samples, {n_nodes}, {expected}]."
        )
    return arr.reshape(arr.shape[0], n_nodes, n_frames, n_components).transpose(0, 2, 1, 3)

def _field_metric(err, valid, reducer=np.nanmean):
    values = np.where(valid, err, np.nan)
    if not np.any(np.isfinite(values)):
        return np.nan
    return float(reducer(values))

def _field_rmse(err, valid):
    mse_value = _field_metric(err ** 2, valid, reducer=np.nanmean)
    return float(np.sqrt(mse_value)) if np.isfinite(mse_value) else np.nan

def _field_mae(err, valid):
    return _field_metric(np.abs(err), valid, reducer=np.nanmean)

def _field_valid_fraction(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid))

def _field_nanstd(arr, axis=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanstd(arr, axis=axis)

def _field_default_components(n_components):
    return [f"U{i + 1}" for i in range(int(n_components))]

def field_performance_diagnostics(
    y_pred,
    y_true,
    field_shape=None,
    frame_values=None,
    components=None,
    node_labels=None,
    node_coords=None,
    train_truth=None,
    eps=1e-12,
):
    """
    Compute masked diagnostics for node-wise field outputs.

    y_pred/y_true may be [samples, nodes, frames*components] or
    [samples, frames, nodes, components]. Invalid truth values should be NaN.
    """
    pred = _field_to_frame_node_component(y_pred, field_shape=field_shape, name="y_pred")
    true = _field_to_frame_node_component(y_true, field_shape=field_shape, name="y_true")
    if pred.shape != true.shape:
        raise ValueError(f"Field prediction and truth shapes must match; got {pred.shape} and {true.shape}.")

    n_samples, n_frames, n_nodes, n_components = pred.shape
    if frame_values is None:
        frame_values = np.arange(n_frames)
    frame_values = np.asarray(frame_values)
    if frame_values.size != n_frames:
        frame_values = np.arange(n_frames)

    if components is None:
        components = _field_default_components(n_components)
    components = [str(c) for c in components]
    if len(components) != n_components:
        components = _field_default_components(n_components)

    if node_labels is None:
        node_labels = np.arange(n_nodes)
    node_labels = np.asarray(node_labels)
    if node_labels.size != n_nodes:
        node_labels = np.arange(n_nodes)

    valid = np.isfinite(true) & np.isfinite(pred)
    err = pred - true

    sample_rows = []
    for i in range(n_samples):
        sample_valid = valid[i]
        sample_rows.append({
            "sample": i,
            "sample_mae": _field_mae(err[i], sample_valid),
            "sample_mse": _field_metric(err[i] ** 2, sample_valid, reducer=np.nanmean),
            "sample_rmse": _field_rmse(err[i], sample_valid),
            "sample_bias": _field_metric(err[i], sample_valid, reducer=np.nanmean),
            "valid_fraction": _field_valid_fraction(sample_valid),
        })
    sample_metrics = pd.DataFrame(sample_rows)

    frame_rows = []
    for frame_idx in range(n_frames):
        frame_valid = valid[:, frame_idx, :, :]
        frame_rows.append({
            "frame": frame_idx,
            "frame_value": frame_values[frame_idx],
            "mae": _field_mae(err[:, frame_idx, :, :], frame_valid),
            "mse": _field_metric(err[:, frame_idx, :, :] ** 2, frame_valid, reducer=np.nanmean),
            "rmse": _field_rmse(err[:, frame_idx, :, :], frame_valid),
            "bias": _field_metric(err[:, frame_idx, :, :], frame_valid, reducer=np.nanmean),
            "valid_fraction": _field_valid_fraction(frame_valid),
        })
    frame_metrics = pd.DataFrame(frame_rows)

    component_rows = []
    for comp_idx, comp_name in enumerate(components):
        comp_valid = valid[:, :, :, comp_idx]
        component_rows.append({
            "component": comp_name,
            "mae": _field_mae(err[:, :, :, comp_idx], comp_valid),
            "mse": _field_metric(err[:, :, :, comp_idx] ** 2, comp_valid, reducer=np.nanmean),
            "rmse": _field_rmse(err[:, :, :, comp_idx], comp_valid),
            "bias": _field_metric(err[:, :, :, comp_idx], comp_valid, reducer=np.nanmean),
            "valid_fraction": _field_valid_fraction(comp_valid),
        })
    component_metrics = pd.DataFrame(component_rows)

    node_rows = []
    for node_idx, node_label in enumerate(node_labels):
        node_valid = valid[:, :, node_idx, :]
        row = {
            "node": node_idx,
            "node_label": node_label,
            "mae": _field_mae(err[:, :, node_idx, :], node_valid),
            "mse": _field_metric(err[:, :, node_idx, :] ** 2, node_valid, reducer=np.nanmean),
            "rmse": _field_rmse(err[:, :, node_idx, :], node_valid),
            "bias": _field_metric(err[:, :, node_idx, :], node_valid, reducer=np.nanmean),
            "valid_fraction": _field_valid_fraction(node_valid),
        }
        if node_coords is not None and len(node_coords) == n_nodes:
            row["x"] = float(node_coords[node_idx][0])
            row["y"] = float(node_coords[node_idx][1])
        node_rows.append(row)
    node_metrics = pd.DataFrame(node_rows)

    pred_masked = np.where(valid, pred, np.nan)
    true_masked = np.where(valid, true, np.nan)
    pred_std = _field_nanstd(pred_masked, axis=0)
    true_std = _field_nanstd(true_masked, axis=0)
    std_ratio = pred_std / np.maximum(true_std, eps)

    if train_truth is not None:
        train_field = _field_to_frame_node_component(train_truth, field_shape=field_shape, name="train_truth")
        baseline_field = np.nanmean(train_field, axis=0)
        baseline_source = "train_mean_field"
    else:
        baseline_field = np.nanmean(true_masked, axis=0)
        baseline_source = "truth_mean_field"
    baseline_err = baseline_field.reshape(1, n_frames, n_nodes, n_components) - true
    baseline_rmse = _field_rmse(baseline_err, np.isfinite(true))

    mse_value = _field_metric(err ** 2, valid, reducer=np.nanmean)
    rmse_value = float(np.sqrt(mse_value)) if np.isfinite(mse_value) else np.nan
    mae_value = _field_mae(err, valid)
    summary = {
        "mae": mae_value,
        "mse": mse_value,
        "rmse": rmse_value,
        "bias": _field_metric(err, valid, reducer=np.nanmean),
        "valid_fraction": _field_valid_fraction(valid),
        "collapse_ratio": float(np.nanmean(std_ratio)) if np.any(np.isfinite(std_ratio)) else np.nan,
        "mean_field_baseline_source": baseline_source,
        "mean_field_baseline_rmse": baseline_rmse,
        "skill_vs_mean_field_rmse": float(1.0 - rmse_value / baseline_rmse)
        if np.isfinite(rmse_value) and baseline_rmse > eps else np.nan,
        "n_samples": int(n_samples),
        "n_frames": int(n_frames),
        "n_nodes": int(n_nodes),
        "n_components": int(n_components),
    }

    return {
        "summary": summary,
        "sample_metrics": sample_metrics,
        "frame_metrics": frame_metrics,
        "component_metrics": component_metrics,
        "node_metrics": node_metrics,
        "y_pred": pred,
        "y_true": true,
        "valid_mask": valid,
        "field_shape": (n_frames, n_nodes, n_components),
        "frame_values": frame_values,
        "components": components,
        "node_labels": node_labels,
        "node_coords": node_coords,
        "pred_std": pred_std,
        "true_std": true_std,
        "std_ratio": std_ratio,
        "baseline_field": baseline_field,
    }


# =============================================================================
# Field Post-Processing
# =============================================================================

def print_field_diagnostics(diagnostics, label="Field"):
    summary = diagnostics.get("summary", diagnostics)
    print(
        f"{label} prediction diagnostics | "
        f"RMSE: {_fmt_metric(summary.get('rmse'))} | "
        f"MAE: {_fmt_metric(summary.get('mae'))} | "
        f"mean-field RMSE: {_fmt_metric(summary.get('mean_field_baseline_rmse'))} | "
        f"skill vs mean field: {_fmt_metric(summary.get('skill_vs_mean_field_rmse'), 3)} | "
        f"collapse ratio: {_fmt_metric(summary.get('collapse_ratio'), 3)} | "
        f"valid: {_fmt_metric(summary.get('valid_fraction'), 3)}"
    )

def plot_field_diagnostics(diagnostics, figsize=(15, 9)):
    frame_metrics = diagnostics["frame_metrics"]
    component_metrics = diagnostics["component_metrics"]
    sample_metrics = diagnostics["sample_metrics"]
    summary = diagnostics["summary"]
    frame_axis = frame_metrics["frame"].to_numpy(dtype=float) + 1

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.reshape(-1)

    ax = axes[0]
    ax.plot(frame_axis, frame_metrics["rmse"], marker="o", label="RMSE")
    if "mae" in frame_metrics.columns:
        ax.plot(frame_axis, frame_metrics["mae"], marker="o", label="MAE")
    ax.set_title("Frame Error")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error")
    ax.legend(fontsize=8)

    ax = axes[1]
    if "bias" in frame_metrics.columns:
        ax.plot(frame_axis, frame_metrics["bias"], color="black", marker="o")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Frame Bias")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Prediction - Truth")

    ax = axes[2]
    if "valid_fraction" in frame_metrics.columns:
        ax.plot(frame_axis, frame_metrics["valid_fraction"], color="tab:green", marker="o")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Valid Fraction")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fraction")

    ax = axes[3]
    component_labels = component_metrics["component"].astype(str)
    ax.bar(component_labels, component_metrics["rmse"], label="RMSE", color="tab:blue", alpha=0.85)
    if "mae" in component_metrics.columns:
        ax.plot(component_labels, component_metrics["mae"], color="black", marker="o", label="MAE")
    ax.set_title("Component Error")
    ax.set_xlabel("Component")
    ax.set_ylabel("Error")
    ax.legend(fontsize=8)

    ax = axes[4]
    values = sample_metrics["sample_rmse"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))) if len(values) else 5)), color="tab:blue", alpha=0.8)
    ax.axvline(summary.get("mean_field_baseline_rmse", np.nan), color="gray", linestyle="--", linewidth=1.5)
    ax.set_title("Sample RMSE")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Count")

    ax = axes[5]
    ax.axis("off")
    lines = [
        f"RMSE: {_fmt_metric(summary.get('rmse'))}",
        f"MAE: {_fmt_metric(summary.get('mae'))}",
        f"Bias: {_fmt_metric(summary.get('bias'))}",
        f"Collapse ratio: {_fmt_metric(summary.get('collapse_ratio'), 3)}",
        f"Skill vs mean field: {_fmt_metric(summary.get('skill_vs_mean_field_rmse'), 3)}",
        f"Valid fraction: {_fmt_metric(summary.get('valid_fraction'), 3)}",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=11)

    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_field_frame_component_trends(diagnostics, figsize=(14, 8)):
    frame_metrics = diagnostics["frame_metrics"]
    component_metrics = diagnostics["component_metrics"]
    frame_axis = frame_metrics["frame"].to_numpy(dtype=float) + 1

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.reshape(-1)

    ax = axes[0]
    ax.plot(frame_axis, frame_metrics["rmse"], marker="o", label="RMSE")
    if "mae" in frame_metrics.columns:
        ax.plot(frame_axis, frame_metrics["mae"], marker="o", label="MAE")
    ax.set_title("Frame Error")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error")
    ax.legend(fontsize=8)

    ax = axes[1]
    if "bias" in frame_metrics.columns:
        ax.plot(frame_axis, frame_metrics["bias"], color="black", marker="o")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Frame Bias")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Prediction - Truth")

    ax = axes[2]
    if "valid_fraction" in frame_metrics.columns:
        ax.plot(frame_axis, frame_metrics["valid_fraction"], color="tab:green", marker="o")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Valid Fraction")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fraction")

    ax = axes[3]
    component_labels = component_metrics["component"].astype(str)
    ax.bar(component_labels, component_metrics["rmse"], label="RMSE", color="tab:blue", alpha=0.85)
    if "mae" in component_metrics.columns:
        ax.plot(component_labels, component_metrics["mae"], color="black", marker="o", label="MAE")
    ax.set_title("Component Error")
    ax.set_xlabel("Component")
    ax.set_ylabel("Error")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_field_diversity(diagnostics, figsize=(16, 4), ratio_reference=1.0, bins=40):
    pred_std = diagnostics.get("pred_std")
    true_std = diagnostics.get("true_std")
    std_ratio = diagnostics.get("std_ratio")
    if pred_std is None or true_std is None or std_ratio is None:
        raise ValueError("Diversity arrays are unavailable for this diagnostics object.")

    pred_std = np.asarray(pred_std, dtype=float)
    true_std = np.asarray(true_std, dtype=float)
    std_ratio = np.asarray(std_ratio, dtype=float)
    frame_axis = np.arange(pred_std.shape[0]) + 1

    pred_frame_std = np.nanmean(pred_std, axis=(1, 2))
    true_frame_std = np.nanmean(true_std, axis=(1, 2))
    ratio_frame = np.nanmean(std_ratio, axis=(1, 2))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].plot(frame_axis, true_frame_std, label="Truth std", color="darkgreen")
    axes[0].plot(frame_axis, pred_frame_std, label="Prediction std", color="orangered")
    axes[0].set_title("Across-Sample Field Diversity")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Mean std")
    axes[0].legend()

    axes[1].plot(frame_axis, ratio_frame, color="tab:purple")
    axes[1].axhline(ratio_reference, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Prediction Std / Truth Std")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Ratio")

    values = std_ratio[np.isfinite(std_ratio)]
    axes[2].hist(values, bins=bins, color="tab:blue", alpha=0.8)
    axes[2].axvline(ratio_reference, color="gray", linestyle="--", linewidth=1)
    axes[2].set_title("Node-Frame-Component Ratio")
    axes[2].set_xlabel("Std ratio")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    plt.show()
    return fig, axes

def _field_frame_component_metric_maps(diagnostics):
    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    valid = np.asarray(diagnostics.get("valid_mask", np.isfinite(y_true) & np.isfinite(y_pred)), dtype=bool)
    err = y_pred - y_true

    n_frames = y_pred.shape[1]
    n_components = y_pred.shape[3]
    components = [str(c) for c in (diagnostics.get("components") or _field_default_components(n_components))]
    frame_labels = [str(idx + 1) for idx in range(n_frames)]

    rmse_map = np.full((n_frames, n_components), np.nan)
    bias_map = np.full((n_frames, n_components), np.nan)
    valid_map = np.full((n_frames, n_components), np.nan)
    collapse_map = np.full((n_frames, n_components), np.nan)
    for frame_idx in range(n_frames):
        for comp_idx in range(n_components):
            mask = valid[:, frame_idx, :, comp_idx]
            values = err[:, frame_idx, :, comp_idx]
            if np.any(mask):
                rmse_map[frame_idx, comp_idx] = np.sqrt(np.nanmean(values[mask] ** 2))
                bias_map[frame_idx, comp_idx] = np.nanmean(values[mask])
                valid_map[frame_idx, comp_idx] = np.mean(mask)
                pred_values = np.where(mask, y_pred[:, frame_idx, :, comp_idx], np.nan)
                true_values = np.where(mask, y_true[:, frame_idx, :, comp_idx], np.nan)
                pred_std = np.nanstd(pred_values, axis=0)
                true_std = np.nanstd(true_values, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    collapse_map[frame_idx, comp_idx] = np.nanmean(pred_std / np.maximum(true_std, 1e-12))

    return {
        "RMSE": rmse_map,
        "Bias": bias_map,
        "Valid Fraction": valid_map,
        "Collapse Ratio": collapse_map,
        "components": components,
        "frame_labels": frame_labels,
    }

def plot_field_frame_component_heatmaps(diagnostics, figsize=(17, 5), cmaps=None, mark_worst=True):
    cmaps = cmaps or {"RMSE": "viridis", "Bias": "coolwarm", "Valid Fraction": "magma"}
    maps = _field_frame_component_metric_maps(diagnostics)
    components = maps["components"]
    frame_labels = maps["frame_labels"]
    rmse_map = maps["RMSE"]
    bias_map = maps["Bias"]
    valid_map = maps["Valid Fraction"]
    n_frames, n_components = rmse_map.shape

    worst = None
    if mark_worst and np.any(np.isfinite(rmse_map)):
        worst = np.unravel_index(np.nanargmax(rmse_map), rmse_map.shape)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, matrix, title in [
        (axes[0], rmse_map, "RMSE"),
        (axes[1], bias_map, "Bias"),
        (axes[2], valid_map, "Valid Fraction"),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap=cmaps.get(title, "viridis"))
        if title == "RMSE" and worst is not None:
            frame_idx, comp_idx = worst
            ax.scatter(comp_idx, frame_idx, marker="x", s=110, color="white", linewidths=2.0)
            ax.scatter(comp_idx, frame_idx, marker="x", s=55, color="black", linewidths=1.4)
            title = f"RMSE (worst: frame {frame_idx + 1}, {components[comp_idx]})"
        ax.set_title(title)
        ax.set_xlabel("Component")
        ax.set_ylabel("Frame")
        ax.set_xticks(np.arange(n_components))
        ax.set_xticklabels(components)
        ax.set_yticks(np.arange(n_frames))
        ax.set_yticklabels(frame_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_field_frame_component_collapse(diagnostics, figsize=(8, 5), cmap="viridis", annotate_worst=True):
    maps = _field_frame_component_metric_maps(diagnostics)
    collapse_map = maps["Collapse Ratio"]
    components = maps["components"]
    frame_labels = maps["frame_labels"]
    n_frames, n_components = collapse_map.shape

    worst = None
    if annotate_worst and np.any(np.isfinite(collapse_map)):
        # Values farthest from 1.0 are the most suspicious: near-zero collapse or inflated diversity.
        worst = np.unravel_index(np.nanargmax(np.abs(collapse_map - 1.0)), collapse_map.shape)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(collapse_map, aspect="auto", cmap=cmap, vmin=0.0)
    title = "Frame-Component Collapse Ratio"
    if worst is not None:
        frame_idx, comp_idx = worst
        ax.scatter(comp_idx, frame_idx, marker="x", s=120, color="white", linewidths=2.0)
        ax.scatter(comp_idx, frame_idx, marker="x", s=60, color="black", linewidths=1.4)
        title += f" (largest deviation: frame {frame_idx + 1}, {components[comp_idx]})"
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Frame")
    ax.set_xticks(np.arange(n_components))
    ax.set_xticklabels(components)
    ax.set_yticks(np.arange(n_frames))
    ax.set_yticklabels(frame_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pred std / true std")
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_field_component_parity(
    diagnostics,
    components=None,
    max_points_per_component=300000,
    gridsize=70,
    cmap="viridis",
    mincnt=1,
    random_seed=0,
    figsize=None,
):
    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    valid = np.asarray(diagnostics.get("valid_mask", np.isfinite(y_true) & np.isfinite(y_pred)), dtype=bool)
    n_components = y_pred.shape[3]
    component_names = [str(c) for c in (diagnostics.get("components") or _field_default_components(n_components))]

    if components is None:
        component_indices = list(range(n_components))
    else:
        requested = [components] if isinstance(components, (str, int, np.integer)) else list(components)
        component_indices = []
        for item in requested:
            if isinstance(item, (int, np.integer)):
                component_indices.append(int(item) % n_components)
            elif str(item) in component_names:
                component_indices.append(component_names.index(str(item)))
        component_indices = list(dict.fromkeys(component_indices))
    if not component_indices:
        raise ValueError("No requested field components are available for parity plotting.")

    figsize = figsize or (5 * len(component_indices), 4.5)
    fig, axes = plt.subplots(1, len(component_indices), figsize=figsize, squeeze=False)
    axes = axes.reshape(-1)
    rng = np.random.default_rng(random_seed)

    for ax, comp_idx in zip(axes, component_indices):
        mask = valid[:, :, :, comp_idx]
        true_values = y_true[:, :, :, comp_idx][mask]
        pred_values = y_pred[:, :, :, comp_idx][mask]
        total_points = true_values.size
        if max_points_per_component is not None and total_points > int(max_points_per_component):
            chosen = rng.choice(total_points, size=int(max_points_per_component), replace=False)
            true_plot = true_values[chosen]
            pred_plot = pred_values[chosen]
        else:
            true_plot = true_values
            pred_plot = pred_values

        if true_plot.size == 0:
            ax.set_title(f"{component_names[comp_idx]}: no valid values")
            ax.axis("off")
            continue

        lo = np.nanpercentile(np.concatenate([true_plot, pred_plot]), 1)
        hi = np.nanpercentile(np.concatenate([true_plot, pred_plot]), 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = np.nanmin(np.concatenate([true_plot, pred_plot]))
            hi = np.nanmax(np.concatenate([true_plot, pred_plot]))
        if lo == hi:
            lo -= 1.0
            hi += 1.0

        hb = ax.hexbin(true_plot, pred_plot, gridsize=gridsize, mincnt=mincnt, cmap=cmap)
        ax.plot([lo, hi], [lo, hi], color="white", linewidth=2.0)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        bias = float(np.nanmean(pred_values - true_values))
        rmse = float(np.sqrt(np.nanmean((pred_values - true_values) ** 2)))
        sample_note = f"{true_plot.size:,}/{total_points:,}" if true_plot.size != total_points else f"{total_points:,}"
        ax.set_title(f"{component_names[comp_idx]} parity\nRMSE={rmse:.4g}, bias={bias:.4g}, n={sample_note}")
        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Count")

    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_field_node_metrics(diagnostics, columns=None, figsize=None, point_size=22):
    node_metrics = diagnostics.get("node_metrics")
    if node_metrics is None or not hasattr(node_metrics, "copy"):
        raise ValueError("Node metrics are unavailable.")

    node = node_metrics.copy()
    if "mean_abs_percent_error" not in node.columns and {"y_pred", "y_true"}.issubset(diagnostics):
        y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
        y_true = np.asarray(diagnostics["y_true"], dtype=float)
        valid = np.asarray(diagnostics.get("valid_mask", np.isfinite(y_true) & np.isfinite(y_pred)), dtype=bool)
        abs_err = np.where(valid, np.abs(y_pred - y_true), np.nan)
        abs_true = np.where(valid, np.abs(y_true), np.nan)
        err_sum = np.nansum(abs_err, axis=(0, 1, 3))
        true_sum = np.nansum(abs_true, axis=(0, 1, 3))
        node_percent_error = np.full_like(err_sum, np.nan, dtype=float)
        denom_ok = true_sum > 1e-12
        node_percent_error[denom_ok] = 100.0 * err_sum[denom_ok] / true_sum[denom_ok]
        node["mean_abs_percent_error"] = node_percent_error

    coords = diagnostics.get("node_coords")
    if ("x" not in node.columns or "y" not in node.columns) and coords is not None and len(coords) == len(node):
        coords = np.asarray(coords, dtype=float)
        node["x"] = coords[:, 0]
        node["y"] = coords[:, 1]
    if "x" not in node.columns or "y" not in node.columns:
        raise ValueError("Node coordinates are unavailable, so spatial node maps cannot be drawn.")

    columns = columns or ["rmse", "mae", "bias", "mean_abs_percent_error"]
    plot_cols = [col for col in columns if col in node.columns]
    if not plot_cols:
        raise ValueError("None of the requested node metric columns are available.")

    figsize = figsize or (5 * len(plot_cols), 4)
    fig, axes = plt.subplots(1, len(plot_cols), figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    labels = {
        "rmse": "RMSE",
        "mae": "MAE",
        "bias": "Bias",
        "valid_fraction": "Valid Fraction",
        "mean_abs_percent_error": "Mean Abs Percent Error",
    }
    for ax, col in zip(axes, plot_cols):
        sc = ax.scatter(node["x"], node["y"], c=node[col], cmap="viridis", s=point_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Node {labels.get(col, col)}")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_field_sample_metric_distributions(diagnostics, columns=None, bins=40, ncols=3, figsize=None):
    sample_metrics = diagnostics.get("sample_metrics")
    if sample_metrics is None or not hasattr(sample_metrics, "copy"):
        raise ValueError("Sample metrics are unavailable.")

    samples = sample_metrics.copy()
    columns = columns or ["sample_mae", "sample_mse", "sample_rmse", "sample_bias", "valid_fraction"]
    numeric_cols = [col for col in columns if col in samples.columns]
    if not numeric_cols:
        raise ValueError("None of the requested sample metric columns are available.")

    ncols = int(ncols)
    nrows = int(np.ceil(len(numeric_cols) / ncols))
    figsize = figsize or (15, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, numeric_cols):
        ax.hist(samples[col].dropna(), bins=bins, color="tab:blue", alpha=0.8)
        ax.set_title(col)
        ax.set_ylabel("Count")
    for ax in axes[len(numeric_cols):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_field_sample(
    diagnostics,
    sample=0,
    frame=-1,
    component=0,
    node_coords=None,
    cmap="coolwarm",
    figsize=(15, 4),
    plot_style="points",
    levels=30,
    point_size=22,
    show=True,
):
    pred = diagnostics["y_pred"]
    true = diagnostics["y_true"]
    valid = diagnostics["valid_mask"]
    coords = node_coords if node_coords is not None else diagnostics.get("node_coords")
    if coords is None:
        raise ValueError("plot_field_sample requires node_coords.")
    coords = np.asarray(coords, dtype=float)

    n_samples, n_frames, _, n_components = pred.shape
    sample = int(sample)
    frame = int(frame) % n_frames
    component = int(component) % n_components
    if sample < 0 or sample >= n_samples:
        raise IndexError(f"sample must be in [0, {n_samples - 1}], got {sample}.")

    truth_values = true[sample, frame, :, component]
    pred_values = pred[sample, frame, :, component]
    err_values = pred_values - truth_values
    valid_values = valid[sample, frame, :, component]
    vmax = np.nanpercentile(np.abs(np.concatenate([truth_values[valid_values], pred_values[valid_values]])), 98) if np.any(valid_values) else 1.0
    vmax = max(float(vmax), 1e-12)
    err_vmax = np.nanpercentile(np.abs(err_values[valid_values]), 98) if np.any(valid_values) else 1.0
    err_vmax = max(float(err_vmax), 1e-12)

    plot_style = str(plot_style).lower()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, values, title, limit in [
        (axes[0], truth_values, "Truth", vmax),
        (axes[1], pred_values, "Prediction", vmax),
        (axes[2], err_values, "Prediction - Truth", err_vmax),
    ]:
        plot_values = np.where(valid_values, values, np.nan)
        finite = np.isfinite(plot_values)
        im = None
        if plot_style in ["continuous", "contour", "tricontour", "tricontourf"] and np.sum(finite) >= 3:
            try:
                level_values = np.linspace(-limit, limit, int(levels))
                im = ax.tricontourf(
                    coords[finite, 0],
                    coords[finite, 1],
                    plot_values[finite],
                    levels=level_values,
                    cmap=cmap,
                    extend="both",
                )
            except Exception:
                im = None
        if im is None:
            im = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=plot_values,
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                s=point_size,
            )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes

def _field_component_index(components, component):
    if isinstance(component, (int, np.integer)):
        return int(component) % len(components)
    component = str(component)
    if component in components:
        return components.index(component)
    return 1 if len(components) > 1 else 0

def _first_sample_index(sample, n_samples):
    if isinstance(sample, (int, np.integer)):
        return int(np.clip(sample, 0, n_samples - 1))
    if isinstance(sample, (list, tuple, set, np.ndarray, pd.Series)) and len(sample) > 0:
        return int(np.clip(list(sample)[0], 0, n_samples - 1))
    text = str(sample).strip()
    if not text:
        return 0
    try:
        import ast
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)) and len(parsed) > 0:
            return int(np.clip(list(parsed)[0], 0, n_samples - 1))
        if isinstance(parsed, (int, np.integer)):
            return int(np.clip(parsed, 0, n_samples - 1))
    except Exception:
        pass
    first = text.replace(";", ",").split(",")[0]
    return int(np.clip(int(first), 0, n_samples - 1))

def plot_field_keyframe_strip(
    diagnostics,
    sample=0,
    component="U2",
    frames=None,
    n_keyframes=5,
    rows=("truth", "prediction", "error"),
    plot_style="continuous",
    cmap="coolwarm",
    levels=24,
    point_size=14,
    figsize=None,
    frame_label_offset=1,
):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None, None

    coords = diagnostics.get("node_coords")
    if coords is None:
        print("Node coordinates are unavailable, so the keyframe strip cannot be drawn.")
        return None, None
    coords = np.asarray(coords, dtype=float)

    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    valid = np.asarray(diagnostics.get("valid_mask", np.isfinite(y_true) & np.isfinite(y_pred)), dtype=bool)
    n_samples, n_frames, _, n_components = y_pred.shape
    components = [str(c) for c in (diagnostics.get("components") or _field_default_components(n_components))]

    sample_idx = _first_sample_index(sample, n_samples)
    component_idx = _field_component_index(components, component)
    if frames is None:
        frame_indices = np.linspace(0, n_frames - 1, num=min(int(n_keyframes), n_frames), dtype=int)
    else:
        frame_indices = np.asarray(frames, dtype=int)
        frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < n_frames)]
        if frame_indices.size == 0:
            frame_indices = np.linspace(0, n_frames - 1, num=min(int(n_keyframes), n_frames), dtype=int)
    frame_indices = np.unique(frame_indices)

    row_specs = []
    for row in rows:
        row_key = str(row).lower()
        if row_key in ["truth", "true", "target"]:
            row_specs.append(("Truth", y_true[sample_idx, frame_indices, :, component_idx]))
        elif row_key in ["prediction", "pred", "output"]:
            row_specs.append(("Prediction", y_pred[sample_idx, frame_indices, :, component_idx]))
        elif row_key in ["error", "residual", "prediction - truth"]:
            row_specs.append(("Prediction - Truth", y_pred[sample_idx, frame_indices, :, component_idx] - y_true[sample_idx, frame_indices, :, component_idx]))
    if not row_specs:
        row_specs = [("Truth", y_true[sample_idx, frame_indices, :, component_idx])]

    sample_valid = valid[sample_idx, frame_indices, :, component_idx]
    field_values = np.concatenate([
        np.ravel(y_true[sample_idx, frame_indices, :, component_idx][sample_valid]),
        np.ravel(y_pred[sample_idx, frame_indices, :, component_idx][sample_valid]),
    ])
    error_values = np.ravel((y_pred[sample_idx, frame_indices, :, component_idx] - y_true[sample_idx, frame_indices, :, component_idx])[sample_valid])
    value_limit = np.nanpercentile(np.abs(field_values), 98) if field_values.size else 1.0
    error_limit = np.nanpercentile(np.abs(error_values), 98) if error_values.size else 1.0
    value_limit = max(float(value_limit), 1e-12)
    error_limit = max(float(error_limit), 1e-12)

    figsize = figsize or (3.0 * len(frame_indices), 2.5 * len(row_specs))
    fig, axes = plt.subplots(
        len(row_specs),
        len(frame_indices),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    plot_style = str(plot_style).lower()

    for row_idx, (row_title, row_values) in enumerate(row_specs):
        limit = error_limit if "Truth" in row_title and "Prediction" in row_title else value_limit
        if row_title == "Prediction - Truth":
            limit = error_limit
        last_im = None
        for col_idx, frame_idx in enumerate(frame_indices):
            ax = axes[row_idx, col_idx]
            frame_valid = sample_valid[col_idx]
            values = np.where(frame_valid, row_values[col_idx], np.nan)
            finite = np.isfinite(values)
            im = None
            if plot_style in ["continuous", "contour", "tricontour", "tricontourf"] and np.sum(finite) >= 3:
                try:
                    im = ax.tricontourf(
                        coords[finite, 0],
                        coords[finite, 1],
                        values[finite],
                        levels=np.linspace(-limit, limit, int(levels)),
                        cmap=cmap,
                        extend="both",
                    )
                except Exception:
                    im = None
            if im is None:
                im = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=values,
                    cmap=cmap,
                    vmin=-limit,
                    vmax=limit,
                    s=point_size,
                )
            last_im = im
            if row_idx == 0:
                ax.set_title(f"Frame {int(frame_idx) + int(frame_label_offset)}")
            if col_idx == 0:
                ax.set_ylabel(row_title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes[row_idx, :], fraction=0.020, pad=0.02)

    fig.suptitle(f"Sample {sample_idx} Keyframes - {components[component_idx]}")
    plt.show()
    return fig, axes

# =============================================================================
# Saved-Run Post-Processing
# =============================================================================
def postprocess_resolve_artifacts(run_path, run_root=None, prefer_hpo_best=True):
    """
    Resolve a saved ML run, model JSON, model checkpoint, or HPO directory into
    the artifact paths used by the post-processing notebook.
    """
    path = Path(run_path).expanduser()
    if not path.is_absolute() and run_root is not None:
        path = Path(run_root).expanduser() / path
    path = path.resolve() if path.exists() else path

    artifacts = {
        "input_path": path,
        "run_dir": None,
        "model_json": None,
        "model_mdl": None,
        "data_json": None,
        "results_dir": None,
        "metrics_json": None,
        "predictions_npz": None,
        "loss_history_csv": None,
        "diagnostics_summary_json": None,
        "hpo_dir": None,
        "hpo_best_params_json": None,
        "hpo_best_trial_user_attrs_json": None,
        "hpo_candidate_model_jsons": [],
        "is_hpo": False,
        "warnings": [],
    }

    if path.suffix.lower() in [".json", ".mdl"]:
        model_json = path.with_suffix(".json")
        model_mdl = path.with_suffix(".mdl")
        run_dir = path.parent
    elif path.is_dir():
        run_dir = path
        model_json, model_mdl = _postprocess_select_model_pair(run_dir, prefer_hpo_best=prefer_hpo_best)
    else:
        run_dir = path
        model_json = None
        model_mdl = None
        artifacts["warnings"].append(f"Input path does not exist yet: {path}")

    artifacts["run_dir"] = run_dir
    artifacts["model_json"] = model_json if model_json is not None and model_json.exists() else None
    artifacts["model_mdl"] = model_mdl if model_mdl is not None and model_mdl.exists() else None

    if model_json is not None:
        data_json = model_json.with_name(f"{model_json.stem}_data.json")
        artifacts["data_json"] = data_json if data_json.exists() else None
        artifacts["hpo_dir"] = _postprocess_nearest_hpo_dir(model_json.parent)
        artifacts["is_hpo"] = artifacts["hpo_dir"] is not None or model_json.stem == "best_model"

    hpo_dir = artifacts["hpo_dir"] if artifacts["hpo_dir"] is not None else run_dir
    if hpo_dir is not None and Path(hpo_dir).is_dir():
        best_params = Path(hpo_dir) / "best_params.json"
        best_attrs = Path(hpo_dir) / "best_trial_user_attrs.json"
        artifacts["hpo_best_params_json"] = best_params if best_params.exists() else None
        artifacts["hpo_best_trial_user_attrs_json"] = best_attrs if best_attrs.exists() else None
        artifacts["hpo_candidate_model_jsons"] = _postprocess_hpo_model_jsons(Path(hpo_dir))
        if artifacts["hpo_best_params_json"] is not None or artifacts["hpo_candidate_model_jsons"]:
            artifacts["is_hpo"] = True

    artifacts["results_dir"] = _postprocess_results_dir(artifacts)
    results_dir = artifacts["results_dir"]
    if results_dir is not None:
        artifacts["metrics_json"] = _postprocess_existing_file(results_dir / "metrics.json")
        artifacts["predictions_npz"] = _postprocess_existing_file(results_dir / "predictions.npz")
        artifacts["loss_history_csv"] = _postprocess_existing_file(results_dir / "loss_history.csv")
        artifacts["diagnostics_summary_json"] = _postprocess_existing_file(results_dir / "diagnostics_summary.json")

    return artifacts

def _postprocess_select_model_pair(run_dir, prefer_hpo_best=True):
    run_dir = Path(run_dir)
    if prefer_hpo_best:
        for candidate in [
            run_dir / "best_model.json",
            run_dir / "model.json",
        ]:
            if candidate.exists() and candidate.with_suffix(".mdl").exists():
                return candidate, candidate.with_suffix(".mdl")

        hpo_candidates = _postprocess_hpo_model_jsons(run_dir)
        if hpo_candidates:
            return hpo_candidates[0], hpo_candidates[0].with_suffix(".mdl")

    jsons = []
    skip_names = {
        "metrics.json",
        "diagnostics_summary.json",
        "best_params.json",
        "best_trial_user_attrs.json",
    }
    for candidate in run_dir.glob("*.json"):
        if candidate.name.endswith("_data.json") or candidate.name in skip_names:
            continue
        if candidate.with_suffix(".mdl").exists():
            jsons.append(candidate)
    if not jsons:
        for candidate in run_dir.rglob("*.json"):
            if candidate.name.endswith("_data.json") or candidate.name in skip_names:
                continue
            if candidate.with_suffix(".mdl").exists():
                jsons.append(candidate)
    if not jsons:
        return None, None
    jsons = sorted(jsons, key=lambda p: p.stat().st_mtime, reverse=True)
    return jsons[0], jsons[0].with_suffix(".mdl")

def _postprocess_hpo_model_jsons(hpo_dir):
    hpo_dir = Path(hpo_dir)
    if not hpo_dir.exists():
        return []
    candidates = []
    for candidate in hpo_dir.rglob("best_model.json"):
        if candidate.with_suffix(".mdl").exists():
            candidates.append(candidate)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

def _postprocess_nearest_hpo_dir(path):
    path = Path(path)
    for candidate in [path] + list(path.parents):
        if candidate.name.upper() == "HPO":
            return candidate
        if (candidate / "best_params.json").exists() or (candidate / "best_trial_user_attrs.json").exists():
            return candidate
    return None

def _postprocess_existing_file(path):
    path = Path(path)
    return path if path.exists() else None

def _postprocess_results_dir(artifacts):
    model_json = artifacts.get("model_json")
    run_dir = artifacts.get("run_dir")
    if model_json is not None:
        model_json = Path(model_json)
        if model_json.stem == "best_model":
            best_results = model_json.parent / "best_model_results"
            if best_results.exists():
                return best_results
        results = model_json.parent / "results"
        if results.exists():
            return results
        best_results = model_json.parent / "best_model_results"
        if best_results.exists():
            return best_results
        return results
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    for candidate in [run_dir / "best_model_results", run_dir / "results"]:
        if candidate.exists():
            return candidate
    return run_dir / "results"

def postprocess_output_dir(artifacts, label=None, create=True):
    """Return the stable post-processing output directory under the run results folder."""
    results_dir = artifacts.get("results_dir")
    if results_dir is None:
        run_dir = artifacts.get("run_dir")
        if run_dir is None:
            raise ValueError("Cannot create post-processing directory without a run or results directory.")
        results_dir = Path(run_dir) / "results"
    out_dir = Path(results_dir) / "postProcessing"
    if label is not None and str(label).strip():
        out_dir = out_dir / str(label).strip()
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _postprocess_output_kind_token(output_kind):
    key = str(output_kind or "Curve").strip()
    if key.lower() in ["fieldtocurve", "field_to_curve", "field-to-curve"]:
        return "FieldToCurve"
    if key.lower() == "field":
        return "Field"
    if key.lower() == "curve":
        return "Curve"
    return key

def _postprocess_layout_output_kinds():
    return {"CURVE", "FIELD", "FIELDTOCURVE"}

def _postprocess_path_parts(path, run_root=None):
    path = Path(path)
    if run_root is None:
        return path.parts
    try:
        return path.relative_to(Path(run_root)).parts
    except ValueError:
        return path.parts

def _postprocess_run_layout_from_path(path, run_root=None):
    parts = _postprocess_path_parts(path, run_root=run_root)
    upper_parts = [str(part).upper() for part in parts]

    layout = {
        "task": None,
        "output_kind": None,
        "model": None,
        "run_name": None,
    }

    if "HPO" in upper_parts:
        hpo_layout = _postprocess_hpo_layout_from_path(path, run_root=run_root)
        layout.update({key: hpo_layout.get(key) for key in layout})
        return layout

    if len(parts) >= 4 and upper_parts[1] in _postprocess_layout_output_kinds():
        layout.update(
            {
                "task": parts[0],
                "output_kind": parts[1],
                "model": parts[2],
                "run_name": parts[3],
            }
        )
    elif len(parts) >= 3:
        layout.update(
            {
                "task": parts[0],
                "model": parts[1],
                "run_name": parts[2],
            }
        )
    return layout

def postprocess_list_runs(run_root="Z:/p2", max_runs=25, include_hpo=True):
    """List recent saved model runs under a local run root."""
    root = Path(run_root).expanduser()
    rows = []
    columns = ["task", "output_kind", "model", "run_name", "run_dir", "model_json", "model_mdl", "results_dir", "is_hpo", "modified"]
    if not root.exists():
        return pd.DataFrame(columns=columns)
    skip_names = {
        "metrics.json",
        "diagnostics_summary.json",
        "best_params.json",
        "best_trial_user_attrs.json",
    }
    for model_json in root.rglob("*.json"):
        if model_json.name.endswith("_data.json") or model_json.name in skip_names:
            continue
        if not model_json.with_suffix(".mdl").exists():
            continue
        artifacts = postprocess_resolve_artifacts(model_json, prefer_hpo_best=include_hpo)
        if artifacts["is_hpo"] and not include_hpo:
            continue
        layout = _postprocess_run_layout_from_path(artifacts.get("run_dir"), run_root=root)
        rows.append(
            {
                "task": layout.get("task"),
                "output_kind": layout.get("output_kind"),
                "model": layout.get("model"),
                "run_name": layout.get("run_name"),
                "run_dir": str(artifacts.get("run_dir")),
                "model_json": str(model_json),
                "model_mdl": str(model_json.with_suffix(".mdl")),
                "results_dir": str(artifacts.get("results_dir")),
                "is_hpo": bool(artifacts.get("is_hpo", False)),
                "modified": datetime.datetime.fromtimestamp(model_json.stat().st_mtime),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_json"])
    df = df.sort_values("modified", ascending=False).reset_index(drop=True)
    if max_runs is not None:
        df = df.head(int(max_runs))
    return df

def postprocess_resolve_hpo_run_path(
    run_root="Z:/p2",
    task="UT",
    output_kind="Curve",
    model="Transformer",
    run_name=None,
    run_type="model_hpo",
    run_path_override=None,
):
    """Resolve the two supported HPO run layouts into one concrete directory."""
    if run_path_override is not None and str(run_path_override).strip():
        return Path(run_path_override).expanduser()

    if run_name is None or not str(run_name).strip():
        raise ValueError("run_name is required when RUN_PATH_OVERRIDE is not set.")

    root = Path(run_root).expanduser()
    task = str(task).upper()
    output_kind = _postprocess_output_kind_token(output_kind)
    model = str(model)
    run_name = str(run_name)
    run_type = str(run_type).lower()

    if run_type == "model_hpo":
        return root / task / output_kind / model / "HPO" / run_name
    if run_type == "cross_model_hpo":
        return root / task / output_kind / "HPO" / run_name / model
    raise ValueError("run_type must be 'model_hpo' or 'cross_model_hpo' for HPO post-processing.")

def _postprocess_hpo_layout_from_path(path, run_root=None):
    parts = _postprocess_path_parts(path, run_root=run_root)

    layout = {
        "run_type": None,
        "task": None,
        "output_kind": None,
        "model": None,
        "run_name": None,
    }
    upper_parts = [str(part).upper() for part in parts]
    if len(parts) >= 5 and upper_parts[3] == "HPO":
        layout.update(
            {
                "run_type": "model_hpo",
                "task": parts[0],
                "output_kind": parts[1],
                "model": parts[2],
                "run_name": parts[4],
            }
        )
    elif len(parts) >= 5 and upper_parts[2] == "HPO":
        layout.update(
            {
                "run_type": "cross_model_hpo",
                "task": parts[0],
                "output_kind": parts[1],
                "model": parts[4],
                "run_name": parts[3],
            }
        )
    elif len(parts) >= 4 and upper_parts[2] == "HPO":
        layout.update(
            {
                "run_type": "model_hpo",
                "task": parts[0],
                "model": parts[1],
                "run_name": parts[3],
            }
        )
    elif len(parts) >= 4 and upper_parts[1] == "HPO":
        layout.update(
            {
                "run_type": "cross_model_hpo",
                "task": parts[0],
                "model": parts[3],
                "run_name": parts[2],
            }
        )
    return layout

def postprocess_hpo_path_summary(
    run_path,
    run_root=None,
    task=None,
    output_kind=None,
    model=None,
    run_name=None,
    run_type=None,
):
    """Return a compact table for the resolved HPO path and user-facing config."""
    run_path = Path(run_path)
    layout = _postprocess_hpo_layout_from_path(run_path, run_root=run_root)
    values = {
        "RUN_ROOT": run_root,
        "OUTPUT_KIND": output_kind if output_kind is not None else layout.get("output_kind"),
        "TASK": task if task is not None else layout.get("task"),
        "MODEL": model if model is not None else layout.get("model"),
        "RUN_NAME": run_name if run_name is not None else layout.get("run_name"),
        "RUN_TYPE": run_type if run_type is not None else layout.get("run_type"),
        "RUN_PATH": run_path,
        "RUN_PATH_EXISTS": run_path.exists(),
        "STUDY_DB": run_path / "full_study.db",
        "STUDY_DB_EXISTS": (run_path / "full_study.db").exists(),
    }
    return pd.DataFrame(values.items(), columns=["item", "value"])

def postprocess_cross_hpo_base_path(run_path, run_type="cross_model_hpo"):
    """Return the comparison root for a cross-model HPO run."""
    if str(run_type).lower() != "cross_model_hpo":
        return None
    return Path(run_path).parent

def postprocess_list_hpo_runs(run_root="Z:/p2", max_runs=25):
    """List recent HPO study directories that contain full_study.db."""
    root = Path(run_root).expanduser()
    columns = [
        "task",
        "output_kind",
        "model",
        "run_name",
        "run_type",
        "hpo_dir",
        "study_db",
        "best_model_json",
        "best_model_results",
        "modified",
    ]
    if not root.exists():
        return pd.DataFrame(columns=columns)

    rows = []
    for study_db in root.rglob("full_study.db"):
        hpo_dir = study_db.parent
        layout = _postprocess_hpo_layout_from_path(hpo_dir, run_root=root)
        artifacts = postprocess_resolve_artifacts(hpo_dir, run_root=root, prefer_hpo_best=True)
        rows.append(
            {
                "task": layout.get("task"),
                "output_kind": layout.get("output_kind"),
                "model": layout.get("model"),
                "run_name": layout.get("run_name"),
                "run_type": layout.get("run_type"),
                "hpo_dir": str(hpo_dir),
                "study_db": str(study_db),
                "best_model_json": str(artifacts.get("model_json")) if artifacts.get("model_json") is not None else None,
                "best_model_results": str(artifacts.get("results_dir")) if artifacts.get("results_dir") is not None else None,
                "modified": datetime.datetime.fromtimestamp(study_db.stat().st_mtime),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows).sort_values("modified", ascending=False).reset_index(drop=True)
    if max_runs is not None:
        df = df.head(int(max_runs))
    return df[columns]

def _postprocess_hpo_storage_url(study_db):
    return f"sqlite:///{Path(study_db).as_posix()}"

def postprocess_load_hpo_study(hpo_dir, study_name=None):
    """
    Load an Optuna study from full_study.db. Missing databases are reported
    clearly and are not replaced by best_params.json fallbacks.
    """
    hpo_dir = Path(hpo_dir).expanduser()
    study_db = hpo_dir / "full_study.db"
    info = {
        "hpo_dir": hpo_dir,
        "study_db": study_db,
        "study_name": study_name,
        "study": None,
        "summaries": pd.DataFrame(columns=["study_name", "direction", "n_trials", "datetime_start"]),
        "message": None,
    }

    if not study_db.exists():
        info["message"] = f"No full_study.db was found at {study_db}."
        return info

    try:
        import optuna
    except Exception as exc:
        info["message"] = f"Optuna is required to load {study_db}: {exc!r}"
        return info

    storage = _postprocess_hpo_storage_url(study_db)
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        summary_rows = []
        for summary in summaries:
            direction = getattr(summary, "direction", None)
            summary_rows.append(
                {
                    "study_name": summary.study_name,
                    "direction": getattr(direction, "name", str(direction)),
                    "n_trials": int(summary.n_trials),
                    "datetime_start": summary.datetime_start,
                }
            )
        info["summaries"] = pd.DataFrame(summary_rows)
        if study_name is None:
            if len(summaries) == 1:
                study_name = summaries[0].study_name
            elif len(summaries) == 0:
                info["message"] = f"No Optuna studies were found in {study_db}."
                return info
            else:
                info["message"] = (
                    f"{study_db} contains multiple studies. Set STUDY_NAME to one of "
                    f"{[summary.study_name for summary in summaries]}."
                )
                return info

        study = optuna.load_study(study_name=study_name, storage=storage)
        info["study_name"] = study_name
        info["study"] = study
        info["message"] = f"Loaded study '{study_name}' from {study_db}."
    except Exception as exc:
        info["message"] = f"Study load failed for {study_db}: {exc!r}"
    return info

def _postprocess_trial_state_name(trial):
    state = getattr(trial, "state", None)
    return getattr(state, "name", str(state))

def _postprocess_completed_trials(study):
    if study is None:
        return []
    trials = getattr(study, "trials", [])
    return [
        trial for trial in trials
        if _postprocess_trial_state_name(trial) == "COMPLETE" and trial.value is not None and np.isfinite(trial.value)
    ]

def _postprocess_hpo_compact_value(value, max_chars=160):
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(_postprocess_json_safe(value), sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    if len(text) > int(max_chars):
        return text[: int(max_chars) - 3] + "..."
    return text

def _postprocess_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _postprocess_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_postprocess_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def postprocess_hpo_study_stats(study):
    if study is None:
        return pd.DataFrame(columns=["item", "value"])
    trials = list(getattr(study, "trials", []))
    state_counts = {}
    for trial in trials:
        state = _postprocess_trial_state_name(trial)
        state_counts[state] = state_counts.get(state, 0) + 1
    completed = _postprocess_completed_trials(study)
    best_trial = None
    try:
        best_trial = study.best_trial
    except Exception:
        best_trial = None
    rows = [
        ("study_name", getattr(study, "study_name", None)),
        ("direction", getattr(getattr(study, "direction", None), "name", getattr(study, "direction", None))),
        ("n_trials", len(trials)),
        ("n_complete", len(completed)),
        ("n_pruned", state_counts.get("PRUNED", 0)),
        ("n_failed", state_counts.get("FAIL", 0)),
        ("best_trial", getattr(best_trial, "number", None)),
        ("best_value", getattr(best_trial, "value", None)),
        ("objective_metric", (best_trial.params or {}).get("objective_metric") if best_trial is not None else None),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])

def postprocess_hpo_best_overview(study_info, artifacts=None, loaded=None):
    """Build compact best-trial tables and best-model artifact paths."""
    study = study_info.get("study") if isinstance(study_info, dict) else study_info
    artifacts = artifacts or {}
    loaded = loaded or {}
    best_trial = None
    try:
        best_trial = study.best_trial if study is not None else None
    except Exception:
        best_trial = None

    summary = postprocess_hpo_study_stats(study)
    params = pd.DataFrame(columns=["parameter", "value"])
    attrs = pd.DataFrame(columns=["attribute", "value"])
    if best_trial is not None:
        params = pd.DataFrame(
            [(key, value) for key, value in sorted((best_trial.params or {}).items())],
            columns=["parameter", "value"],
        )
        attrs = pd.DataFrame(
            [
                (key, _postprocess_hpo_compact_value(value))
                for key, value in sorted((best_trial.user_attrs or {}).items())
                if key != "model_instance"
            ],
            columns=["attribute", "value"],
        )

    artifact_keys = [
        "model_json",
        "model_mdl",
        "data_json",
        "results_dir",
        "metrics_json",
        "predictions_npz",
        "loss_history_csv",
        "diagnostics_summary_json",
        "hpo_best_params_json",
        "hpo_best_trial_user_attrs_json",
    ]
    artifact_rows = [(key, artifacts.get(key)) for key in artifact_keys]
    if isinstance(loaded.get("metrics"), dict):
        metadata = loaded["metrics"].get("metadata", {})
        if isinstance(metadata, dict):
            artifact_rows.extend(
                [
                    ("saved_trial_number", metadata.get("trial_number")),
                    ("saved_objective_value", metadata.get("objective_value")),
                ]
            )
    artifact_paths = pd.DataFrame(
        [(key, str(value) if value is not None else None) for key, value in artifact_rows],
        columns=["artifact", "path_or_value"],
    )

    return {
        "summary": summary,
        "best_params": params,
        "best_user_attrs": attrs,
        "artifact_paths": artifact_paths,
        "best_trial": best_trial,
    }

def display_hpo_best_overview(overview, max_attrs=20):
    from IPython.display import Markdown, display

    display(Markdown("### Best Trial Summary"))
    display(overview["summary"])
    if not overview["best_params"].empty:
        display(Markdown("### Best Parameters"))
        display(overview["best_params"])
    if not overview["best_user_attrs"].empty:
        display(Markdown("### Best User Attributes"))
        display(overview["best_user_attrs"].head(int(max_attrs)))
    display(Markdown("### Best Model Artifacts"))
    display(overview["artifact_paths"])

def _postprocess_hpo_key_params(study, max_params=8):
    trials = list(getattr(study, "trials", [])) if study is not None else []
    values_by_key = {}
    for trial in trials:
        for key, value in (trial.params or {}).items():
            if key == "objective_metric":
                continue
            values_by_key.setdefault(key, set()).add(str(value))
    ranked = sorted(values_by_key.items(), key=lambda item: (-len(item[1]), item[0]))
    return [key for key, values in ranked[: int(max_params)]]

def postprocess_hpo_trial_leaderboard(study, top_n=20, key_params=None):
    """Return a compact objective leaderboard for Optuna trials."""
    columns = ["trial", "value", "state", "model_family", "duration_s"]
    if study is None:
        return pd.DataFrame(columns=columns)
    key_params = list(key_params) if key_params is not None else _postprocess_hpo_key_params(study)
    rows = []
    for trial in getattr(study, "trials", []):
        duration = None
        if trial.datetime_start is not None and trial.datetime_complete is not None:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        row = {
            "trial": int(trial.number),
            "value": trial.value,
            "state": _postprocess_trial_state_name(trial),
            "model_family": (trial.user_attrs or {}).get("typ"),
            "duration_s": duration,
        }
        for key in key_params:
            row[key] = (trial.params or {}).get(key)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns + key_params)
    df = pd.DataFrame(rows)
    df["_sort_value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values(["_sort_value", "trial"], ascending=[True, True], na_position="last")
    df = df.drop(columns=["_sort_value"]).reset_index(drop=True)
    if top_n is not None:
        df = df.head(int(top_n))
    return df

def plot_hpo_optimization_history(study, figsize=(9, 4), show=True):
    """Plot objective value by trial number and mark the best completed trial."""
    completed = _postprocess_completed_trials(study)
    if not completed:
        print("No completed Optuna trials are available for optimization history.")
        return None, None

    trial_numbers = np.asarray([trial.number for trial in completed], dtype=int)
    values = np.asarray([trial.value for trial in completed], dtype=float)
    best_idx = int(np.nanargmin(values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(trial_numbers, values, color="tab:blue", linewidth=1.0, alpha=0.65)
    ax.scatter(trial_numbers, values, color="tab:blue", s=34, alpha=0.8, label="Completed trial")
    ax.scatter(
        [trial_numbers[best_idx]],
        [values[best_idx]],
        color="crimson",
        s=80,
        marker="*",
        label=f"Best trial {trial_numbers[best_idx]}",
        zorder=5,
    )
    ax.set_title("HPO Optimization History")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

def postprocess_hpo_param_importance(study):
    """Compute Optuna parameter importance when the completed trials support it."""
    columns = ["parameter", "importance"]
    if study is None:
        return pd.DataFrame(columns=columns), "No study is loaded."
    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:
        return pd.DataFrame(columns=columns), f"Parameter importance could not be computed: {exc!r}"
    if not importance:
        return pd.DataFrame(columns=columns), "Parameter importance is empty for this study."
    df = pd.DataFrame(importance.items(), columns=columns).sort_values("importance", ascending=False).reset_index(drop=True)
    return df, None

def plot_hpo_param_importance(importance, top_n=15, figsize=None, show=True):
    if importance is None or not hasattr(importance, "empty") or importance.empty:
        print("No parameter importance values are available to plot.")
        return None, None
    plot_df = importance.head(int(top_n)).iloc[::-1]
    figsize = figsize or (9, max(3, 0.35 * len(plot_df)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df["parameter"], plot_df["importance"], color="tab:green")
    ax.set_title("HPO Parameter Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

def _postprocess_metric_scalar(value):
    if isinstance(value, (list, tuple)):
        numeric = [item for item in value if isinstance(item, (int, float, np.integer, np.floating))]
        if numeric:
            return float(numeric[-1])
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

def _postprocess_hpo_metric_extract(metrics, task=None):
    if not isinstance(metrics, dict):
        return {}
    modes = [str(task).upper()] if task is not None and str(task).upper() in ["UT", "FT"] else ["UT", "FT"]
    extracted = {"evaluation_split": metrics.get("evaluation_split")}
    for mode in modes:
        for key in ["best_loss", "best_mse", "best_rmse", "val_mae", "val_mse", "val_rmse", "mae", "mse", "rmse"]:
            metric_key = f"{mode}_{key}"
            if metric_key in metrics:
                extracted[metric_key] = _postprocess_metric_scalar(metrics[metric_key])
        summary = metrics.get(f"{mode}_prediction_summary")
        if isinstance(summary, dict):
            for key in ["rmse", "mae", "mse", "collapse_ratio", "skill_vs_mean_curve_rmse", "skill_vs_mean_field_rmse"]:
                if key in summary:
                    extracted[f"{mode}_summary_{key}"] = _postprocess_metric_scalar(summary[key])
    return extracted

def postprocess_cross_model_hpo_comparison(cross_hpo_dir, model_names=None, task=None):
    """
    Compare model-family HPO folders under Z:/p2/{task}/{output_kind}/HPO/{run_descriptor}.
    """
    base_dir = Path(cross_hpo_dir).expanduser()
    columns = [
        "model",
        "best_value",
        "best_trial",
        "n_trials",
        "n_complete",
        "n_pruned",
        "n_failed",
        "study_db",
        "best_model_json",
        "results_dir",
    ]
    if not base_dir.exists():
        return pd.DataFrame(columns=columns)

    if model_names is None:
        candidates = sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    else:
        candidates = [base_dir / str(name) for name in model_names]

    rows = []
    for hpo_dir in candidates:
        if not hpo_dir.exists():
            continue
        study_info = postprocess_load_hpo_study(hpo_dir)
        study = study_info.get("study")
        stats = postprocess_hpo_study_stats(study)
        stats_map = dict(stats.values.tolist()) if not stats.empty else {}
        artifacts = postprocess_resolve_artifacts(hpo_dir, prefer_hpo_best=True)
        loaded = postprocess_load_artifacts(artifacts)
        row = {
            "model": hpo_dir.name,
            "best_value": stats_map.get("best_value"),
            "best_trial": stats_map.get("best_trial"),
            "n_trials": stats_map.get("n_trials"),
            "n_complete": stats_map.get("n_complete"),
            "n_pruned": stats_map.get("n_pruned"),
            "n_failed": stats_map.get("n_failed"),
            "study_db": str(study_info.get("study_db")) if study_info.get("study_db") is not None else None,
            "best_model_json": str(artifacts.get("model_json")) if artifacts.get("model_json") is not None else None,
            "results_dir": str(artifacts.get("results_dir")) if artifacts.get("results_dir") is not None else None,
        }
        row.update(_postprocess_hpo_metric_extract(loaded.get("metrics"), task=task))
        if study_info.get("message") and study is None:
            row["study_message"] = study_info.get("message")
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    df["_sort_value"] = pd.to_numeric(df["best_value"], errors="coerce")
    df = df.sort_values(["_sort_value", "model"], ascending=[True, True], na_position="last")
    return df.drop(columns=["_sort_value"]).reset_index(drop=True)

def postprocess_select_cross_model_hpo_model(comparison, preferred_metric="best_value", task=None):
    """Select one model family from a cross-model HPO table using lower-is-better metrics."""
    if comparison is None or not hasattr(comparison, "empty") or comparison.empty:
        return None, None, None

    mode = str(task).upper() if task is not None and str(task).upper() in ["UT", "FT"] else None
    metric_candidates = []

    def _add_metric(metric):
        if metric and metric in comparison.columns and metric not in metric_candidates:
            metric_candidates.append(metric)

    _add_metric(preferred_metric)
    _add_metric("best_value")
    if mode is not None:
        _add_metric(f"{mode}_best_rmse")
        _add_metric(f"{mode}_summary_rmse")
        _add_metric(f"{mode}_val_rmse")
    for column in comparison.columns:
        if str(column).endswith(("_best_rmse", "_summary_rmse", "_val_rmse")):
            _add_metric(column)

    for metric in metric_candidates:
        ranked = comparison.copy()
        ranked["_selection_value"] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=["_selection_value"])
        if ranked.empty:
            continue
        ranked = ranked.sort_values(["_selection_value", "model"], ascending=[True, True])
        row = ranked.iloc[0].drop(labels=["_selection_value"])
        return row.get("model"), metric, row

    return None, None, None

def plot_cross_model_hpo_comparison(comparison, figsize=(9, 4), show=True):
    if comparison is None or not hasattr(comparison, "empty") or comparison.empty:
        print("No cross-model HPO comparison rows are available.")
        return None, None
    plot_df = comparison.dropna(subset=["best_value"]).copy()
    if plot_df.empty:
        print("No best objective values are available for cross-model plotting.")
        return None, None
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(plot_df["model"], plot_df["best_value"], color="tab:blue", alpha=0.85)
    ax.set_title("Cross-Model HPO Best Objective")
    ax.set_xlabel("Model family")
    ax.set_ylabel("Best objective value")
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        n_complete = row.get("n_complete")
        if pd.notna(n_complete):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={int(n_complete)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

def postprocess_load_artifacts(artifacts):
    """Load saved metadata, scalar metrics, predictions, HPO files, and diagnostic CSVs."""
    loaded = {
        "descriptor": None,
        "data_descriptor": None,
        "metrics": {},
        "diagnostics_summary": {},
        "predictions": {},
        "loss_history": None,
        "diagnostic_tables": {},
        "hpo": {},
    }
    for key, target in [
        ("descriptor", artifacts.get("model_json")),
        ("data_descriptor", artifacts.get("data_json")),
        ("metrics", artifacts.get("metrics_json")),
        ("diagnostics_summary", artifacts.get("diagnostics_summary_json")),
    ]:
        if target is not None and Path(target).exists():
            with open(target, "r", encoding="utf-8") as f:
                loaded[key] = json.load(f)

    for key, target in [
        ("best_params", artifacts.get("hpo_best_params_json")),
        ("best_trial_user_attrs", artifacts.get("hpo_best_trial_user_attrs_json")),
    ]:
        if target is not None and Path(target).exists():
            with open(target, "r", encoding="utf-8") as f:
                loaded["hpo"][key] = json.load(f)

    predictions_npz = artifacts.get("predictions_npz")
    if predictions_npz is not None and Path(predictions_npz).exists():
        with np.load(predictions_npz, allow_pickle=False) as npz:
            loaded["predictions"] = {key: npz[key] for key in npz.files}

    loss_history_csv = artifacts.get("loss_history_csv")
    if loss_history_csv is not None and Path(loss_history_csv).exists():
        loaded["loss_history"] = pd.read_csv(loss_history_csv)

    results_dir = artifacts.get("results_dir")
    if results_dir is not None and Path(results_dir).exists():
        for csv_file in Path(results_dir).glob("*_metrics.csv"):
            try:
                loaded["diagnostic_tables"][csv_file.stem] = pd.read_csv(csv_file, index_col=0)
            except Exception:
                loaded["diagnostic_tables"][csv_file.stem] = pd.read_csv(csv_file)

    return loaded

def postprocess_load_data(
    data_json,
    data_path_override=None,
    auto_path_root=None,
    default_data_path=r"Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata",
    **overrides,
):
    """
    Load a DATA sidecar JSON while allowing the saved DATA constructor path to
    be replaced by a local path such as Z:/p2.
    """
    from resources.MLdata import DATA

    data_json = Path(data_json)
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    config = payload.get("data_config", payload.get("config", payload))
    if not isinstance(config, dict):
        raise ValueError(f"DATA JSON at '{data_json}' does not contain a valid data_config dictionary.")

    config = dict(config)
    if data_path_override is None or str(data_path_override).strip().lower() in ["", "auto"]:
        config["path"] = _postprocess_normalize_data_path(default_data_path)
    elif data_path_override is not None:
        config["path"] = _postprocess_normalize_data_path(data_path_override)
    config.update(overrides)
    return DATA(**config)

def _postprocess_normalize_data_path(path):
    path = Path(path)
    if path.name.lower() == "mldata":
        path = path.parent
    return str(path)

def _postprocess_saved_output_kind(loaded):
    descriptor = loaded.get("descriptor", {}) if isinstance(loaded, dict) else {}
    if isinstance(descriptor, dict):
        run_layout = descriptor.get("run_layout", {})
        if isinstance(run_layout, dict):
            output_kind = run_layout.get("output_kind")
            if output_kind is not None:
                return str(output_kind).lower()

    descriptor = loaded.get("data_descriptor", {}) if isinstance(loaded, dict) else {}
    if not isinstance(descriptor, dict):
        return None
    config = descriptor.get("data_config", descriptor.get("config", descriptor))
    if isinstance(config, dict):
        input_kind = str(config.get("input_kind", "geometry") or "geometry").lower()
        output_kind = config.get("output_kind")
        if output_kind is not None:
            if str(output_kind).lower() == "curve" and input_kind == "field":
                return "fieldtocurve"
            return str(output_kind).lower()
    return None

def _postprocess_is_field_output(data, loaded=None, outputs=None):
    if data is not None and str(getattr(data, "output_kind", "curve")).lower() == "field":
        return True
    if loaded is not None and _postprocess_saved_output_kind(loaded) == "field":
        return True
    if outputs is not None and np.asarray(outputs).ndim == 4:
        return True
    return False

def _postprocess_saved_summary(loaded, mode):
    diag_summary = loaded.get("diagnostics_summary", {}) if isinstance(loaded, dict) else {}
    if not isinstance(diag_summary, dict):
        return {}
    saved_summary = diag_summary.get(mode, diag_summary.get(mode.lower(), {}))
    return saved_summary if isinstance(saved_summary, dict) else {}

def _postprocess_field_metadata(data, loaded, mode, outputs=None):
    summary = _postprocess_saved_summary(loaded, mode)
    field_shape = getattr(data, f"{mode}_field_shape", None) if data is not None else None
    outputs_arr = np.asarray(outputs) if outputs is not None else None

    if field_shape is None and outputs_arr is not None and outputs_arr.ndim == 4:
        field_shape = outputs_arr.shape[1:4]
    if field_shape is None:
        keys = ("n_frames", "n_nodes", "n_components")
        if all(key in summary for key in keys):
            field_shape = tuple(int(summary[key]) for key in keys)
    if field_shape is not None:
        field_shape = tuple(int(v) for v in field_shape)

    frame_values = getattr(data, f"{mode}_field_frame_values", None) if data is not None else None
    components = getattr(data, f"{mode}_field_components", None) if data is not None else None
    node_labels = getattr(data, f"{mode}_field_node_labels", None) if data is not None else None
    node_coords = getattr(data, f"{mode}_field_node_coords", None) if data is not None else None

    if field_shape is not None:
        n_frames, n_nodes, n_components = field_shape
        if frame_values is None:
            frame_values = np.arange(n_frames)
        if components is None:
            components = _field_default_components(n_components)
        if node_labels is None:
            node_labels = np.arange(n_nodes)

    return {
        "field_shape": field_shape,
        "frame_values": frame_values,
        "components": components,
        "node_labels": node_labels,
        "node_coords": node_coords,
    }

def _postprocess_reconstruct_output(data, mode, values):
    reconstructor = getattr(data, f"{mode}_OUTreconstructor", None) if data is not None else None
    return reconstructor(values) if callable(reconstructor) else values

def _postprocess_train_truth(data, mode):
    if data is None:
        return None
    values = getattr(data, f"{mode}_train_out", None)
    if values is None:
        return None
    return _postprocess_reconstruct_output(data, mode, values)

def postprocess_available_evaluations(loaded):
    """Summarize which mode/split prediction arrays and diagnostic tables are available."""
    row_map = {}
    predictions = loaded.get("predictions", {})
    tables = loaded.get("diagnostic_tables", {})
    for key in set(predictions.keys()):
        parts = key.split("_")
        if len(parts) < 3:
            continue
        mode, split, kind = parts[0].upper(), parts[1].lower(), "_".join(parts[2:])
        if kind not in ["outputs", "truth"]:
            continue
        row_map.setdefault((mode, split), {"mode": mode, "split": split})

    for key in set(tables.keys()):
        parts = key.split("_")
        if len(parts) < 3:
            continue
        mode, split = parts[0].upper(), parts[1].lower()
        row_map.setdefault((mode, split), {"mode": mode, "split": split})

    rows = []
    for (mode, split), row in row_map.items():
        row.update(
            {
                "outputs": f"{mode}_{split}_outputs" in predictions,
                "truth": f"{mode}_{split}_truth" in predictions,
                "sample_metrics": f"{mode}_{split}_sample_metrics" in tables,
                "point_metrics": f"{mode}_{split}_point_metrics" in tables,
                "zone_metrics": f"{mode}_{split}_zone_metrics" in tables,
                "frame_metrics": f"{mode}_{split}_frame_metrics" in tables,
                "component_metrics": f"{mode}_{split}_component_metrics" in tables,
                "node_metrics": f"{mode}_{split}_node_metrics" in tables,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mode", "split"]).reset_index(drop=True) if rows else pd.DataFrame()

def postprocess_attach_results(model_obj, loaded):
    """Attach saved predictions and scalar metrics to a MODEL object using framework attribute names."""
    if model_obj is None:
        return None
    metrics = loaded.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(key, str) and (key.startswith("UT_") or key.startswith("FT_")):
            try:
                setattr(model_obj, key, value)
            except Exception:
                pass

    for key, value in loaded.get("predictions", {}).items():
        setattr(model_obj, key, value)
        parts = key.split("_")
        if len(parts) >= 3 and parts[1].lower() == "test" and parts[-1] == "truth":
            setattr(model_obj, f"{parts[0].upper()}_truth", value)
        if len(parts) >= 3 and parts[1].lower() == "test" and parts[-1] == "outputs":
            setattr(model_obj, f"{parts[0].upper()}_test_outputs", value)
    return model_obj

def postprocess_build_diagnostics(
    data,
    loaded,
    mode="UT",
    split="test",
    zone_boundaries=None,
    prefer_saved_tables=True,
    recompute_from_predictions=True,
):
    """
    Build an in-memory diagnostics dictionary from saved predictions and saved
    diagnostic CSVs. This does not run the model.
    """
    mode = str(mode).upper()
    split = str(split).lower()
    predictions = loaded.get("predictions", {})
    outputs = predictions.get(f"{mode}_{split}_outputs")
    truth = predictions.get(f"{mode}_{split}_truth")
    if outputs is None or truth is None:
        return None

    is_field = _postprocess_is_field_output(data, loaded=loaded, outputs=outputs)
    if recompute_from_predictions:
        train_truth = _postprocess_train_truth(data, mode)
        if is_field:
            metadata = _postprocess_field_metadata(data, loaded, mode, outputs=outputs)
            if np.asarray(outputs).ndim == 3 and metadata["field_shape"] is None:
                raise ValueError(
                    "Field predictions are saved as [samples, nodes, outputs], so "
                    "postprocess_build_diagnostics needs a loaded field DATA object "
                    "or saved diagnostics_summary with n_frames, n_nodes, and n_components."
                )
            diagnostics = field_performance_diagnostics(
                outputs,
                truth,
                train_truth=train_truth,
                **metadata,
            )
        else:
            try:
                diagnostics = curve_performance_diagnostics(
                    outputs,
                    truth,
                    x_values=getattr(data, f"{mode}_OUT_df", None) if data is not None else None,
                    train_truth=train_truth,
                    zone_boundaries=zone_boundaries,
                )
            except ValueError:
                diagnostics = curve_performance_diagnostics(
                    outputs,
                    truth,
                    x_values=getattr(data, f"{mode}_OUT_df", None) if data is not None else None,
                    train_truth=train_truth,
                    zone_boundaries=None,
                )
    elif is_field:
        metadata = _postprocess_field_metadata(data, loaded, mode, outputs=outputs)
        pred = _field_to_frame_node_component(outputs, field_shape=metadata["field_shape"], name="y_pred")
        true = _field_to_frame_node_component(truth, field_shape=metadata["field_shape"], name="y_true")
        diagnostics = {
            "summary": {},
            "sample_metrics": None,
            "frame_metrics": None,
            "component_metrics": None,
            "node_metrics": None,
            "y_pred": pred,
            "y_true": true,
            "valid_mask": np.isfinite(true) & np.isfinite(pred),
            **metadata,
        }
        pred_masked = np.where(diagnostics["valid_mask"], pred, np.nan)
        true_masked = np.where(diagnostics["valid_mask"], true, np.nan)
        diagnostics["pred_std"] = _field_nanstd(pred_masked, axis=0)
        diagnostics["true_std"] = _field_nanstd(true_masked, axis=0)
        diagnostics["std_ratio"] = diagnostics["pred_std"] / np.maximum(diagnostics["true_std"], 1e-12)
    else:
        diagnostics = {
            "summary": {},
            "sample_metrics": None,
            "point_metrics": None,
            "zone_metrics": None,
            "x": np.arange(np.asarray(outputs).shape[1], dtype=float),
            "y_pred": np.asarray(outputs, dtype=float),
            "y_true": np.asarray(truth, dtype=float),
        }

    saved_summary = _postprocess_saved_summary(loaded, mode)
    if saved_summary:
        diagnostics["summary"].update(saved_summary)

    if prefer_saved_tables:
        tables = loaded.get("diagnostic_tables", {})
        table_keys = ["sample_metrics", "frame_metrics", "component_metrics", "node_metrics"] if is_field else [
            "sample_metrics",
            "point_metrics",
            "zone_metrics",
        ]
        for key in table_keys:
            saved = tables.get(f"{mode}_{split}_{key}")
            if saved is not None:
                diagnostics[key] = saved
        if is_field:
            node_metrics = diagnostics.get("node_metrics")
            if (
                diagnostics.get("node_coords") is None
                and hasattr(node_metrics, "columns")
                and {"x", "y"}.issubset(node_metrics.columns)
            ):
                ordered_nodes = node_metrics
                if "node" in ordered_nodes.columns:
                    ordered_nodes = ordered_nodes.sort_values("node")
                coords = ordered_nodes[["x", "y"]].to_numpy(dtype=float)
                field_shape = diagnostics.get("field_shape") or (None, None, None)
                if coords.shape[0] == field_shape[1]:
                    diagnostics["node_coords"] = coords
                    if "node_label" in ordered_nodes.columns:
                        diagnostics["node_labels"] = ordered_nodes["node_label"].to_numpy()
        if not is_field and diagnostics.get("point_metrics") is not None and "x" in diagnostics["point_metrics"].columns:
            diagnostics["x"] = diagnostics["point_metrics"]["x"].to_numpy(dtype=float)

    return diagnostics

def postprocess_artifact_table(artifacts, keys=None):
    keys = keys or [
        "input_path",
        "run_dir",
        "model_json",
        "model_mdl",
        "data_json",
        "results_dir",
        "metrics_json",
        "predictions_npz",
        "loss_history_csv",
        "diagnostics_summary_json",
        "is_hpo",
    ]
    rows = []
    for key in keys:
        value = artifacts.get(key)
        rows.append((key, str(value) if value is not None else None))
    return pd.DataFrame(rows, columns=["artifact", "path_or_value"])

def postprocess_load_field_run(
    run_path,
    run_root=None,
    prefer_hpo_best=True,
    load_data=True,
    load_model=True,
    data_path_override=None,
    device="cpu",
    verbose=True,
):
    return _postprocess_load_run(
        run_path,
        run_root=run_root,
        prefer_hpo_best=prefer_hpo_best,
        load_data=load_data,
        load_model=load_model,
        data_path_override=data_path_override,
        device=device,
        verbose=verbose,
    )

def postprocess_field_run_overview(
    artifacts,
    loaded,
    data=None,
    run_name=None,
    run_type=None,
    mech_mode=None,
    view_mode="UT",
    model_name=None,
    device=None,
    active_split=None,
):
    descriptor = loaded.get("descriptor") or {}
    metrics = loaded.get("metrics") or {}
    hpo = loaded.get("hpo") or {}

    saved_output_kind = _postprocess_saved_output_kind(loaded)
    loaded_output_kind = getattr(data, "output_kind", None) if data is not None else saved_output_kind

    run_summary_fields = [
        "run",
        "run_type",
        "mechMode",
        "VIEW_MODE",
        "model",
        "output_kind",
        "device",
        "evaluation_split",
        "is_hpo",
        "model_json",
        "results_dir",
    ]
    summary_values = {
        "run": run_name,
        "run_type": run_type,
        "mechMode": mech_mode,
        "VIEW_MODE": view_mode,
        "model": model_name,
        "output_kind": loaded_output_kind,
        "device": device,
        "evaluation_split": metrics.get("evaluation_split"),
        "is_hpo": artifacts.get("is_hpo"),
        "model_json": artifacts.get("model_json"),
        "results_dir": artifacts.get("results_dir"),
    }
    summary_table = pd.DataFrame([(key, summary_values.get(key)) for key in run_summary_fields], columns=["item", "value"])

    run_descriptor = descriptor.get("run_descriptor") if isinstance(descriptor, dict) else None
    model_setup = pd.DataFrame(columns=["item", "value"])
    if isinstance(run_descriptor, dict):
        setup_fields = ["model_type", "name", "in_size", "out_size", "hidden_size", "n_layers", "n_heads", "dropout"]
        compact_descriptor = {
            "model_type": descriptor.get("model_type"),
            "name": descriptor.get("name"),
            "in_size": run_descriptor.get("in_size"),
            "out_size": run_descriptor.get("out_size"),
            "hidden_size": run_descriptor.get("hidden_size"),
            "n_layers": run_descriptor.get("n_layers"),
            "n_heads": run_descriptor.get("n_heads"),
            "dropout": run_descriptor.get("dropout"),
        }
        compact_descriptor = {k: v for k, v in compact_descriptor.items() if k in setup_fields and v is not None}
        if compact_descriptor:
            model_setup = pd.DataFrame(compact_descriptor.items(), columns=["item", "value"])

    active_metric_keys = ["best_epoch", "best_loss", "best_mse", "best_rmse", "mae", "mse", "rmse", "best", "worst"]
    metric_rows = []
    for key in [f"{str(view_mode).upper()}_{metric}" for metric in active_metric_keys]:
        if key in metrics:
            metric_rows.append((key, metrics[key]))
    active_metrics = pd.DataFrame(metric_rows, columns=["metric", "value"])

    available_evals = postprocess_available_evaluations(loaded)
    field_eval_columns = ["mode", "split", "outputs", "truth", "sample_metrics", "frame_metrics", "component_metrics", "node_metrics"]
    available_field_evals = available_evals[[col for col in field_eval_columns if col in available_evals.columns]] if not available_evals.empty else pd.DataFrame(columns=field_eval_columns)

    resolved_split = active_split
    if resolved_split is None:
        resolved_split = metrics.get("evaluation_split", None)
    if resolved_split is None and not available_evals.empty:
        matching = available_evals[available_evals["mode"].astype(str).str.upper() == str(view_mode).upper()]
        resolved_split = (matching.iloc[0] if not matching.empty else available_evals.iloc[0])["split"]

    warnings = []
    if loaded_output_kind is not None and str(loaded_output_kind).lower() != "field":
        warnings.append("This notebook is intended for field-output runs, but the selected run does not look like output_kind='field'.")

    return {
        "summary": summary_table,
        "model_setup": model_setup,
        "active_metrics": active_metrics,
        "available_evals": available_evals,
        "available_field_evals": available_field_evals,
        "active_split": resolved_split,
        "hpo": hpo,
        "warnings": warnings,
    }

def display_field_run_overview(overview):
    from IPython.display import Markdown, display

    for warning in overview.get("warnings", []):
        print("WARNING:", warning)

    display(overview["summary"])
    if not overview["model_setup"].empty:
        display(Markdown("### Model Setup"))
        display(overview["model_setup"])
    if not overview["active_metrics"].empty:
        display(Markdown("### Saved Metrics For Active View"))
        display(overview["active_metrics"])

    display(Markdown("### Available Field Predictions / Diagnostic Tables"))
    display(overview["available_field_evals"])

    hpo = overview.get("hpo", {})
    if hpo:
        display(Markdown("### HPO Summary"))
        for key, value in hpo.items():
            if isinstance(value, dict):
                display(Markdown(f"#### {key}"))
                display(pd.DataFrame(value.items(), columns=["parameter", "value"]))
    else:
        print("No HPO files were found for this run.")

def postprocess_build_active_field_diagnostics(
    data,
    loaded,
    available_evals,
    view_mode="UT",
    active_split=None,
    model=None,
):
    diagnostics = {}
    for _, row in available_evals.iterrows():
        mode = str(row["mode"]).upper()
        split = str(row["split"]).lower()
        diag = postprocess_build_diagnostics(
            data,
            loaded,
            mode=mode,
            split=split,
            prefer_saved_tables=True,
            recompute_from_predictions=False,
        )
        if diag is None:
            print(f"No saved diagnostics are available for {mode} {split}.")
            continue
        if "field_shape" not in diag:
            print(f"Skipping {mode} {split}: diagnostics are not field diagnostics.")
            continue

        diagnostics[(mode, split)] = diag
        if model is not None:
            setattr(model, f"{mode}_{split}_diagnostics", diag)
            if split == "test":
                setattr(model, f"{mode}_diagnostics", diag)
                setattr(model, f"{mode}_prediction_summary", diag.get("summary"))

    active_key = (str(view_mode).upper(), str(active_split).lower()) if view_mode and active_split else None
    active_diag = diagnostics.get(active_key) if active_key is not None else None
    return diagnostics, active_key, active_diag

def field_summary_table(diagnostics, metrics=None):
    if diagnostics is None:
        return pd.DataFrame(columns=["metric", "value"])
    metrics = metrics or [
        "rmse",
        "mae",
        "mse",
        "bias",
        "collapse_ratio",
        "mean_field_baseline_rmse",
        "skill_vs_mean_field_rmse",
        "valid_fraction",
        "n_samples",
        "n_frames",
        "n_nodes",
        "n_components",
    ]
    summary = diagnostics.get("summary", {})
    return pd.DataFrame([(key, summary.get(key)) for key in metrics if key in summary], columns=["metric", "value"])

def display_field_sample_error_summary(
    diagnostics,
    bins=40,
    ncols=3,
    top_n=5,
    columns=None,
):
    from IPython.display import Markdown, display

    if diagnostics is None:
        print("No active diagnostics are available.")
        return
    sample_metrics = diagnostics.get("sample_metrics")
    if sample_metrics is None or not hasattr(sample_metrics, "copy"):
        print("Sample metrics are unavailable.")
        return

    samples = sample_metrics.copy()
    if "sample_rmse" in samples.columns:
        display(Markdown("### Best And Worst Samples By RMSE"))
        best_worst = pd.concat([
            samples.sort_values("sample_rmse").head(int(top_n)),
            samples.sort_values("sample_rmse").tail(int(top_n)),
        ]).drop_duplicates()
        display(best_worst)

    columns = columns or ["sample_mae", "sample_mse", "sample_rmse", "sample_bias", "valid_fraction"]
    try:
        plot_field_sample_metric_distributions(diagnostics, columns=columns, bins=bins, ncols=ncols)
    except ValueError as exc:
        print(exc)

    describe_cols = [col for col in columns if col in samples.columns]
    if describe_cols:
        display(Markdown("### Sample Metric Summary"))
        display(samples[describe_cols].describe().T)

def display_field_node_error_summary(
    diagnostics,
    metric="rmse",
    top_n=20,
    plot_columns=None,
    point_size=22,
):
    from IPython.display import Markdown, display

    if diagnostics is None:
        print("No active diagnostics are available.")
        return
    node_metrics = diagnostics.get("node_metrics")
    if node_metrics is None or not hasattr(node_metrics, "copy"):
        print("Node metrics are unavailable.")
        return

    node = node_metrics.copy()
    metric = metric if metric in node.columns else "rmse"
    display(Markdown(f"### Worst Nodes By {metric}"))
    display(node.sort_values(metric, ascending=False).head(int(top_n)))

    try:
        plot_field_node_metrics(diagnostics, columns=plot_columns, point_size=point_size)
    except ValueError as exc:
        print(exc)

def field_sample_viewer(
    diagnostics,
    sample_mode="selected",
    selected_samples=0,
    frame=10,
    component="U2",
    ranking_metric="rmse",
    plot_style="continuous",
    random_count=5,
):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None

    try:
        import ipywidgets as widgets
        widgets.Widget.close_all()
    except Exception:
        widgets = None

    import ast
    from io import BytesIO
    from IPython.display import Image as IPyImage, display

    state = {"last_key": None}
    samples = diagnostics["sample_metrics"].copy()
    coords = diagnostics.get("node_coords")
    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    n_samples, n_frames, _, n_components = y_pred.shape
    components = [str(c) for c in (diagnostics.get("components") or _field_default_components(n_components))]
    metric_options = [metric for metric in ["rmse", "mae", "mse", "bias"] if f"sample_{metric}" in samples.columns]
    if not metric_options:
        metric_options = ["rmse"]

    def _as_sample_list(value):
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            return [int(v) for v in list(value)]

        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (int, np.integer)):
                return [int(parsed)]
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                return [int(v) for v in list(parsed)]
        except Exception:
            pass
        return [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]

    def _ranked_samples(metric, chosen_mode, count=1):
        col = f"sample_{metric}"
        if col not in samples.columns:
            col = "sample_rmse"
        ranked = samples[["sample", col]].dropna().copy()
        ranked["_rank_value"] = ranked[col].abs() if metric == "bias" else ranked[col]
        ranked = ranked.sort_values("_rank_value", ascending=(chosen_mode == "best"))
        return ranked["sample"].astype(int).head(int(count)).tolist()

    def _sample_metric_text(sample_idx):
        row = samples.loc[samples["sample"].astype(int) == int(sample_idx)]
        if row.empty:
            return ""
        parts = []
        for metric in ["rmse", "mae", "mse", "bias"]:
            col = f"sample_{metric}"
            if col in row.columns:
                parts.append(f"{metric}={row[col].iloc[0]:.4g}")
        return ", " + ", ".join(parts) if parts else ""

    def _choose_samples(chosen_mode, sample_text, metric, count):
        if chosen_mode == "selected":
            chosen = _as_sample_list(sample_text)
        elif chosen_mode in ["best", "worst"]:
            chosen = _ranked_samples(metric, chosen_mode, count=1)
        elif chosen_mode == "random":
            valid_samples = samples["sample"].astype(int).to_numpy()
            n_random = min(max(5, int(count)), len(valid_samples))
            chosen = np.random.default_rng().choice(valid_samples, size=n_random, replace=False).astype(int).tolist()
        else:
            chosen = []
        return [idx for idx in dict.fromkeys(chosen) if 0 <= int(idx) < n_samples]

    def _render_key(chosen_mode, sample_text, frame_value, component_value, metric, style, count):
        return (chosen_mode, str(sample_text), int(frame_value), int(component_value), metric, style, int(count))

    def _figure_png(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return IPyImage(data=buffer.getvalue())

    def _draw_field_samples(chosen_mode, sample_text, frame_value, component_value, metric, style, count, out=None, force=False):
        render_key = _render_key(chosen_mode, sample_text, frame_value, component_value, metric, style, count)
        if not force and render_key == state["last_key"]:
            return
        state["last_key"] = render_key

        def _write(text):
            if out is None:
                print(text)
            else:
                out.append_stdout(text + "\n")

        if coords is None:
            _write("Node coordinates are unavailable, so field sample maps cannot be drawn.")
            return
        if metric == "bias":
            _write("Bias = mean(prediction - truth); positive values over-predict and negative values under-predict.")

        frame_label = int(np.clip(frame_value, 1, n_frames))
        frame_idx = frame_label - 1
        component_idx = int(component_value)
        chosen = _choose_samples(chosen_mode, sample_text, metric, count)
        if not chosen:
            _write("No valid sample indices were selected.")
            return

        for sample_idx in chosen:
            _write(
                f"Sample {sample_idx}{_sample_metric_text(sample_idx)}, "
                f"frame={frame_label}, component={components[component_idx]}, style={style}"
            )
            fig, _ = plot_field_sample(
                diagnostics,
                sample=sample_idx,
                frame=frame_idx,
                component=component_idx,
                node_coords=coords,
                plot_style=style,
                show=False,
            )
            if out is None:
                display(_figure_png(fig))
            else:
                out.append_display_data(_figure_png(fig))

    try:
        if widgets is None:
            import ipywidgets as widgets

        sample_mode_widget = widgets.ToggleButtons(
            options=["selected", "best", "worst", "random"],
            value=sample_mode,
            description="Samples",
        )
        selected_samples_widget = widgets.Text(
            value=str(selected_samples),
            description="Selected",
            placeholder="0 or 0, 4, 12",
            layout=widgets.Layout(width="260px"),
        )
        frame_widget = widgets.IntSlider(
            value=int(np.clip(frame, 1, n_frames)),
            min=1,
            max=n_frames,
            step=1,
            description="",
            continuous_update=False,
            layout=widgets.Layout(width="190px"),
        )
        component_default = components.index(component) if component in components else (1 if n_components > 1 else 0)
        component_widget = widgets.Dropdown(
            options=[(name, idx) for idx, name in enumerate(components)],
            value=component_default,
            description="Component",
            layout=widgets.Layout(width="185px"),
            style={"description_width": "80px"},
        )
        metric_widget = widgets.Dropdown(
            options=metric_options,
            value=ranking_metric if ranking_metric in metric_options else metric_options[0],
            description="Rank by",
            layout=widgets.Layout(width="165px"),
            style={"description_width": "58px"},
        )
        plot_style_widget = widgets.ToggleButtons(
            options=["continuous", "points"],
            value=plot_style,
            description="Style",
            layout=widgets.Layout(width="320px", margin="0 0 0 28px"),
            style={"description_width": "48px"},
        )
        random_count_widget = widgets.BoundedIntText(
            value=min(max(5, int(random_count)), n_samples),
            min=min(5, n_samples),
            max=max(5, min(20, n_samples)),
            step=1,
            description="Random n",
            layout=widgets.Layout(width="150px"),
        )
        refresh_widget = widgets.Button(
            description="Refresh",
            button_style="primary",
            layout=widgets.Layout(width="80px", margin="0 0 0 16px"),
        )
        frame_label_widget = widgets.Label(value="Frame", layout=widgets.Layout(width="44px"))
        frame_minus_widget = widgets.Button(description="-", layout=widgets.Layout(width="30px", margin="0 2px 0 0"))
        frame_plus_widget = widgets.Button(description="+", layout=widgets.Layout(width="30px", margin="0 0 0 2px"))
        frame_box = widgets.HBox(
            [frame_label_widget, frame_minus_widget, frame_widget, frame_plus_widget],
            layout=widgets.Layout(align_items="center", margin="0 14px 0 0"),
        )

        controls = widgets.VBox([
            widgets.HBox([sample_mode_widget, selected_samples_widget, random_count_widget, refresh_widget]),
            widgets.HBox(
                [frame_box, component_widget, metric_widget, plot_style_widget],
                layout=widgets.Layout(margin="12px 0 0 0"),
            ),
        ])
        output = widgets.Output()

        def _current_widget_values():
            return {
                "chosen_mode": sample_mode_widget.value,
                "sample_text": selected_samples_widget.value,
                "frame_value": frame_widget.value,
                "component_value": component_widget.value,
                "metric": metric_widget.value,
                "style": plot_style_widget.value,
                "count": random_count_widget.value,
            }

        def _update_view(_=None, force=False):
            output.clear_output(wait=False)
            _draw_field_samples(**_current_widget_values(), out=output, force=force)

        def _step_frame(delta):
            frame_widget.value = int(np.clip(frame_widget.value + delta, frame_widget.min, frame_widget.max))

        frame_minus_widget.on_click(lambda _: _step_frame(-1))
        frame_plus_widget.on_click(lambda _: _step_frame(1))
        refresh_widget.on_click(lambda _: _update_view(force=True))

        for widget in [
            sample_mode_widget,
            selected_samples_widget,
            frame_widget,
            component_widget,
            metric_widget,
            plot_style_widget,
            random_count_widget,
        ]:
            widget.observe(lambda change: _update_view(force=True), names="value")

        display(controls, output)
        _update_view()
        return controls, output
    except Exception as exc:
        print("ipywidgets are unavailable; using editable variables in this cell instead.")
        print(repr(exc))
        field_component = components.index(component) if component in components else (1 if n_components > 1 else 0)
        _draw_field_samples(
            sample_mode,
            selected_samples,
            frame,
            field_component,
            ranking_metric,
            plot_style,
            random_count,
        )
        return None

def plot_field_sample_frame_evolution(diagnostics, sample=0, figsize=None):
    if diagnostics is None:
        print("No active diagnostics are available.")
        return None, None

    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    valid = np.asarray(diagnostics.get("valid_mask", np.isfinite(y_true) & np.isfinite(y_pred)), dtype=bool)
    n_samples, n_frames, _, n_components = y_pred.shape
    sample_idx = int(np.clip(sample, 0, n_samples - 1))
    x = np.arange(n_frames) + 1
    components = [str(c) for c in (diagnostics.get("components") or _field_default_components(n_components))]

    pred_s = np.where(valid[sample_idx], y_pred[sample_idx], np.nan)
    true_s = np.where(valid[sample_idx], y_true[sample_idx], np.nan)
    err_s = pred_s - true_s

    pred_mean = np.nanmean(pred_s, axis=1)
    true_mean = np.nanmean(true_s, axis=1)
    mae_frame = np.nanmean(np.abs(err_s), axis=1)

    figsize = figsize or (6 * n_components, 4)
    fig, axes = plt.subplots(1, n_components, figsize=figsize, squeeze=False)
    for comp_idx in range(n_components):
        ax = axes[0, comp_idx]
        ax.plot(x, true_mean[:, comp_idx], label="Truth", color="darkgreen")
        ax.plot(x, pred_mean[:, comp_idx], label="Prediction", color="orangered")
        ax.plot(x, mae_frame[:, comp_idx], label="Mean abs error", color="gray", linestyle="--")
        ax.set_title(f"Sample {sample_idx} - {components[comp_idx]}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean node value")
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_loss_history(loss_history, metrics=None, figsize=(9, 4)):
    metrics = metrics or ["train_loss", "val_loss"]
    if loss_history is None or not hasattr(loss_history, "empty") or loss_history.empty:
        print("No loss_history.csv was found for this run.")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)
    for mode, mode_history in loss_history.groupby("mode"):
        for metric in metrics:
            if metric in mode_history.columns:
                ax.plot(mode_history["epoch"], mode_history[metric], label=f"{mode} {metric}")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig, ax

def postprocess_save_open_figures(out_dir, prefix="", formats=("png",), dpi=250, close=False):
    """Save all currently open matplotlib figures into a post-processing folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    prefix = f"{prefix}_" if prefix else ""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        title = ""
        if fig.axes:
            title = fig.axes[0].get_title()
        stem = _postprocess_slug(title or f"figure_{num:02d}")
        for fmt in formats:
            path = out_dir / f"{prefix}{stem}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(path)
        if close:
            plt.close(fig)
    return saved

def _postprocess_slug(text, default="figure"):
    text = str(text or default).strip()
    text = "".join(ch if ch.isalnum() or ch in ["-", "_", "."] else "-" for ch in text)
    text = "-".join(part for part in text.split("-") if part)
    return text[:96] if text else default
