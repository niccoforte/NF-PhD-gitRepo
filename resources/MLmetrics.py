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
        components = [f"c{i}" for i in range(n_components)]
    components = [str(c) for c in components]
    if len(components) != n_components:
        components = [f"c{i}" for i in range(n_components)]

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

def plot_field_frame_component_heatmaps(diagnostics, figsize=(17, 5), cmaps=None):
    cmaps = cmaps or {"RMSE": "viridis", "Bias": "coolwarm", "Valid Fraction": "magma"}
    y_pred = np.asarray(diagnostics["y_pred"], dtype=float)
    y_true = np.asarray(diagnostics["y_true"], dtype=float)
    valid = np.asarray(diagnostics.get("valid_mask", np.isfinite(y_true) & np.isfinite(y_pred)), dtype=bool)
    err = y_pred - y_true

    n_frames = y_pred.shape[1]
    n_components = y_pred.shape[3]
    components = [str(c) for c in diagnostics.get("components", [f"c{i}" for i in range(n_components)])]
    frame_labels = [str(idx + 1) for idx in range(n_frames)]

    rmse_map = np.full((n_frames, n_components), np.nan)
    bias_map = np.full((n_frames, n_components), np.nan)
    valid_map = np.full((n_frames, n_components), np.nan)
    for frame_idx in range(n_frames):
        for comp_idx in range(n_components):
            mask = valid[:, frame_idx, :, comp_idx]
            values = err[:, frame_idx, :, comp_idx]
            if np.any(mask):
                rmse_map[frame_idx, comp_idx] = np.sqrt(np.nanmean(values[mask] ** 2))
                bias_map[frame_idx, comp_idx] = np.nanmean(values[mask])
                valid_map[frame_idx, comp_idx] = np.mean(mask)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, matrix, title in [
        (axes[0], rmse_map, "RMSE"),
        (axes[1], bias_map, "Bias"),
        (axes[2], valid_map, "Valid Fraction"),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap=cmaps.get(title, "viridis"))
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

def plot_field_node_metrics(diagnostics, columns=None, figsize=None, point_size=22):
    node_metrics = diagnostics.get("node_metrics")
    if node_metrics is None or not hasattr(node_metrics, "copy"):
        raise ValueError("Node metrics are unavailable.")

    node = node_metrics.copy()
    coords = diagnostics.get("node_coords")
    if ("x" not in node.columns or "y" not in node.columns) and coords is not None and len(coords) == len(node):
        coords = np.asarray(coords, dtype=float)
        node["x"] = coords[:, 0]
        node["y"] = coords[:, 1]
    if "x" not in node.columns or "y" not in node.columns:
        raise ValueError("Node coordinates are unavailable, so spatial node maps cannot be drawn.")

    columns = columns or ["rmse", "mae", "bias", "valid_fraction"]
    plot_cols = [col for col in columns if col in node.columns]
    if not plot_cols:
        raise ValueError("None of the requested node metric columns are available.")

    figsize = figsize or (5 * len(plot_cols), 4)
    fig, axes = plt.subplots(1, len(plot_cols), figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, plot_cols):
        sc = ax.scatter(node["x"], node["y"], c=node[col], cmap="viridis", s=point_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Node {col}")
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

def postprocess_list_runs(run_root="Z:/p2", max_runs=25, include_hpo=True):
    """List recent saved model runs under a local run root."""
    root = Path(run_root).expanduser()
    rows = []
    columns = ["run_dir", "model_json", "model_mdl", "results_dir", "is_hpo", "modified"]
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
        rows.append(
            {
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
    descriptor = loaded.get("data_descriptor", {}) if isinstance(loaded, dict) else {}
    if not isinstance(descriptor, dict):
        return None
    config = descriptor.get("data_config", descriptor.get("config", descriptor))
    if isinstance(config, dict):
        output_kind = config.get("output_kind")
        if output_kind is not None:
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
            components = [f"c{i}" for i in range(n_components)]
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
