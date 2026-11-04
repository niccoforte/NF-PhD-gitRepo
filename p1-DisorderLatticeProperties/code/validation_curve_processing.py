"""Read-only curve-processing qualification pilot; never rewrites MLdata.

Population transfer CSVs omit time and can contain artificial tails. Their
sample-domain filter sensitivities are not source-qualified modal cutoffs.
Stable regression is a numerical screen, not proof of an elastic body modulus.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from resources.calculations import smooth


TOPOLOGIES = {"tri": 30, "kagome": 20, "hex": 20, "FCC": 20}
IDS = (1, 50, 500, 1000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sqrt(np.mean((y-slope*x-intercept)**2))/np.ptp(y))
    return float(slope), float(intercept), residual


def initial_fit(x, y, max_strain=.005, residual_limit=.02, slope_limit=.02):
    """Find earliest three consecutive stable initial windows, with an intercept.

    Start indices 0/1/2 test startup sensitivity. Every chosen end must have
    span, positive slope and a low span-normalised residual, and every tested
    slope must lie within the declared stability limit. No averaging can rescue
    inconsistent windows. The strain cap is an explicit pilot assumption.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    candidates = []
    if len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "insufficient_finite_samples", "candidates": []}
    if np.any(np.diff(x) <= 0):
        return {"status": "nonmonotone_initial_strain", "candidates": []}
    cap = np.flatnonzero(x-x[0] <= max_strain)
    end_max = min(int(cap[-1]), 19) if len(cap) else 0
    for end in range(7, end_max+1):
        slopes, residuals = [], []
        for start in range(3):
            xx, yy = x[start:end+1], y[start:end+1]
            if len(xx) < 6 or np.ptp(yy) <= 0 or np.ptp(xx) <= 0:
                break
            slope, intercept, residual = fit_line(xx, yy)
            slopes.append(slope)
            residuals.append(residual)
        if len(slopes) != 3:
            continue
        median = float(np.median(slopes))
        spread = float(max(abs(np.asarray(slopes)/median-1))) if median > 0 else None
        candidates.append({"end_index": end, "end_strain": float(x[end]),
                           "slope": median, "startup_slopes": slopes,
                           "max_residual_over_span": max(residuals),
                           "startup_relative_slope_spread": spread,
                           "passes": bool(median > 0 and max(residuals) <= residual_limit and spread <= slope_limit)})
        if len(candidates) < 3:
            continue
        group = candidates[-3:]
        slopes = np.array([row["slope"] for row in group])
        stability = float(max(abs(slopes/np.median(slopes)-1)))
        if all(row["passes"] for row in group) and stability <= slope_limit:
            # Choose the central qualified interval, not an average of windows.
            chosen = dict(group[1])
            return {"status": "numerically_stable_candidate", "selected": chosen,
                    "neighbour_window_relative_slope_spread": stability,
                    "material_elasticity_verified": False, "body_gauge_verified": False,
                    "candidates": candidates}
    return {"status": "unresolved_initial_window", "selected": None, "candidates": candidates}


def paired_filter(x, y, corner):
    """Second-order Butterworth, forward/backward on observed sample positions.

    corner is a fraction of sample Nyquist, not a frequency inferred from strain.
    Missing values/gaps must be split before calling. No STATUS interpolation.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) <= 12 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("A finite paired observed prefix of at least 13 samples is required")
    if not 0 < corner < 1:
        raise ValueError("Corner must lie inside sample Nyquist")
    sos = butter(2, corner, output="sos")
    return sosfiltfilt(sos, x), sosfiltfilt(sos, y)


def first_crossing(y, fraction):
    peak = int(np.argmax(y))
    values = np.flatnonzero(np.asarray(y)[peak:] <= fraction*y[peak])
    return int(peak+values[0]) if len(values) else None


def numeric_metrics(x, y, fixed_end):
    crossing = first_crossing(y, .25)
    span = float(np.ptp(y))
    return {"peak": float(max(y)),
            "work_fixed_raw_endpoint": float(np.trapz(y[:fixed_end+1], x[:fixed_end+1])),
            "end_load_fixed_raw_endpoint": float(y[fixed_end]),
            "ductility_25": float(x[crossing]) if crossing is not None else None,
            "ductility_25_sample": crossing,
            "second_difference_RMS_over_span": float(np.sqrt(np.mean(np.diff(y, n=2)**2))/span) if span else None}


def transfer_prefix(path, known_n=None):
    """Read continuous columns; keep transfer support distinct from native proof."""
    points, first_deletion, first_all_zero_status = [], None, None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for index, row in enumerate(csv.reader(stream)):
            points.append([float(row[0]), float(row[1])])
            if len(row) > 2:
                status = np.asarray([float(v) for v in row[2:]], float)
                if first_all_zero_status is None and len(status) and np.all(status == 0):
                    first_all_zero_status = index
                if first_deletion is None and np.any(status == 0) and not np.all(status == 0):
                    first_deletion = index
    points = np.asarray(points)
    stop = len(points) if known_n is None else int(known_n)
    reason = "native_prefix_bound_from_retained_replay" if known_n is not None else "transfer_only_native_support_not_attested"
    if first_all_zero_status is not None:
        stop = min(stop, first_all_zero_status)
    # A zero-valued transfer tail is never accepted as observed structural failure.
    tail = np.flatnonzero(np.any(points != 0., axis=1))
    if len(tail):
        stop = min(stop, int(tail[-1]+1))
    finite = np.flatnonzero(~np.isfinite(points[:stop]).all(axis=1))
    if len(finite):
        stop = int(finite[0])
    return points[:stop], {"CSV_rows": len(points), "used_prefix_rows": stop,
                           "native_support": reason, "first_partial_STATUS_deletion_row": first_deletion,
                           "first_all_zero_STATUS_row": first_all_zero_status,
                           "no_tail_imputed": True}


def native_filter_controls(root):
    """Use only existing source-matched modal controls and fixed material events."""
    from validation_calculations import filter_force
    modal = Path(root)/"audit/modal/summary.json"
    records = []
    for control in json.loads(modal.read_text()):
        source = Path(control["source_odb"])
        relative = source.relative_to(root)
        event_path = Path(root)/"audit/failure"/relative.parent/source.stem/"initiation.json"
        curve_path = Path(root)/"audit/history"/relative.parent/source.stem/"native_curve.csv"
        if not event_path.is_file() or not curve_path.is_file():
            continue
        event = json.loads(event_path.read_text())
        if not event.get("prev_native_history"):
            continue
        with curve_path.open(newline="") as stream:
            curve = list(csv.DictReader(stream))
        time = np.array([float(row["time_s"]) for row in curve])
        x = np.array([float(row["U2_mm"]) for row in curve])
        y = np.array([float(row["RF2_N"]) for row in curve])
        endpoint = int(event["prev_native_history"]["index"])
        raw_work = float(np.trapz(y[:endpoint+1], x[:endpoint+1]))
        for factor in (.5, 1., 2.):
            cutoff = control["cutoff_2fn_Hz"]*factor
            for support in ("through_fixed_material_endpoint_only", "whole_native_history"):
                stop = endpoint+1 if support.startswith("through") else len(time)
                row = {"case_id": control["case_id"], "modal_control": control["control"],
                       "cutoff_Hz": cutoff, "cutoff_multiplier": factor,
                       "filter_support": support, "fixed_endpoint_index": endpoint,
                       "fixed_endpoint_source": str(event_path), "event_source_sha256": sha(event_path),
                       "curve_source": str(curve_path), "curve_source_sha256": sha(curve_path),
                       "modal_source_sha256": sha(modal)}
                try:
                    xx, info = filter_force(time[:stop], x[:stop], cutoff, zero_phase=True)
                    yy, _ = filter_force(time[:stop], y[:stop], cutoff, zero_phase=True)
                    work = float(np.trapz(yy[:endpoint+1], xx[:endpoint+1]))
                    row.update(status="evaluated", filter_info=info,
                               work_relative_change=work/raw_work-1,
                               fixed_endpoint_force_relative_change=float(yy[endpoint]/y[endpoint]-1),
                               filtered_endpoint_force_N=float(yy[endpoint]),
                               raw_endpoint_force_N=float(y[endpoint]))
                except ValueError as error:
                    row.update(status="rejected_sampling_or_support", reason=str(error))
                records.append(row)
    return records


def run(root, data_root, out):
    root, data_root, out = Path(root), Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    replay = json.loads((root/"audit/production/native-normalisation-replay-20261102.json").read_text())
    known = {row["raw_OUT"]: row["compared_genuine_prefix_samples"] for row in replay["cases"]}
    specifications = []
    for topology, nx in TOPOLOGIES.items():
        for mode in ("Ductile", "Fracture"):
            specifications.append((topology, "periodic", 0, mode,
                                   data_root/"per"/topology/"transfer"/f"OUT-{mode}-{topology}-{nx}-per-1.csv"))
            for design, part, token in (("full", "", "all"), ("mechanism", "Target-xs", "xs")):
                for identity in IDS:
                    specifications.append((topology, design, identity, mode,
                        data_root/"disNodes"/part/"0.2"/topology/"transfer"/
                        f"OUT-{mode}-{topology}-{nx}-20disNodes-lhs-{token}-{identity}.csv"))
    cases, filters, sources = [], [], {}
    for topology, design, identity, mode, path in specifications:
        if not path.is_file():
            cases.append({"topology": topology, "design": design, "id": identity, "mode": mode,
                          "path": str(path), "status": "missing_source"})
            continue
        before = sha(path)
        sources[str(path)] = before
        points, support = transfer_prefix(path, known.get(str(path)))
        x, y = points.T
        if len(x) < 13:
            cases.append({"topology": topology, "design": design, "id": identity, "mode": mode,
                          "path": str(path), "support": support, "status": "insufficient_prefix"})
            continue
        if mode == "Fracture" and support["first_partial_STATUS_deletion_row"] is not None:
            endpoint = max(1, min(len(x)-1, support["first_partial_STATUS_deletion_row"]-1))
            endpoint_role = "last_transfer_sample_before_partial_STATUS_loss_not_native_initiation"
        else:
            crossing = first_crossing(y, .25)
            endpoint = crossing if crossing is not None else len(x)-1
            endpoint_role = "fixed_raw_25pct_crossing_or_last_observed_prefix_not_material_initiation"
        raw = numeric_metrics(x, y, endpoint)
        record = {"topology": topology, "design": design, "id": identity, "mode": mode,
                  "path": str(path), "source_sha256": before, "support": support,
                  "fixed_endpoint_index": endpoint, "endpoint_role": endpoint_role,
                  "raw_metrics": raw, "status": "analysed"}
        if mode == "Ductile":
            record["initial_fit"] = initial_fit(x[:20], y[:20])
            if topology == "hex":
                record["expanded_strain_cap_sensitivity"] = {
                    str(cap): initial_fit(x[:30], y[:30], max_strain=cap)
                    for cap in (.01, .02, .03)}
        cases.append(record)
        methods = [("double_moving_average", x, np.asarray(smooth(smooth(y))), "whole_observed_prefix")]
        for corner in (.1, .2, .4):
            xx, yy = paired_filter(x, y, corner)
            name = f"butterworth2_zero_phase_corner_{corner}_Nyquist"
            methods.append((name, xx, yy, "whole_observed_prefix"))
            if endpoint >= 12:
                xx, yy = paired_filter(x[:endpoint+1], y[:endpoint+1], corner)
                methods.append((name, xx, yy, "through_fixed_endpoint_only"))
        for name, xx, yy, scope in methods:
            values = numeric_metrics(xx, yy, endpoint)
            baseline = raw if scope == "whole_observed_prefix" else numeric_metrics(x[:endpoint+1], y[:endpoint+1], endpoint)
            work_change = (values["work_fixed_raw_endpoint"]/baseline["work_fixed_raw_endpoint"]-1) if baseline["work_fixed_raw_endpoint"] else None
            peak_change = values["peak"]/baseline["peak"]-1 if baseline["peak"] else None
            duct_change = (values["ductility_25"]/baseline["ductility_25"]-1) if values["ductility_25"] is not None and baseline["ductility_25"] else None
            load_change = values["end_load_fixed_raw_endpoint"]/raw["end_load_fixed_raw_endpoint"]-1 if raw["end_load_fixed_raw_endpoint"] else None
            # Numerical screening thresholds declared a priori; not proof of physics.
            passes = bool(work_change is not None and abs(work_change) <= .01 and abs(peak_change) <= .02
                          and (mode != "Ductile" or (duct_change is not None and abs(duct_change) <= .02)))
            filters.append({"topology": topology, "design": design, "id": identity, "mode": mode,
                            "method": name, "filter_support": scope, "work_relative_change": work_change,
                            "peak_relative_change": peak_change, "ductility_relative_change": duct_change,
                            "fixed_endpoint_load_relative_change": load_change,
                            "numerical_preservation_screen_passes": passes, "metrics": values,
                            "x_monotone": bool(np.all(np.diff(xx) >= 0)),
                            "initial_fit_slope_change": None})
            if mode == "Ductile" and record["initial_fit"].get("selected"):
                end = record["initial_fit"]["selected"]["end_index"]
                filtered_slope = fit_line(xx[:end+1], yy[:end+1])[0]
                native_slope = fit_line(x[:end+1], y[:end+1])[0]
                filters[-1]["initial_fit_slope_change"] = filtered_slope/native_slope-1
    assert all(sha(Path(path)) == digest for path, digest in sources.items()), "Source changed during pilot"
    result = {"status": "Completed source-preserving processing pilot; production route unchanged",
              "scope": "Four topologies, periodic plus full/mechanism original IDs 1,50,500,1000; UAT and FT",
              "source_sha256": sources, "sources_unchanged": True, "cases": cases, "filters": filters,
              "native_fixed_material_event_filter_controls": native_filter_controls(root),
              "filter_rule": {"order": 2, "application": "forward-backward SOS, paired continuous coordinates/response",
                              "corners_fraction_Nyquist": [.1,.2,.4], "physical_Hz_cutoff_qualified": False,
                              "screen": "|work change|<=1%, |peak change|<=2%, |UAT ductility change|<=2% at fixed raw endpoint",
                              "screen_interpretation": "Sensitivity flag only. Raw noisy maxima are not ground truth, so failing raw-peak preservation does not itself disqualify denoising.",
                              "limitations": "Sample positions are not recorded time. Zero-phase endpoint/future-information and aliasing remain checks; no STATUS or missing tails filtered."},
              "fit_rule": {"minimum_samples_after_startup": 6, "starts": [0,1,2], "end_indices": [7,19],
                           "maximum_strain_span": .005, "span_normalised_RMS_limit": .02,
                           "startup_and_neighbour_slope_limit": .02, "consecutive_windows": 3,
                           "selection": "Central member of earliest stable initial family; no averaging invalid windows",
                           "material_preplasticity_and_body_gauge_verified": False},
              "mask_migration": ["Keep native timestamps and observation masks; do not zero-fill missing fracture observations.",
                  "Fit scalers only to observed training-fold values. Store finite placeholders plus explicit masks for tensor operations.",
                  "Apply masks before loss arithmetic, including derivatives, work integrals and metrics; require contiguous support for differences.",
                  "Do not give unknown test-time termination masks to the network as an input: that leaks a response-derived failure endpoint.",
                  "Compare identical architecture/folds/seeds on raw+mask, legacy smoothing+mask and qualified Butterworth+mask; maintain existing winner until holdout evidence.",
                  "Evaluate observed-support curve errors, peak/ductility/work bias, recovered-property errors and results versus observed coverage; no arbitrary training of missing tails as zero.",
                  "Retain independently observed initiation/property labels; filtering a training target must not redefine material onset."],
              "limits": ["Transfer-only support for most cases is not a native ODB attestation.",
                         "Numerically stable grip-based slopes are candidates, not accepted elastic material moduli.",
                         "No fitted cutoff is chosen to improve reference agreement; no source dataset or trained model is changed."]}
    (out/"curve-processing.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--data-root", type=Path, default=Path("Z:/p1/data/Ti"))
    args = parser.parse_args()
    result = run(args.root, args.data_root, args.root/"audit/curve-processing")
    print("Completed", len(result["source_sha256"]), "source reads and", len(result["filters"]), "filter comparisons")
