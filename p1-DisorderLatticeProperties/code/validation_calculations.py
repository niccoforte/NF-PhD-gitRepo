"""Compare native C(T) validation calculations without changing source exports.

Run after validation_history_audit.py and validation_modal.py. Units are N, mm,
seconds and MPa; exported K is MPa sqrt(m). Butterworth comparisons use the
recorded sample grid, not the unavailable Explicit integration-increment signal.
No negative plastic area is clipped or published as a valid toughness.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from resources.calculations import calc_FaW, smooth
from resources.lattices import Geometry, effProperties


def filter_force(time, force, cutoff, order=2, zero_phase=False):
    """Uniform-grid Butterworth; restore genuine times, never interpolate STATUS.

    DC startup and causal second-order filtering follow Abaqus defaults. Small
    output-time jitter is interpolated explicitly; gaps >1.5 nominal intervals
    are rejected. A noncausal sensitivity is separately labelled.
    """
    time, force = np.asarray(time, float), np.asarray(force, float)
    if len(time) < 4 or len(time) != len(force):
        raise ValueError("Need at least four paired samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(force)):
        raise ValueError("Missing/nonfinite continuous histories cannot be filtered")
    dt = np.diff(time)
    if np.any(dt <= 0) or np.max(dt) > 1.5 * np.median(dt):
        raise ValueError("Time grid is nonmonotone or contains a missing interval")
    grid = np.linspace(time[0], time[-1], len(time))
    fs = (len(time) - 1) / (time[-1] - time[0])
    if not 0 < cutoff < fs / 2:
        raise ValueError("Cutoff must be below native-history Nyquist frequency")
    uniform_force = np.interp(grid, time, force)
    sos = butter(order, cutoff, fs=fs, output="sos")
    if zero_phase:
        filtered = sosfiltfilt(sos, uniform_force)
    else:
        filtered, _ = sosfilt(sos, uniform_force,
                              zi=sosfilt_zi(sos) * uniform_force[0])
    return np.interp(time, grid, filtered), {
        "fs_Hz": fs, "nyquist_Hz": fs / 2, "cutoff_Hz": cutoff,
        "order": order, "zero_phase": zero_phase,
        "max_time_remap_s": float(np.max(np.abs(grid - time))),
    }


def fit_stiffness(displacement, force, end):
    """Free-intercept early-loading regression, matching the legacy convention."""
    x, y = np.asarray(displacement[:end + 1]), np.asarray(force[:end + 1])
    if len(x) < 3 or np.ptp(x) == 0:
        return {"stiffness_N_mm": None, "intercept_N": None, "r_squared": None}
    slope, intercept = np.polyfit(x, y, 1)
    resid = np.sum((y - slope * x - intercept) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    return {"stiffness_N_mm": float(slope), "intercept_N": float(intercept),
            "r_squared": float(1 - resid / total) if total else None}


def fracture_metrics(displacement, force, index, geom, modulus, stiffness):
    """ASTM C(T) stationary-crack work decomposition at an explicit sample.

    index is never inferred from an arbitrary global force maximum. A negative
    Apl remains diagnostic and yields an unavailable KJ, rather than KJ<KI.
    """
    d, f = np.asarray(displacement, float), np.asarray(force, float)
    if not np.all(np.isfinite(d[:index + 1])) or not np.all(np.isfinite(f[:index + 1])):
        raise ValueError("Nonfinite history in the integration interval")
    if not 0 < index < len(d) or np.any(np.diff(d[:index + 1]) < 0):
        raise ValueError("Endpoint must bound a monotone loading history")
    if not np.isfinite(stiffness) or stiffness <= 0 or modulus <= 0:
        raise ValueError("Positive stiffness and conversion modulus required")
    p = float(f[index])
    work = float(trapezoid(f[:index + 1], d[:index + 1]))
    elastic_work = p * p / (2 * stiffness)
    apl = work - elastic_work
    a, b, w = float(geom.ai[0]), float(geom.B), float(geom.W)
    ki = p * calc_FaW(a, w) / (b * np.sqrt(w))
    jel = ki * ki / modulus
    jpl = (2 + .522 * (w - a) / w) * apl / (b * (w - a))
    total = jel + jpl
    return {"sample_index": int(index), "displacement_mm": float(d[index]),
            "force_N": p, "work_N_mm": work, "elastic_work_N_mm": elastic_work,
            "Apl_N_mm": apl, "stiffness_N_mm": float(stiffness),
            "modulus_MPa": float(modulus), "Jel_N_per_mm": float(jel),
            "Jpl_N_per_mm": float(jpl), "KI_MPa_sqrt_m": float(ki / np.sqrt(1000)),
            "KJ_signed_area_diagnostic_MPa_sqrt_m": float(np.sqrt(modulus * total) / np.sqrt(1000))
            if total >= 0 else None,
            "KJ_MPa_sqrt_m": float(np.sqrt(modulus * total) / np.sqrt(1000))
            if apl >= 0 and total >= 0 else None,
            "valid_plastic_area": bool(apl >= 0)}


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def native_path(root, case):
    odb = Path(case["odb"])
    return root / "audit/history" / odb.parent / odb.stem


def case_comparison(root, case, modal, output):
    directory = native_path(root, case)
    rows = read_table(directory / "native_curve.csv")
    t = np.array([float(row["time_s"]) for row in rows])
    d = np.array([float(row["U2_mm"]) for row in rows])
    f = np.array([float(row["RF2_N"]) for row in rows])
    zeros = np.array([float(row["observed_STATUS_zero"]) for row in rows])
    if np.any(~np.isfinite(zeros)) or np.any(zeros < 0) or np.any(zeros != np.floor(zeros)):
        raise ValueError("Invalid observed deletion counts: " + case["case_id"])
    first = np.flatnonzero(zeros > 0)
    if not len(first):
        return [], {"case_id": case["case_id"], "limit": "No monitored deletion observed"}
    deletion = int(first[0])
    if deletion < 3:
        raise ValueError("Insufficient intact loading samples: " + case["case_id"])
    # Preserve the exact old centred-filter window as a labelled control. Its
    # future dependence differs from causal Butterworth and must remain visible.
    f[0] = 0
    full_moving_average = np.array(smooth(smooth(f)))[:deletion + 1]
    legacy_curve = np.loadtxt(root / case["curve"], delimiter=",", usecols=(0, 1))
    legacy_curve[0] = 0
    legacy_force = np.array(smooth(smooth(legacy_curve[:, 1])))[:deletion + 1]
    legacy_d = legacy_curve[:deletion + 1, 0]
    # Butterworth comparison ends at observed deletion. A separately labelled
    # centred-prefix control measures sensitivity to the available endpoint.
    t, d, f = t[:deletion + 1], d[:deletion + 1], f[:deletion + 1]
    geom = Geometry(case["topology"], float(case["l_mm"]), int(case["nnx"]),
                    float(case["nominal_relative_density"]))
    geom.FTcalc()
    affine = effProperties(case["topology"], geom, E_s=123000, mode="stiff")[2]
    bending = effProperties(case["topology"], geom, E_s=123000, mode="scaling")[2]
    moduli = {"current_affine": affine, "literature_scaling": bending,
              "legacy_parent_override": 123000.}
    variants = {"raw": f, "moving_average_twice": full_moving_average,
                "moving_average_twice_prefix_sensitivity": np.array(smooth(smooth(f)))}
    displacement_variants = {}
    if len(legacy_force) == len(f):
        variants["legacy_export_moving_average_twice"] = legacy_force
    controls = []
    for mode in modal:
        if mode["case_id"] != case["case_id"] or not mode.get("cutoff_2fn_Hz"):
            continue
        for order in (2, 4):
            name = "butterworth_{}_order{}".format(mode["control"], order)
            try:
                variants[name], info = filter_force(t, f, mode["cutoff_2fn_Hz"], order)
                controls.append(dict(method=name, **info))
                paired_name = name + "_paired_U_RF"
                paired_d, paired_info = filter_force(t, d, mode["cutoff_2fn_Hz"], order)
                variants[paired_name] = variants[name]
                displacement_variants[paired_name] = paired_d
                controls.append(dict(method=paired_name, filtered_components="U2 and RF2", **paired_info))
                # Zero phase is a sensitivity, not a real-time reproduction.
                if order == 2:
                    name_z = name + "_zero_phase_through_deletion_sensitivity"
                    variants[name_z], info_z = filter_force(t, f, mode["cutoff_2fn_Hz"], order, True)
                    controls.append(dict(method=name_z, **info_z))
                    intact_name = name + "_zero_phase_intact_prefix_sensitivity"
                    variants[intact_name], intact_info = filter_force(t[:-1], f[:-1],
                                                                      mode["cutoff_2fn_Hz"], order, True)
                    controls.append(dict(method=intact_name, **intact_info))
            except ValueError as error:
                controls.append({"method": name, "unavailable": str(error),
                                 "cutoff_Hz": mode["cutoff_2fn_Hz"]})
    static = [mode for mode in modal if mode["case_id"] == case["case_id"]
              and mode.get("static_stiffness_N_mm")]
    answers = []
    stiffness_rows = []
    for name, force in variants.items():
        method_d = displacement_variants.get(name, legacy_d if name == "legacy_export_moving_average_twice" else d)
        if not np.all(np.isfinite(force)) or not np.all(np.isfinite(method_d)):
            controls.append({"method": name, "unavailable": "Nonfinite values in the event prefix"})
            continue
        endpoints = {"last_monitored_intact": deletion - 1,
                     "first_monitored_deletion": deletion,
                     "method_pre_deletion_peak": int(np.argmax(force[:deletion]))}
        for endpoint, index in endpoints.items():
            if index >= len(force):
                continue
            for fraction in (.05, .10, .15, .20, .25, .30):
                end = max(2, int(fraction * index))
                fit = fit_stiffness(method_d, force, end)
                stiffness_rows.append({"method": name, "endpoint": endpoint,
                                       "fit_fraction": fraction, "fit_end_index": end, **fit})
                slope = fit["stiffness_N_mm"]
                if slope is None or slope <= 0:
                    continue
                for modulus_name, modulus in moduli.items():
                    answers.append({"case_id": case["case_id"], "topology": case["topology"],
                                    "l_mm": float(case["l_mm"]),
                                    "relative_density": float(case["nominal_relative_density"]),
                                    "method": name, "endpoint": endpoint,
                                    "fit_fraction": fraction, "modulus": modulus_name,
                                    "endpoint_time_s": float(t[index]), **fit,
                                    **fracture_metrics(method_d, force, index, geom, modulus, slope)})
            for control in static:
                for modulus_name, modulus in moduli.items():
                    answers.append({"case_id": case["case_id"], "topology": case["topology"],
                                    "l_mm": float(case["l_mm"]),
                                    "relative_density": float(case["nominal_relative_density"]),
                                    "method": name, "endpoint": endpoint,
                                    "fit_fraction": "elastic_unit_load", "modulus": modulus_name,
                                    "endpoint_time_s": float(t[index]),
                                    "compliance_source_job": control["job"],
                                    "compliance_source_inp_sha256": control["source_inp_sha256"],
                                    **fracture_metrics(method_d, force, index, geom, modulus,
                                                       control["static_stiffness_N_mm"])})
    safe = "{}_{}_rho{}_nx{}".format(case["topology"], case["l_mm"],
                                    case["nominal_relative_density"], case["nnx"])
    continuous = [{"time_s": float(t[i]), "displacement_mm": float(d[i]),
                   "legacy_displacement_mm": float(legacy_d[i]) if i < len(legacy_d) else None,
                   **{name + "_displacement_mm": float(values[i])
                      for name, values in displacement_variants.items()},
                   **{name: float(values[i]) if i < len(values) else None
                      for name, values in variants.items()}}
                  for i in range(len(t))]
    write_table(output / (safe + "-filtered.csv"), continuous)
    write_table(output / (safe + "-stiffness.csv"), stiffness_rows)
    plot_comparison(output / (safe + ".png"), safe, d, legacy_d, variants, deletion, stiffness_rows,
                    displacement_variants)
    return answers, {"case_id": case["case_id"], "source_curve": str(directory / "native_curve.csv"),
                     "native_curve_sha256": hashlib.sha256((directory / "native_curve.csv").read_bytes()).hexdigest(),
                     "original_curve_sha256": hashlib.sha256((root / case["curve"]).read_bytes()).hexdigest(),
                     "indexed_source_odb_sha256": case["odb_sha256"],
                     "indexed_source_inp_sha256": case["inp_sha256"],
                     "samples": len(rows), "monitored_deletion_index": deletion,
                     "initiation_time_bracket_s": [float(t[deletion - 1]), float(t[deletion])],
                     "geometry_mm": {"W": geom.W, "B": geom.B, "a": geom.ai[0]},
                     "filters": controls,
                     "legacy_displacement_matches_native_prefix": bool(
                         len(legacy_d) == len(d) and np.allclose(legacy_d, d, rtol=1e-7, atol=1e-5)),
                     "legacy_force_uses_original_displacement": True,
                     "moduli_are_sensitivity_controls": True,
                     "limits": "Monitored deletion is not first section damage; modulus selection awaits lattice review. Offline filtering cannot remove pre-existing aliasing."}


def plot_comparison(path, title, displacement, legacy_displacement, variants, deletion, stiffness_rows,
                    displacement_variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    labels = {"raw": "Raw native", "moving_average_twice": "Moving average, native",
              "legacy_export_moving_average_twice": "Moving average, old CSV",
              "butterworth_released_order2": "Causal BW2, force only",
              "butterworth_released_order2_paired_U_RF": "Causal BW2, force + displacement"}
    for name, force in variants.items():
        if name not in labels:
            continue
        method_d = displacement_variants.get(name, legacy_displacement if name == "legacy_export_moving_average_twice" else displacement)
        axes[0].plot(method_d[:len(force)], force / 1000, label=labels[name], lw=1.2)
    axes[0].axvspan(displacement[deletion - 1], displacement[deletion], color="red", alpha=.15,
                   label="monitored deletion bracket")
    axes[0].set(xlabel="Loading point displacement (mm)", ylabel="Force (kN)", title=title)
    axes[0].legend(fontsize=7)
    for name in variants:
        rows = [row for row in stiffness_rows if row["method"] == name
                and row["endpoint"] == "last_monitored_intact"]
        if name not in labels:
            continue
        axes[1].plot([100 * row["fit_fraction"] for row in rows],
                     [row["stiffness_N_mm"] / 1000 for row in rows], marker="o", label=labels[name])
    axes[1].set(xlabel="Initial fit window (% of endpoint index)", ylabel="Initial stiffness (kN/mm)",
                title="Sensitivity of the early fit")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def reconcile_workbook(workbook, root, cases, results, output):
    """Read-only saved-value join; preserve original cell contents and missingness."""
    sheet = pd.read_excel(workbook, sheet_name="FTval", header=None)
    joined = []
    for row_number in range(43, 75):
        row = sheet.iloc[row_number - 1]
        if row.iloc[0] not in ("tri", "kagome", "hex"):
            continue
        def number(column, factor=1):
            value = row.iloc[column - 1]
            return float(value) * factor if pd.notna(value) else None
        record = {"workbook_row": row_number, "topology": row.iloc[0],
                  "nnx": number(2), "relative_density": number(3), "l_mm": number(9, 1000),
                  "saved_force_N": number(15), "saved_displacement_mm": number(16, 1000),
                  "saved_KI_MPa_sqrt_m": number(17), "saved_KJ_MPa_sqrt_m": number(18)}
        record["placeholder_KJ"] = record["saved_KJ_MPa_sqrt_m"] == 1 and record["saved_force_N"] is None
        matches = [case for case in cases if case["topology"] == record["topology"]
                   and float(case["l_mm"]) == record["l_mm"]
                   and float(case["nominal_relative_density"]) == record["relative_density"]
                   and int(case["nnx"]) == record["nnx"]]
        record["matching_native_cases"] = len(matches)
        if len(matches) == 1:
            case = matches[0]
            record["case_id"] = case["case_id"]
            native = read_table(native_path(root, case) / "native_curve.csv")
            if record["saved_displacement_mm"] is not None:
                dd = np.array([float(r["U2_mm"]) for r in native])
                error = np.abs(dd - record["saved_displacement_mm"])
                index = int(np.nanargmin(error))
                record["nearest_native_index"] = index
                record["saved_displacement_error_mm"] = float(error[index])
                record["native_raw_force_at_saved_displacement_N"] = float(native[index]["RF2_N"])
            controls = [value for value in results if value["case_id"] == case["case_id"]]
            for method, endpoint, fit, modulus, prefix in (
                ("legacy_export_moving_average_twice", "method_pre_deletion_peak", .15,
                 "legacy_parent_override", "current_legacy_replay"),
                ("butterworth_released_order2", "last_monitored_intact", "elastic_unit_load",
                 "literature_scaling", "native_butterworth_static_scaling_control"),
            ):
                found = [value for value in controls if value["method"] == method
                         and value["endpoint"] == endpoint and value["fit_fraction"] == fit
                         and value["modulus"] == modulus]
                if len(found) == 1:
                    for key in ("force_N", "KI_MPa_sqrt_m", "KJ_MPa_sqrt_m", "Apl_N_mm"):
                        record[prefix + "_" + key] = found[0][key]
        joined.append(record)
    write_table(output / "workbook-reconciliation.csv", joined)


def run(root, output, workbook=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = root / "audit/modal/summary.json"
    modal = json.loads(modes.read_text(encoding="utf-8")) if modes.exists() else []
    static = root / "audit/modal/static-summary.json"
    if static.exists():
        modal += json.loads(static.read_text(encoding="utf-8"))
    cases = [case for case in read_table(root / "case-index.csv")
             if case["campaign"] == "Density campaign" and case["mode"] == "Fracture"
             and case["topology"] in ("tri", "kagome", "hex")
             and (native_path(root, case) / "native_curve.csv").exists()]
    results, provenance = [], []
    for case in cases:
        values, record = case_comparison(root, case, modal, output)
        results.extend(values)
        provenance.append(record)
    write_table(output / "comparison.csv", results)
    if workbook:
        reconcile_workbook(workbook, root, cases, results, output)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    sources = [Path(__file__), Path(sys.modules[Geometry.__module__].__file__),
               Path(sys.modules[calc_FaW.__module__].__file__)]
    run_info = {"calculation_code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in sources},
                "case_count": len(cases), "sensitivity_rows": len(results),
                "workbook_read_only": str(workbook) if workbook else None,
                "source_exports_modified": False, "negative_area_clipped": False}
    (output / "calculation-run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("Compared {} native cases; {} sensitivity rows".format(len(cases), len(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out", default="Z:/p1/sims/Ti/Validation/audit/calculations")
    parser.add_argument("--workbook", help="Read-only FTval saved-value reconciliation; explicit canonical XLSX path")
    args = parser.parse_args()
    run(args.root, args.out, args.workbook)
