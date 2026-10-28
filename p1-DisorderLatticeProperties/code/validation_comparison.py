"""Dimensional, explicitly provisional p1 / Hsieh Figure 10 comparison.

The digitised CSV is the sole new reference input. Saved FTval values come from
the read-only workbook-reconciliation export, with row identities and defects
preserved. No Excel write, FEA submission, fitted reference correction or
automatic promotion to a validated material toughness is performed.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TOPOLOGIES = ("tri", "kagome", "hex")
DENSITIES = (.13, .16, .20)
LENGTHS = (3., 10.)
METHODS = ("raw", "butterworth_released_order2")
MODULUS_PREFERENCE = ("beam_rve", "literature_scaling")


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_lookup(rows):
    result = {}
    for row in rows:
        if row["stage"] != "initial":
            continue
        key = (row["topology"], float(row["nominal_relative_density"]))
        if key in result or float(row["bar_length_mm"]) != 3:
            raise ValueError("Figure 10 initial reference identity is ambiguous")
        result[key] = row
    if len(result) != 9:
        raise ValueError("Expected nine initial Figure 10 points")
    return result


def add_reference(record, reference):
    key = (record["topology"], record["relative_density"])
    if key not in reference or record["l_mm"] not in LENGTHS:
        record["reference_status"] = "outside_digitised_reference_domain"
        return record
    ref = reference[key]
    factor = math.sqrt(record["l_mm"] / 3.)
    record.update(reference_point_id=ref["point_id"], reference_length_factor=factor,
                  reference_basis="direct_Fig10_3mm" if factor == 1 else
                  "SCALING_INFERENCE_Fig10_3mm_times_sqrt_10_over_3",
                  reference_K_MPa_sqrt_m=float(ref["K_MPa_sqrt_m"]) * factor,
                  reference_graphical_lower=float(ref["K_graphical_lower"]) * factor,
                  reference_graphical_upper=float(ref["K_graphical_upper"]) * factor,
                  reference_status="available")
    value = record.get("KJ_MPa_sqrt_m")
    if record["value_status"] == "available_provisional" and value is not None:
        record["difference_from_reference_percent"] = 100 * (value / record["reference_K_MPa_sqrt_m"] - 1)
        record["difference_reference_graphical_lower_percent"] = 100 * (
            value / record["reference_graphical_upper"] - 1)
        record["difference_reference_graphical_upper_percent"] = 100 * (
            value / record["reference_graphical_lower"] - 1)
    return record


def historical_records(rows, reference):
    output = []
    for row in rows:
        if row["topology"] not in TOPOLOGIES:
            continue
        ki, kj = number(row["saved_KI_MPa_sqrt_m"]), number(row["saved_KJ_MPa_sqrt_m"])
        missing = any(number(row.get(key)) is None for key in
                      ("nnx", "saved_force_N", "saved_displacement_mm",
                       "saved_KI_MPa_sqrt_m", "saved_KJ_MPa_sqrt_m"))
        if row.get("placeholder_KJ", "").lower() == "true":
            status = "placeholder_not_a_result"
        elif missing:
            status = "missing_saved_input"
        elif kj < ki:
            status = "negative_area_saved_KJ_below_KI"
        elif number(row.get("matching_native_cases")) != 1:
            status = "unmatched_historic_case_metadata"
        else:
            status = "available_provisional"
        record = {"source": "historic_FTval", "workbook_row": int(row["workbook_row"]),
                  "case_id": row.get("case_id", ""), "topology": row["topology"],
                  "nnx": number(row.get("nnx")), "l_mm": number(row["l_mm"]),
                  "relative_density": number(row["relative_density"]),
                  "method": "saved_workbook_value", "compliance": "historic_unverified",
                  "modulus": "historic_parent_override_workflow",
                  "KI_MPa_sqrt_m": ki, "KJ_MPa_sqrt_m": kj,
                  "value_status": status,
                  "saved_literal_KJ_preserved": kj,
                  "matching_native_cases": number(row.get("matching_native_cases"))}
        output.append(add_reference(record, reference))
    return output


def native_records(rows, historic, reference):
    # Existing paper selections and independently prepared static controls only.
    # No specimen size or fit window is selected by its agreement with Hsieh.
    selected = {r["case_id"] for r in historic if r.get("case_id")}
    selected.update(r["case_id"] for r in rows if r["fit_fraction"] == "elastic_unit_load")
    output = []
    for case_id in sorted(selected):
        case_rows = [r for r in rows if r["case_id"] == case_id
                     and r["endpoint"] == "last_monitored_intact"]
        if not case_rows:
            continue
        modulus = next((name for name in MODULUS_PREFERENCE
                        if any(r["modulus"] == name for r in case_rows)), None)
        if modulus is None:
            raise ValueError("No explicit permitted modulus control for " + case_id)
        has_static = any(r["fit_fraction"] == "elastic_unit_load" and r["modulus"] == modulus
                         for r in case_rows)
        fit = "elastic_unit_load" if has_static else "0.15"
        for method in METHODS:
            found = [r for r in case_rows if r["method"] == method and r["modulus"] == modulus
                     and r["fit_fraction"] == fit]
            if not found:
                # A missing modal control is not borrowed from another case.
                continue
            if len(found) != 1:
                raise ValueError("Ambiguous native control: " + repr((case_id, method, modulus, fit)))
            row = found[0]
            apl, kj = number(row["Apl_N_mm"]), number(row["KJ_MPa_sqrt_m"])
            status = ("negative_Apl_unresolved" if apl is not None and apl < 0 else
                      "available_provisional" if kj is not None else "unavailable_native_KJ")
            record = {"source": "native_audit", "case_id": case_id,
                      "topology": row["topology"], "l_mm": number(row["l_mm"]),
                      "relative_density": number(row["relative_density"]),
                      "nnx": int(case_id.split("/Fracture-")[-1].split("-")[1]),
                      "method": method, "endpoint": row["endpoint"],
                      "endpoint_time_s": number(row["endpoint_time_s"]),
                      "compliance": "initial_elastic_unit_load" if has_static else "initial_curve_fit_15_percent",
                      "fit_fraction": fit, "modulus": modulus,
                      "modulus_MPa": number(row["modulus_MPa"]),
                      "stiffness_N_mm": number(row["stiffness_N_mm"]),
                      "compliance_source_job": row.get("compliance_source_job", ""),
                      "compliance_source_inp_sha256": row.get("compliance_source_inp_sha256", ""),
                      "KI_MPa_sqrt_m": number(row["KI_MPa_sqrt_m"]),
                      "KJ_MPa_sqrt_m": kj, "Apl_N_mm": apl,
                      "signed_area_KJ_diagnostic_MPa_sqrt_m": number(
                          row.get("KJ_signed_area_diagnostic_MPa_sqrt_m")),
                      "value_status": status,
                      "modulus_scope": "exact_case_beam_RVE" if modulus == "beam_rve" else
                      "explicit_literature_scaling_control_not_accepted_modulus"}
            output.append(add_reference(record, reference))
    return output


def add_old_new_changes(records):
    historic = {r["case_id"]: r for r in records if r["source"] == "historic_FTval"
                and r["case_id"] and r["value_status"] == "available_provisional"}
    for row in records:
        if row["source"] != "native_audit" or row["value_status"] != "available_provisional":
            continue
        old = historic.get(row["case_id"])
        if old is not None and old["KJ_MPa_sqrt_m"] > 0:
            row["historic_workbook_row"] = old["workbook_row"]
            row["historic_KJ_MPa_sqrt_m"] = old["KJ_MPa_sqrt_m"]
            row["change_from_same_historic_case_percent"] = 100 * (
                row["KJ_MPa_sqrt_m"] / old["KJ_MPa_sqrt_m"] - 1)


def coverage(records):
    output = []
    for length in LENGTHS:
        for topology in TOPOLOGIES:
            for rho in DENSITIES:
                matches = [r for r in records if r["topology"] == topology
                           and r["l_mm"] == length and r["relative_density"] == rho]
                old = [r for r in matches if r["source"] == "historic_FTval"]
                native = [r for r in matches if r["source"] == "native_audit"]
                output.append({"topology": topology, "l_mm": length, "relative_density": rho,
                               "historic_status": ";".join(r["value_status"] for r in old) or "absent",
                               "native_status": ";".join(sorted(set(r["value_status"] for r in native))) or
                               "no_exact_selected_case",
                               "native_methods": ";".join(sorted(set(r["method"] for r in native))),
                               "native_moduli": ";".join(sorted(set(r["modulus"] for r in native))),
                               "reference_basis": "direct_3mm" if length == 3 else "sqrt_l_inference_10mm"})
    return output


def plot_style(row):
    if row["source"] == "historic_FTval":
        return "Historic FTval", "#777777", "s", "none", "--"
    static = row["compliance"] == "initial_elastic_unit_load"
    rve = row["modulus"] == "beam_rve"
    if row["method"] == "raw":
        label = "Native raw: static C0 / beam RVE" if static and rve else (
            "Native raw: static C0 / scaling E" if static else "Native raw: 15% fit / scaling E")
        return label, "#2379b5", "o", "#2379b5" if static and rve else "none", "-"
    label = "Causal BW: static C0 / beam RVE" if static and rve else "Causal BW: static C0 / scaling E"
    return label, "#de811a", "D", "#de811a" if rve else "none", "-"


def plot_comparison(records, reference, out, differences=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.8), constrained_layout=True)
    handles, labels = [], []
    for i, length in enumerate(LENGTHS):
        for j, topology in enumerate(TOPOLOGIES):
            ax = axes[i, j]
            factor = math.sqrt(length / 3.)
            refs = [reference[topology, rho] for rho in DENSITIES]
            x = 100*np.array(DENSITIES)
            k = factor*np.array([float(r["K_MPa_sqrt_m"]) for r in refs])
            low = factor*np.array([float(r["K_graphical_lower"]) for r in refs])
            high = factor*np.array([float(r["K_graphical_upper"]) for r in refs])
            if differences:
                ax.axhline(0, color="#222222", lw=1, ls="--")
            else:
                ax.plot(x, k, color="#222222", marker="^", lw=1.5, label="Figure 10 initial KJIC")
                ax.fill_between(x, low, high, color="#222222", alpha=.12)
            panel = [r for r in records if r["topology"] == topology and r["l_mm"] == length
                     and r["relative_density"] in DENSITIES]
            groups = {}
            for row in panel:
                if row["value_status"] == "available_provisional":
                    groups.setdefault(plot_style(row), []).append(row)
            for (label, color, marker, face, line), points in groups.items():
                points.sort(key=lambda r: r["relative_density"])
                xs = np.array([100*r["relative_density"] for r in points])
                if differences:
                    ys = np.array([r["difference_from_reference_percent"] for r in points])
                    err = np.array([[r["difference_from_reference_percent"] -
                                     r["difference_reference_graphical_lower_percent"] for r in points],
                                    [r["difference_reference_graphical_upper_percent"] -
                                     r["difference_from_reference_percent"] for r in points]])
                    ax.errorbar(xs, ys, yerr=err, color=color, marker=marker,
                                mfc=face, mec=color, ls=line, capsize=3, label=label)
                else:
                    ax.plot(xs, [r["KJ_MPa_sqrt_m"] for r in points], color=color,
                            marker=marker, mfc=face, mec=color, ls=line, label=label)
                if label not in labels:
                    labels.append(label)
                    handles.append(Line2D([], [], color=color, marker=marker, mfc=face,
                                          mec=color, ls=line, label=label))
            problems = []
            for rho in DENSITIES:
                subset = [r for r in panel if r["relative_density"] == rho]
                if not any(r["source"] == "native_audit" for r in subset):
                    problems.append("{}%: no exact audited selection".format(round(100*rho)))
                for row in subset:
                    status = row["value_status"]
                    if status in ("missing_saved_input", "placeholder_not_a_result"):
                        problems.append("{}%: saved {}".format(round(100*rho),
                                        "placeholder" if status.startswith("placeholder") else "absent"))
                    elif status in ("negative_area_saved_KJ_below_KI", "negative_Apl_unresolved"):
                        problems.append("{}%: {} negative Apl".format(round(100*rho),
                                        "saved" if row["source"] == "historic_FTval" else "BW"))
                        diagnostic = (row["KJ_MPa_sqrt_m"] if row["source"] == "historic_FTval" else
                                      row.get("signed_area_KJ_diagnostic_MPa_sqrt_m"))
                        if diagnostic is not None and not differences:
                            ax.plot(100*rho, diagnostic, "x", color="#b22222", ms=9, mew=1.5)
            if problems:
                ax.text(.02, .98, "\n".join(dict.fromkeys(problems)), transform=ax.transAxes,
                        va="top", fontsize=7.2, color="#7b2929",
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
            ax.set_title("{} | {} mm - {}".format(
                {"tri": "Triangular", "kagome": "Kagome", "hex": "Hexagonal"}[topology],
                int(length), "DIRECT reference" if length == 3 else "SCALING INFERENCE"), fontsize=10)
            ax.set(xlabel="Nominal relative density (%)", xticks=[13, 16, 20], xlim=(12.4, 20.6))
            if j == 0:
                ax.set_ylabel("Difference from reference (%)" if differences else r"$K_J$ (MPa$\sqrt{\mathrm{m}}$)")
            ax.grid(alpha=.22)
            ax.margins(y=.18 if differences else .12)
    if not differences:
        handles.insert(0, Line2D([], [], color="#222222", marker="^", label="Fig.10 initial + graphical bounds"))
        labels.insert(0, "Fig.10 initial + graphical bounds")
        handles.append(Line2D([], [], color="#b22222", marker="x", ls="none"))
        labels.append("Negative-area diagnostic; not accepted KJ")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(("Signed difference: 100 (p1 / reference - 1)" if differences else
                  "Dimensional comparison: Fig.10 initiation and p1 endpoint controls") +
                 "\n3 mm: direct Fig.10; 10 mm: Fig.10 x sqrt(10/3), not digitised 10 mm data\n"
                 "Native endpoint: last before monitored STATUS deletion; earlier section onset is separate. All KJ provisional.", fontsize=10)
    name = "validation-relative-differences.png" if differences else "validation-dimensional.png"
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def run(root="Z:/p1/sims/Ti/Validation", output=None):
    root = Path(root).resolve()
    out = Path(output).resolve() if output else root / "audit/benchmark"
    out.mkdir(parents=True, exist_ok=True)
    sources = {"reference": root / "audit/benchmark/fig10-digitised.csv",
               "native": root / "audit/calculations/comparison.csv",
               "native_provenance": root / "audit/calculations/provenance.json",
               "historic_FTval_export": root / "audit/calculations/workbook-reconciliation.csv"}
    reference = reference_lookup(read_rows(sources["reference"]))
    historic = historical_records(read_rows(sources["historic_FTval_export"]), reference)
    native = native_records(read_rows(sources["native"]), historic, reference)
    case_provenance = {row["case_id"]: row for row in json.loads(
        sources["native_provenance"].read_text(encoding="utf-8"))}
    for row in native:
        proof = case_provenance.get(row["case_id"], {})
        if row["modulus"] == "beam_rve" and not proof.get("beam_rve_job"):
            raise ValueError("Beam-RVE comparison lacks case-specific provenance: " + row["case_id"])
        row["indexed_source_inp_sha256"] = proof.get("indexed_source_inp_sha256")
        row["indexed_source_odb_sha256"] = proof.get("indexed_source_odb_sha256")
        row["beam_rve_job"] = proof.get("beam_rve_job") if row["modulus"] == "beam_rve" else None
        row["beam_rve_odb_sha256"] = proof.get("beam_rve_odb_sha256") if row["modulus"] == "beam_rve" else None
    records = historic + native
    add_old_new_changes(records)
    write_rows(out / "validation-comparison.csv", records)
    write_rows(out / "validation-coverage.csv", coverage(records))
    plot_comparison(records, reference, out)
    plot_comparison(records, reference, out, differences=True)
    provenance = {"inputs": {name: {"path": str(path), "sha256": sha256(path)}
                             for name, path in sources.items()},
                  "selection": "Existing FTval exact case IDs plus prepared exact static controls; no fit to benchmark",
                  "native_control": "last_monitored_intact; raw and available released order2 Butterworth; static C0 preferred, otherwise explicitly labelled 15% curve-fit sensitivity",
                  "modulus_preference_per_exact_case": list(MODULUS_PREFERENCE),
                  "moduli_in_use": sorted({r["modulus"] for r in native}),
                  "historic_rows_retained": len(historic), "native_controls": len(native),
                  "unmatched_historic_rows": [r["workbook_row"] for r in historic
                                              if r["value_status"] == "unmatched_historic_case_metadata"],
                  "limits": ["No toughness definition is promoted to validated paper truth",
                             "Last-monitored-intact STATUS is later than section/strain onset for the checked hex cases; these are endpoint controls, not certified like-for-like initiation values",
                             "Initial-compliance subtraction can give negative apparent Apl under elastic geometric stiffening",
                             "A negative-area point is diagnostic only and has no relative-error estimate",
                             "10mm references are sqrt(length) scaling inferences, with only digitisation bounds propagated",
                             "Figure10 final KJFC remains available in the canonical reference CSV but is excluded here",
                             "Reference density uses nominal manuscript assignments; printed density axis is inconsistent"]}
    (out / "validation-comparison-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out")
    args = parser.parse_args()
    values = run(args.root, args.out)
    print("Wrote {} labelled historical/native controls".format(len(values)))
