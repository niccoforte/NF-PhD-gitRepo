"""Provisional p1 / Hsieh Figures 8a, 8b and 10 comparisons.

The digitised CSV is the sole new reference input. Saved FTval values come from
the read-only workbook-reconciliation export, with row identities and defects
preserved. No Excel write, FEA submission, fitted reference correction or
automatic promotion to a validated material toughness is performed.
"""

import argparse
import csv
import hashlib
import io
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
        if row["source"] not in ("native_audit", "recalculated_rule") or row["value_status"] != "available_provisional":
            continue
        old = historic.get(row["case_id"])
        if old is not None and old["KJ_MPa_sqrt_m"] > 0:
            row["historic_workbook_row"] = old["workbook_row"]
            row["historic_KJ_MPa_sqrt_m"] = old["KJ_MPa_sqrt_m"]
            row["change_from_same_historic_case_percent"] = 100 * (
                row["KJ_MPa_sqrt_m"] / old["KJ_MPa_sqrt_m"] - 1)


def recalculated_records(rows, reference):
    """Adapter for the explicitly selected rule; never upgrade a diagnostic."""
    required = {"case_id", "topology", "l_mm", "relative_density", "endpoint", "method",
                "KI_MPa_sqrt_m", "KJ_MPa_sqrt_m", "value_status"}
    output = []
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError("Recalculated-result columns missing: " + repr(sorted(missing)))
        kj, ki = number(row["KJ_MPa_sqrt_m"]), number(row["KI_MPa_sqrt_m"])
        upstream = row["value_status"]
        status = ("available_provisional" if upstream.startswith("available") and kj is not None else
                  "diagnostic_only" if upstream.startswith("diagnostic") else "unavailable_recalculation")
        rec = dict(row)
        for key in list(rec):
            if key.startswith(("difference_", "reference_", "change_from_", "historic_")):
                rec.pop(key)
        rec.update(source="recalculated_rule", l_mm=number(row["l_mm"]),
                   relative_density=number(row["relative_density"]), nnx=number(row.get("nnx")),
                   KI_MPa_sqrt_m=ki, KJ_MPa_sqrt_m=kj,
                   upstream_value_status=upstream, value_status=status,
                   modulus=row.get("modulus_source", "explicit_recalculated_rule"),
                   compliance=row.get("work_partition", "explicit_recalculated_rule"),
                   signed_area_KJ_diagnostic_MPa_sqrt_m=number(
                       row.get("KJ_signed_area_diagnostic_MPa_sqrt_m", row.get("KJ_MPa_sqrt_m"))))
        output.append(add_reference(rec, reference))
    return output


def criterion_controls(rows, native, reference):
    """Use interim native criterion brackets only until a selected-results CSV exists."""
    metadata = {r["case_id"]: r for r in native}
    prepared = []
    for row in rows:
        if row["endpoint"] not in ("last_before_section_damage_bracket", "first_after_section_damage_bracket"):
            continue
        meta = metadata.get(row["case_id"])
        if meta is None:
            continue
        kj = number(row.get("KJ_MPa_sqrt_m"))
        prepared.append({**meta, **row,
                         "endpoint": "criterion_last_intact" if row["endpoint"].startswith("last_") else "criterion_first_crossed",
                         "topology": meta["topology"], "l_mm": meta["l_mm"],
                         "relative_density": meta["relative_density"], "nnx": meta["nnx"],
                         "work_partition": "initial_triangle_criterion_control_unresolved",
                         "value_status": "diagnostic_interim_criterion_control",
                         "quality_note": "Section-damage bracket, not a completed final-rule calculation",
                         "KJ_MPa_sqrt_m": kj})
    records = recalculated_records(prepared, reference)
    for row in records:
        row["source"] = "interim_criterion_control"
    return records


def coverage(records):
    output = []
    for length in LENGTHS:
        for topology in TOPOLOGIES:
            for rho in DENSITIES:
                matches = [r for r in records if r["topology"] == topology
                           and r["l_mm"] == length and r["relative_density"] == rho]
                old = [r for r in matches if r["source"] == "historic_FTval"]
                native = [r for r in matches if r["source"] == "native_audit"]
                criterion = [r for r in matches if r["source"] in ("recalculated_rule", "interim_criterion_control")]
                output.append({"topology": topology, "l_mm": length, "relative_density": rho,
                               "historic_status": ";".join(r["value_status"] for r in old) or "absent",
                               "native_status": ";".join(sorted(set(r["value_status"] for r in native))) or
                               "no_exact_selected_case",
                               "native_methods": ";".join(sorted(set(r["method"] for r in native))),
                               "native_moduli": ";".join(sorted(set(r["modulus"] for r in native))),
                               "criterion_status": ";".join(sorted(set(r["value_status"] for r in criterion))) or "not_yet_calculated",
                               "reference_basis": "direct_3mm" if length == 3 else "sqrt_l_inference_10mm"})
    return output


def selected_method_style(method):
    """Identify the work definition and loading path without ranking agreement."""
    lower = method.lower()
    if "hsieh_initial_triangle" in lower:
        return (("Damage-off QS initial triangle", "#73509e", "P") if lower.endswith("_qs") else
                ("Native Hsieh initial triangle", "#1d5b92", "s"))
    if "native_energy_eta" in lower:
        return "Native (IE-SE) eta work estimate", "#c88128", "o"
    if lower == "qs_measured_recovery_eta" or any(term in lower for term in ("qs", "matched", "unload")):
        return "Damage-off QS recovery eta estimate", "#278752", "*"
    return method.replace("_", " "), "#555555", "D"


def negative_area(row):
    area = number(row.get("Apl_Nmm", row.get("Apl_N_mm")))
    return area is not None and area < 0


def plot_style(row):
    if row["source"] in ("recalculated_rule", "interim_criterion_control"):
        crossed = row["endpoint"] == "criterion_first_crossed"
        endpoint_label = ("first-crossed sample" if crossed else
                          "verified precriterion peak" if row["endpoint"] == "QS_verified_precriterion_peak" else
                          "lower recorded sample")
        prefix, color, marker = selected_method_style(row["method"])
        if row["source"] == "interim_criterion_control":
            prefix = "Interim initial-triangle control"
        diagnostic = row["value_status"] != "available_provisional"
        marker = "x" if diagnostic else marker
        if negative_area(row):
            prefix += " (negative Apl)"
        return (prefix + ": " + endpoint_label,
                color, marker, "none" if crossed or diagnostic else color, "none")
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


def selected_method_note(records):
    methods = {row["method"] for row in records}
    if any("hsieh_initial_triangle" in method.lower() for method in methods):
        return ("Initial triangles follow Hsieh's subtraction convention; energy/recovery rows are separate eta-work estimates.\n"
                "Damage-off QS is a separate counterfactual state; no certified ASTM J claim.")
    if any("native_energy_eta" in method for method in methods):
        return "Native (IE-SE) eta estimates; nonlinear unloading checks continue. Conditional estimates, not certified ASTM J."
    return "Selected work-partition estimates retain their recorded method limits; no certified ASTM J claim."


def plot_comparison(records, reference, out, differences=False, principal=False):
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
                if not any(r["source"] in ("native_audit", "recalculated_rule") for r in subset):
                    problems.append("{}%: {}".format(round(100*rho),
                                    "No corrected result yet" if principal else "no exact current calculation"))
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
                    elif row["source"] in ("recalculated_rule", "interim_criterion_control") and status != "available_provisional":
                        problems.append("{}%: {}".format(round(100*rho),
                                        "negative initial-triangle area" if negative_area(row) else
                                        "diagnostic only" if status == "diagnostic_only" else "No corrected result yet"))
                        if not differences and row.get("signed_area_KJ_diagnostic_MPa_sqrt_m") is not None:
                            label, color, marker, face, line = plot_style(row)
                            ax.plot(100*rho, row["signed_area_KJ_diagnostic_MPa_sqrt_m"], marker=marker,
                                    color=color, ls="none", ms=7)
                            if label not in labels:
                                labels.append(label)
                                handles.append(Line2D([], [], color=color, marker=marker, ls="none"))
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
        if any(row["value_status"] != "available_provisional" and
               row.get("signed_area_KJ_diagnostic_MPa_sqrt_m") is not None for row in records):
            handles.append(Line2D([], [], color="#b22222", marker="x", ls="none"))
            labels.append("Negative-area diagnostic; not accepted KJ")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(("Signed difference: 100 (p1 / reference - 1)" if differences else
                  "Dimensional comparison: Fig.10 initiation and p1 " +
                  ("selected recalculations" if principal else "historical/native controls")) +
                 "\n3 mm: direct Fig.10; 10 mm: Fig.10 x sqrt(10/3), not digitised 10 mm data\n"
                 + (selected_method_note(records) + "\nLower/crossed points are recorded samples, not continuous bounds."
                    if principal else "STATUS endpoint controls and criterion samples are distinct; all KJ retain their recorded quality limits."), fontsize=10)
    name = "validation-relative-differences" if differences else "validation-dimensional"
    name += ".png" if principal else "-controls.png"
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def normalised_records(records, fig8, material):
    output = []
    uts, yield_stress = material["parent_UTS_workbook_MPa"], material["parent_yield_MPa"]
    for row in records:
        length, rho = row["l_mm"], row["relative_density"]
        if length not in LENGTHS or rho not in DENSITIES:
            continue
        for figure in ("8a", "8b"):
            if figure == "8b" and rho != .2:
                continue
            refs = [r for r in fig8 if r["figure"] == figure and r["topology"] == row["topology"]
                    and float(r["relative_density"]) == rho and float(r["l_mm"]) == length]
            if len(refs) != 1:
                raise ValueError("Ambiguous preserved Fig8 coordinate")
            factor = math.sqrt(length/1000.) if figure == "8a" else 1.
            kj = row["KJ_MPa_sqrt_m"]
            valid = row["value_status"] == "available_provisional" and kj is not None
            q = kj / (uts*factor) if valid else None
            qyield = kj / (yield_stress*factor) if valid else None
            record = {**row, "source_figure": figure,
                      "source_literal_ordinate": float(refs[0]["literal_ordinate"]),
                      "source_literal_cell": refs[0]["source_cell"],
                      "normalisation_parent_UTS_MPa": uts,
                      "normalisation_parent_yield_MPa_sensitivity": yield_stress,
                      "p1_ordinate_parent_UTS": q, "p1_ordinate_parent_yield_sensitivity": qyield,
                      "source_ordinate_altered": False,
                      "difference_from_literal_UTS_convention_percent":
                          100*(q/float(refs[0]["literal_ordinate"])-1) if valid else None,
                      "normalisation_limit": "Source sigma_TS numeric is ambiguous; yield-convention branch is sensitivity only"}
            output.append(record)
    return output


def plot_fig8(records, fig8, material, out, panel, principal=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2 if panel == "8a" else 1, 3,
                             figsize=(13.5, 8.2 if panel == "8a" else 5.7),
                             squeeze=False, constrained_layout=True)
    legend = {"Hsieh: original literal ordinate": Line2D([], [], color="#222222", marker="^", lw=1.4)}
    uts = float(material["parent_UTS_workbook_MPa"])
    for i, length in enumerate(LENGTHS if panel == "8a" else (None,)):
        for j, topology in enumerate(TOPOLOGIES):
            ax = axes[i, j]
            original = [r for r in fig8 if r["figure"] == panel and r["topology"] == topology
                        and (panel == "8b" or float(r["l_mm"]) == length)]
            original.sort(key=lambda r: float(r["abscissa_value"]))
            xs = [100*float(r["abscissa_value"]) if panel == "8a" else float(r["abscissa_value"]) for r in original]
            ax.plot(xs, [float(r["literal_ordinate"]) for r in original], color="#222222", marker="^", lw=1.4)
            selected = [r for r in records if r["topology"] == topology
                        and r["relative_density"] in DENSITIES and r["l_mm"] in LENGTHS
                        and (r["l_mm"] == length if panel == "8a" else r["relative_density"] == .2)]
            groups = {}
            unavailable = []
            for row in selected:
                if row["value_status"] == "available_provisional":
                    groups.setdefault(plot_style(row), []).append(row)
                else:
                    unavailable.append(row)
            for style, points in groups.items():
                label, color, marker, face, line = style
                points.sort(key=lambda r: (r["relative_density"] if panel == "8a" else r["l_mm"]))
                x = [100*r["relative_density"] if panel == "8a" else math.sqrt(r["l_mm"]) for r in points]
                y = [r["KJ_MPa_sqrt_m"]/(uts*(math.sqrt(r["l_mm"]/1000.) if panel == "8a" else 1)) for r in points]
                # Multiple specimen sizes at the same density are separate points.
                if len(set(x)) != len(x):
                    line = "none"
                ax.plot(x, y, color=color, marker=marker, mfc=face, mec=color,
                        ls=line, ms=8 if marker == "*" else 5, alpha=.85)
                legend.setdefault(label, Line2D([], [], color=color, marker=marker,
                                              mfc=face, mec=color, ls=line))
            diagnostics = [r for r in unavailable if r.get("signed_area_KJ_diagnostic_MPa_sqrt_m") is not None
                           and r["source"] in ("recalculated_rule", "interim_criterion_control")]
            for row in diagnostics:
                x = 100*row["relative_density"] if panel == "8a" else math.sqrt(row["l_mm"])
                factor = math.sqrt(row["l_mm"]/1000.) if panel == "8a" else 1
                value = row["signed_area_KJ_diagnostic_MPa_sqrt_m"] / (uts*factor)
                if value > 0:
                    label, color, marker, face, line = plot_style(row)
                    ax.plot(x, value, marker=marker, color=color, ls="none", ms=7)
                    legend.setdefault(label, Line2D([], [], color=color, marker=marker, ls="none"))
            if unavailable or principal:
                locations = {}
                for row in unavailable:
                    key = ("Negative initial-triangle area" if negative_area(row) else
                           "Diagnostic only" if row["value_status"] == "diagnostic_only" else
                           "No corrected result yet" if principal else "Saved/control unavailable")
                    value = "{}%".format(round(100*row["relative_density"])) if panel == "8a" else "{}mm".format(int(row["l_mm"]))
                    locations.setdefault(key,set()).add(value)
                if principal:
                    expected = DENSITIES if panel == "8a" else LENGTHS
                    present = {r["relative_density"] if panel == "8a" else r["l_mm"] for r in selected}
                    missing = set(expected) - present
                    if missing:
                        locations.setdefault("No corrected result yet", set()).update(
                            "{}%".format(round(100*v)) if panel == "8a" else "{}mm".format(int(v)) for v in missing)
                ax.text(.02, .98, "\n".join(key+": "+", ".join(sorted(values))
                                               for key,values in locations.items()),
                        transform=ax.transAxes, va="top", fontsize=7.2, color="#7b2929",
                        bbox={"facecolor":"white","alpha":.8,"edgecolor":"none","pad":1})
            ax.set_yscale("log")
            ax.grid(alpha=.2, which="both")
            ax.set_title("{}{}".format({"tri":"Triangular", "kagome":"Kagome", "hex":"Hexagonal"}[topology],
                                      " | {} mm".format(int(length)) if panel == "8a" else " | rho = 20%"))
            ax.set_xlabel("Nominal relative density (%)" if panel == "8a" else r"$\sqrt{\ell}$ (mm$^{1/2}$)")
            if panel == "8a":
                ax.set_xscale("log")
                ax.set(xticks=[13,16,20], xticklabels=["13","16","20"], xlim=(12.4,20.6))
                from matplotlib.ticker import NullFormatter
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_xticks([math.sqrt(3),math.sqrt(10)], ["sqrt(3)","sqrt(10)"])
            if j == 0:
                ax.set_ylabel(r"$K_J/(\sigma_{TS}\sqrt{\ell})$" if panel == "8a" else r"$K_J/\sigma_{TS}$ (m$^{1/2}$)")
    fig.legend(list(legend.values()), list(legend), loc="outside lower center", ncol=3, fontsize=8)
    note = ("Nine original Fig.8a ordinates are repeated in the two length panels; no source values were corrected." if panel == "8a" else
            "Original Fig.8b ordinates retained at nominal 3/10mm assignments; sqrt(mm) is used despite the source sqrt(m) label.")
    fig.suptitle("Figure {} reconstruction: Hsieh and p1 {}\n".format(panel,
                 "selected recalculations" if principal else "historical/native controls") + note +
                 "\np1 uses parent UTS = {:.3f} MPa; source sigma_TS remains ambiguous. Recorded endpoints are not continuous bounds.".format(uts)
                 + ("\n" + selected_method_note(records) if principal else ""), fontsize=10)
    fig.savefig(out / ("figure{}-validation{}.png".format(panel, "" if principal else "-controls")), dpi=210)
    plt.close(fig)


def plot_figure10_density(records, reference_rows, out):
    """Reconstruct source K-density axes without altering literal digitisation.

    The manuscript's nominal density assignment is used for physical density;
    printed source x coordinates remain separate, unchanged CSV fields. A 10mm
    p1 result projected to 3mm is a scaling prediction, never direct 3mm data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter

    colors = {"tri": "#b8443e", "kagome": "#327aa5", "hex": "#4c8c50"}
    names = {"tri": "Triangular", "kagome": "Kagome", "hex": "Hexagonal"}
    fig, ax = plt.subplots(figsize=(11.2, 7.4), constrained_layout=True)
    output = []
    for topology in TOPOLOGIES:
        for stage in ("initial", "final"):
            points = sorted((r for r in reference_rows if r["topology"] == topology
                             and r["stage"] == stage), key=lambda r: float(r["nominal_relative_density"]))
            x = np.array([float(r["nominal_relative_density"])*4500 for r in points])
            y = np.array([float(r["K_MPa_sqrt_m"]) for r in points])
            low = np.array([float(r["K_graphical_lower"]) for r in points])
            high = np.array([float(r["K_graphical_upper"]) for r in points])
            ax.errorbar(x, y, yerr=[y-low, high-y], color=colors[topology],
                        marker="^" if stage == "initial" else "v", ls="-" if stage == "initial" else "--",
                        mfc=colors[topology] if stage == "initial" else "white", capsize=3, lw=1.2, ms=6)
            for point in points:
                output.append({**point, "source": "Hsieh_Fig10", "plot_density_kg_m3": float(point["nominal_relative_density"])*4500,
                               "plot_density_basis": "MANUSCRIPT_INTERPRETATION_nominal_rho_times_4500_not_literal_printed_x",
                               "plot_K_MPa_sqrt_m": float(point["K_MPa_sqrt_m"]),
                               "length_basis": "published_3mm", "source_ordinate_altered": False})
    method_handles = {}
    for row in records:
        if row["l_mm"] not in LENGTHS or row["relative_density"] not in DENSITIES:
            continue
        length = row["l_mm"]
        factor = math.sqrt(length/3.)
        value = row.get("KJ_MPa_sqrt_m")
        signed = row.get("signed_area_KJ_diagnostic_MPa_sqrt_m")
        item = {**row, "stage": "initial_estimate", "quantity": "initial_KJ_estimate",
                "plot_density_kg_m3": row["relative_density"]*4430,
                "plot_density_basis": "p1_nominal_rho_times_source_parent_density_4430",
                "plot_K_MPa_sqrt_m": value/factor if value is not None else None,
                "plot_signed_area_diagnostic_MPa_sqrt_m": signed/factor if signed is not None else None,
                "length_basis": "direct_3mm_simulation" if length == 3 else "PREDICTION_10mm_K_divided_by_sqrt_10_over_3",
                "length_scale_factor_applied": 1/factor}
        output.append(item)
        family, _, marker = selected_method_style(row["method"])
        if row["value_status"] != "available_provisional" or value is None:
            if signed is not None and signed > 0:
                ax.plot(item["plot_density_kg_m3"], signed/factor, marker="x", color="#a12a2a", ls="none", ms=7)
                label = family + ": invalid area diagnostic"
                method_handles.setdefault(label, Line2D([], [], color="#a12a2a", marker="x", ls="none"))
            continue
        if row["method"] == "native_energy_eta" and length == 10:
            marker = "D"
        face = colors[row["topology"]] if length == 3 else "none"
        ax.plot(item["plot_density_kg_m3"], item["plot_K_MPa_sqrt_m"], color=colors[row["topology"]],
                marker=marker, mfc=face, mec=colors[row["topology"]], ls="none", ms=8, mew=1.4)
        label = family + (
            "\ndirect 3mm" if length == 3 else "\n10mm -> 3mm scaling prediction")
        method_handles.setdefault(label, Line2D([], [], color="#333333", marker=marker, mfc="#333333" if length == 3 else "none", ls="none"))
    # Retain exact coordinates when two direct observations visually coincide.
    direct = sorted((r for r in records if r["l_mm"] == 3 and r["method"] == "native_energy_eta"
                     and r["value_status"] == "available_provisional"
                     and r.get("KJ_MPa_sqrt_m") is not None), key=lambda r: (r["relative_density"], r["KJ_MPa_sqrt_m"]))
    for first, second in zip(direct, direct[1:]):
        if (first["relative_density"] == second["relative_density"] and first["topology"] != second["topology"]
                and abs(first["KJ_MPa_sqrt_m"]/second["KJ_MPa_sqrt_m"]-1) < .005):
            ax.annotate("3mm {}/{}: {:.2f}/{:.2f}".format(first["topology"], second["topology"],
                        first["KJ_MPa_sqrt_m"], second["KJ_MPa_sqrt_m"]),
                        xy=(first["relative_density"]*4430, first["KJ_MPa_sqrt_m"]), xytext=(-30, 27),
                        textcoords="offset points", ha="right", fontsize=8,
                        bbox={"facecolor":"white", "alpha":.9, "edgecolor":"none", "pad":1},
                        arrowprops={"arrowstyle":"-", "color":"#666666", "lw":.7})
    topology_handles = [Line2D([], [], color=colors[t], lw=2, label=names[t]) for t in TOPOLOGIES]
    source_handles = [Line2D([], [], color="#333333", marker="^", ls="-", label="Hsieh initial KJIC (3mm)"),
                      Line2D([], [], color="#333333", marker="v", mfc="white", ls="--", label="Hsieh final KJFC (3mm)")]
    fig.legend(topology_handles+source_handles+list(method_handles.values()),
               [h.get_label() for h in topology_handles+source_handles]+list(method_handles),
               loc="outside lower center", ncol=3, fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel=r"Physical density (kg m$^{-3}$)",
           ylabel=r"$K_J$ (MPa$\sqrt{\mathrm{m}}$)", xlim=(530, 1000), ylim=(1.5, 65))
    ax.set_xticks([585, 720, 900], ["585", "720", "900"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.22, which="both")
    unavailable = [row for row in records if row["value_status"] != "available_provisional"]
    status_note = ("\n{} selected calculations have unavailable KJ{}.".format(len(unavailable),
                   " because their initial-triangle areas are negative" if all(negative_area(row) for row in unavailable) else "")
                   if unavailable else "")
    if any(row.get("signed_area_KJ_diagnostic_MPa_sqrt_m") is not None for row in unavailable):
        status_note += " Crosses show finite signed-area diagnostics only."
    ax.text(.02, .98, "Source graphical bounds apply to K only.\nP1 initiation is compared with source initial KJIC; p1 final toughness is not computed." + status_note,
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor":"white", "alpha":.9, "edgecolor":"none", "pad":3})
    fig.suptitle("Figure 10 reconstruction - manuscript density interpretation\n"
                 "Hsieh: nominal rho x 4500 kg/m3; p1: nominal rho x 4430 kg/m3\n"
                 "Printed source density x coordinates differ by about 10x and remain unchanged in the CSV.\n"
                 + selected_method_note(records), fontsize=10)
    fig.savefig(out / "figure10-validation.png", dpi=210)
    plt.close(fig)
    write_rows(out / "figure10-density-comparison.csv", output)


def plot_normalisation_sensitivity(records, fig8, material, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    candidates = [r for r in records if r["source"] == "recalculated_rule" and r["value_status"] == "available_provisional"
                  and r["endpoint"] == "criterion_last_intact" and r["relative_density"] in DENSITIES]
    if candidates:
        priority = ("Hsieh_initial_triangle_native", "Hsieh_initial_triangle_QS", "native_energy_eta", "qs_measured_recovery_eta")
        chosen = next((method for method in priority if any(r["method"] == method for r in candidates)), None)
        candidates = [r for r in candidates if r["method"] == chosen] if chosen else []
    if not candidates:
        candidates = [r for r in records if r["source"] == "native_audit" and r["method"] == "raw"
                      and r["value_status"] == "available_provisional" and r["relative_density"] in DENSITIES]
    fig, axes = plt.subplots(1,3,figsize=(12,4.6),constrained_layout=True)
    for ax, topology in zip(axes,TOPOLOGIES):
        refs=[r for r in fig8 if r["figure"]=='8a' and r["topology"]==topology and float(r['l_mm'])==3]
        ax.plot([100*float(r['relative_density']) for r in refs],[float(r['literal_ordinate']) for r in refs],
                color='#222222',marker='^')
        for row in candidates:
            if row['topology']!=topology:continue
            for strength,color,marker in [(material['parent_UTS_workbook_MPa'],'#2379b5','o'),
                                          (material['parent_yield_MPa'],'#9b4098','+')]:
                ax.plot(100*row['relative_density'],row['KJ_MPa_sqrt_m']/(strength*math.sqrt(row['l_mm']/1000)),
                        marker=marker,color=color,ls='none',mfc='none',ms=7)
        ax.set(title=topology,xlabel='Nominal relative density (%)',xticks=[13,16,20],yscale='log')
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r'$K_J/(\sigma\sqrt{\ell})$')
    fig.legend([Line2D([],[],color='#222222',marker='^'),Line2D([],[],color='#2379b5',marker='o',mfc='none',ls='none'),
                Line2D([],[],color='#9b4098',marker='+',ls='none')],
               ['Hsieh Fig8a literal (unchanged)','p1 / parent UTS1111.792MPa','p1 / parent yield932MPa sensitivity'],
               loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Separate normalisation sensitivity: source ordinates are never rescaled\n'
                 'Changing the p1 denominator is an interpretation test, not proof of the source convention.',fontsize=10)
    fig.savefig(out/'figure8-normalisation-sensitivity.png',dpi=210)
    plt.close(fig)


def paper_exports(records, fig8, reference_rows, material, destination, inputs, source_pdf_sha):
    """Initiation-only paper figures; audit calculations and values stay intact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter

    destination.mkdir(parents=True, exist_ok=True)
    method = "Hsieh_initial_triangle_native"
    simulations = {}
    for row in records:
        if row["method"] != method:
            continue
        key = (row["topology"], row["l_mm"], row["relative_density"])
        if key in simulations:
            raise ValueError("Duplicate paper simulation identity: " + repr(key))
        simulations[key] = row
    reference = reference_lookup(reference_rows)
    uts = float(material["parent_UTS_workbook_MPa"])
    points = []
    for figure in ("8a", "8b", "10"):
        for length in LENGTHS:
            for topology in TOPOLOGIES:
                for rho in ((.2,) if figure == "8b" else DENSITIES):
                    key = (topology, length, rho)
                    if figure == "10":
                        source = reference[topology, rho]
                        factor = math.sqrt(length/3.)
                        ref_value = float(source["K_MPa_sqrt_m"])*factor
                        ref_low = float(source["K_graphical_lower"])*factor
                        ref_high = float(source["K_graphical_upper"])*factor
                        basis = "direct_published_3mm" if length == 3 else "published_3mm_times_sqrt_10_over_3_inference"
                        source_id, source_cell = source["point_id"], ""
                    else:
                        source = [r for r in fig8 if r["figure"] == figure and r["topology"] == topology
                                  and float(r["l_mm"]) == length and float(r["relative_density"]) == rho]
                        if len(source) != 1:
                            raise ValueError("Ambiguous paper Figure8 source identity: " + repr(key))
                        source = source[0]
                        ref_value, ref_low, ref_high = float(source["literal_ordinate"]), None, None
                        basis = "original_literal_ordinate_nominal_length_assignment"
                        source_id, source_cell = source["reference_identity"], source["source_cell"]
                    x = math.sqrt(length) if figure == "8b" else 100*rho
                    unit = "MPa sqrt(m)" if figure == "10" else "sqrt(m)" if figure == "8b" else "dimensionless"
                    common = {"figure": figure, "topology": topology, "l_mm": length, "relative_density": rho,
                              "x_value": x, "x_unit": "sqrt(mm)" if figure == "8b" else "relative density (%)",
                              "ordinate_unit": unit, "initiation_only": True,
                              "reference_basis": basis, "reference_point_id": source_id,
                              "reference_source_cell": source_cell, "published_pdf_sha256": source_pdf_sha,
                              "normalisation_parent_UTS_MPa": uts if figure != "10" else None,
                              "source_ordinates_rescaled": figure == "10" and length == 10,
                              "vertical_fit_or_offset": False}
                    points.append({**common, "series": "Hsieh et al.", "plotted": True, "ordinate": ref_value,
                                   "graphical_lower": ref_low, "graphical_upper": ref_high,
                                   "source_input_sha256": inputs["reference" if figure == "10" else "original_fig8"]["sha256"]})
                    row = simulations.get(key)
                    valid = (row is not None and row["value_status"] == "available_provisional"
                             and row.get("KJ_MPa_sqrt_m") is not None and not negative_area(row))
                    value = row["KJ_MPa_sqrt_m"] if valid else None
                    denominator = uts*(math.sqrt(length/1000.) if figure == "8a" else 1) if figure != "10" else 1
                    reason = ("" if valid else "negative initial-triangle area" if row is not None and negative_area(row) else
                              "no selected simulation" if row is None else "unavailable recalculation")
                    points.append({**common, "series": "Simulation", "plotted": bool(valid),
                                   "ordinate": value/denominator if valid else None, "exclusion_reason": reason,
                                   "case_id": row.get("case_id") if row else None, "method": method,
                                   "endpoint": row.get("endpoint") if row else None,
                                   "endpoint_time_s": row.get("endpoint_time_s") if row else None,
                                   "KI_MPa_sqrt_m": row.get("KI_MPa_sqrt_m") if row else None,
                                   "KJ_MPa_sqrt_m": row.get("KJ_MPa_sqrt_m") if row else None,
                                   "Apl_Nmm": row.get("Apl_Nmm") if row else None,
                                   "qualified_ASTM_J": row.get("qualified_ASTM_J") if row else False,
                                   "source_input_sha256": inputs.get("selected_recalculation", {}).get("sha256"),
                                   **{name: row.get(name) if row else None for name in
                                      ("source_inp_sha256", "source_odb_sha256", "modulus_odb_sha256", "modulus_job",
                                       "compliance_odb_sha256", "initial_stiffness_N_per_mm", "modulus_MPa", "quality_note")}})
    write_rows(destination / "points.csv", points)
    write_rows(destination / "exclusions.csv", [r for r in points if r["series"] == "Simulation" and not r["plotted"]])
    captions = {
        "8a": ("Initial-fracture comparison at nominal relative densities 13%, 16%, 20%. Simulation values use the original Explicit histories, "
               "the last recorded undamaged tensile-strain sample and Hsieh's initial-slope area convention with the C(T) calibration. "
               "Simulation normalisation uses parent UTS 1111.792 MPa and bar length in metres. Hsieh's original normalised ordinates are unchanged "
               "and repeated at the two nominal length assignments; the source strength convention remains unresolved. Undefined areas and absent cases are omitted, "
               "with every exclusion recorded in exclusions.csv. No vertical shift, fitted factor or replacement work estimate is used."),
        "8b": ("Initial-fracture comparison at nominal relative density 20%. The original Hsieh ordinates are retained at nominal 3/10 mm assignments. "
               "The x coordinate is sqrt(bar length in mm), following the source tick convention despite its metre label; source marker x coordinates were not digitised. "
               "Simulation values use parent UTS 1111.792 MPa. The inconsistency between Figures 8a/8b remains unresolved; this is not independent proof of absolute toughness. "
               "The undefined 3 mm hexagonal simulation is omitted."),
        "10": ("Dimensional comparison based on Figure 10 initiation data at matched nominal relative densities, avoiding the different constituent mass densities 4430/4500 kg/m3. "
               "The 3 mm Hsieh initial-toughness points are directly digitised; the 10 mm reference is an explicitly declared sqrt(10/3) scaling inference. "
               "Error bars bound graphical extraction only and are not statistical confidence intervals. Simulation values use the original Explicit histories and initial-slope area convention; "
               "no final-fracture, dissipation or nonlinear-recovery estimates are included. Negative-area values and absent cases are omitted; data and source fingerprints are in points.csv. "
               "This comparison does not establish ASTM qualification or agreement for all lattices.")}
    colors = {"tri": "#c43b3b", "kagome": "#303030", "hex": "#3465a4"}
    names = {"tri": "Triangular", "kagome": "Kagome", "hex": "Hexagonal"}
    style = {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8.5,
             "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .8}
    with plt.rc_context(style):
        for figure in ("8a", "8b", "10"):
            count = 1 if figure == "8b" else 2
            fig, axes = plt.subplots(1, count, figsize=(7.2, 3.5), squeeze=False, sharey=count==2)
            fig.subplots_adjust(left=.145, right=.98, top=.89, bottom=.30, wspace=.22)
            for panel, length in enumerate((None,) if figure == "8b" else LENGTHS):
                ax = axes[0, panel]
                for topology in TOPOLOGIES:
                    for series in ("Hsieh et al.", "Simulation"):
                        selected = [r for r in points if r["figure"] == figure and r["topology"] == topology
                                    and r["series"] == series and r["plotted"] and (length is None or r["l_mm"] == length)]
                        selected.sort(key=lambda r: r["x_value"])
                        if not selected:
                            continue
                        xx, yy = [r["x_value"] for r in selected], [r["ordinate"] for r in selected]
                        sim = series == "Simulation"
                        options = dict(color=colors[topology], marker="o" if sim else "^", ms=4.8,
                                       mfc=colors[topology] if sim else "white", mew=1., lw=1.,
                                       ls="--" if sim else "-", zorder=3 if sim else 2)
                        if figure == "10" and not sim:
                            ax.errorbar(xx, yy, yerr=[[r["ordinate"]-r["graphical_lower"] for r in selected],
                                                    [r["graphical_upper"]-r["ordinate"] for r in selected]],
                                        capsize=2, elinewidth=.7, **options)
                        else:
                            ax.plot(xx, yy, **options)
                ax.set_yscale("log")
                ax.grid(alpha=.18, which="major", lw=.6)
                ax.set_axisbelow(True)
                if figure == "8b":
                    ax.set_xticks([math.sqrt(3), math.sqrt(10)], [r"$\sqrt{3}$", r"$\sqrt{10}$"])
                    ax.set_xlabel(r"$\sqrt{\ell}$ (mm$^{1/2}$)")
                    ax.set_title(r"$\bar{\rho}=20\%$")
                else:
                    ax.set_xscale("log")
                    ax.set_xticks([13,16,20], ["13","16","20"])
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    ax.set_xlim(12.4,20.6)
                    ax.set_xlabel("Relative density (%)")
                    title = r"$\ell={}$ mm".format(int(length))
                    if figure == "10":
                        title += " | direct reference" if length == 3 else " | scaled reference"
                    ax.set_title(title)
                if panel == 0:
                    ax.set_ylabel(r"$K_J/(\sigma_{\mathrm{TS}}\sqrt{\ell})$" if figure == "8a" else
                                  r"$K_J/\sigma_{\mathrm{TS}}$ (m$^{1/2}$)" if figure == "8b" else
                                  r"$K_J$ (MPa$\sqrt{\mathrm{m}}$)")
                if figure == "10":
                    ax.set_ylim(1.5,65)
                if count == 2:
                    ax.text(.02,.95,"({})".format("ab"[panel]),transform=ax.transAxes,va="top",fontsize=9)
            handles = [Line2D([],[],color=colors[t],lw=1.6,label=names[t]) for t in TOPOLOGIES]
            handles += [Line2D([],[],color="#555555",marker="^",mfc="white",ls="-",lw=1,label="Hsieh et al."),
                        Line2D([],[],color="#555555",marker="o",mfc="#555555",ls="--",lw=1,label="Simulation")]
            fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.08),ncol=5,frameon=False,
                       columnspacing=1.15,handlelength=1.5,handletextpad=.45)
            note = (r"10 mm reference: 3 mm data $\times\sqrt{10/3}$. " if figure == "10" else "")
            note += "Initial fracture only."
            fig.text(.5,.025,note,ha="center",va="bottom",fontsize=7.5)
            for extension in ("png","pdf","svg"):
                fig.savefig(destination / ("figure{}.{}".format(figure,extension)),dpi=300)
            plt.close(fig)
    metadata = {"selection_method":method,"case_selection":"Existing selected largest model per topology/length/nominal density; no agreement-based selection",
                "simulation_cases":len(simulations),"point_rows":len(points),
                "plotted_simulation_rows":sum(r["series"]=="Simulation" and r["plotted"] for r in points),
                "excluded_simulation_rows":sum(r["series"]=="Simulation" and not r["plotted"] for r in points),
                "inputs":{key:inputs[key] for key in ("reference", "original_fig8", "original_fig8_provenance", "selected_recalculation") if key in inputs},
                "published_pdf_sha256":source_pdf_sha,"captions":captions,
                "limits":["Original Explicit initial-triangle recalculation only; no recovery/IE-SE/QS/final-fracture series",
                          "No ordinate shift or agreement fit; same nominal densities have identical horizontal coordinates",
                          "Figure8 source normalisation and length ambiguities remain unresolved",
                          "Only3mm Figure10 references are direct;10mm is an explicit length-scaling inference",
                          "Negative area remains unavailable; positive area does not establish qualified ASTM J"]}
    (destination / "metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (destination / "captions.txt").write_text("\n\n".join("Figure {}. {}".format(k,v) for k,v in captions.items())+"\n",encoding="utf-8")


def run(root="Z:/p1/sims/Ti/Validation", output=None, final_results=None):
    root = Path(root).resolve()
    out = Path(output).resolve() if output is not None else root / "audit/benchmark"
    out.mkdir(parents=True, exist_ok=True)
    paper_out = out / "paper" if output is not None else root / "audit/paper"
    sources = {"reference": root / "audit/benchmark/fig10-digitised.csv",
               "native": root / "audit/calculations/comparison.csv",
               "native_provenance": root / "audit/calculations/provenance.json",
               "historic_FTval_export": root / "audit/calculations/workbook-reconciliation.csv"}
    snapshots = {}
    def capture(name, json_data=False):
        # The exporter may update between refreshes. Every output in this run
        # uses one captured version, whose exact bytes are hashed below.
        if name not in snapshots:
            snapshots[name] = sources[name].read_bytes()
        source = snapshots[name].decode("utf-8-sig")
        return json.loads(source) if json_data else list(csv.DictReader(io.StringIO(source)))
    reference_rows = capture("reference")
    reference = reference_lookup(reference_rows)
    historic = historical_records(capture("historic_FTval_export"), reference)
    native = native_records(capture("native"), historic, reference)
    case_provenance = {row["case_id"]: row for row in capture("native_provenance", json_data=True)}
    for row in native:
        proof = case_provenance.get(row["case_id"], {})
        if row["modulus"] == "beam_rve" and not proof.get("beam_rve_job"):
            raise ValueError("Beam-RVE comparison lacks case-specific provenance: " + row["case_id"])
        row["indexed_source_inp_sha256"] = proof.get("indexed_source_inp_sha256")
        row["indexed_source_odb_sha256"] = proof.get("indexed_source_odb_sha256")
        row["beam_rve_job"] = proof.get("beam_rve_job") if row["modulus"] == "beam_rve" else None
        row["beam_rve_odb_sha256"] = proof.get("beam_rve_odb_sha256") if row["modulus"] == "beam_rve" else None
    records = historic + native
    selected_path = Path(final_results).resolve() if final_results else root / "audit/final/validation-results.csv"
    if selected_path.exists():
        sources["selected_recalculation"] = selected_path
        records += recalculated_records(capture("selected_recalculation"), reference)
    else:
        interim = root / "audit/diagnostics/criterion-endpoint-metrics.csv"
        if interim.exists():
            sources["interim_criterion_controls"] = interim
            records += criterion_controls(capture("interim_criterion_controls"), native, reference)
    add_old_new_changes(records)
    write_rows(out / "validation-comparison.csv", records)
    write_rows(out / "validation-coverage.csv", coverage(records))
    principal = [row for row in records if row["source"] == "recalculated_rule"]
    controls = [row for row in records if row["source"] != "recalculated_rule"]
    plot_comparison(principal, reference, out, principal=True)
    plot_comparison(principal, reference, out, differences=True, principal=True)
    plot_comparison(controls, reference, out)
    plot_comparison(controls, reference, out, differences=True)
    plot_figure10_density(principal, reference_rows, out)
    summary_fields = ("case_id", "topology", "l_mm", "relative_density", "nnx", "endpoint", "endpoint_time_s",
                      "method", "work_partition", "loading_state", "source_work_convention", "initial_stiffness_N_per_mm",
                      "compliance_source", "KI_MPa_sqrt_m", "KJ_MPa_sqrt_m", "Apl_Nmm", "Apl_N_mm",
                      "KJ_signed_area_diagnostic_MPa_sqrt_m", "upstream_value_status",
                      "reference_basis", "reference_K_MPa_sqrt_m", "reference_graphical_lower", "reference_graphical_upper",
                      "difference_from_reference_percent", "difference_reference_graphical_lower_percent",
                      "difference_reference_graphical_upper_percent", "qualified_ASTM_J", "quality_note")
    write_rows(out / "validation-selected-summary.csv", [{key: row.get(key) for key in summary_fields} for row in principal])
    fig8_path = root / "audit/benchmark/fig8-original-digitisation.csv"
    if fig8_path.exists():
        sources["original_fig8"] = fig8_path
        sources["original_fig8_provenance"] = root / "audit/benchmark/fig8-source-provenance.json"
        fig8 = capture("original_fig8")
        fig8_provenance = capture("original_fig8_provenance", json_data=True)
        material = fig8_provenance["material"]
        write_rows(out / "fig8-normalised-comparison.csv", normalised_records(records,fig8,material))
        for panel in ("8a", "8b"):
            plot_fig8(principal,fig8,material,out,panel,principal=True)
            plot_fig8(controls,fig8,material,out,panel)
        plot_normalisation_sensitivity(records,fig8,material,out)
        paper_exports(principal,fig8,reference_rows,material,paper_out,
                      {name:{"path":str(path),"sha256":hashlib.sha256(snapshots[name]).hexdigest()}
                       for name,path in sources.items()},fig8_provenance["published_pdf_sha256"])
    provenance = {"inputs": {name: {"path": str(path), "sha256": hashlib.sha256(snapshots[name]).hexdigest()}
                             for name, path in sources.items()},
                  "paper_output_directory": str(paper_out),
                  "inputs_changed_after_capture": [name for name, path in sources.items()
                       if sha256(path) != hashlib.sha256(snapshots[name]).hexdigest()],
                  "selection": "Existing FTval exact case IDs plus prepared exact static controls; no fit to benchmark",
                  "native_control": "last_monitored_intact; raw and available released order2 Butterworth; static C0 preferred, otherwise explicitly labelled 15% curve-fit sensitivity",
                  "modulus_preference_per_exact_case": list(MODULUS_PREFERENCE),
                  "moduli_in_use": sorted({r["modulus"] for r in native}),
                  "historic_rows_retained": len(historic), "native_controls": len(native),
                  "recalculated_rows": sum(r["source"] == "recalculated_rule" for r in records),
                  "interim_criterion_controls": sum(r["source"] == "interim_criterion_control" for r in records),
                  "principal_plot_selection": "Only rows explicitly supplied by selected_recalculation; historical/native controls are separate -controls PNGs, never substituted for pending corrected results",
                  "figure10_density_interpretation": "Source nominalrho*4500 and p1 nominalrho*4430; source literal printed density coordinates preserved; p1 10mm projected to3mm explicitly a scaling prediction",
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
    parser.add_argument("--final-results", help="Explicit selected-rule CSV; defaults to audit/final/validation-results.csv when present")
    args = parser.parse_args()
    values = run(args.root, args.out, args.final_results)
    print("Wrote {} labelled historical/native controls".format(len(values)))
