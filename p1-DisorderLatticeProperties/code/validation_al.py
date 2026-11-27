"""Reproduce the Al triangular UAT benchmark overlay without changing its sources."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

from resources.calculations import filter_force
import resources.calculations as calculations


def run(results, workbook, output=None):
    results, workbook = Path(results), Path(workbook)
    native = results.with_name("al-native.json")
    registry = Path(__file__).with_name("response_filter_cutoffs.json")
    paths = (results, native, workbook, registry, Path(calculations.__file__), Path(__file__))
    captured = {str(p.resolve()): p.read_bytes() for p in paths}
    fingerprints = {p: hashlib.sha256(data).hexdigest() for p, data in captured.items()}
    doc = json.loads(captured[str(results.resolve())])
    source = json.loads(captured[str(native.resolve())])
    cutoffs = json.loads(captured[str(registry.resolve())])["cutoffs_hz"]
    if doc["study"] != "al" or source["study"] != "al":
        raise ValueError("Expected Al convergence extraction and replay")
    if doc["native_sha256"] != fingerprints[str(native.resolve())]:
        raise ValueError("Native source does not match the replay fingerprint")
    if doc["registry_sha256"] != fingerprints[str(registry.resolve())]:
        raise ValueError("Cutoff registry changed after the replay")
    rows = [r for r in doc["records"] if r["variant"] == "18-1.1"]
    original = [r for r in source["records"] if r["variant"] == "18-1.1"]
    if len(rows) != 1 or len(original) != 1:
        raise ValueError("Expected one active 18-1.1 source")
    row, original = rows[0], original[0]
    if row["input_sha256"] != original["input_sha256"]:
        raise ValueError("Extraction and replay input identities differ")
    if hashlib.sha256(Path(row["input"]).read_bytes()).hexdigest() != row["input_sha256"]:
        raise ValueError("Archived model input changed after extraction")
    time = np.asarray(original["time"], float)
    strain = np.asarray(row["curve"]["strain"], float)
    stress = np.asarray(row["curve"]["stress_MPa"], float)
    gauge = row["source_gauge"]
    if not np.allclose(strain, np.asarray(original["x"])/gauge["height_mm"], rtol=0, atol=1e-14):
        raise ValueError("Stored curve does not use its declared strain normalisation")
    if not np.allclose(stress, np.asarray(original["y"])/gauge["area_mm2"], rtol=0, atol=1e-12):
        raise ValueError("Stored curve does not use its declared stress normalisation")
    cutoff = row["filter"]["cutoff_Hz"]
    if cutoff != cutoffs["tri"]["UT"]:
        raise ValueError("Stored filter differs from the approved triangular UAT cutoff")
    filtered_strain, filter_info = filter_force(time, strain, cutoff)
    filtered_stress, _ = filter_force(time, stress, cutoff)
    peak = int(np.argmax(filtered_stress))
    if not np.isclose(filtered_stress[peak], row["peak_response"], rtol=0, atol=1e-10):
        raise ValueError("Reconstructed filter does not reproduce the saved peak")
    source_dimensions = {"raw_peak_stress_MPa": float(np.max(stress)),
                         "filtered_peak_stress_MPa": float(filtered_stress[peak]),
                         "filtered_strain_at_peak": float(filtered_strain[peak]),
                         "ductility25_coordinate": row["evaluation"]["ductility25_coordinate"],
                         "initial_work_stiffness_40_MPa": row["evaluation"]["initial_work_fits"]["0.4"]["estimate"]}
    benchmark = {"height_mm": 176., "area_mm2": 414., "nominal_width_mm": 207., "B_mm": 2.,
                 "convention": "Author-agreed nominal benchmark dimensions; distinct from actual deck dimensions",
                 "strain": "clamp travel divided by nominal benchmark height",
                 "stress": "total clamp reaction divided by nominal benchmark gross area"}
    strain = np.asarray(original["x"], float)/benchmark["height_mm"]
    stress = np.asarray(original["y"], float)/benchmark["area_mm2"]
    filtered_strain, _ = filter_force(time, strain, cutoff)
    filtered_stress, _ = filter_force(time, stress, cutoff)
    peak = int(np.argmax(filtered_stress))
    raw_peak = int(np.argmax(stress))
    running_peak = np.maximum.accumulate(stress)
    collapses = np.flatnonzero((np.arange(len(stress)) > raw_peak) & (stress <= .5*running_peak))
    if not len(collapses):
        raise ValueError("No observed post-peak 50%-force collapse for the requested display endpoint")
    collapse = int(collapses[0])
    minima = [i for i in range(max(collapse, 1), len(stress)-1)
              if filtered_stress[i] < filtered_stress[i-1] and filtered_stress[i+1] >= filtered_stress[i]]
    if not minima:
        raise ValueError("First filtered minimum after force collapse was not observed")
    stop = minima[0]
    truncation = {"rule": "First native post-peak reaction <=50% of its preceding running peak; include filtered decline through its first subsequent local minimum, before rebound",
                  "native_collapse_index": collapse, "native_collapse_time_s": float(time[collapse]),
                  "native_collapse_fraction_of_running_peak": float(stress[collapse]/running_peak[collapse]),
                  "last_plotted_native_index": stop, "last_plotted_time_s": float(time[stop]),
                  "last_plotted_strain": float(filtered_strain[stop]),
                  "last_plotted_stress_MPa": float(filtered_stress[stop]),
                  "plotted_samples": stop+1, "retained_native_samples": len(time),
                  "purpose": "Display truncation only; full native histories and calculation replay remain unchanged"}

    book = load_workbook(io.BytesIO(captured[str(workbook.resolve())]), read_only=True, data_only=True)
    sheet = book["UTval"]
    values = list(sheet.iter_rows(values_only=True))
    if values[0][1] != "18-1.1":
        raise ValueError("Workbook no longer identifies 18-1.1 as its current curve")
    series = {}
    for name, columns in (("historic_current", (0, 1)), ("Bhuwal et al. (2023)", (12, 13)),
                          ("Gu et al. (2018), FE", (15, 16)), ("Gu et al. (2018), experiment", (18, 19))):
        pairs = [(r[columns[0]], r[columns[1]]) for r in values[1:]
                 if isinstance(r[columns[0]], (int, float)) and isinstance(r[columns[1]], (int, float))]
        series[name] = np.asarray(pairs, float)
        if not len(pairs) or not np.isfinite(series[name]).all():
            raise ValueError("Missing or nonfinite workbook reference observations")
    book.close()
    historic = series["historic_current"]
    historic_scaling = {"status": "not_resolved"}
    if len(time) == 2*len(historic)-1:
        sampled_u = np.asarray(original["x"])[::2]
        sampled_p = np.asarray(original["y"])[::2]
        height = float(np.dot(historic[:, 0], sampled_u)/np.dot(historic[:, 0], historic[:, 0]))
        area = float(np.dot(historic[:, 1], sampled_p)/np.dot(historic[:, 1], historic[:, 1]))
        residual_u = float(np.max(np.abs(height*historic[:, 0]-sampled_u))/np.max(np.abs(sampled_u)))
        residual_p = float(np.max(np.abs(area*historic[:, 1]-sampled_p))/np.max(np.abs(sampled_p)))
        historic_scaling = {"status": "inferred_from_every_second_native_sample",
                            "height_mm": height, "area_mm2": area,
                            "inferred_width_at_source_B_mm": area/gauge["B_mm"],
                            "maximum_relative_displacement_residual": residual_u,
                            "maximum_relative_force_residual": residual_p,
                            "corrected_over_historic_stress_scale": area/gauge["area_mm2"],
                            "corrected_over_historic_strain_scale": height/gauge["height_mm"]}
        if max(residual_u, residual_p) > 5e-6:
            historic_scaling["status"] = "unresolved_scaling_mismatch"
    destination = Path(output) if output is not None else results.parent
    destination.mkdir(parents=True, exist_ok=True)
    image = destination / "al-validation.png"
    receipt = destination / "al-validation.json"
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "mathtext.fontset": "stix"}):
        fig, ax = plt.subplots(figsize=(8.5, 5.3), layout="constrained")
        ax.plot(filtered_strain[:stop+1], filtered_stress[:stop+1], color="red", linewidth=1.8,
                label="Current study", zorder=3)
        for name, color, style in (("Bhuwal et al. (2023)", ".55", "--"),
                                   ("Gu et al. (2018), FE", ".25", "-")):
            xy = series[name]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=style, linewidth=1.45,
                    label=name, zorder=2)
        xy = series["Gu et al. (2018), experiment"]
        ax.plot(xy[:, 0], xy[:, 1], color="black", linestyle=":", linewidth=1.1, marker="+", markersize=6,
                markeredgewidth=1.1, label="Gu et al. (2018), experiment", zorder=4)
        ax.set(xlabel=r"Macroscopic strain, $\varepsilon_L$", ylabel=r"Macroscopic stress, $\Sigma$ (MPa)",
               xlim=(0, .02), ylim=(0, 20))
        ax.grid(alpha=.18, linewidth=.6)
        ax.legend(frameon=False, fontsize=10, loc="lower center", ncol=2)
        fig.savefig(image, dpi=240)
        plt.close(fig)
    comparisons = {}
    for name, xy in series.items():
        i = int(np.argmax(xy[:, 1]))
        comparisons[name] = {"points": len(xy), "peak_stress_MPa": float(xy[i, 1]),
                             "strain_at_peak": float(xy[i, 0]), "last_point": xy[-1].tolist(),
                             "present_peak_difference_percent": float(100*(filtered_stress[peak]/xy[i, 1]-1))}
    caption = (
        "Al1050A triangular lattice UAT comparison. Present simulations use archived variant 18-1.1, "
        "the author-agreed nominal benchmark height H=176mm and gross area A=414mm2; "
        "actual deck dimensions remain separately recorded. "
        "Only the approved paired causal second-order Butterworth curve is shown in red. "
        "It ends at the first filtered minimum after the native force falls below 50% of its preceding peak, "
        "including the first collapse and excluding the rebound tail. "
        "References are the preserved workbook digitisation of Gu et al. (2018), Fig.13, 0-degree orientation, "
        "and Bhuwal et al. (2023), Supplementary Fig.B.2B. Gu's curves do not include a full failure drop; "
        "they do not validate a 25%-stress ductility endpoint. Exact reference strain-gauge equivalence is unestablished."
    )
    result = {"figure": str(image.resolve()), "caption": caption,
              "status": "diagnostic_author_agreed_benchmark_nominal_normalisation",
              "active_variant": "18-1.1", "source_gauge": gauge,
              "benchmark_normalisation": benchmark, "source_dimension_diagnostics": source_dimensions,
              "display_truncation": truncation, "filter": filter_info,
              "samples": len(time), "last_time_s": float(time[-1]),
              "raw_peak_stress_MPa": float(np.max(stress)),
              "filtered_peak_stress_MPa": float(filtered_stress[peak]),
              "filtered_strain_at_peak": float(filtered_strain[peak]),
              "filter_peak_change_percent": float(100*(filtered_stress[peak]/np.max(stress)-1)),
              "present_ductility25_observed": row["evaluation"]["ductility25_observed"],
              "present_ductility25_coordinate": row["evaluation"]["ductility25_coordinate"]*gauge["height_mm"]/benchmark["height_mm"]
                                               if row["evaluation"]["ductility25_coordinate"] is not None else None,
              "historic_current_scaling": historic_scaling,
              "initial_work_stiffness_40_MPa": source_dimensions["initial_work_stiffness_40_MPa"]*
                                               gauge["area_mm2"]/benchmark["area_mm2"]*
                                               benchmark["height_mm"]/gauge["height_mm"],
              "comparisons": comparisons,
              "reference_ranges": {"historic_current": "UTval!A2:B102", "Bhuwal et al. (2023)": "UTval!M2:N43",
                                   "Gu et al. (2018), FE": "UTval!P2:Q93", "Gu et al. (2018), experiment": "UTval!S2:T27"},
              "reference_dois": {"Gu et al. (2018)": "10.1016/j.ijsolstr.2018.06.028",
                                 "Bhuwal et al. (2023)": "10.1016/j.jmps.2023.105294"},
              "source_sha256": fingerprints, "input_sha256": row["input_sha256"],
              "odb": row["odb"], "sampled_odb_fingerprint": row["sampled_odb_fingerprint"],
              "figure_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    for path, digest in fingerprints.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Source changed during figure generation: " + path)
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.results, args.workbook, args.output)
    print(json.dumps({k: result[k] for k in ("figure", "filtered_peak_stress_MPa", "comparisons")}, indent=2))
