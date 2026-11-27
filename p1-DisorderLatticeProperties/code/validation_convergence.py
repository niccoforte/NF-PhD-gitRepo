"""Read-only mesh, periodic size and aluminium tensile benchmark replay."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from resources.calculations import compact_tension_initiation, filter_force
from resources.lattices import prismatic_plane_strain
from validation_controls import body_tensor_source
from validation_curve_processing import sampled_odb_fingerprint
from validation_failure_audit import tensile_strain_event
from validation_fcc import source_geometry
from validation_fixed_filter_pilot import REGISTRY, ROOT, metrics, read_json
from validation_production_replay import compact_tension_properties
from validation_stiffness import _write_immutable_targets, sha

MATERIALS = Path("audit/controls/population-materials/material-catalogue.json")
MESH_SELECTED = {
    "tri": {"UT": "1-2-5", "FT": "1-2-5"},
    "kagome": {"UT": "1-1-10", "FT": "1-1-10"},
    "hex": {"UT": "1-1-15", "FT": "1-1-10"},
    "FCC": {"UT": "1-5-5", "FT": "1-2-5"},
}
SELECTED = {"tri": 30, "kagome": 20, "hex": 20, "FCC": 20}
COLOR = {"tri": "red", "kagome": "black", "hex": "green", "FCC": "tab:blue"}
LABEL = {"tri": "Triangular", "kagome": "Kagome", "hex": "Hexagonal", "FCC": "FCC"}


def sources(root, study):
    shared = root/"audit/shared"
    if study == "mesh":
        return read_json(shared/"mesh-source-inventory.json")["cases"]
    if study == "psc":
        base = root.parent/"PSC/10"
        files = sorted(base.glob("*-per-1.inp"))
        result = [dict(input=str(p), input_sha256=sha(p), odb=str(p.with_suffix(".odb")),
                       export=str(base/"transfer"/("OUT-"+p.stem+".csv")), job=p.stem)
                  for p in files]
        if len(result) != 56:
            raise ValueError("Expected the retained 56-case PSC cohort")
        return result
    al = read_json(shared/"ortho-material-and-psc-inventory.json")["UT_validation"]
    return [dict(input=str(p), input_sha256=sha(p), odb=str(p.with_suffix(".odb")),
                 export=str(p.parent/"transfer"/("OUT-"+p.stem+".csv")), job=p.stem,
                 variant=p.parent.name) for p in map(Path, al["INPs"])]


def source_transfer(x, y, path):
    """Check existing export normalization on its real overlap with U2/RF2."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        old = np.asarray([list(map(float, line.split(",", 2)[:2])) for line in stream if line.strip()])
    n = min(len(x)-1, len(old))
    factors, mismatches = [], []
    for original, retained in zip((x, y), old[:n].T):
        a = original[:n]
        inside = np.isfinite(retained) & (abs(a) > 1e-7*max(abs(a))) & (retained != 0.)
        if not np.any(inside):
            raise ValueError("No nonzero native/transfer normalization overlap")
        scale = float(np.median(retained[inside]/a[inside]))
        mismatch = float(max(abs(retained[inside]-scale*a[inside]))/max(abs(retained[inside])))
        if mismatch > 2e-5 or scale <= 0:
            raise ValueError("Native/transfer normalization mismatch: {:.6g}".format(mismatch))
        factors.append(scale)
        mismatches.append(mismatch)
    return factors, mismatches


def read_native_case(odb_file, task, scratch):
    """Open the archive directly or copy one ODB to local scratch for random reads."""
    from odbAccess import openOdb
    from resources.abaqus import native_reaction_history

    before = sampled_odb_fingerprint(odb_file)
    staged = None
    try:
        path = odb_file
        if scratch is not None:
            with tempfile.NamedTemporaryFile(prefix="p1-convergence-", suffix=".odb",
                                             dir=scratch, delete=False) as temporary:
                staged = Path(temporary.name)
            shutil.copyfile(odb_file, staged)
            copied = sampled_odb_fingerprint(staged)
            if any(before[k] != copied[k] for k in ("bytes", "first_and_last_64KiB_sha256")):
                raise ValueError("Staged ODB fingerprint differs from the archive")
            path = staged
        odb = openOdb(str(path), readOnly=True)
        try:
            t, x, y = native_reaction_history(odb, task)
            step, = odb.steps.values()
            frames = len(step.frames)
            event = tensile_strain_event(step, t, x, y) if task == "FT" else None
        finally:
            odb.close()
        if sampled_odb_fingerprint(odb_file) != before:
            raise ValueError("ODB identity changed during read-only extraction")
        return t, x, y, frames, event, before
    finally:
        if staged is not None:
            staged.unlink()


def collect(root, study, out, scratch=None):
    exporter_sha = sha(__file__)
    configs = sources(root, study)
    result = []
    for number, config in enumerate(configs, start=1):
        inp, odb_file, transfer = (Path(config[k]) for k in ("input", "odb", "export"))
        label, lat, nx, _, _ = inp.stem.split("-")
        task = {"Ductile": "UT", "Fracture": "FT"}[label]
        row = dict(config, topology=lat, nx=int(nx), task=task)
        try:
            if sha(inp) != config["input_sha256"]:
                raise ValueError("Archive input changed")
            t, x, y, frames, event, before = read_native_case(odb_file, task, scratch)
            row.update(field_frames=frames, native_event=event)
            row.update(native_status="observed", time=t.tolist(), x=x.tolist(), y=y.tolist(),
                       sampled_odb_fingerprint=before)
            if transfer.is_file():
                row["transfer_sha256"] = sha(transfer)
                try:
                    row["normalisation"], row["relative_transfer_error"] = source_transfer(x, y, transfer)
                except ValueError as error:
                    row["normalisation_note"] = str(error)
            if task == "FT":
                row["geometry"] = source_geometry(inp)[3]
                if row.get("normalisation") and not np.allclose(row["normalisation"], 1., rtol=2e-5):
                    raise ValueError("FT archive export units differ from native N/mm")
            if task == "UT" and not row.get("normalisation"):
                row["normalisation_note"] = row.get("normalisation_note", "No retained transfer for gauge validation")
        except Exception as error:
            row.update(native_status="unavailable", reason=str(error), error_type=type(error).__name__)
        result.append(row)
        if number % 8 == 0 or number == len(configs):
            print(study, number, len(configs), dict(Counter(r["native_status"] for r in result)), flush=True)
    doc = dict(study=study, records=result, exporter_sha256=exporter_sha)
    _write_immutable_targets({out:json.dumps(doc, indent=2, allow_nan=False)+"\n"})


def convergence_event(event, time):
    """Admit tiny timestamp round-off when the intact field bracket is observed."""
    endpoint = {"status": event["status"]}
    if endpoint["status"] == "unqualified_observed_event" and not event["nonfinite_before_threshold"]:
        low, high = event["prev_native_history"], event["cross_native_history"]
        # The original absolute gate is 0.2 us. Limit the convergence replay
        # correction to 0.1% of the measured history interval, at most 1 us.
        allowed = min(1e-6, .001 * float(np.median(np.diff(time))))
        error = max(abs(v["time_error_to_field_s"]) for v in (low, high))
        if (0 < low["index"] < high["index"] < len(time) and error <= allowed
                and event["prev_max_LE11"] < .1105 <= event["cross_max_LE11"]):
            endpoint.update(status="qualified_observed_event", time_match="native_timestamp_roundoff",
                            maximum_time_error_s=error, permitted_time_error_s=allowed)
    if endpoint["status"] == "qualified_observed_event":
        point = event["prev_native_history"]
        endpoint.update(native_index=point["index"], time_s=point["time_s"])
    return endpoint


def archive_raw_fallback(filtered, fits, x, y):
    """Recompute at the same native event and screen the three k0 windows."""
    i = filtered["sample_index"]
    a, B, W = (filtered["geometry"][k] for k in ("a_mm", "B_mm", "W_mm"))
    candidates = {}
    for fraction in ("0.3", "0.4", "0.5"):
        fit = fits[fraction]
        if fit["status"] != "resolved_estimate" or fit["end_index"] > i:
            raise ValueError("Raw fallback lacks a qualified initial-stiffness window")
        result = compact_tension_initiation(x, y, i, a, B, W, filtered["modulus_MPa"],
                                            fit["estimate"], filtered["geometry_factor"])
        if not result["valid_plastic_area"]:
            raise ValueError("Raw fallback fails plastic-work sensitivity at " + fraction)
        candidates[fraction] = result
    primary = candidates["0.4"]
    return dict(filtered, **primary, status="calculated_raw_fallback",
                response_method="paired_archive_raw_fallback",
                minimum_k0_for_nonnegative_area_N_per_mm=primary["force_N"]**2/(2*primary["work_N_mm"]),
                raw_Apl_by_filtered_stiffness_window_N_mm={
                    fraction: values["Apl_N_mm"] for fraction, values in candidates.items()},
                filtered_diagnostic=filtered)


def calculate(raw, materials, raw_fallbacks=()):
    registry = read_json(REGISTRY)["cutoffs_hz"]
    fcc = prismatic_plane_strain(materials["FCC"]["C_MPa"], .2, 123000., .3)
    result = []
    for row in raw["records"]:
        stripped = {k: v for k, v in row.items() if k not in ("time", "x", "y")}
        result.append(stripped)
        if row["native_status"] != "observed":
            continue
        task, lat = row["task"], row["topology"]
        t, x, y = (np.asarray(row[k]) for k in ("time", "x", "y"))
        if task == "UT":
            try:
                if sha(row["input"]) != row["input_sha256"]:
                    raise ValueError("Archive input changed since native extraction")
                gauge = body_tensor_source(row["input"])
                width, height = np.diff(np.asarray(gauge["bounds_mm"]), axis=0)[0]
                thickness = gauge["section_B_t_mm"][0]
            except ValueError as error:
                stripped.update(calculation_status="unavailable_source_gauge", calculation_reason=str(error))
                continue
            factors = np.array([1/height, 1/(width*thickness)])
            x, y = x*factors[0], y*factors[1]
            stripped["source_gauge"] = dict(height_mm=float(height), width_mm=float(width),
                                            B_mm=thickness, area_mm2=float(width*thickness),
                                            strain="clamp travel divided by body height")
            if "normalisation" in row:
                stripped["source_to_retained_scale_ratio"] = (factors/row["normalisation"]).tolist()
        cutoff = registry[lat][task]
        raw_psc_uat = task == "UT" and raw["study"] == "psc"
        try:
            if raw_psc_uat:
                evaluation = metrics(x, y, task)
                filt = None
            else:
                fx, filt = filter_force(t, x, cutoff)
                fy, _ = filter_force(t, y, cutoff)
                evaluation = metrics(fx, fy, task)
        except ValueError as error:
            stripped.update(calculation_status="unavailable_sampling_or_response", calculation_reason=str(error))
            continue
        stripped.update(calculation_status="calculated", filter=filt, recorded_samples=len(t),
                        last_time_s=float(t[-1]), peak_response=evaluation["peak_response"],
                        evaluation=evaluation)
        if raw_psc_uat:
            stripped["response_method"] = "native_raw"
        if task == "FT":
            endpoint = convergence_event(row["native_event"], t)
            processed = {"material_endpoint": endpoint, "methods": {"fixed": evaluation}}
            stripped["compact_tension"] = compact_tension_properties(
                processed, t, x, y, cutoff, row["geometry"], materials[lat]["M_MPa"],
                fcc if lat == "FCC" else None)
            if "time_match" in endpoint:
                stripped["compact_tension"]["event_time_match"] = endpoint
            if row["input_sha256"] in raw_fallbacks:
                baseline = stripped["compact_tension"]
                if baseline["status"] != "unavailable_negative_plastic_work":
                    raise ValueError("Archived fallback no longer matches the reviewed negative work")
                stripped["compact_tension"] = archive_raw_fallback(
                    baseline, evaluation["initial_work_fits"], x, y)
        elif raw["study"] == "al":
            stripped["curve"] = {"strain": x.tolist(), "stress_MPa": y.tolist()}
    return dict(study=raw["study"], materials=materials, records=result,
                method_scope="Fixed ordered conversion M by topology at 10 mm/20%; actual source a, B, W and tensile gross gauge.",
                selected_size=SELECTED, selected_mesh=MESH_SELECTED,
                comparisons=convergence_comparisons(result, raw["study"]),
                calculation_counts=dict(Counter(r.get("calculation_status",r["native_status"]) for r in result)),
                FT_endpoint_counts=dict(Counter(r.get("compact_tension",{}).get("status", "unavailable_curve")
                                               for r in result if r["task"] == "FT")))


def stiffness(row):
    return row.get("evaluation", {}).get("initial_work_fits", {}).get("0.4", {}).get("estimate")


UT_METRICS = {
    "Peak stress (MPa)": lambda r: r.get("peak_response"),
    "Initial modulus (MPa)": stiffness,
    "Ductility (%)": lambda r: (100*r["evaluation"]["ductility25_coordinate"]
                               if r.get("evaluation", {}).get("ductility25_observed") else None),
    "Work of fracture (MJ/m³)": lambda r: r.get("evaluation", {}).get("work1"),
}
FT_METRICS = {
    "Initiation toughness, $K_J$ (MPa√m)": lambda r: r.get("compact_tension", {}).get("KJ_MPa_sqrt_m"),
    "Initiation $J$ (N/mm)": lambda r: r.get("compact_tension", {}).get("J_N_per_mm"),
    "Initial load stiffness (N/mm)": stiffness,
    "Initiation load (N)": lambda r: r.get("compact_tension", {}).get("force_N"),
}


def convergence_comparisons(records, study):
    if study == "al":
        return []
    output = []
    for lat in SELECTED:
        for task, functions in (("UT", UT_METRICS), ("FT", FT_METRICS)):
            family = [r for r in records if r["topology"] == lat and r["task"] == task]
            if study == "psc":
                selected, reference = SELECTED[lat], 40
                key = "nx"
            else:
                selected = MESH_SELECTED[lat][task]
                reference = selected.rsplit("-",1)[0]+"-25"
                key = "configuration"
            cases = {r[key]:r for r in family}
            for label, getter in functions.items():
                chosen, finest = getter(cases[selected]), getter(cases[reference])
                if chosen is not None and finest is not None and finest != 0:
                    difference = 100*(chosen/finest-1)
                else:
                    difference = None
                output.append(dict(topology=lat, task=task, metric=label, selected=selected,
                                   reference=reference, selected_value=chosen, reference_value=finest,
                                   selected_relative_to_reference_percent=difference))
    return output



def apply_rate_replay(report, evidence_path, native_path):
    """Replace one archived negative-work result with its qualified rerun."""
    evidence = read_json(evidence_path)
    if (evidence["prior_native"]["sha256"] != sha(native_path)
            or evidence["registry_sha256"] != sha(REGISTRY)):
        raise ValueError("Rate replay is bound to different inputs")
    row, = [row for row in report["records"]
            if row["input_sha256"] == evidence["source"]["source_sha256"]]
    old = row["compact_tension"]
    event = evidence["event"]
    selected = evidence["selected_method"]
    native_only = selected is None and report["study"] == "psc"
    if native_only:
        # Use the measured dense history if only its decimation is sensitive.
        selected = "approved_butterworth"
        sensitivity = evidence["response_comparisons"]["40kHz_native+approved_butterworth"][
            "last_intact_native_field_time"]
        required = {"rerun_peak_fit": {"0.3", "0.4", "0.5"},
                    "archive_fixed": {"0.3", "0.4", "0.5"}, "exact_1N": {"exact_1N"}}
        if ({scheme: set(fits) for scheme, fits in sensitivity.items()} != required
                or not all(value["Apl_N_mm"] > 0 and value["valid_plastic_area"]
                           for fits in sensitivity.values() for value in fits.values())):
            raise ValueError("Native dense PSC evidence remains work-partition sensitive")
    result = evidence["qualification"].get(selected, {})
    point = result.get("native_refit_40pct")
    if (old["status"] != "unavailable_negative_plastic_work"
            or event["status"] != "qualified_observed_event"
            or (not native_only and not result.get("positive_work_robust_across_stiffness_windows_and_sampling"))
            or not point or not point["valid_plastic_area"]
            or point["modulus_MPa"] != old["modulus_MPa"]
            or sha(evidence["delivered_input"]) != evidence["delivered_input_sha256"]):
        raise ValueError("Rate replay lacks compatible initiation evidence")
    row["compact_tension"] = {
        **old, **point, "status": "calculated",
        "response_method": selected, "event_time_s": event["prev_native_history"]["time_s"],
        "minimum_k0_for_nonnegative_area_N_per_mm": point["force_N"]**2/(2*point["work_N_mm"]),
    }
    row["compact_tension"].pop("stiffness_fit_end_index")
    row["compact_tension_replay"] = {
        "comparison": str(evidence_path.resolve()), "comparison_sha256": sha(evidence_path),
        "input": evidence["delivered_input"], "input_sha256": evidence["delivered_input_sha256"],
        "odb": evidence["delivered_odb"], "odb_sha256": evidence["delivered_odb_sha256"],
        "archive_diagnostic": old,
        "qualification": "native_dense_history" if native_only else "native_and_resampled",
    }


def apply_first_damage_evidence(report, evidence_path, materials_path):
    """Add the observed first-damage indices for the four PSC exceptions."""
    evidence = read_json(evidence_path)
    if (report["study"] != "psc" or evidence["registry_sha256"] != sha(REGISTRY)
            or evidence["material_sha256"] != sha(materials_path)):
        raise ValueError("PSC first-damage evidence uses different processing inputs")
    for row in report["records"]:
        if row["task"] != "FT":
            continue
        key = f'psc-{row["topology"]}-{row["nx"]}'
        if key not in evidence["cases"]:
            continue
        control = evidence["cases"][key]
        archived = row["compact_tension"]
        point = control["first_damage_mechanics"]
        section_strain = control["first_damage"]["LE11_at_matching_section"]
        if (control["input_sha256"] != row["input_sha256"]
                or control["odb_fingerprint"] != row["sampled_odb_fingerprint"]
                or archived["status"] != "unqualified_observed_event"
                or not point["valid_plastic_area"]
                or not section_strain or not all(value < 0 for value in section_strain)):
            raise ValueError("PSC first-damage evidence does not match " + key)
        row["compact_tension"] = dict(point, status="calculated_first_damage_away_from_notch")
        row["compact_tension_replay"] = {
            "comparison": str(evidence_path.resolve()), "comparison_sha256": sha(evidence_path),
            "archive_diagnostic": archived, "first_damage": control["first_damage"],
        }


def apply_fcc_full_edge_evidence(report, evidence_path, materials_path):
    """Use one consistent C5 boundary condition and preserve the archive comparison."""
    evidence = read_json(evidence_path)
    sources = evidence_path.with_name("paired-source.json")
    if (report["study"] != "mesh" or evidence["paired_source_sha256"] != sha(sources)
            or evidence["registry_sha256"] != sha(REGISTRY)
            or evidence["material_catalogue_sha256"] != sha(materials_path)):
        raise ValueError("FCC full-edge evidence uses different processing inputs")
    expected = {case["configuration"]: case for case in read_json(sources)["cases"]}
    if set(expected) != set(evidence["cases"]):
        raise ValueError("The seven FCC C5 cases must be processed together")
    for row in report["records"]:
        key = row.get("configuration")
        if row["task"] != "FT" or row["topology"] != "FCC" or key not in expected:
            continue
        control = evidence["cases"][key]
        source = expected[key]
        if (row["input_sha256"] != source["source_sha256"]
                or control["input_sha256"] != source["input_sha256"]
                or control["verified_rotation_frames"] < 2
                or control["maximum_rear_rotation_rad"] != 0.
                or row["compact_tension"] != control["original"]):
            raise ValueError("FCC full-edge evidence does not match " + key)
        point = control["boundary_full_result"]
        if point["status"] == "unqualified_observed_event":
            first_damage = control["first_observed_damage_mechanics"]
            if first_damage and first_damage["valid_plastic_area"]:
                point = dict(first_damage, status="calculated_first_damage_away_from_notch")
        row["compact_tension"] = point
        row["compact_tension_replay"] = {
            "comparison": str(evidence_path.resolve()), "comparison_sha256": sha(evidence_path),
            "archive_diagnostic": control["original"], "boundary_condition": "full_right_edge_UR3_zero",
            "first_damage": control["first_damage"], "odb": control["odb"],
        }


def plot_convergence(report, task, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

    psc = report["study"] == "psc"
    rows = [row for row in report["records"] if not psc or row["nx"] <= 36]
    if task == "UT":
        panels = {
            "Peak stress (MPa)": UT_METRICS["Peak stress (MPa)"],
            "Stiffness (GPa)": stiffness,
            r"Ductility, $\mathcal{D}$": lambda r: r.get("evaluation", {}).get("ductility25_coordinate"),
        }
        nrows, ncols, figsize = 1, 3, (13.4, 4.4)
    elif psc:
        panels = dict(list(FT_METRICS.items())[:2])
        nrows, ncols, figsize = 1, 2, (11.6, 4.1)
    else:
        panels = {
            **dict(list(FT_METRICS.items())[:2]),
            "Initiation load-line displacement (mm)":
                lambda r: r.get("compact_tension", {}).get("displacement_mm"),
            "Initiation load (kN)": FT_METRICS["Initiation load (N)"],
        }
        nrows, ncols, figsize = 2, 2, (11.6, 7.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
    styles = {None: "-", 1: "-", 2: "--", 5: ":"}
    for ax, (label, getter) in zip(axes.ravel(), panels.items()):
        scale = .001 if label in ("Stiffness (GPa)", "Initiation load (kN)") else 1.
        shape_preserving = psc and (task == "FT" or label.startswith("Ductility"))
        for lat in SELECTED:
            subset = [r for r in rows if r["task"] == task and r["topology"] == lat]
            for coarse in ([None] if psc else [1, 2, 5]):
                series = [(r["nx"] if psc else int(r["configuration"].split("-")[2]), getter(r))
                          for r in subset if psc or int(r["configuration"].split("-")[1]) == coarse]
                x, y = np.asarray(sorted((a, np.nan if b is None else b) for a, b in series)).T
                y *= scale
                observed = np.flatnonzero(np.isfinite(y))
                ax.plot(x[observed], y[observed], color=COLOR[lat],
                        linestyle="none", marker="o", ms=3.8)
                # Break the guide where the underlying observation is unavailable.
                for group in np.split(observed, np.flatnonzero(np.diff(observed) > 1) + 1):
                    if len(group) >= 3:
                        xs = np.linspace(x[group[0]], x[group[-1]], 100)
                        if shape_preserving:
                            guide = PchipInterpolator(x[group], y[group])
                        else:
                            guide = Akima1DInterpolator(x[group], y[group], method="makima")
                        ax.plot(xs, guide(xs), color=COLOR[lat], linestyle=styles[coarse])
                    elif len(group) == 2:
                        ax.plot(x[group], y[group], color=COLOR[lat], linestyle=styles[coarse])
                selected = SELECTED[lat] if psc else int(MESH_SELECTED[lat][task].rsplit("-",1)[1])
                selected_coarse = int(MESH_SELECTED[lat][task].split("-")[1])
                if (psc or coarse == selected_coarse) and selected in x:
                    v = y[list(x).index(selected)]
                    if np.isfinite(v):
                        ax.scatter(selected, v, edgecolor=COLOR[lat], facecolor="white",
                                   marker="D", zorder=4, s=52, linewidth=1.4)
        ax.set_ylabel(label)
        ax.set_xlabel("Unit cells in the $X$-direction" if psc else "Fine mesh divisions per strut (nominal)")
        ax.set_xticks(sorted(set(r["nx"] if psc else int(r["configuration"].split("-")[2]) for r in rows)))
        ax.grid(False)
    topology = [Line2D([], [], color=COLOR[lat], marker="o", markersize=4, label=LABEL[lat])
                for lat in SELECTED]
    selected_handle = Line2D([], [], linestyle="none", marker="D", markersize=6, color="0.25",
                             markerfacecolor="white", label="Selected model")
    fig.legend(handles=[*topology, selected_handle], loc="outside upper center", frameon=False, ncol=5)
    if not psc:
        coarse_labels = {1: "1/strut", 2: "2/strut", 5: "5/strut"}
        full_edge = task == "FT" and any(r.get("compact_tension_replay", {}).get("boundary_condition")
                                        == "full_right_edge_UR3_zero" for r in rows)
        if full_edge:
            coarse_labels[5] = "5/strut (FCC: full edge)"
        coarse_handles = [Line2D([], [], color="0.3", linestyle=styles[c], label=coarse_labels[c])
                          for c in (1, 2, 5)]
        axes[0, -1].legend(handles=coarse_handles, title="Coarse mesh", frameon=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "replay"))
    parser.add_argument("--study", choices=("mesh", "psc", "al"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scratch", type=Path, help="Local directory for one removable ODB copy")
    parser.add_argument("--replace-result-sha256", help="Replace this exact existing convergence receipt")
    parser.add_argument("--raw-fallback-evidence", type=Path,
                        help="Reviewed set of archive input SHA-256 values eligible for raw fallback")
    parser.add_argument("--rate-comparison", "--mesh-rate-comparison", type=Path,
                        action="append", default=[], help="Qualified rate-rerun receipt; repeat for multiple cases")
    parser.add_argument("--psc-first-damage-evidence", type=Path)
    parser.add_argument("--mesh-fcc-full-edge-evidence", type=Path)
    args = parser.parse_args()
    base = args.root/"audit/production/convergence"
    base.mkdir(parents=True, exist_ok=True)
    raw = base/(args.study+"-native.json")
    if args.action == "extract":
        if raw.is_file():
            print("Existing native evidence:", raw)
        else:
            collect(args.root, args.study, raw, args.scratch)
    else:
        material = args.root/MATERIALS
        fallbacks = ()
        if args.raw_fallback_evidence and args.study != "al":
            evidence = read_json(args.raw_fallback_evidence)
            if (evidence["registry_sha256"] != sha(REGISTRY)
                    or evidence["source_result_receipts"][args.study]["native_sha256"] != sha(raw)):
                raise ValueError("Reviewed raw-fallback evidence requires the same filter and native histories")
            fallbacks = {case["archived_input_sha256"] for case in evidence["archived_cases"]
                         if case["study"] == args.study and case["proposed_raw_fallback"]}
        report = calculate(read_json(raw), read_json(material)["materials"], fallbacks)
        for evidence in args.rate_comparison:
            apply_rate_replay(report, evidence, raw)
        if args.psc_first_damage_evidence:
            apply_first_damage_evidence(report, args.psc_first_damage_evidence, material)
        if args.mesh_fcc_full_edge_evidence:
            apply_fcc_full_edge_evidence(report, args.mesh_fcc_full_edge_evidence, material)
        report["comparisons"] = convergence_comparisons(report["records"], report["study"])
        report["FT_endpoint_counts"] = dict(Counter(
            row.get("compact_tension", {}).get("status", "unavailable_curve")
            for row in report["records"] if row["task"] == "FT"))
        report.update(native_sha256=sha(raw), materials_sha256=sha(material), registry_sha256=sha(REGISTRY))
        if fallbacks:
            report["raw_fallback_evidence"] = dict(path=str(args.raw_fallback_evidence.resolve()),
                                                   sha256=sha(args.raw_fallback_evidence))
        path = base/(args.study+"-results.json")
        payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
        if args.replace_result_sha256 is None:
            _write_immutable_targets({path: payload})
        else:
            if not path.is_file() or sha(path) != args.replace_result_sha256.lower():
                raise ValueError("Existing convergence receipt differs from reviewed version")
            path.write_text(payload, encoding="utf-8")
        if args.study != "al":
            for task in ("UT", "FT"):
                plot_convergence(report, task, base/f"{args.study}-{task}-convergence.png")
        print(report["calculation_counts"], report["FT_endpoint_counts"], flush=True)


if __name__ == "__main__":
    main()
