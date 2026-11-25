"""Read-only mesh, periodic size and aluminium tensile benchmark replay."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from resources.calculations import filter_force
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


def calculate(raw, materials):
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
        try:
            fx, filt = filter_force(t, x, cutoff)
            fy, _ = filter_force(t, y, cutoff)
            filtered = metrics(fx, fy, task)
        except ValueError as error:
            stripped.update(calculation_status="unavailable_sampling_or_response", calculation_reason=str(error))
            continue
        stripped.update(calculation_status="calculated", filter=filt, recorded_samples=len(t),
                        last_time_s=float(t[-1]), peak_response=filtered["peak_response"],
                        evaluation=filtered)
        if task == "FT":
            event = row["native_event"]
            endpoint = dict(status=event["status"])
            if event["status"] == "qualified_observed_event":
                point = event["prev_native_history"]
                endpoint.update(native_index=point["index"], time_s=point["time_s"])
            processed = {"material_endpoint": endpoint, "methods": {"fixed": filtered}}
            stripped["compact_tension"] = compact_tension_properties(
                processed, t, x, y, cutoff, row["geometry"], materials[lat]["M_MPa"],
                fcc if lat == "FCC" else None)
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


def plot_convergence(report, task, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psc = report["study"] == "psc"
    rows = report["records"]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0), constrained_layout=True)
    for ax, (label, getter) in zip(axes.ravel(), (UT_METRICS if task=="UT" else FT_METRICS).items()):
        for lat in SELECTED:
            subset = [r for r in rows if r["task"] == task and r["topology"] == lat]
            for coarse in ([None] if psc else [1, 2, 5]):
                coords = [(r["nx"] if psc else int(r["configuration"].split("-")[2]), getter(r))
                          for r in subset if psc or int(r["configuration"].split("-")[1]) == coarse]
                valid = sorted((a, np.nan if b is None else b) for a,b in coords)
                if valid:
                    x, y = np.asarray(valid).T
                    ax.plot(x, y, color=COLOR[lat],
                            label=LABEL[lat] if psc or coarse == 1 else None,
                            linestyle={None:"-",1:"-",2:"--",5:":"}[coarse], marker="o", ms=3.8)
                    selected = SELECTED[lat] if psc else int(MESH_SELECTED[lat][task].rsplit("-",1)[1])
                    selected_coarse = int(MESH_SELECTED[lat][task].split("-")[1])
                    if (psc or coarse == selected_coarse) and selected in x:
                        v = y[list(x).index(selected)]
                        if np.isfinite(v):
                            ax.scatter(selected, v, edgecolor=COLOR[lat], facecolor="white",
                                       marker="D", zorder=4, s=52, linewidth=1.4)
        ax.set_ylabel(label)
        ax.set_xlabel("Unit cells in the $X$-direction" if psc else "Fine mesh divisions per strut (nominal)")
        ax.grid(alpha=.22)
    from matplotlib.lines import Line2D
    handles, labels = axes[0, 0].get_legend_handles_labels()
    selected_handle = Line2D([], [], linestyle="none", marker="D", markersize=6, color="0.25",
                             markerfacecolor="white", label="Selected model")
    fig.legend(handles=[*handles, selected_handle], labels=[*labels, "Selected model"],
               loc="outside upper center", frameon=False, ncol=5)
    if not psc:
        handles = [Line2D([], [], color="0.3", linestyle=ls, label=f"{coarse}/strut")
                   for coarse, ls in ((1,"-"),(2,"--"),(5,":"))]
        axes[0, 1].legend(handles=handles, title="Coarse mesh", frameon=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "replay"))
    parser.add_argument("--study", choices=("mesh", "psc", "al"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scratch", type=Path, help="Local directory for one removable ODB copy")
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
        report = calculate(read_json(raw), read_json(material)["materials"])
        report.update(native_sha256=sha(raw), materials_sha256=sha(material), registry_sha256=sha(REGISTRY))
        path = base/(args.study+"-results.json")
        _write_immutable_targets({path:json.dumps(report, indent=2, allow_nan=False)+"\n"})
        if args.study != "al":
            for task in ("UT", "FT"):
                plot_convergence(report, task, base/f"{args.study}-{task}-convergence.png")
        print(report["calculation_counts"], report["FT_endpoint_counts"], flush=True)


if __name__ == "__main__":
    main()
