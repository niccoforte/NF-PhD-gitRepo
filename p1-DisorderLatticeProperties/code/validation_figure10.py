"""Stage and install the author-approved fixed-cutoff Figure 10 comparison."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops
from scipy.integrate import trapezoid

from validation_calculations import filter_force
from validation_comparison import paper_exports, read_rows, recalculated_records, reference_lookup, write_rows
from validation_results import geometry_and_ki, sha


METHOD = "Hsieh_initial_triangle_native"
REGISTRY = Path(__file__).with_name("response_filter_cutoffs.json")
FIGURES = tuple("figure10." + ext for ext in ("png", "pdf", "svg"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def verify(entry):
    path = Path(entry["path"])
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Source fingerprint changed: {path}")
    return path


def identity(row):
    return row["topology"], float(row["l_mm"]), float(row["relative_density"])


def result(u, force, row, geom):
    stiffness = float(row["initial_stiffness_N_per_mm"])
    modulus = float(row["modulus_MPa"])
    applied_work = float(trapezoid(force, u))
    area = applied_work - float(force[-1])**2/(2*stiffness)
    ligament = geom.W-geom.ai[0]
    eta = 2 + .522*ligament/geom.W
    ki = float(row["KI_MPa_sqrt_m"])*float(force[-1])/float(row["force_N"])
    kj = math.sqrt(ki**2 + eta*area*modulus/(geom.B*ligament*1000)) if area >= 0 else None
    return dict(displacement_mm=float(u[-1]), force_N=float(force[-1]),
                loading_work_Nmm=applied_work, Apl_Nmm=area, KI_MPa_sqrt_m=ki, KJ_MPa_sqrt_m=kj)


def transfer(row, geometry, case, event, audit, cutoff, old_filtered):
    case_id = row["case_id"]
    native_root = audit / "new-validation/results/audit" if case_id.startswith("new:") else audit
    odb = Path(case["odb"])
    curve_path = native_root / "history" / odb.parent / odb.stem / "native_curve.csv"
    deck_path = audit.parent / case["inp"]
    if (sha(curve_path) != row["curve_csv_sha256"] or sha(deck_path) != row["source_inp_sha256"]
            or event["source_sha256"] != row["source_odb_sha256"]
            or case["odb_sha256"] != row["source_odb_sha256"]):
        raise ValueError(f"Native event or archived files do not match {case_id}")
    if (int(event["prev_native_index"]) != int(row["native_index"])
            or event["undamaged_prefix_verified"] != "True" or event["hash_unchanged"] != "True"
            or not float(event["prevmaxLE11"]) < .1105 <= float(event["crossmaxLE11"])
            or not 0 <= float(row["KE_IE"]) <= .05
            or geometry["value_status"] != "available_derived_assumptions"):
        raise ValueError(f"Native event eligibility not established for {case_id}")
    deck = deck_path.read_text(encoding="utf-8-sig")
    section = re.search(r"^\*Beam Section,[^\n]*\bsection=RECT\s*\n\s*([\d.eE+-]+)", deck, re.I | re.M)
    geom, ki = geometry_and_ki(case, float(row["force_N"]))
    if (not section or not math.isclose(float(section[1]), geom.B, rel_tol=5e-6)
            or not math.isclose(ki, float(row["KI_MPa_sqrt_m"]), rel_tol=1e-10)):
        raise ValueError(f"Geometry/thickness differs from the archived calculation for {case_id}")
    curve = read_rows(curve_path)
    lo, hi = int(event["prev_native_index"]), int(event["cross_native_index"])
    for i, tm in ((lo, "prevtime_s"), (hi, "crosstime_s")):
        if abs(float(curve[i]["time_s"])-float(event[tm])) > 2e-7:
            raise ValueError(f"Unmatched bracket timestamp for {case_id}")
    time, u, force = (np.array([float(value[k]) for value in curve[:lo+1]])
                      for k in ("time_s", "U2_mm", "RF2_N"))
    if (time[-1] != float(row["endpoint_time_s"]) or force[-1] != float(row["force_N"])
            or len(time) < 13):
        raise ValueError(f"Native event or transfer support differs for {case_id}")
    raw = result(u, force, row, geom)
    for k in ("loading_work_Nmm", "Apl_Nmm", "KJ_MPa_sqrt_m"):
        if row[k] and not math.isclose(raw[k], float(row[k]), rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"Raw replay differs for {case_id}: {k}")
    uf, info = filter_force(time, u, cutoff)
    pf, _ = filter_force(time, force, cutoff)
    filtered = result(uf, pf, row, geom)
    sensitivity_areas = []
    for factor in (.9, 1.1):
        ux, _ = filter_force(time, u, cutoff*factor)
        px, _ = filter_force(time, force, cutoff*factor)
        sensitivity_areas.append(result(ux, px, row, geom)["Apl_Nmm"])
    frame_spacing = hi-lo
    # Comparable earlier sample spacing is a sensitivity only. The event is fixed above.
    earlier = result(uf[:-frame_spacing], pf[:-frame_spacing], row, geom)
    available = filtered["KJ_MPa_sqrt_m"] is not None
    raw_fallback = not available and raw["KJ_MPa_sqrt_m"] is not None and row["value_status"].startswith("available")
    value = dict(row, **(raw if raw_fallback else filtered))
    value.update(response_processing="original_unfiltered_native_raw_fallback" if raw_fallback else "causal_Butterworth_paired_fixed_topology",
                 filter_applied=not raw_fallback, filter_cutoff_Hz=cutoff, filter_natural_frequency_Hz="",
                 filter_status="explicit_raw_fallback_filtered_area_negative" if raw_fallback else "applied_causal_Butterworth_fixed_topology",
                 filter_order=2, filter_zero_phase=False, filter_startup="DC", filter_native_prefix_samples=len(time),
                 saved_Nyquist_Hz=info["nyquist_Hz"], filter_max_time_remap_s=info["max_time_remap_s"],
                 filtered_Apl_Nmm=filtered["Apl_Nmm"], filtered_KJ_status="available" if available else "unavailable_negative_plastic_area",
                 fallback_reason="Approved native raw fallback; fixed-cutoff plastic work is negative" if raw_fallback else "",
                 value_status="available_source_work_convention" if available or raw_fallback else "unavailable_negative_plastic_area",
                 quality_note="Native intact event and original compliance and conversion modulus retained. "
                 + ("The plotted native raw fallback is disclosed; fixed-cutoff signed plastic work is negative. " if raw_fallback else
                    "Paired causal second-order Butterworth with DC startup at the approved topology cutoff. ")
                 + "No ASTM qualification claimed.")
    proof = dict(case_id=case_id, topology=row["topology"], l_mm=row["l_mm"], relative_density=row["relative_density"],
                 source_inp=str(deck_path), source_inp_sha256=row["source_inp_sha256"], source_odb_sha256=row["source_odb_sha256"],
                 source_curve=str(curve_path), source_curve_sha256=row["curve_csv_sha256"],
                 event_lower_index=lo, event_upper_index=hi, event_lower_s=time[-1], event_upper_s=float(event["crosstime_s"]),
                 previous_max_LE11=event["prevmaxLE11"], crossing_max_LE11=event["crossmaxLE11"],
                 W_mm=geom.W, B_mm=geom.B, a_mm=geom.ai[0], modulus_MPa=row["modulus_MPa"],
                 compliance_N_per_mm=row["initial_stiffness_N_per_mm"],
                 native_prefix_samples=len(time), sampling_Hz=info["fs_Hz"], nyquist_Hz=info["nyquist_Hz"],
                 max_gap_over_median=float(np.max(np.diff(time))/np.median(np.diff(time))),
                 time_grid_max_remap_s=info["max_time_remap_s"], cutoff_Hz=cutoff,
                 selected_processing=value["response_processing"],
                 endpoint_KE_IE=float(row["KE_IE"]), startup_window_max_KE_IE=row.get("startup_window_max_KE_IE"),
                 filtered_plastic_work_fraction=filtered["Apl_Nmm"]/filtered["loading_work_Nmm"],
                 cutoff_minus_10pct_Apl_Nmm=sensitivity_areas[0], cutoff_plus_10pct_Apl_Nmm=sensitivity_areas[1],
                 earlier_bracket_spacing_samples=frame_spacing, earlier_sample_Apl_Nmm=earlier["Apl_Nmm"],
                 **{f"{label}_{key}": metric for label, values in (("raw", raw), ("filtered", filtered))
                    for key, metric in values.items()})
    limits = {"force_N": 1., "displacement_mm": 1., "loading_work_Nmm": 2.}
    if old_filtered:
        proof["archive_baseline_cutoff_Hz"] = float(old_filtered["filter_cutoff_Hz"])
        proof["archive_baseline_KJ_MPa_sqrt_m"] = old_filtered["KJ_MPa_sqrt_m"]
        changes = {key: 100*(filtered[key]/float(old_filtered[key])-1) for key in limits}
        proof.update({f"change_{key}_percent": change for key, change in changes.items()})
        proof["outside_baseline_response_limits"] = any(abs(changes[key]) > limits[key] for key in limits)
    return value, proof


def render(stage, audits, results, supplement):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    reference = audits / "benchmark/fig10-digitised.csv"
    fig8 = audits / "benchmark/fig8-original-digitisation.csv"
    provenance = audits / "benchmark/fig8-source-provenance.json"
    source = read_json(provenance)
    ref = read_rows(reference)
    inputs = {name: binding(path) for name, path in (("reference", reference), ("original_fig8", fig8),
              ("original_fig8_provenance", provenance), ("selected_recalculation", results))}
    save = Figure.savefig

    def save_figure(self, target, *args, **kwargs):
        if Path(target).name in FIGURES:
            save(self, target, *args, **kwargs)

    def reference_outlines(self, x, y, **kwargs):
        for key in ("yerr", "xerr", "capsize", "elinewidth", "ecolor", "capthick"):
            kwargs.pop(key, None)
        return self.plot(x, y, **kwargs)

    with patch.object(Figure, "savefig", save_figure), patch.object(Axes, "errorbar", reference_outlines):
        paper_exports(recalculated_records(read_rows(results), reference_lookup(ref)), read_rows(fig8), ref,
                      source["material"], stage, inputs, source["published_pdf_sha256"], supplement_path=supplement)
    return inputs


def prepare(audit, stage):
    paper = audit / "paper"
    stage.mkdir(parents=True, exist_ok=True)
    recipe = read_json(paper / "approved-figure-recipe.json")
    baseline = {name: sha(paper / name) for name in (*recipe["figures"], "points.csv", "captions.txt", "metadata.json",
                "approved-figure-recipe.json", "figure10-finalization.json", "qa.json", "figure10-results.csv", "figure10-supplement.json")}
    if any(baseline[name] != digest for name, digest in recipe["figures"].items()):
        raise ValueError("Approved baseline figure changed before staging")
    verify(recipe["figure10_display_source"])
    prior = read_json(paper / "figure10-finalization.json")
    published_before = read_rows(paper / "points.csv")
    expected = {identity(r): r for r in published_before if r["figure"] == "10" and r["series"] == "Simulation"}
    current_raw = read_rows(verify(recipe["approved_display_source"]))
    baseline_results = verify(prior.get("archive_baseline_results", prior["source_filtered_results"]))
    modal_baseline = {r["case_id"]: r for r in read_rows(baseline_results)
                      if r["method"] == METHOD and r.get("filter_status") == "applied_causal_Butterworth_min_2fn_fs6"}
    for row in read_rows(verify(recipe["forward_calculation_source"])):
        # A previously archived 2fn result is the same baseline if it lies below the archive cap.
        if (row["method"] == METHOD and row.get("filter_status") == "applied_causal_Butterworth_2fn"
                and float(row["filter_cutoff_Hz"]) <= float(row["saved_Nyquist_Hz"])/3):
            modal_baseline.setdefault(row["case_id"], row)
    archive = (audit.parent, audit / "new-validation/results")
    index = {r["case_id"]: r for folder in archive for r in read_rows(folder / "case-index.csv")}
    events = {r["case_id"]: r for folder in archive for r in read_rows(folder / "audit/failure/failure-initiation.csv")}
    energy = {r["case_id"]: r for r in current_raw if r["method"] == "native_energy_eta" and r["endpoint"] == "criterion_last_intact"}
    registry = read_json(REGISTRY)["cutoffs_hz"]
    rows, proofs = [], []
    for row in current_raw:
        if row["method"] == METHOD and identity(row) in expected:
            key = row["case_id"]
            previous_basis = expected[identity(row)]["evaluation_basis"]
            row, proof = transfer(row, energy[key], index[key], events[key], audit,
                                  float(registry[row["topology"]]["FT"]), modal_baseline.get(key))
            if (previous_basis == "nonlinear_recovery" and proof["filtered_Apl_Nmm"] >= 0
                    and min(proof[k] for k in ("cutoff_minus_10pct_Apl_Nmm", "cutoff_plus_10pct_Apl_Nmm",
                                              "earlier_sample_Apl_Nmm")) < 0):
                # Replacing a recovery estimate requires a stable sign near the measured bracket.
                # Record the fixed-cutoff numerical value in the transfer table either way.
                proof["selected_processing"] = "retained_recovery_fragile_native_plastic_work"
                row.update(KJ_MPa_sqrt_m=None, value_status="unavailable_fragile_plastic_work_transfer",
                           filtered_KJ_status="numerical_only_sensitive_plastic_work",
                           quality_note=row["quality_note"] + " Near-bracket sensitivity reverses the plastic-work sign; the prior recovery index is retained in the display.")
            proofs.append(proof)
        rows.append(row)
    if len({identity(r) for r in proofs}) != len(proofs) or len(proofs) != 16:
        raise ValueError("Unexpected set of attested Figure 10 histories")
    results = stage / "figure10-results.csv"
    write_rows(results, rows)
    supplement = read_json(verify(recipe["figure10_supplement"]))
    supplement.update(source_results=str(results), source_results_sha256=sha(results),
                      display_policy="A verified native value takes precedence. Retained recovery values and predictions are disclosed in the caption and point records.")
    staged_supplement = stage / "figure10-supplement.json"
    write_json(staged_supplement, supplement)
    inputs = render(stage, audit, results, staged_supplement)
    new_points = [r for r in read_rows(stage / "points.csv") if r["figure"] == "10"]
    old_refs = [r for r in published_before if r["figure"] == "10" and r["series"] == "Hsieh et al."]
    new_refs = [r for r in new_points if r["series"] == "Hsieh et al."]
    if old_refs != new_refs or len(new_refs) != 18:
        raise ValueError("Reference Figure 10 coordinates or provenance changed")
    new_study = [r for r in new_points if r["series"] == "Simulation"]
    if {identity(r) for r in new_study} != set(expected) or not all(r["plotted"] == "True" for r in new_study):
        raise ValueError("Figure 10 does not have the agreed set of 18 visible Current study points")
    by_identity = {identity(r): r for r in proofs}
    for row in new_study:
        proof = by_identity.get(identity(row))
        if proof:
            row.update(**{key: proof[key] for key in ("cutoff_Hz", "filtered_Apl_Nmm", "filtered_KJ_MPa_sqrt_m",
                        "selected_processing", "source_curve_sha256", "event_lower_index", "event_upper_index")})
    write_rows(stage / "points.csv", [r for r in published_before if r["figure"] != "10"] + new_points)
    count = {name: sum(r["evaluation_basis"] == name for r in new_study)
             for name in ("original_fracture", "nonlinear_recovery", "prediction")}
    fallbacks = [r for r in proofs if r["selected_processing"] == "original_unfiltered_native_raw_fallback"]
    count.update(raw_native=len(fallbacks), filtered_native=count["original_fracture"]-len(fallbacks), total_current_study=18)
    caption = figure_caption(count, new_study, fallbacks, proofs)
    write_rows(stage / "figure10-transfer-checks.csv", proofs)
    frozen = read_json(paper / "metadata.json")
    generated = read_json(stage / "metadata.json")
    if any(generated[k] != frozen[k] for k in ("topology_colors", "display_labels", "marker_design")):
        raise ValueError("Approved marker or topology styling changed")
    frozen["captions"]["10"] = caption
    frozen["figure10_selected_inputs"] = inputs
    frozen["figure10_processing_contract"] = count
    write_json(stage / "metadata.json", frozen)
    captions = (paper / "captions.txt").read_text(encoding="utf-8")
    captions, n = re.subn(r"Figure 10\..*?(?=\n\nFigure |\Z)", lambda _: "Figure 10. " + caption + "\n", captions, flags=re.S)
    if n != 1:
        raise ValueError("Cannot surgically replace the Figure 10 caption")
    (stage / "captions.txt").write_text(captions, encoding="utf-8")
    with Image.open(paper / "figure10.png") as old, Image.open(stage / "figure10.png") as new:
        if new.size != old.size:
            raise ValueError("Figure 10 canvas size changed")
        pixels = ImageChops.difference(old.convert("RGB"), new.convert("RGB"))
        bounds = pixels.getbbox()
        if bounds and (bounds[1] < 90 or bounds[3] >= 900):
            raise ValueError(f"Figure 10 titles, legend, or note changed: {bounds}")
    evidence = {"baseline_files": baseline, "inputs": inputs, "cutoff_registry": binding(REGISTRY),
                "script": binding(Path(__file__).resolve()),
                "archive_baseline_results": binding(baseline_results),
                "cutoffs_FT_Hz": {k: v["FT"] for k, v in registry.items()}, "counts": count, "caption": caption,
                "filtered_negative_cases": [identity(r) for r in proofs if r["filtered_Apl_Nmm"] < 0],
                "near_bracket_sensitive_recovery_cases": [identity(r) for r in proofs
                      if r["selected_processing"] == "retained_recovery_fragile_native_plastic_work"],
                "cases_outside_archive_sensitivity_limits": [identity(r) for r in proofs if r.get("outside_baseline_response_limits")],
                "unchanged_canvas_and_static_regions": True, "changed_pixels_bounds": bounds}
    files = (*FIGURES, "points.csv", "captions.txt", "metadata.json", "figure10-results.csv", "figure10-supplement.json",
             "figure10-transfer-checks.csv")
    # Canonical references must resolve after the staging files are moved.
    supplement["source_results"] = str(paper / results.name)
    write_json(staged_supplement, supplement)
    evidence["inputs"]["selected_recalculation"]["path"] = str(paper / results.name)
    frozen["figure10_selected_inputs"] = evidence["inputs"]
    write_json(stage / "metadata.json", frozen)
    final_supplement_hash = sha(staged_supplement)
    for row in new_points:
        if row["evaluation_basis"] in ("nonlinear_recovery", "prediction"):
            row["source_input_sha256"] = final_supplement_hash
    write_rows(stage / "points.csv", [r for r in published_before if r["figure"] != "10"] + new_points)
    evidence["staged_files"] = {name: sha(stage / name) for name in files}
    write_json(stage / "prepared.json", evidence)
    print(json.dumps({k: evidence[k] for k in ("counts", "filtered_negative_cases", "cases_outside_archive_sensitivity_limits")}, indent=2))


def figure_caption(count, points, fallbacks, proofs):
    def describe(rows):
        return ", ".join(f"{float(r['l_mm']):g} mm/{float(r['relative_density']):.0%}" for r in rows)
    recovery = describe([r for r in points if r["evaluation_basis"] == "nonlinear_recovery"])
    predicted = describe([r for r in points if r["evaluation_basis"] == "prediction"])
    transient = next((r for r in proofs if identity(r) == ("hex", 10., .16)
                      and r.get("outside_baseline_response_limits")), None)
    return ("Dimensional initiation comparison at nominal relative densities 13%, 16%, and 20%. Hsieh's 3 mm values are "
            "directly digitised. The 10 mm reference uses the declared sqrt(10/3) length scaling. "
            f"{count['filtered_native']} Current study values use paired causal second-order Butterworth filtering with DC startup "
            "at the approved fixed C(T) cutoffs (triangular 266.667 Hz, kagome 291.667 Hz, hexagonal 302.252 Hz). "
            "The recorded initiation brackets, source compliance and conversion moduli are retained. "
            + (f"The hexagonal {describe(fallbacks)} values use the raw histories because their filtered initial-triangle areas are negative. " if fallbacks else "")
            + (f"Hexagonal {recovery} uses source-matched nonlinear-recovery equivalent toughness because the near-bracket sensitivity reverses the sign of the native plastic work. " if recovery else "")
            + (f"Hexagonal {predicted} are density and length predictions based on the two 3 mm nonlinear-recovery indices. " if predicted else "")
            + (f"The 10 mm/16% hexagonal record has a pre-initiation kinetic transient and its filtered work changes by {transient['change_loading_work_Nmm_percent']:+.2f}% from the archive baseline. " if transient else "")
            + "Case-specific provenance is recorded in the companion tables. Connecting lines guide the eye.")


def promote(audit, stage):
    paper = audit / "paper"
    evidence = read_json(stage / "prepared.json")
    review = read_json(stage / "visual-review.json")
    if not (review["png_review_passed"] and review["pdf_review_passed"] and all(
            review["hashes"][name] == evidence["staged_files"][name] for name in ("figure10.png", "figure10.pdf"))):
        raise ValueError("Staged Figure 10 image and PDF require matching visual review")
    for root, hashes in ((paper, evidence["baseline_files"]), (stage, evidence["staged_files"])):
        for name, digest in hashes.items():
            if sha(root / name) != digest:
                raise ValueError(f"Preflight fingerprint changed: {root / name}")
    verify(evidence["cutoff_registry"])
    verify(evidence["script"])
    recipe = read_json(paper / "approved-figure-recipe.json")
    recipe["figure10_authority"] = "Author-approved 23 November 2026 fixed-cutoff Figure 10 update"
    for key, name in (("figure10_display_source", "figure10-results.csv"), ("figure10_supplement", "figure10-supplement.json"),
                      ("figure10_transfer_checks", "figure10-transfer-checks.csv")):
        recipe[key] = {"path": str(paper / name), "sha256": evidence["staged_files"][name]}
    recipe["figure10_processing_contract"] = evidence["counts"]
    recipe["figure10_cutoff_registry"] = evidence["cutoff_registry"]
    recipe["figures"].update({name: evidence["staged_files"][name] for name in FIGURES})
    write_json(stage / "approved-figure-recipe.json", recipe)
    evidence.update(status="final_author_selected_display", visual_review=review, script=binding(Path(__file__).resolve()),
                    source_filtered_results=recipe["figure10_display_source"], final_results=recipe["figure10_display_source"],
                    supplement=recipe["figure10_supplement"], other_figures_unchanged=True,
                    final_figures={name: recipe["figures"][name] for name in FIGURES})
    write_json(stage / "figure10-finalization.json", evidence)
    qa = read_json(paper / "qa.json")
    qa["figure10_selection_20261123"] = review
    write_json(stage / "qa.json", qa)
    names = (*evidence["staged_files"], "qa.json", "figure10-finalization.json", "approved-figure-recipe.json")
    rollback = stage / "rollback"
    rollback.mkdir(exist_ok=False)
    for name in names:
        if (paper / name).exists():
            shutil.copy2(paper / name, rollback / name)
    installed = []
    try:
        for name in names:
            (stage / name).replace(paper / name)
            installed.append(name)
        for name, digest in {**recipe["figures"], **evidence["staged_files"]}.items():
            if sha(paper / name) != digest:
                raise RuntimeError(f"Installed output fingerprint differs: {name}")
    except Exception:
        for name in reversed(installed):
            backup = rollback / name
            if backup.exists():
                backup.replace(paper / name)
            else:
                (paper / name).replace(stage / name)
        raise
    print(json.dumps({"figure": str(paper / "figure10.png"), "counts": evidence["counts"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "promote"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "promote": promote}[args.action](args.audit.resolve(), args.stage.resolve())


if __name__ == "__main__":
    main()
