"""Finite-geometry work partitions for the validation audit, without FEA launch.

Measured load/unload work is a direct matched-state result. ALLIE-ALLSE is a
different, internally nonrecoverable energy quantity; absent decomposition
channels remain unknown. Neither is silently relabelled ASTM plastic area.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Abaqus also ships a namespace named resources. Resolve the maintained project
# package explicitly before importing its geometry/calibration helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


INTERNAL_TERMS = ("ALLPD", "ALLCD", "ALLAE", "ALLDMD", "ALLDC", "ALLFC")
SELECTED_INPUTS = ("3/0.16/Fracture-hex-80-per-1.inp", "10/0.2/Fracture-hex-60-per-1.inp",
                   "3/0.2/Fracture-tri-70-per-1.inp", "3/0.2/Fracture-kagome-40-per-1.inp",
                   "10/0.2/Fracture-tri-70-per-1.inp", "10/0.2/Fracture-kagome-40-per-1.inp",
                   "10/0.16/Fracture-tri-80-per-1.inp", "10/0.16/Fracture-kagome-50-per-1.inp",
                   "10/0.13/Fracture-tri-120-per-1.inp", "10/0.13/Fracture-kagome-80-per-1.inp",
                   "3/0.2/Fracture-hex-50-per-1.inp")


def paired_arrays(displacement, force):
    u, p = np.asarray(displacement, dtype=float), np.asarray(force, dtype=float)
    if u.ndim != 1 or len(u) < 2 or u.shape != p.shape or not np.isfinite(u).all() or not np.isfinite(p).all():
        raise ValueError("Need at least two finite paired displacement/force observations")
    return u, p


def distinct_history_times(arrays, *, allow_terminal_ambiguity=False):
    """Remove only identical repeated static history rows, retaining their indices."""
    reference = next(iter(arrays.values()))[:, 0]
    if any(not np.array_equal(value[:, 0], reference) for value in arrays.values()):
        raise ValueError("History channels do not share one time grid")
    removed = []
    for index in np.flatnonzero(np.diff(reference) <= 0)+1:
        if reference[index] != reference[index-1] or any(not np.array_equal(value[index], value[index-1]) for value in arrays.values()):
            if allow_terminal_ambiguity and reference[index] == reference[index-1]:
                beginning = int(index-1)
                while beginning and reference[beginning-1] == reference[index]: beginning -= 1
                removed = [row for row in removed if row < beginning]+list(range(beginning, len(reference)))
                break
            raise ValueError("Nonmonotone or differing repeated-time static history records")
        removed.append(int(index))
    keep = np.ones(len(reference), dtype=bool); keep[removed] = False
    return {name: value[keep] for name, value in arrays.items()}, removed


def work(displacement, force):
    """Signed trapezoidal work, Nmm for mm and N input; preserves unloading sign."""
    u, p = paired_arrays(displacement, force)
    return float(np.sum(.5*(p[1:]+p[:-1])*np.diff(u)))


def ct_rotation_factor(displacement_mm, W_mm, a_mm, half_pin_span_mm, half_gauge_span_mm):
    """ASTM E1820-08a A2.14-15 denominator: C_corrected=C_measured/factor.

    This geometrical compliance correction does not itself define an elastic
    recovery area for a geometrically nonlinear lattice specimen.
    """
    if not 0 < a_mm < W_mm or half_pin_span_mm <= 0 or half_gauge_span_mm <= 0:
        raise ValueError("Positive C(T) dimensions and 0<a<W required")
    u = np.asarray(displacement_mm, dtype=float)
    R, D, H = (W_mm+a_mm)/2, half_gauge_span_mm, half_pin_span_mm
    argument = (u/2+D)/np.hypot(D, R)
    if not np.isfinite(u).all() or np.any(abs(argument) > 1):
        raise ValueError("Rotation construction outside its geometrical domain")
    theta = np.arcsin(argument)-np.arctan2(D, R)
    factor = (H/R*np.sin(theta)-np.cos(theta))*(D/R*np.sin(theta)-np.cos(theta))
    if np.any(factor <= 0):
        raise ValueError("Rotation factor is nonpositive")
    return factor


def measured_recovery_partition(load_u, load_p, unload_u, unload_p, *, initial_stiffness=None):
    """Direct full load/release-force-unload area, with no linear triangle fit.

    Applies only to this source and peak state. Residual elastic self-stress and
    any new plasticity during unloading are part of the non-recovered area.
    """
    lu, lp = paired_arrays(load_u, load_p)
    uu, up = paired_arrays(unload_u, unload_p)
    force_scale = max(abs(lp[-1]), 1.)
    if abs(lu[0]) > 1e-5 or abs(lp[0]) > 1e-4*force_scale:
        raise ValueError("Full loading work requires the initial zero-displacement/zero-force state")
    if np.min(np.diff(lu)) < -1e-6 or np.max(np.diff(uu)) > 1e-6:
        raise ValueError("Loading must increase and unloading must decrease displacement")
    if abs(lu[-1]-uu[0]) > 1e-5 or abs(lp[-1]-up[0]) > 1e-4*force_scale:
        raise ValueError("Unload must begin at the same measured load/displacement state")
    if np.min(up) < -1e-4*force_scale or abs(up[-1]) > 1e-4*force_scale:
        raise ValueError("Unload must end at zero force without a reverse-loading tail")
    loading, recovered = work(lu, lp), -work(uu, up)
    result = {"loading_work_Nmm": loading, "measured_recovery_Nmm": recovered,
              "nonrecovered_work_Nmm": loading-recovered,
              "residual_displacement_mm": float(uu[-1]),
              "method": "matched_source_finite_geometry_load_unload",
              "is_certified_ASTM_Apl": False,
              "limit": "Measured finite-geometry hysteresis area; eta-to-J calibration/standard qualification is a separate assumption."}
    if initial_stiffness is not None:
        if initial_stiffness <= 0 or not np.isfinite(initial_stiffness):
            raise ValueError("Initial stiffness must be positive")
        result["initial_triangle_Nmm"] = float(lp[-1]**2/(2*initial_stiffness))
        result["initial_triangle_apparent_Apl_Nmm"] = loading-result["initial_triangle_Nmm"]
    return result


def internal_energy_partition(ALLIE, ALLSE, *, channels=None, excluded_terms=()):
    """Bound plastic dissipation from the documented Abaqus internal identity.

    IE-SE=PD+CD+AE+DMD+DC+FC. KE and viscous bulk damping are outside ALLIE.
    Unknown components cannot be zeroed. Nonnegative-contribution bounds require
    FC absent or known, since fluid-cavity energy can be signed.
    """
    channels = channels or {}
    excluded = set(excluded_terms)
    if not excluded <= set(INTERNAL_TERMS)-{"ALLPD"}:
        raise ValueError("Only documented nonplastic internal terms may be excluded")
    values = np.asarray([ALLIE, ALLSE], dtype=float)
    if not np.isfinite(values).all() or ALLIE < 0 or ALLSE < 0:
        raise ValueError("Finite nonnegative internal/recoverable energies required")
    total = float(ALLIE-ALLSE)
    known, unknown = {}, []
    for term in INTERNAL_TERMS:
        value = channels.get(term)
        if term in excluded:
            if value is not None and np.isfinite(value) and abs(value) > 1e-8:
                raise ValueError("Recorded energy contradicts source-card exclusion: "+term)
            known[term] = 0.
        elif value is not None and np.isfinite(value):
            known[term] = float(value)
        else:
            unknown.append(term)
    tolerance = 1e-6*max(ALLIE, ALLSE, 1.)
    if total < -tolerance:
        raise ValueError("Internal energy is below recoverable energy beyond rounding tolerance")
    result = {"internal_nonrecoverable_Nmm": total, "known_components_Nmm": known,
              "unknown_components": unknown, "is_ASTM_Apl": False,
              "identity": "ALLIE-ALLSE=ALLPD+ALLCD+ALLAE+ALLDMD+ALLDC+ALLFC"}
    if "ALLPD" in known:
        lower = upper = known["ALLPD"]
        status = "plastic_dissipation_recorded"
    elif "ALLFC" in unknown:
        lower = upper = None
        status = "unbounded_without_signed_fluid_cavity_component"
    else:
        upper = total-sum(value for name, value in known.items() if name != "ALLPD")
        lower = upper if not unknown or unknown == ["ALLPD"] else 0.
        status = "plastic_dissipation_identified" if lower == upper else "plastic_dissipation_upper_bound"
        if upper < -tolerance:
            raise ValueError("Known internal components exceed IE-SE")
    result.update(plastic_dissipation_lower_Nmm=lower, plastic_dissipation_upper_Nmm=upper, status=status)
    return result


def eta_energy_metrics(KI_MPa_sqrt_m, conversion_MPa, eta, B_mm, ligament_mm, area_Nmm):
    """Compute a labelled eta energy estimate; provenance/eligibility stays outside.

    Uses K(mm)=sqrt(1000)*K(m); a negative input area remains unavailable.
    """
    if any(not np.isfinite(value) or value <= 0 for value in (conversion_MPa, eta, B_mm, ligament_mm)):
        raise ValueError("Positive finite geometry/conversion parameters required")
    if not np.isfinite(KI_MPa_sqrt_m) or KI_MPa_sqrt_m < 0:
        raise ValueError("Nonnegative finite KI required")
    if area_Nmm is None or not np.isfinite(area_Nmm) or area_Nmm < 0:
        return {"J_eta_N_per_mm": None, "K_eta_MPa_sqrt_m": None, "available": False}
    Jel = 1000*KI_MPa_sqrt_m**2/conversion_MPa
    Jpl = eta*area_Nmm/(B_mm*ligament_mm)
    return {"Jel_N_per_mm": Jel, "Jpl_eta_N_per_mm": Jpl, "J_eta_N_per_mm": Jel+Jpl,
            "K_eta_MPa_sqrt_m": float(np.sqrt(conversion_MPa*(Jel+Jpl)/1000)),
            "available": True, "qualified_ASTM_J": False}


def initial_triangle_metrics(loading_work_Nmm, force_N, stiffness_N_mm, KI_MPa_sqrt_m,
                             conversion_MPa, eta, B_mm, ligament_mm):
    """Hsieh initial-slope triangle, adapted only to C(T)'s eta/calibration."""
    if not np.isfinite(stiffness_N_mm) or stiffness_N_mm <= 0:
        raise ValueError("Positive finite initial stiffness required")
    if not np.isfinite(loading_work_Nmm) or not np.isfinite(force_N) or force_N < 0:
        raise ValueError("Finite work and positive-opening force required")
    elastic = force_N**2/(2*stiffness_N_mm)
    signed = loading_work_Nmm-elastic
    calculated = eta_energy_metrics(KI_MPa_sqrt_m, conversion_MPa, eta, B_mm, ligament_mm, signed)
    return {"method": "Hsieh_initial_slope_triangle_CT_geometry_adaptation",
            "initial_stiffness_N_mm": stiffness_N_mm, "loading_work_Nmm": loading_work_Nmm,
            "elastic_triangle_Nmm": elastic, "signed_Apl_Nmm": signed,
            "Jpl_signed_N_per_mm": eta*signed/(B_mm*ligament_mm),
            "KI_MPa_sqrt_m": KI_MPa_sqrt_m,
            "KJ_MPa_sqrt_m": calculated["K_eta_MPa_sqrt_m"], "available": calculated["available"],
            "unavailable_reason": "Negative initial-triangle plastic area; not clipped or called valid toughness" if signed < 0 else None,
            "qualified_ASTM_J": False,
            "definition_limit": "Prescribed initial-slope partition; nonlinear recovery/PD are separate robustness calculations, not substitutions for this Hsieh definition"}


def early_elastic_stiffness_windows(curve, frames):
    """Report fixed small-strain windows as sensitivity, never as a chosen C0."""
    eligible = []
    for frame in frames:
        if float(frame["maximum_PEEQ"]) > 1e-10: break
        if float(frame["U2_mm"]) > 0: eligible.append(frame)
    records = []
    for threshold in (.001, .002, .003, .005):
        selected = [frame for frame in eligible if float(frame["maximum_tensile_LE11"]) <= threshold]
        if not selected: continue
        indices = [int(frame["history_index"]) for frame in selected]
        u = np.array([float(curve[index]["U2_mm"]) for index in indices])
        p = np.array([float(curve[index]["force_N"]) for index in indices])
        stiffness = float(np.dot(u, p)/np.dot(u, u))
        records.append({"maximum_tensile_LE11_window": threshold, "samples": len(u),
                        "last_U2_mm": float(u[-1]), "stiffness_through_origin_N_mm": stiffness,
                        "maximum_PEEQ_in_window": max(float(frame["maximum_PEEQ"]) for frame in selected),
                        "purpose": "Fixed elastic early-window sensitivity, not selected by agreement with Hsieh"})
    return records


def triangle_window_sensitivity(loading_work, force, metrics, windows):
    records = []
    for window in windows:
        triangle = initial_triangle_metrics(loading_work, force, window["stiffness_through_origin_N_mm"],
            metrics["KI_MPa_sqrt_m"], metrics["conversion_modulus_MPa"], metrics["eta"], metrics["B_mm"], metrics["W_mm"]-metrics["a_mm"])
        records.append(dict(window, signed_Apl_Nmm=triangle["signed_Apl_Nmm"],
                            KJ_MPa_sqrt_m=triangle["KJ_MPa_sqrt_m"], available=triangle["available"]))
    return records


def enrich_stiffness(args):
    """Recalculate saved numerical results with exact C0; no ODB/FEA needed."""
    from validation_diagnostics import sha
    root, out = Path(args.root), Path(args.out)
    moduli, stiffness, loads = available_moduli(root), available_stiffness(root), []
    for path in out.glob("QSload*/result.json"):
        result = json.loads(path.read_text()); directory = path.parent
        if sha(result["source_inp"]) != result["source_sha256"]: raise ValueError("Original source changed")
        curve, frames = read_csv(directory/"qs-load-curve.csv"), read_csv(directory/"qs-section-frames.csv")
        result["early_elastic_stiffness_windows"] = early_elastic_stiffness_windows(curve, frames)
        result["stiffness_sensitivity_curve_sha256"] = sha(directory/"qs-load-curve.csv")
        result["stiffness_sensitivity_fields_sha256"] = sha(directory/"qs-section-frames.csv")
        for point in result.get("criterion_points", {}).values():
            metrics = static_point_metrics(root, result, point, moduli[result["case_id"]]["E_x_MPa"])
            point.update(metrics)
            point["Hsieh_initial_triangle_window_sensitivity"] = triangle_window_sensitivity(
                point["integrated_work_Nmm"], point["force_N"], metrics, result["early_elastic_stiffness_windows"])
        save_result_versioned(path, result); loads.append(result)
    unload_count = 0
    for path in out.glob("QSunload*/result.json"):
        result = json.loads(path.read_text()); initial = stiffness.get(result["case_id"])
        if not initial: continue
        if initial["source_sha256"] != result["source_sha256"]: raise ValueError("C0/unload sources differ")
        peak = result["peak_metrics"]; c0 = initial["static_stiffness_N_mm"]
        result["Hsieh_initial_triangle"] = initial_triangle_metrics(result["work_partition"]["loading_work_Nmm"],
            result["loading_peak_force_N"], c0, peak["KI_MPa_sqrt_m"], peak["conversion_modulus_MPa"], peak["eta"], peak["B_mm"], peak["W_mm"]-peak["a_mm"])
        result["Hsieh_initial_triangle"]["stiffness_source"] = "Exact-source undeformed linear unit-load compliance"
        matched = json.loads(Path(result["matched_load_result"]).read_text())
        if matched["source_odb_sha256"] != result["matched_loading_odb_sha256"]: raise ValueError("Unload/window sources differ")
        result["Hsieh_initial_triangle_window_sensitivity"] = triangle_window_sensitivity(
            result["work_partition"]["loading_work_Nmm"], result["loading_peak_force_N"], peak,
            matched.get("early_elastic_stiffness_windows", []))
        save_result_versioned(path, result); unload_count += 1
    (out/"qs-load-results.json").write_text(json.dumps(loads, indent=2))
    grouped = {}
    for path in out.glob("QSunload*/result.json"):
        result = json.loads(path.read_text())
        grouped.setdefault(Path(result["source_odb"]).parents[2].name, []).append(result)
    for queue_name, records in grouped.items():
        (out/(queue_name+"-results.json")).write_text(json.dumps(records, indent=2))
    print("Enriched", len(loads), "retained load and", unload_count, "unload records with explicit Hsieh triangle/stiffness sensitivity; no ODB opened")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, records):
    if not records:
        return
    names = list(dict.fromkeys(name for row in records for name in row))
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names); writer.writeheader(); writer.writerows(records)


def save_result_versioned(path, result):
    """Keep any previous small extraction referenced by downstream provenance."""
    path = Path(path); content = json.dumps(result, indent=2).encode("utf-8")
    if path.exists() and path.read_bytes() != content:
        previous = path.read_bytes(); digest = hashlib.sha256(previous).hexdigest()
        archive = path.parent/"extraction-history"; archive.mkdir(exist_ok=True)
        retained = archive/(digest+".json")
        if retained.exists() and retained.read_bytes() != previous: raise ValueError("Extraction archive hash collision")
        if not retained.exists(): retained.write_bytes(previous)
    path.write_bytes(content)


def delivered_odb_hash(path):
    """Require the queue's verified delivery before opening a retained ODB."""
    path = Path(path)
    records = json.loads((path.parent/"delivery.json").read_text())["files"]
    entries = [row for row in records if row["file"] == path.name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(entries) != 1 or entries[0]["sha256"] != digest or entries[0]["bytes"] != path.stat().st_size:
        raise ValueError("Retained ODB differs from completed queue delivery")
    return digest


def audit_controls(root, out):
    """Recompute both exact elastoplastic unload controls and the elastic control."""
    diagnostics = root / "audit/diagnostics"
    result = []
    for name in ("epUnload_hex_3_rho0p16_nx80", "epUnloadPre_hex_3_rho0p16_nx80"):
        directory = diagnostics / name
        metadata = json.loads((directory / "result.json").read_text())
        loading, unloading = read_csv(directory / "load-curve.csv"), read_csv(directory / "unload-curve.csv")
        u = lambda table: [float(row["U2_mm"]) for row in table]
        p = lambda table: [float(row["force_from_fixed_pin_N"]) for row in table]
        baseline = next(row for row in json.loads((root / "audit/modal/static-summary.json").read_text())
                        if row["case_id"] == metadata["case_id"])
        partition = measured_recovery_partition(u(loading), p(loading), u(unloading), p(unloading),
                                               initial_stiffness=baseline["static_stiffness_N_mm"])
        last_load, last_unload = loading[-1], unloading[-1]
        channels = {name: float(last_load[name+"_Nmm"]) for name in INTERNAL_TERMS if name+"_Nmm" in last_load}
        internal = internal_energy_partition(float(last_load["ALLIE_Nmm"]), float(last_load["ALLSE_Nmm"]),
                                             channels=channels, excluded_terms=("ALLCD", "ALLDC", "ALLFC", "ALLDMD"))
        final_IE, initial_IE = float(last_unload["ALLIE_Nmm"]), float(loading[0]["ALLIE_Nmm"])
        closure = partition["nonrecovered_work_Nmm"]-(final_IE-initial_IE)
        _, geom, _ = source_geometry(root, metadata)
        H = source_pin_halfspan(metadata["source_inp"])
        row = dict(job=name, **partition, **internal,
                   residual_elastic_energy_Nmm=float(last_unload["ALLSE_Nmm"]),
                   unload_added_plastic_dissipation_Nmm=float(last_unload["ALLPD_Nmm"])-float(last_load["ALLPD_Nmm"]),
                   finite_geometry_energy_closure_error_Nmm=closure,
                   relative_cycle_closure_error=abs(closure)/partition["loading_work_Nmm"],
                   passed_original_source_criterion=not metadata["loading_section_check"]["max_PEEQ_exceeds_original_threshold"],
                   ALLAE_loading_Nmm=float(last_load["ALLAE_Nmm"]),
                   rotation_factor=float(ct_rotation_factor(metadata["target_U2_mm"], float(geom.W),
                                                           float(geom.ai[0]), H, H)))
        row["rotation_only_apparent_Apl_Nmm"] = partition["loading_work_Nmm"]-partition["initial_triangle_Nmm"]*row["rotation_factor"]
        modulus = available_moduli(root)[metadata["case_id"]]
        if modulus["source_sha256"] != metadata["source_sha256"]: raise ValueError("Control/RVE sources differ")
        metrics = static_point_metrics(root, metadata, {"force_N": p(loading)[-1], "ALLPD_Nmm": float(last_load["ALLPD_Nmm"])}, modulus["E_x_MPa"])
        recovery_metrics = eta_energy_metrics(metrics["KI_MPa_sqrt_m"], modulus["E_x_MPa"], metrics["eta"], metrics["B_mm"], metrics["W_mm"]-metrics["a_mm"], partition["nonrecovered_work_Nmm"])
        row.update(KI_MPa_sqrt_m=metrics["KI_MPa_sqrt_m"], PD_eta_K_MPa_sqrt_m=metrics["plastic_dissipation_K_eta_MPa_sqrt_m"],
                   measured_recovery_eta_K_MPa_sqrt_m=recovery_metrics["K_eta_MPa_sqrt_m"],
                   recovery_vs_PD_K_percent=100*(recovery_metrics["K_eta_MPa_sqrt_m"]/metrics["plastic_dissipation_K_eta_MPa_sqrt_m"]-1))
        if row["relative_cycle_closure_error"] > 1e-4:
            raise ValueError("Matched-control energy closure failed")
        result.append(row)
    (out / "matched-control-checks.json").write_text(json.dumps(result, indent=2))
    return result


def audit(args):
    import re
    from validation_diagnostics import sha
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    controls = audit_controls(root, out)
    ae_evidence = None
    if args.accept_b21_ae_zero:
        probe_path = out/"B21ExplicitAE_axial_bend/result.json"
        ae_evidence = json.loads(probe_path.read_text())
        if not ae_evidence["AE_zero_to_recorded_precision"] or ae_evidence["maximum_PEEQ"] <= 0 or ae_evidence["maximum_within_element_surface_LE11_range"] <= 1e-4:
            raise ValueError("Completed plastic axial/bending Explicit probe does not corroborate AE exclusion")
        ae_evidence = {"probe_result": str(probe_path), "probe_result_sha256": sha(probe_path),
                       "basis": "B21-only planar two-node beam has three independent axial/shear/curvature modes and three rigid null modes; no shell drill or nonrigid hourglass mode. Explicit axial+bending/shear probe corroborates zero AE. This is a formulation/source exclusion, not observed native AE."}
    moduli = available_moduli(root)
    records = []
    events = {row["case_id"]: row for row in read_csv(root / "audit/failure/failure-initiation.csv")}
    for case in read_csv(root / "case-index.csv"):
        if case["mode"] != "Fracture" or case["campaign"] != "Density campaign" or case["topology"] not in ("tri", "kagome", "hex"):
            continue
        directory = root / "audit/history" / Path(case["odb"]).parent / Path(case["odb"]).stem
        if not (directory / "native_curve.csv").exists():
            continue
        curve, energies = read_csv(directory / "native_curve.csv"), read_csv(directory / "native_stored_energy.csv")
        event = events[case["case_id"]]
        if event["threshold_found"] != "True":
            raise ValueError("Native tensile criterion was not observed")
        source = root / case["inp"]
        if sha(source) != case["inp_sha256"]: raise ValueError("Original native source input changed")
        text = source.read_text().lower()
        exclusions = []
        if not any(card in text for card in ("*viscoelastic", "*creep", "*swelling")):
            exclusions.append("ALLCD")
        if "*fluid" not in text:
            exclusions.append("ALLFC")
        if "distortion control" not in text and "type=b21" in text.replace(" ", ""):
            exclusions.append("ALLDC")
        element_types = set(re.findall(r"(?im)^\*element\s*,[^\n]*?type\s*=\s*([^,\s]+)", text))
        if ae_evidence:
            if element_types != {"b21"}: raise ValueError("AE source argument is valid only for a B21-only original model")
            exclusions.append("ALLAE")
        for endpoint, i in (("native_last_before_tensile_criterion_field_bracket", int(event["prev_native_index"])),
                            ("native_first_after_tensile_criterion_field_bracket", int(event["cross_native_index"]))):
            energy = energies[i]
            if abs(float(energy["time_s"])-float(curve[i]["time_s"])) > 1e-7:
                raise ValueError("Native energy/load times differ")
            channels = {name: float(energy[name+"_Nmm"]) for name in INTERNAL_TERMS if name+"_Nmm" in energy}
            endpoint_exclusions = list(exclusions)
            if "last_before" in endpoint and event.get("undamaged_prefix_verified") == "True" and event.get("damage_before_primary_frame") == "False":
                endpoint_exclusions.append("ALLDMD")
            partition = internal_energy_partition(float(energy["ALLIE_Nmm"]), float(energy["ALLSE_Nmm"]),
                                                 channels=channels, excluded_terms=endpoint_exclusions)
            u = [float(row["U2_mm"]) for row in curve[:i+1]]; p = [float(row["RF2_N"]) for row in curve[:i+1]]
            W = work(u, p)
            deltaKE = float(energy["ALLKE_Nmm"])-float(energies[0]["ALLKE_Nmm"])
            record = dict(case_id=case["case_id"], endpoint=endpoint,
                            index=i, time_s=curve[i]["time_s"], W_Nmm=W,
                            W_minus_SE_Nmm=W-float(energy["ALLSE_Nmm"]),
                            W_minus_IE_minus_deltaKE_Nmm=W-float(energy["ALLIE_Nmm"])-deltaKE,
                            KE_over_IE=float(energy["ALLKE_Nmm"])/float(energy["ALLIE_Nmm"]),
                            plastic_dissipation_lower_Nmm=partition["plastic_dissipation_lower_Nmm"],
                            plastic_dissipation_upper_Nmm=partition["plastic_dissipation_upper_Nmm"],
                            IE_minus_SE_Nmm=partition["internal_nonrecoverable_Nmm"],
                            unknown_components=";".join(partition["unknown_components"]),
                            source_card_excluded_components=";".join(endpoint_exclusions),
                            AE_exclusion_basis=ae_evidence["basis"] if ae_evidence else "unknown native component; not assumed zero",
                            accepted_ASTM_Apl=False, method_status=partition["status"])
            modulus = moduli.get(case["case_id"])
            if modulus:
                if modulus["source_sha256"] != case["inp_sha256"]: raise ValueError("Native/RVE sources differ")
                meta = {"case_id": case["case_id"], "source_inp": str(source)}
                point = {"force_N": p[-1]}
                base = static_point_metrics(root, meta, point, modulus["E_x_MPa"])
                record.update({"KI_MPa_sqrt_m": base["KI_MPa_sqrt_m"], "conversion_modulus_MPa": modulus["E_x_MPa"]})
                for label in ("lower", "upper"):
                    bound = partition["plastic_dissipation_"+label+"_Nmm"]
                    if base["KI_MPa_sqrt_m"] < 0:
                        record["plastic_dissipation_eta_K_"+label+"_MPa_sqrt_m"] = None
                        record["eta_energy_unavailable_reason"] = "Negative signed loading-pin force at this native endpoint; no absolute-value correction"
                    else:
                        calculated = eta_energy_metrics(base["KI_MPa_sqrt_m"], modulus["E_x_MPa"], base["eta"], base["B_mm"], base["W_mm"]-base["a_mm"], bound)
                        record["plastic_dissipation_eta_K_"+label+"_MPa_sqrt_m"] = calculated["K_eta_MPa_sqrt_m"]
                record["eta_energy_qualification"] = "PD-based C(T) eta-J estimate; not automatically ASTM plastic-area or size/constraint qualification"
            records.append(record)
    write_csv(out / "native-energy-partitions.csv", records)
    receipt = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "native_cases": len(set(row["case_id"] for row in records)),
               "native_endpoint_records": len(records), "matched_unload_controls": len(controls),
               "B21_AE_exclusion_evidence": ae_evidence,
               "decision": ("Peak PD is derived from IE-SE only for verified undamaged B21 prefixes under the explicitly recorded source/formulation AE exclusion; crossing damaged prefixes remain bounded. " if ae_evidence else "Native IE-SE is bounded internal dissipation while AE is unknown. ")+
                           "Measured load-unload work applies only at its matched state. Neither quantity is automatically accepted ASTM Apl. Rotation-only sensitivity does not cure the checked negative triangles.",
               "sources": {"ASTM_local_edition": "E1820-08a,AnnexA2.4.2andA2.4.4,pages18-19",
                           "Hsieh": "2020,AppendixD,D.1-D.5",
                           "Abaqus_internal_identity": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEOUTRefMap/simaout-c-exp-totalenergyoutput.htm",
                           "NIST_ASTME1820_explanation": "https://doi.org/10.1016/j.tafmec.2023.104139"}}
    (out / "work-partition-receipt.json").write_text(json.dumps(receipt, indent=2))
    print("Audited", len(records), "native endpoint energy partitions and", len(controls), "matched unload controls")


def prepare_loads(args):
    """Stage exact-source damage-off static loads at measured native event targets.

    Does not submit. Existing different/pinned inputs or manifests are rejected;
    source plasticity/elasticity, sections, joints and geometry remain intact.
    """
    import re
    from validation_diagnostics import blocks, key, rows, sha, source_mesh

    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = read_csv(root / "case-index.csv")
    events = read_csv(root / "audit/failure/failure-initiation.csv")
    jobs, staged = [], []
    for relative in SELECTED_INPUTS:
        case = next(row for row in cases if row["inp"] == relative)
        event = next((row for row in events if row["case_id"] == case["case_id"]), None)
        if event is None or event["threshold_found"] != "True":
            raise ValueError("A verified native tensile criterion crossing is required: "+relative)
        native = root / "audit/history" / Path(relative).parent / Path(relative).stem / "native_curve.csv"
        curve = read_csv(native)
        index = int(event["cross_native_index"])
        if abs(float(curve[index]["time_s"])-float(event["crosstime_s"])) > 1e-7:
            raise ValueError("Native crossing row/time mismatch")
        target = float(curve[index]["U2_mm"])*1.08
        source = root / relative
        if sha(source) != case["inp_sha256"]:
            raise ValueError("Indexed source hash changed")
        original, _, elements, _, _ = source_mesh(source)
        prefix = []
        for block in blocks(original):
            if key(block) == "*step": break
            if key(block) not in {"*amplitude", "*damage initiation", "*damage evolution"}:
                prefix.append(block)
        name = "QSload_{}_{}_rho{}_nx{}".format(case["topology"], case["l_mm"],
                    case["nominal_relative_density"].replace(".", "p"), case["nnx"])
        directory = out / name; directory.mkdir(exist_ok=True)
        inp = directory / (name+".inp")
        text = "".join(prefix).rstrip()+"\n"
        text += ("*Step, name=QSLoad, nlgeom=YES, inc=2000, amplitude=RAMP\n"
                 "*Static\n0.002, 1., 1.e-10, 0.01\n*Boundary\n"
                 "fixity, 1, 2\nload, 1, 1\nload, 2, 2, {:.12g}\n"
                 "*Output, field, frequency=1\n*Node Output, nset=load\nU, RF\n"
                 "*Node Output, nset=fixity\nU, RF\n"
                 "*Element Output, elset=Set-cracktip\nLE, PEEQ, S\n"
                 "*Output, history, frequency=1\n*Node Output, nset=load\nU1, U2, UR3, RF1, RF2\n"
                 "*Node Output, nset=fixity\nU1, U2, UR3, RF1, RF2\n"
                 "*Energy Output\nALLSE, ALLPD, ALLIE, ALLWK, ALLVD, ALLAE, ALLKE, ALLDMD, ALLCD, ALLSD\n"
                 "*End Step\n").format(target)
        if inp.exists() and inp.read_text() != text:
            raise ValueError("Refusing different existing static loading input")
        if not inp.exists() and any(directory.iterdir()):
            raise ValueError("Static loading directory is occupied without input")
        hash_expected = sha(inp) if inp.exists() else hashlib.sha256(text.encode("ascii")).hexdigest()
        sta = source.with_suffix(".sta")
        times = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([\d.]+)", sta.read_text()) if sta.exists() else []
        meta = {"job": name, "case_id": case["case_id"], "source_inp": str(source), "source_sha256": sha(source),
                "input": str(inp), "input_sha256": hash_expected, "state": "prepared_not_submitted",
                "target_U2_mm": target, "target_basis": "1.08times first native field-crossing loading displacement; actual static criterion must be located independently, with no silent extrapolation",
                "native_crossing_index": index, "native_crossing_time_s": float(event["crosstime_s"]),
                "native_crossing_U2_mm": float(curve[index]["U2_mm"]), "native_event_record": event,
                "elements": len(elements), "source_original_wall_seconds": float(times[-1]) if times else None,
                "estimated_wall_seconds_range": [max(60, int(120*len(elements)/53096)), max(300, int(600*len(elements)/53096))],
                "estimate_basis": "Exact53k-element elastoplastic two-step controls finished136-175seconds;100+increment load-only controls expectedroughly2-10minutes scaledbyelementcount. Nonlinearity makes runtime uncertain.",
                "changes": ["Exact source B21 mesh/pins/sections and full elastic/plastic law retained",
                            "Remove damage initiation/evolution and obsolete Explicit amplitude only from model cards",
                            "Standard NLGEOM quasi-static loading; no stabilization; no filtering",
                            "Every-increment reference-point displacement/reaction/rotation, all available energy channels and original crack-tip section LE/PEEQ/S fields"],
                "criterion": "First maximum tensile saved LE11 crossing.1105 in the original crack-tip region; comparison keeps strain-measure/source-subroutine limitation explicit.",
                "analysis_limit": "Damage-off quasistatic model control; only the pre-criterion/crossing bracket is eligible. ALLPD remains plastic dissipation, not automatically ASTM Apl."}
        source_json = directory / "source.json"
        if source_json.exists():
            old = json.loads(source_json.read_text())
            if any(old.get(field) != meta[field] for field in ("input_sha256", "source_sha256", "target_U2_mm")):
                raise ValueError("Static-load provenance differs")
        else:
            if inp.exists(): raise ValueError("Existing static load lacks provenance")
            inp.write_bytes(text.encode("ascii")); source_json.write_text(json.dumps(meta, indent=2))
        staged.append(meta)
        jobs.append({"job": name, "input": str(inp), "input_sha256": hash_expected,
                     "kind": "exact_source_elastoplastic_quasistatic_criterion_load",
                     "purpose": "Resolve source-consistent tensile initiation and direct energy decomposition without changing the original Explicit simulation",
                     "cpus": 8, "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]})
    proposal = {"jobs": jobs, "queue_root": str(out / "queue-loads-20261028"), "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    manifest = out / "serial-loads-proposal.json"
    proposed = json.dumps(proposal, indent=2)
    if manifest.exists() and manifest.read_text() != proposed:
        raise ValueError("Refuse different existing loading manifest")
    if not manifest.exists(): manifest.write_text(proposed)
    (out / "loading-controls-manifest.json").write_text(json.dumps(staged, indent=2))
    print("Prepared", len(jobs), "exact-source static criterion loads; no jobs submitted")


def source_geometry(root, metadata):
    from resources.lattices import Geometry
    from validation_diagnostics import source_mesh
    case = next(row for row in read_csv(root/"case-index.csv") if row["case_id"] == metadata["case_id"])
    geom = Geometry(case["topology"], float(case["l_mm"]), int(case["nnx"]), float(case["nominal_relative_density"]))
    geom.FTcalc()
    _, _, _, section, _ = source_mesh(metadata["source_inp"])
    return case, geom, section[0]


def source_pin_halfspan(path):
    from validation_diagnostics import blocks, key, rows
    coordinates, nested = {}, False
    for block in blocks(Path(path).read_text()):
        keyword = key(block)
        if keyword in {"*part", "*instance"}: nested = True
        if keyword in {"*end part", "*end instance"}: nested = False
        if keyword != "*node" or nested: continue
        values = [line.split(",") for line in rows(block)]
        for row in values:
            label = int(row[0])
            if label in coordinates: raise ValueError("Duplicate assembly reference-point node")
            coordinates[label] = (float(row[1]), float(row[2]))
    if set(coordinates) != {1, 2}: raise ValueError("Expected exactly two assembly reference points")
    halfspan = abs(coordinates[1][1]-coordinates[2][1])/2
    if halfspan <= 0: raise ValueError("Nonpositive original C(T) pin span")
    return halfspan


def available_moduli(root):
    path = root/"audit/controls/campaign-rve-results.json"
    if not path.exists(): path = root/"audit/diagnostics/abaqus-rve-results.json"
    records = json.loads(path.read_text())
    return {record["case_id"]: record for record in records}


def available_stiffness(root):
    records = []
    legacy = root/"audit/modal/static-summary.json"
    if legacy.exists():
        records = [dict(row, source_sha256=row["source_inp_sha256"]) for row in json.loads(legacy.read_text())]
    current = root/"audit/controls/compliance-summary.json"
    if current.exists():
        incoming = json.loads(current.read_text())
        if any(row.get("state") != "completed_and_extracted" or not row.get("source_input_hash_verified")
               or not row.get("read_only") or row["static_stiffness_N_mm"]<=0 for row in incoming):
            raise ValueError("Only verified completed source-specific initial stiffness is usable")
        records += incoming
    return {record["case_id"]: record for record in records}


def static_point_metrics(root, metadata, point, modulus):
    from resources.calculations import calc_FaW
    case, geom, B = source_geometry(root, metadata)
    a, W = float(geom.ai[0]), float(geom.W)
    KI = point["force_N"]*calc_FaW(a, W)/(B*np.sqrt(W*1000))
    eta = 2+.522*(W-a)/W
    metrics = {"KI_MPa_sqrt_m": float(KI), "W_mm": W, "a_mm": a, "B_mm": B,
               "eta": eta, "conversion_modulus_MPa": modulus,
               "geometry_calibration": "ASTM alternate C(T) geometry; cellular/discrete-height adaptation remains a modelling limit"}
    if point.get("ALLPD_Nmm") is not None:
        metrics.update({"plastic_dissipation_"+name: value for name, value in eta_energy_metrics(
            KI, modulus, eta, B, W-a, point["ALLPD_Nmm"]).items()})
    initial = available_stiffness(root).get(metadata["case_id"])
    if initial and point.get("integrated_work_Nmm") is not None:
        if initial["source_sha256"] != case["inp_sha256"]: raise ValueError("Initial-stiffness/source inputs differ")
        metrics["Hsieh_initial_triangle"] = initial_triangle_metrics(point["integrated_work_Nmm"], point["force_N"],
            initial["static_stiffness_N_mm"], KI, modulus, eta, B, W-a)
        metrics["Hsieh_initial_triangle"]["stiffness_source"] = "Exact-source undeformed linear unit-load compliance"
    return metrics


def interpolated_peak(frames, threshold=.11045):
    """Interpolate only within the first observed positive tensile crossing."""
    if not 0 < threshold <= .1105:
        raise ValueError("Unload target must be at or below the source criterion")
    high_index = next((i for i, row in enumerate(frames)
                       if float(row["maximum_tensile_LE11"]) >= threshold), None)
    if high_index is None or high_index == 0:
        raise ValueError("Need an observed pre-target/cross-target strain bracket")
    low, high = frames[high_index-1], frames[high_index]
    elo, ehi = float(low["maximum_tensile_LE11"]), float(high["maximum_tensile_LE11"])
    fraction = (threshold-elo)/(ehi-elo)
    target = float(low["U2_mm"])+fraction*(float(high["U2_mm"])-float(low["U2_mm"]))
    if not 0 <= fraction <= 1 or target <= 0:
        raise ValueError("No extrapolated or nonpositive unload target permitted")
    return target, {"target_LE11": threshold, "interpolation_fraction": fraction,
                    "strain_bracket": [elo, ehi],
                    "displacement_bracket_mm": [float(low["U2_mm"]), float(high["U2_mm"])],
                    "method": "linear displacement interpolation within the first observed tensile LE11 bracket; actual new peak strain must be checked"}


def retain_prepared_input(directory, name, text, metadata):
    """Immutable solver input/provenance; same-byte preparation is idempotent."""
    directory = Path(directory)
    inp, source_file = directory/(name+".inp"), directory/"source.json"
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()
    metadata = dict(metadata, job=name, input=str(inp), input_sha256=digest)
    if directory.exists():
        if not inp.exists() or not source_file.exists():
            raise ValueError("Refuse occupied job directory without complete provenance")
        old = json.loads(source_file.read_text())
        if inp.read_bytes() != text.encode("ascii") or any(old.get(k) != metadata[k]
                for k in ("job", "source_sha256", "input_sha256", "target_U2_mm")):
            raise ValueError("Refuse changed existing/pinned solver input or target")
        return old
    directory.mkdir(parents=True)
    inp.write_bytes(text.encode("ascii"))
    source_file.write_text(json.dumps(metadata, indent=2))
    return metadata


def prepare_unloads(args):
    """Prepare matched full unload controls; no solver submission or fitted transfer."""
    from validation_diagnostics import blocks, key, sha
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    selected = args.jobs.split(",") if args.jobs else None
    summaries = json.loads((out/"qs-load-results.json").read_text())
    jobs, staged = [], []
    for result in summaries:
        if selected and result["job"] not in selected: continue
        reviewed_usable = result.get("state") == "failed_with_reviewed_converged_prefix" and result.get("reviewed_partial_usable")
        if not result.get("criterion_found") or (result["state"] != "completed_and_extracted" and not reviewed_usable):
            if selected: raise ValueError("Verified completed loading/criterion output is required")
            continue
        source = Path(result["source_inp"])
        if sha(source) != result["source_sha256"]: raise ValueError("Original source input changed")
        frames = read_csv(out/result["job"]/"qs-section-frames.csv")
        target, interpolation = interpolated_peak(frames, args.unload_criterion)
        prefix = []
        for block in blocks(source.read_text()):
            if key(block) == "*step": break
            if key(block) not in {"*amplitude", "*damage initiation", "*damage evolution"}:
                prefix.append(block)
        name = result["job"].replace("QSload", "QSunload", 1)
        text = "".join(prefix).rstrip()+"\n"
        text += ("*Step, name=Loading, nlgeom=YES, inc=2000, amplitude=RAMP\n"
                 "*Static\n0.002, 1., 1.e-10, 0.01\n*Boundary\n"
                 "fixity, 1, 2\nload, 1, 1\nload, 2, 2, {:.12g}\n").format(target)
        output = ("*Output, field, frequency=100000\n*Node Output, nset=load\nU, RF\n"
                  "*Node Output, nset=fixity\nU, RF\n"
                  "*Element Output, elset=Set-cracktip\nLE, PEEQ, S\n"
                  "*Output, history, frequency=1\n*Node Output, nset=load\nU1, U2, UR3, RF1, RF2\n"
                  "*Node Output, nset=fixity\nU1, U2, UR3, RF1, RF2\n"
                  "*Energy Output\nALLSE, ALLPD, ALLIE, ALLWK, ALLVD, ALLAE, ALLKE, ALLDMD, ALLCD, ALLSD\n")
        text += output+"*End Step\n*Step, name=Unloading, nlgeom=YES, inc=2000, amplitude=RAMP\n"
        text += "*Static\n0.002, 1., 1.e-10, 0.01\n*Boundary, op=NEW\nfixity, 1, 2\nload, 1, 1\n"
        text += output+"*End Step\n"
        metadata = {"case_id": result["case_id"], "source_inp": str(source),
                    "source_sha256": result["source_sha256"], "state": "prepared_not_submitted",
                    "target_U2_mm": target, "peak_selection": interpolation,
                    "matched_load_result": str(out/result["job"]/"result.json"),
                    "matched_load_result_sha256": sha(out/result["job"]/"result.json"),
                    "matched_loading_odb_sha256": result["source_odb_sha256"],
                    "matched_loading_solver_state": result.get("solver_state", "completed"),
                    "matched_loading_reviewed_criterion_valid": bool(result.get("reviewed_partial_usable", False)),
                    "changes": ["Original exact B21 mesh, sections, material Elastic/Plastic and pin constraints retained",
                                "Damage cards removed for the counterfactual intact pre-criterion control",
                                "Standard NLGEOM load at independently interpolated target then release only loading U2",
                                "Removed U2 constraint becomes previous reaction force ramped to zero; fixed-pin reaction is the unload force",
                                "Final Loading LE/PEEQ/S and every-increment pin/global-energy histories retained"],
                    "acceptance": "Verify actual peak maximum tensile LE11 <= .1105 (report any overshoot); no extrapolation, no reverse-force tail, pin equilibrium and full energy closure",
                    "method_limit": "Direct matched finite-geometry recovery; nonrecovered work is not automatically qualified ASTM Apl. No cross-case fitted correction.",
                    "estimated_wall_seconds_range": [120, 600],
                    "source_boundary_release_reference": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-boundary.htm"}
        metadata = retain_prepared_input(out/name, name, text, metadata); staged.append(metadata)
        jobs.append({"job": name, "input": metadata["input"], "input_sha256": metadata["input_sha256"],
                     "kind": "matched_source_precriterion_finite_geometry_unload",
                     "purpose": "Measure recoverable work at this source-specific tensile initiation state, independently of the initial linear triangle",
                     "cpus": 8, "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]})
    if selected and len(staged) != len(selected): raise ValueError("Selected loading jobs are not all extracted")
    if not jobs: raise ValueError("No completed loading controls selected")
    manifest = out/(args.manifest_name or "serial-unloads-proposal.json")
    if manifest.name != str(args.manifest_name or "serial-unloads-proposal.json"):
        raise ValueError("Manifest name must be a single filename")
    queue = args.queue_root or str(out/"queue-unloads-20261028")
    proposal = json.dumps({"jobs": jobs, "queue_root": queue, "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}, indent=2)
    if manifest.exists() and manifest.read_text() != proposal:
        raise ValueError("Refuse overwriting a different existing unloading proposal; choose a unique manifest/queue")
    if not manifest.exists(): manifest.write_text(proposal)
    print("Prepared", len(jobs), "matched unload controls; no jobs submitted:", manifest)


def prepare_fine_retry(args):
    """One authorised Newton/line-search retry; no stabilization or new physics."""
    from validation_diagnostics import sha
    out = Path(args.out).resolve()
    baseline_job = args.retry_source_job or "QSload_tri_10_rho0p16_nx80"
    if Path(baseline_job).name != baseline_job or not baseline_job.startswith("QSload_"):
        raise ValueError("Specify one original loading job name; no automatic repeated retries")
    parent = out/baseline_job
    baseline = json.loads((parent/"source.json").read_text())
    if sha(baseline["source_inp"]) != baseline["source_sha256"] or sha(baseline["input"]) != baseline["input_sha256"]:
        raise ValueError("Original/pinned baseline changed")
    original = Path(baseline["input"]).read_text()
    search = "*Static\n0.002, 1., 1.e-10, 0.01\n"
    if original.count(search) != 1: raise ValueError("Expected exactly one baseline static-control line")
    text = original.replace(search, "*Static\n0.001, 1., 1.e-10, 0.0025\n"
                            "*Controls, parameters=LINE SEARCH\n5\n"
                            "*Controls, parameters=TIME INCREMENTATION\n,,,,,,,,12\n")
    name = baseline_job.replace("QSload_", "QSloadFine_", 1)
    metadata = dict(baseline, prepared_from_job=baseline["job"],
                    prepared_parent_input_sha256=baseline["input_sha256"], state="prepared_not_submitted",
                    solver_controls={"initial_increment": .001, "maximum_increment": .0025,
                                     "minimum_increment": 1e-10, "line_search_iterations_Nls": 5,
                                     "maximum_attempts_IA_ninth_time_incrementation_field": 12,
                                     "force_residual_tolerances": "unchanged defaults",
                                     "stabilization": "none; no added damping or material changes"},
                    solver_controls_reference="https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-controls.htm",
                    estimated_wall_seconds_range=[120, 900],
                    retry_limit="One physics-preserving solver-control retry. A failure before criterion remains unavailable; no automatic damped/Riks substitution.")
    metadata = retain_prepared_input(out/name, name, text, metadata)
    proposal = {"jobs": [{"job": name, "input": metadata["input"], "input_sha256": metadata["input_sha256"],
                          "kind": "exact_source_static_algorithm_retry_no_stabilization",
                          "purpose": "Test numerical convergence at unchanged displacement/material/geometry before treating the baseline negative-eigenvalue stop as unavoidable static instability",
                          "cpus": 8, "allow_solver_failure": True, "required_outputs": [".odb", ".dat"]}],
                "queue_root": args.queue_root or str(out/"queue-fine-retry-20261028"), "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    filename = args.manifest_name or "serial-load-fine-tri-proposal.json"
    if Path(filename).name != filename: raise ValueError("Manifest name must be a single filename")
    manifest = out/filename; proposed = json.dumps(proposal, indent=2)
    if manifest.exists() and manifest.read_text() != proposed: raise ValueError("Refuse a different existing static retry manifest")
    if not manifest.exists(): manifest.write_text(proposed)
    print("Prepared one authorised static solver-control retry; no jobs submitted:", manifest)


def pin_step_curve(step, *, force_from_fixed=True):
    """Matched reference-point histories with force from the constrained pin."""
    regions = [region for region in step.historyRegions.values()
               if "U2" in region.historyOutputs and "RF2" in region.historyOutputs]
    if len(regions) != 2: raise ValueError("Exactly two pin histories required")
    load = max(regions, key=lambda region: region.historyOutputs["U2"].data[0][1]+region.historyOutputs["U2"].data[-1][1])
    fixed = next(region for region in regions if region != load)
    arrays = {"U2_mm": np.array(load.historyOutputs["U2"].data),
              "loading_RF2_N": np.array(load.historyOutputs["RF2"].data),
              "fixed_RF2_N": np.array(fixed.historyOutputs["RF2"].data)}
    for label, region in (("loading", load), ("fixed", fixed)):
        for channel in ("U1", "UR3"):
            if channel in region.historyOutputs:
                arrays[label+"_"+channel] = np.array(region.historyOutputs[channel].data)
    energy = [region for region in step.historyRegions.values() if "ALLSE" in region.historyOutputs]
    if len(energy) != 1: raise ValueError("Exactly one global energy history required")
    arrays.update({name+"_Nmm": np.array(value.data) for name, value in energy[0].historyOutputs.items() if name.startswith("ALL")})
    time = arrays["U2_mm"][:, 0]
    if any(not np.array_equal(value[:, 0], time) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Pin/energy histories are nonfinite or not aligned")
    return [dict(time_s=float(t), force_N=float(-arrays["fixed_RF2_N"][i, 1] if force_from_fixed else arrays["loading_RF2_N"][i, 1]),
                 **{name: float(array[i, 1]) for name, array in arrays.items()}) for i, t in enumerate(time)]


def final_loading_section(frame):
    field = frame.fieldOutputs["LE"]
    component = list(field.componentLabels).index("LE11")
    axial = [float(value.data[component]) for value in field.values]
    if not np.isfinite(axial).all(): raise ValueError("Nonfinite peak LE11")
    maximum = max(axial)
    return {"maximum_tensile_LE11": maximum, "minimum_LE11": min(axial),
            "maximum_PEEQ": max(float(value.data) for value in frame.fieldOutputs["PEEQ"].values),
            "passed_tensile_criterion": maximum <= .1105,
            "tensile_criterion_overshoot": max(0., maximum-.1105),
            "scope": "Original Set-cracktip; damage disabled only in this counterfactual control"}


def extract_unloads(args):
    """Read completed matched unloads only; never fit another case's correction."""
    from odbAccess import openOdb
    from validation_diagnostics import sha
    root, out = Path(args.root), Path(args.out)
    if not args.queue_root: raise ValueError("Specify the unique matched-unload queue root")
    queue = Path(args.queue_root); state = json.loads((queue/"state.json").read_text())
    moduli, summaries = available_moduli(root), []
    baselines = available_stiffness(root)
    for item in state["jobs"]:
        if not item["job"].startswith("QSunload"): continue
        if item["status"] != "completed": continue
        directory = out/item["job"]; metadata = json.loads((directory/"source.json").read_text())
        result_file = directory/"result.json"
        if result_file.exists() and not args.refresh:
            summaries.append(json.loads(result_file.read_text())); continue
        odb_path = queue/"runs"/item["job"]/(item["job"]+".odb")
        before = delivered_odb_hash(odb_path)
        if sha(metadata["source_inp"]) != metadata["source_sha256"] or sha(odb_path.with_suffix(".inp")) != metadata["input_sha256"]:
            raise ValueError("Original source or pinned unload input changed")
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            loading, unloading = pin_step_curve(odb.steps["Loading"]), pin_step_curve(odb.steps["Unloading"])
            u = lambda rows: [row["U2_mm"] for row in rows]
            p = lambda rows: [row["force_N"] for row in rows]
            baseline = baselines.get(metadata["case_id"])
            stiffness = baseline["static_stiffness_N_mm"] if baseline else None
            if baseline and baseline["source_sha256"] != metadata["source_sha256"]:
                raise ValueError("Initial stiffness belongs to another source")
            partition = measured_recovery_partition(u(loading), p(loading), u(unloading), p(unloading), initial_stiffness=stiffness)
            scale = max(abs(loading[-1]["force_N"]), 1.)
            equilibrium = max(abs(row["loading_RF2_N"]+row["fixed_RF2_N"]) for row in loading)/scale
            release = max(abs(row["force_N"]-scale*(1-row["time_s"])) for row in unloading)/scale
            free_pin = max(abs(row["loading_RF2_N"]) for row in unloading)/scale
            if max(equilibrium, release, free_pin)>1e-4: raise ValueError("Pin equilibrium/released-reaction ramp failed")
            if abs(loading[-1]["U2_mm"]-metadata["target_U2_mm"])>1e-5: raise ValueError("Unload peak target not matched")
            section = final_loading_section(odb.steps["Loading"].frames[-1])
            peak, final = loading[-1], unloading[-1]
            channels = {name: peak.get(name+"_Nmm") for name in INTERNAL_TERMS}
            internal = internal_energy_partition(peak["ALLIE_Nmm"], peak["ALLSE_Nmm"], channels=channels,
                         excluded_terms=("ALLCD", "ALLDC", "ALLFC", "ALLDMD"))
            closure = partition["nonrecovered_work_Nmm"]-(final["ALLIE_Nmm"]-loading[0]["ALLIE_Nmm"])
            result = dict(metadata, state="completed_and_extracted", preparation_state=metadata["state"],
                          source_odb=str(odb_path), source_odb_sha256=before, read_only=True,
                          source_input_hash_verified=True, retained_solver_input_hash_verified=True,
                          loading_samples=len(loading), unloading_samples=len(unloading),
                          force_balance_relative_error=equilibrium, release_force_ramp_relative_error=release,
                          free_loading_pin_RF2_relative=free_pin, loading_section_check=section,
                          loading_peak_U2_mm=peak["U2_mm"], loading_peak_force_N=peak["force_N"],
                          loading_peak_state=peak,
                          work_partition=partition, internal_partition=internal,
                          residual_elastic_energy_Nmm=final["ALLSE_Nmm"],
                          unload_added_plastic_dissipation_Nmm=final["ALLPD_Nmm"]-peak["ALLPD_Nmm"],
                          finite_geometry_energy_closure_error_Nmm=closure,
                          relative_cycle_closure_error=abs(closure)/partition["loading_work_Nmm"],
                          maximum_abs_other_energy_channels_Nmm={name: max(abs(row[name+"_Nmm"]) for row in loading+unloading) if all(name+"_Nmm" in row for row in loading+unloading) else None
                            for name in ("ALLAE", "ALLCD", "ALLDMD", "ALLVD", "ALLSD", "ALLKE")})
            modulus = moduli.get(metadata["case_id"])
            if modulus:
                if modulus["source_sha256"] != metadata["source_sha256"]: raise ValueError("Unload/RVE sources differ")
                metrics = static_point_metrics(root, metadata, peak, modulus["E_x_MPa"])
                result["peak_metrics"] = metrics
                matched = json.loads(Path(metadata["matched_load_result"]).read_text())
                if matched["source_odb_sha256"] != metadata["matched_loading_odb_sha256"]:
                    raise ValueError("Unload/window source ODBs differ")
                result["Hsieh_initial_triangle_window_sensitivity"] = triangle_window_sensitivity(
                    partition["loading_work_Nmm"], peak["force_N"], metrics, matched.get("early_elastic_stiffness_windows", []))
                if stiffness:
                    result["Hsieh_initial_triangle"] = initial_triangle_metrics(partition["loading_work_Nmm"], peak["force_N"], stiffness,
                        metrics["KI_MPa_sqrt_m"], modulus["E_x_MPa"], metrics["eta"], metrics["B_mm"], metrics["W_mm"]-metrics["a_mm"])
                    result["Hsieh_initial_triangle"]["stiffness_source"] = "Exact-source undeformed linear unit-load compliance"
                result["measured_recovery_metrics"] = eta_energy_metrics(metrics["KI_MPa_sqrt_m"], modulus["E_x_MPa"],
                    metrics["eta"], metrics["B_mm"], metrics["W_mm"]-metrics["a_mm"], partition["nonrecovered_work_Nmm"])
                result["precriterion_peak_verified"] = section["passed_tensile_criterion"]
                result["energy_closure_review_required"] = True
                result["qualification"] = "Matched finite-geometry eta-J estimate; not ASTM size/constraint-qualified and not a fitted native-history correction"
            write_csv(directory/"load-curve.csv", loading); write_csv(directory/"unload-curve.csv", unloading)
        finally:
            odb.close()
        if sha(odb_path) != before: raise ValueError("Unload ODB changed during read-only extraction")
        save_result_versioned(result_file, result); summaries.append(result)
    target = out/(queue.name+"-results.json")
    target.write_text(json.dumps(summaries, indent=2))
    print("Extracted", len(summaries), "completed matched unloads; running ODBs left unopened:", target)


def prepare_ae_probe(args):
    """Stage a tiny B21 axial+bending/shear plastic Explicit implementation check."""
    from validation_diagnostics import blocks, key, sha, source_mesh
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    source = root/SELECTED_INPUTS[0]
    original, _, _, section, _ = source_mesh(source)
    material = "".join(block for block in blocks(original) if key(block) in {"*material", "*density", "*elastic", "*plastic"})
    if material.lower().count("*material") != 1 or "*plastic" not in material.lower():
        raise ValueError("Need exactly one original elastoplastic material")
    name = "B21ExplicitAE_axial_bend"
    text = ("*Heading\nB21 Explicit artificial-energy check; not a lattice validation point\n"
            "*Node\n1,0.,0.,0.\n2,1.,0.,0.\n3,2.,0.,0.\n4,3.,0.,0.\n"
            "*Element, type=B21, elset=BODY\n1,1,2\n2,2,3\n3,3,4\n"
            "*Nset, nset=FIX\n1\n*Nset, nset=TIP\n4\n"
            "*Beam Section, elset=BODY, material=ti, temperature=GRADIENTS, section=RECT\n"
            "{:.12g}, {:.12g}\n0.,0.,-1.\n").format(*section)
    text += material.rstrip()+"\n*Amplitude, name=RAMP, definition=SMOOTH STEP\n0.,0.,0.002,1.\n"
    text += ("*Step, name=AxialBend, nlgeom=YES\n*Dynamic, Explicit\n,0.002\n"
             "*Bulk Viscosity\n0.06,1.2\n*Boundary, amplitude=RAMP\nFIX,1,2\nFIX,6,6\nTIP,1,1,0.15\nTIP,2,2,0.9\n"
             "*Output, field, number interval=20\n*Node Output\nU,RF\n*Element Output\nLE,PEEQ,S\n"
             "*Output, history, time interval=0.00001\n*Node Output, nset=TIP\nU1,U2,RF1,RF2,UR3\n"
             "*Energy Output\nALLIE,ALLSE,ALLPD,ALLAE,ALLCD,ALLDMD,ALLDC,ALLFC,ALLKE,ALLVD,ALLWK\n*End Step\n")
    metadata = {"source_inp": str(source), "source_sha256": sha(source), "target_U2_mm": .9,
                "state": "prepared_not_submitted", "target_U1_mm": .15,
                "purpose": "Exercise B21 axial plasticity and finite bending/shear/rotation while recording all internal energy terms",
                "source_features": "Exact original Ti Elastic/Plastic/Density, rectangular section and B21 orientation; damage omitted only in this formulation probe",
                "not_validation_data": True,
                "interpretation_limit": "Zero AE in this probe is implementation evidence alongside the B21 mode-rank/absence-of-shell-drill argument; it is not a missing native channel observation.",
                "estimated_wall_seconds_range": [10, 120]}
    metadata = retain_prepared_input(out/name, name, text, metadata)
    proposal = {"jobs": [{"job": name, "input": metadata["input"], "input_sha256": metadata["input_sha256"],
                          "kind": "tiny_B21_Explicit_artificial_energy_probe", "purpose": metadata["purpose"],
                          "cpus": 1, "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]}],
                "queue_root": str(out/"queue-ae-probe-20261028"), "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    manifest = out/"serial-ae-probe-proposal.json"; proposed = json.dumps(proposal, indent=2)
    if manifest.exists() and manifest.read_text() != proposed: raise ValueError("Refuse a different existing AE probe manifest")
    if not manifest.exists(): manifest.write_text(proposed)
    print("Prepared axial+bending/shear Explicit probe; no jobs submitted:", manifest)


def extract_ae_probe(args):
    from odbAccess import openOdb
    from validation_diagnostics import sha
    out = Path(args.out); queue = Path(args.queue_root) if args.queue_root else out/"queue-ae-probe-20261028"
    name = "B21ExplicitAE_axial_bend"; directory = out/name
    state = json.loads((queue/"state.json").read_text())
    item = next(row for row in state["jobs"] if row["job"] == name)
    if item["status"] != "completed":
        print("AE probe is", item["status"], "; ODB left unopened"); return
    metadata = json.loads((directory/"source.json").read_text())
    odb_path = queue/"runs"/name/(name+".odb"); before = delivered_odb_hash(odb_path)
    if sha(metadata["source_inp"]) != metadata["source_sha256"] or sha(odb_path.with_suffix(".inp")) != metadata["input_sha256"]:
        raise ValueError("AE probe source or pinned input changed")
    odb = openOdb(str(odb_path), readOnly=True)
    try:
        step = odb.steps["AxialBend"]
        region = next(region for region in step.historyRegions.values() if "ALLIE" in region.historyOutputs)
        values = {name: np.array(channel.data) for name, channel in region.historyOutputs.items() if name.startswith("ALL")}
        required = {"ALLIE", "ALLSE", "ALLPD", "ALLAE"}
        if not required <= values.keys(): raise ValueError("Probe internal-energy decomposition was not recorded")
        time = values["ALLIE"][:, 0]
        if any(not np.array_equal(value[:, 0], time) or not np.isfinite(value).all() for value in values.values()):
            raise ValueError("Probe energy times/channels are unaligned or nonfinite")
        write_csv(directory/"energy-curve.csv", [dict(time_s=float(t), **{name+"_Nmm": float(value[i, 1]) for name, value in values.items()}) for i, t in enumerate(time)])
        maximum = {name: float(np.max(abs(value[:, 1]))) for name, value in values.items()}
        final = step.frames[-1]
        le = final.fieldOutputs["LE"]; component = list(le.componentLabels).index("LE11")
        by_element = {}
        for value in le.values: by_element.setdefault(value.elementLabel, []).append(float(value.data[component]))
        spread = max(max(axial)-min(axial) for axial in by_element.values())
        peeq = max(float(value.data) for value in final.fieldOutputs["PEEQ"].values)
        if maximum["ALLPD"] <= 0 or peeq <= 0 or spread <= 1e-4:
            raise ValueError("Probe did not exercise both plasticity and section bending")
        recorded_internal = {"ALLSE"}|(set(INTERNAL_TERMS)&values.keys())
        residual = values["ALLIE"][:, 1]-sum(values[name][:, 1] for name in recorded_internal)
        result = dict(metadata, state="completed_and_extracted", preparation_state=metadata["state"],
                      source_odb=str(odb_path), source_odb_sha256=before, read_only=True,
                      source_input_hash_verified=True, retained_solver_input_hash_verified=True,
                      history_samples=len(time), maximum_abs_energy_channels_Nmm=maximum,
                      internal_components_unrecorded=sorted(set(INTERNAL_TERMS)-values.keys()),
                      unrecorded_component_source_exclusion="Probe has no viscoelastic/creep, damage, distortion-control or fluid-cavity cards; unrecorded optional channels are not claimed as measured zeros",
                      maximum_PEEQ=peeq, maximum_within_element_surface_LE11_range=spread,
                      maximum_internal_identity_residual_Nmm=float(np.max(abs(residual))),
                      AE_zero_to_recorded_precision=maximum["ALLAE"] == 0.,
                      IE_minus_SE_minus_PD_relative=float(np.max(abs(values["ALLIE"][:, 1]-values["ALLSE"][:, 1]-values["ALLPD"][:, 1]))/max(maximum["ALLIE"], 1.)))
    finally:
        odb.close()
    if sha(odb_path) != before: raise ValueError("Probe ODB changed while read-only")
    (directory/"result.json").write_text(json.dumps(result, indent=2))
    print("Extracted completed B21 Explicit axial+bending/shear probe:", directory/"result.json")


def extract_loads(args):
    """Read completed QS jobs only; locate tensile criterion without filtering."""
    from odbAccess import openOdb
    from validation_diagnostics import sha

    root, out = Path(args.root), Path(args.out)
    queue = Path(args.queue_root) if args.queue_root else out/"queue-loads-20261028"
    state = json.loads((queue/"state.json").read_text())
    moduli = available_moduli(root)
    summaries = []
    for item in state["jobs"]:
        if not item["job"].startswith("QSload"): continue
        reviewed_partial = item["job"] == args.reviewed_partial_job
        if item["status"] != "completed" and not reviewed_partial: continue
        if reviewed_partial:
            import ctypes
            import os
            if item["status"] != "failed" or not item.get("finished_at"):
                raise ValueError("Reviewed partial requires a terminated failed job with finished solver")
            if os.name == "nt" and item.get("solver_launcher_pid"):
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
                kernel.OpenProcess.restype = ctypes.c_void_p
                kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                handle = kernel.OpenProcess(0x1000, False, int(item["solver_launcher_pid"]))
                if handle:
                    code = ctypes.c_ulong()
                    try:
                        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259:
                            raise ValueError("Failed-job launcher may still be running")
                    finally:
                        kernel.CloseHandle(handle)
                elif ctypes.get_last_error() not in (0, 87, 1168):
                    raise ValueError("Cannot verify failed-job launcher has exited")
        directory = out/item["job"]
        metadata = json.loads((directory/"source.json").read_text())
        result_file = directory/"result.json"
        if result_file.exists() and not args.refresh:
            summaries.append(json.loads(result_file.read_text())); continue
        odb_path = queue/"runs"/item["job"]/(item["job"]+".odb")
        if reviewed_partial and odb_path.with_suffix(".lck").exists():
            raise ValueError("Failed-job ODB still has an active lock")
        before = delivered_odb_hash(odb_path)
        if sha(metadata["source_inp"]) != metadata["source_sha256"]:
            raise ValueError("QS source input changed")
        retained_input = odb_path.with_suffix(".inp")
        if sha(retained_input) != metadata["input_sha256"]:
            raise ValueError("QS retained solver input differs from pinned preparation")
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            step = odb.steps["QSLoad"]
            regions = {name: region for name, region in step.historyRegions.items()
                       if "U2" in region.historyOutputs and "RF2" in region.historyOutputs}
            if len(regions) != 2: raise ValueError("Expected exactly two QS pin histories")
            load_name = max(regions, key=lambda name: regions[name].historyOutputs["U2"].data[-1][1])
            fixed_name = next(name for name in regions if name != load_name)
            arrays = {"U2_mm": np.array(regions[load_name].historyOutputs["U2"].data),
                      "force_N": np.array(regions[load_name].historyOutputs["RF2"].data),
                      "fixed_RF2_N": np.array(regions[fixed_name].historyOutputs["RF2"].data)}
            for name in ("U1", "UR3"):
                if name in regions[load_name].historyOutputs:
                    arrays["loading_"+name] = np.array(regions[load_name].historyOutputs[name].data)
                if name in regions[fixed_name].historyOutputs:
                    arrays["fixed_"+name] = np.array(regions[fixed_name].historyOutputs[name].data)
            er = [region for region in step.historyRegions.values() if "ALLSE" in region.historyOutputs]
            if len(er) != 1: raise ValueError("Expected one global QS energy region")
            arrays.update({name+"_Nmm": np.array(channel.data) for name, channel in er[0].historyOutputs.items() if name.startswith("ALL")})
            arrays, duplicate_rows = distinct_history_times(arrays, allow_terminal_ambiguity=reviewed_partial)
            time = arrays["U2_mm"][:, 0]
            if any(not np.array_equal(array[:, 0], time) or not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("Nonfinite or unaligned QS histories")
            u, p = arrays["U2_mm"][:, 1], arrays["force_N"][:, 1]
            if np.any(np.diff(time)<=0) or np.any(np.diff(u)<-1e-6): raise ValueError("QS loading is not monotone")
            balance = float(np.max(abs(p+arrays["fixed_RF2_N"][:, 1]))/max(np.max(abs(p)), 1.))
            if balance > 1e-4: raise ValueError("QS vertical pin equilibrium failed")
            integrated = np.r_[0., np.cumsum(.5*(p[1:]+p[:-1])*np.diff(u))]
            energy_scale = max(float(np.max(abs(arrays["ALLIE_Nmm"][:, 1]))), 1.)
            decomp = arrays["ALLIE_Nmm"][:, 1]-arrays["ALLSE_Nmm"][:, 1]-arrays["ALLPD_Nmm"][:, 1]
            work_closure = arrays["ALLWK_Nmm"][:, 1]-arrays["ALLIE_Nmm"][:, 1]
            energy_checks = {"maximum_abs_IE_minus_SE_minus_PD_Nmm": float(np.max(abs(decomp))),
                             "relative_IE_decomposition_residual": float(np.max(abs(decomp))/energy_scale),
                             "maximum_abs_WK_minus_IE_Nmm": float(np.max(abs(work_closure))),
                             "relative_WK_IE_closure_residual": float(np.max(abs(work_closure))/energy_scale),
                             "maximum_abs_integrated_work_minus_WK_Nmm": float(np.max(abs(integrated-arrays["ALLWK_Nmm"][:, 1]))),
                             "maximum_abs_other_recorded_channels_Nmm": {name: float(np.max(abs(array[:, 1])))
                                 for name, array in arrays.items() if name in {"ALLAE_Nmm", "ALLCD_Nmm", "ALLDMD_Nmm", "ALLVD_Nmm", "ALLSD_Nmm", "ALLKE_Nmm"}}}
            curve = [dict(time_s=float(t), **{name: float(array[i, 1]) for name, array in arrays.items()},
                          integrated_work_Nmm=float(integrated[i])) for i, t in enumerate(time)]
            write_csv(directory/"qs-load-curve.csv", curve)
            fields = []
            for index, frame in enumerate(step.frames):
                if "LE" not in frame.fieldOutputs: continue
                if reviewed_partial and frame.frameValue > time[-1]+1e-7: continue
                le = frame.fieldOutputs["LE"]; component = list(le.componentLabels).index("LE11")
                values = le.values
                axial = np.array([value.data[component] for value in values])
                if not np.isfinite(axial).all(): raise ValueError("Nonfinite QS section strain")
                location = int(np.argmax(axial)); critical = values[location]
                def point(value): return (value.instance.name, value.elementLabel, value.sectionPoint.number, value.integrationPoint)
                identity = point(critical)
                pp = {point(value): float(value.data) for value in frame.fieldOutputs["PEEQ"].values}
                ss = frame.fieldOutputs["S"] if "S" in frame.fieldOutputs else None
                stress = None
                if ss is not None and "S11" in ss.componentLabels:
                    sc = list(ss.componentLabels).index("S11")
                    stress = next((float(value.data[sc]) for value in ss.values if point(value)==identity), None)
                hi = int(np.argmin(abs(time-frame.frameValue)))
                if abs(time[hi]-frame.frameValue)>1e-7: raise ValueError("QS frame/history times differ")
                fields.append(dict(frame_index=index, history_index=hi, time_s=float(frame.frameValue),
                                   maximum_tensile_LE11=float(axial[location]), minimum_LE11=float(axial.min()),
                                   maximum_PEEQ=max(pp.values()), PEEQ_at_max_tensile=pp.get(identity),
                                   S11_MPa_at_max_tensile=stress, element=critical.elementLabel,
                                   section_point=critical.sectionPoint.number, integration_point=critical.integrationPoint,
                                   U2_mm=float(u[hi]), force_N=float(p[hi])))
            write_csv(directory/"qs-section-frames.csv", fields)
            crossing = next((i for i, row in enumerate(fields) if row["maximum_tensile_LE11"]>=.1105), None)
            result = dict(metadata, preparation_state=metadata["state"],
                          state="failed_with_reviewed_converged_prefix" if reviewed_partial else "completed_and_extracted",
                          solver_state=item["status"], reviewed_partial=reviewed_partial,
                          source_odb=str(odb_path), source_odb_sha256=before, read_only=True,
                          source_input_hash_verified=True, retained_solver_input_hash_verified=True,
                          energy_checks=energy_checks, history_samples=len(curve), field_frames=len(fields),
                          duplicate_history_rows_removed=duplicate_rows,
                          duplicate_policy="Only identical rows may be deduplicated; for a reviewed failed job, the complete suffix beginning at the first differing repeated-time group is excluded, with corresponding later field frames omitted",
                          force_balance_relative_error=balance, criterion_threshold_LE11=.1105,
                          criterion_found=crossing is not None,
                          region_scope="Original Set-cracktip output region; no assertion about unrequested global fields",
                          source_case_modulus_verified=metadata["case_id"] in moduli)
            result["early_elastic_stiffness_windows"] = early_elastic_stiffness_windows(curve, fields)
            if crossing is not None:
                if crossing == 0: raise ValueError("No pre-criterion QS frame")
                low, high = fields[crossing-1], fields[crossing]
                fraction = (.1105-low["maximum_tensile_LE11"])/(high["maximum_tensile_LE11"]-low["maximum_tensile_LE11"])
                points = {}
                for label, frame in (("last_before_criterion", low), ("first_after_criterion", high)):
                    points[label] = dict(curve[frame["history_index"]], **{name: value for name, value in frame.items() if name not in curve[frame["history_index"]]})
                estimated = {name: float(points["last_before_criterion"][name]+fraction*(points["first_after_criterion"][name]-points["last_before_criterion"][name]))
                             for name in curve[0]}
                estimated.update(maximum_tensile_LE11=.1105, interpolation_fraction=float(fraction),
                                 method="linear interpolation inside first observed QS tensile criterion bracket; no extrapolation")
                points["interpolated_criterion"] = estimated
                if metadata["case_id"] in moduli:
                    modulus = moduli[metadata["case_id"]]
                    if modulus["source_sha256"] != metadata["source_sha256"]:
                        raise ValueError("QS/RVE source inputs differ")
                    for point_record in points.values():
                        point_record.update(static_point_metrics(root, metadata, point_record, modulus["E_x_MPa"]))
                result["criterion_points"] = points
                result["criterion_displacement_bracket_mm"] = [low["U2_mm"], high["U2_mm"]]
                if reviewed_partial: result["reviewed_partial_usable"] = True
            else:
                result["unavailable_reason"] = "Tensile criterion not reached; do not extrapolate or silently use the load endpoint"
                if reviewed_partial: result["reviewed_partial_usable"] = False
        finally:
            odb.close()
        if sha(odb_path)!=before: raise ValueError("QS ODB changed during read-only extraction")
        save_result_versioned(result_file, result); summaries.append(result)
    extracted = {result["job"]: result for result in summaries}
    for path in out.glob("QSload*/result.json"):
        record = json.loads(path.read_text())
        extracted.setdefault(record["job"], record)
    summaries = list(extracted.values())
    (out/"qs-load-results.json").write_text(json.dumps(summaries, indent=2))
    print("Extracted", len(summaries), "completed or explicitly reviewed converged-prefix QS records; running ODBs left unopened")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare_loads", "extract_loads", "prepare_unloads", "extract_unloads", "prepare_ae_probe", "extract_ae_probe", "prepare_fine_retry", "enrich_stiffness"), nargs="?", default="audit")
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out", default="Z:/p1/sims/Ti/Validation/audit/work-partition")
    parser.add_argument("--queue-root", help="Explicit existing serial-queue result root")
    parser.add_argument("--refresh", action="store_true", help="Re-read completed QS output rather than reuse its checked extraction")
    parser.add_argument("--jobs", help="Comma-separated completed loading job names to include in a new unload manifest")
    parser.add_argument("--unload-criterion", type=float, default=.11045,
                        help="Positive tensile LE11 target at or below .1105; actual new peak strain must be checked")
    parser.add_argument("--manifest-name", help="Unique proposal filename within the output directory")
    parser.add_argument("--accept-b21-ae-zero", action="store_true",
                        help="Explicitly adopt the B21 mode-rank/source exclusion only after the recorded plastic Explicit probe corroborates zero AE")
    parser.add_argument("--reviewed-partial-job", help="Explicitly authorised terminated failed loading job: inspect its converged prefix, preserve failed solver state")
    parser.add_argument("--retry-source-job", help="Explicitly authorised original QSload_ job for one unique physics-preserving numerical retry")
    options = parser.parse_args()
    globals()[options.action](options)
