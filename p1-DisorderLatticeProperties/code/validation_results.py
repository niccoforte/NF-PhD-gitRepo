"""Recalculate native validation prefixes with one explicit event/energy rule.

Native IE-SE is a derived predamage plastic-dissipation estimate, conditional
on the B21 formulation/source-card exclusions. It is not falsely called measured
ALLPD or ASTM plastic area. Source-matched nonlinear unloading, when available,
is recorded as a separate method and can replace the selected comparison point.
No source workbook, original simulation, shared calcFT or PSC value is changed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from resources.calculations import calc_FaW
from resources.lattices import Geometry
from validation_calculations import beam_conversion_modulus
from validation_work_partition import SELECTED_INPUTS, eta_energy_metrics, work


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_loading_snapshot(result):
    """Resolve pinned extraction bytes even after later additive enrichment."""
    path = Path(result["matched_load_result"])
    digest = result["matched_load_result_sha256"]
    for candidate in (path, path.parent / "extraction-history" / (digest + ".json")):
        if candidate.exists() and sha(candidate) == digest:
            proof = json.loads(candidate.read_text())
            if (proof["source_odb_sha256"] != result["matched_loading_odb_sha256"]
                    or proof["source_sha256"] != result["source_sha256"]):
                raise ValueError("Pinned loading extraction belongs to a different model/ODB")
            return str(candidate)
    raise ValueError("Pinned loading extraction version is unavailable")


def geometry_and_ki(case, force):
    geom = Geometry(case["topology"], float(case["l_mm"]), int(case["nnx"]),
                    float(case["nominal_relative_density"]))
    geom.FTcalc()
    a, W, B = geom.ai[0], geom.W, geom.B
    KI = force*calc_FaW(a, W)/(B*np.sqrt(W*1000))
    return geom, float(KI)


def native_record(root, case, event, rve, endpoint):
    if endpoint not in ("criterion_last_intact", "criterion_first_crossed"):
        raise ValueError("Unknown initiation endpoint")
    if (rve["source_sha256"] != case["inp_sha256"] or not rve.get("source_input_hash_verified")
            or rve.get("state") != "completed_and_extracted" or not rve.get("read_only")):
        raise ValueError("Conversion control source differs from the indexed input")
    if event["source_sha256"] != case["odb_sha256"]:
        raise ValueError("Initiation event differs from the indexed native ODB")
    directory = root / "audit/history" / Path(case["odb"]).parent / Path(case["odb"]).stem
    curve, energy = read_csv(directory / "native_curve.csv"), read_csv(directory / "native_stored_energy.csv")
    i = int(event["prev_native_index"] if endpoint == "criterion_last_intact" else event["cross_native_index"])
    low, high = int(event["prev_native_index"]), int(event["cross_native_index"])
    if not (0 <= low < high < len(curve) and float(event["prevmaxLE11"]) < .1105 <= float(event["crossmaxLE11"])):
        raise ValueError("Invalid native initiation bracket")
    time = float(curve[i]["time_s"])
    for index, key in ((low, "prevtime_s"), (high, "crosstime_s")):
        # Explicit field/history float timestamps differ by up to1.12e-7 s
        # in this campaign, versus5e-4 s between history observations.
        if abs(float(curve[index]["time_s"])-float(event[key])) > 2e-7:
            raise ValueError("Native initiation index/time differs from event bracket")
    if abs(time-float(energy[i]["time_s"])) > 1e-7:
        raise ValueError("Native reaction and energy are not time aligned")
    geom, KI = geometry_and_ki(case, float(curve[i]["RF2_N"]))
    modulus = beam_conversion_modulus(rve["C_MPa"])
    IE, SE, KE = (float(energy[i][name+"_Nmm"]) for name in ("ALLIE", "ALLSE", "ALLKE"))
    derived_PD = IE-SE
    before = endpoint == "criterion_last_intact"
    eligible = (before and event.get("undamaged_prefix_verified") == "True"
                and event.get("hash_unchanged") == "True" and IE > 0 and KE/IE <= .05
                and np.isfinite([IE, SE, KE, derived_PD, KI]).all() and min(SE, KE, derived_PD, KI) >= 0)
    metric = (eta_energy_metrics(KI_MPa_sqrt_m=KI, conversion_MPa=modulus,
                                area_Nmm=derived_PD if eligible else None,
                                B_mm=geom.B, ligament_mm=geom.W-geom.ai[0],
                                eta=2+.522*(geom.W-geom.ai[0])/geom.W)
              if KI >= 0 else {"K_eta_MPa_sqrt_m": None})
    return {"case_id": case["case_id"], "topology": case["topology"], "l_mm": float(case["l_mm"]),
            "relative_density": float(case["nominal_relative_density"]), "nnx": int(case["nnx"]),
            "endpoint": endpoint, "endpoint_time_s": time, "native_index": i,
            "method": "native_energy_eta", "work_partition": "IE_minus_SE_derived_predamage_PD_eta_estimate",
            "modulus_source": "exact_source_B21_RVE_1_over_S11", "modulus_MPa": modulus,
            "KI_MPa_sqrt_m": KI, "KJ_MPa_sqrt_m": metric["K_eta_MPa_sqrt_m"],
            "value_status": "available_derived_assumptions" if eligible else "unavailable_damaged_or_energy_eligibility",
            "quality_note": "Derived PD, not recorded ALLPD or Hsieh/ASTM Apl. B21 zero AE is corroborated by a separate plastic bending probe; CD/DC/FC absent; saved outer sections undamaged before bracket. Work estimates do not establish Hsieh method agreement.",
            "qualified_ASTM_J": False, "displacement_mm": float(curve[i]["U2_mm"]),
            "loading_state": "native_dynamic", "source_work_convention": "derived_IE_minus_SE",
            "loading_work_Nmm": work([float(row["U2_mm"]) for row in curve[:i+1]],
                                       [float(row["RF2_N"]) for row in curve[:i+1]]),
            "force_N": float(curve[i]["RF2_N"]), "ALLIE_Nmm": IE, "ALLSE_Nmm": SE,
            "ALLKE_Nmm": KE, "KE_IE": KE/IE if IE > 0 else None,
            "derived_predamage_PD_Nmm": derived_PD, "recorded_ALLPD_available": False,
            "surface_threshold_LE11": .1105, "previous_max_LE11": float(event["prevmaxLE11"]),
            "crossing_max_LE11": float(event["crossmaxLE11"]),
            "criterion_lower_time_s": float(event["prevtime_s"]), "criterion_upper_time_s": float(event["crosstime_s"]),
            "source_inp_sha256": case["inp_sha256"], "source_odb_sha256": case["odb_sha256"],
            "modulus_odb_sha256": rve["source_odb_sha256"], "modulus_job": rve["job"],
            "energy_csv_sha256": sha(directory / "native_stored_energy.csv"),
            "curve_csv_sha256": sha(directory / "native_curve.csv")}


def verify_native_exports(root, case, attestations):
    """Bind derived CSVs to the existing read-only ODB extraction receipts."""
    directory = root / "audit/history" / Path(case["odb"]).parent / Path(case["odb"]).stem
    proof = attestations[str((root / case["odb"]).resolve()).lower()]
    curve_proof = json.loads((directory / "summary.json").read_text())
    for receipt in (proof, curve_proof):
        if (not receipt.get("read_only") or not receipt.get("source_unchanged")
                or receipt["before"]["sha256"] != case["odb_sha256"]
                or receipt["after"]["sha256"] != case["odb_sha256"]):
            raise ValueError("Native extraction is not attested to the indexed ODB")
    if proof["csv_sha256"] != sha(directory / "native_stored_energy.csv"):
        raise ValueError("Native energy CSV differs from extraction attestation")
    curve = read_csv(directory / "native_curve.csv")
    with np.load(directory / "native_curve.npz", allow_pickle=False) as packed:
        for column, key in (("time_s", "time"), ("U2_mm", "U2"), ("RF2_N", "RF2")):
            values = np.array([float(row[column]) for row in curve])
            if not np.array_equal(values, packed[key], equal_nan=True):
                raise ValueError("Native curve CSV differs from retained NPZ export")


def triangle_record(record, compliance):
    """Reproduce Appendix D's work convention, retaining negative area as invalid."""
    if (compliance["source_sha256"] != record["source_inp_sha256"]
            or compliance["state"] != "completed_and_extracted"
            or not compliance["source_input_hash_verified"] or not compliance["read_only"]):
        raise ValueError("Initial compliance belongs to a different/unverified source")
    stiffness = float(compliance["static_stiffness_N_mm"])
    if not np.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("Nonpositive initial stiffness")
    rec = dict(record)
    area = rec["loading_work_Nmm"]-rec["force_N"]**2/(2*stiffness)
    # C(T) eta and f(a/W), not Hsieh's SENB specimen calibration.
    geom, _ = geometry_and_ki({"topology": rec["topology"], "l_mm": rec["l_mm"],
                              "nnx": rec["nnx"], "nominal_relative_density": rec["relative_density"]}, rec["force_N"])
    eta = 2+.522*(geom.W-geom.ai[0])/geom.W
    metric = eta_energy_metrics(rec["KI_MPa_sqrt_m"], rec["modulus_MPa"], eta,
                                geom.B, geom.W-geom.ai[0], area) if rec["KI_MPa_sqrt_m"] >= 0 else {}
    valid = rec["value_status"].startswith("available") and area >= 0
    rec.update(method="Hsieh_initial_triangle_native" if rec["loading_state"] == "native_dynamic" else "Hsieh_initial_triangle_QS",
               source_work_convention="initial_triangle", work_partition="integrated_pin_work_minus_initial_stiffness_triangle",
               Apl_Nmm=area, initial_stiffness_N_per_mm=stiffness,
               compliance_source=compliance["job"], compliance_odb_sha256=compliance["source_odb_sha256"],
               KJ_MPa_sqrt_m=metric.get("K_eta_MPa_sqrt_m") if valid else None,
               value_status="available_source_work_convention" if valid else "unavailable_negative_area_or_state",
               quality_note="Hsieh Appendix D initial-slope work convention with this C(T) specimen's f(a/W), eta and exact-source linear compliance. Negative area remains invalid; finite-rotation/ASTM qualification is not established.")
    return rec


def measured_record(result, case, rve):
    """A separate damage-off QS work estimate; never transfer recovery to Explicit."""
    if (result["case_id"] != case["case_id"] or result["source_sha256"] != case["inp_sha256"]
            or rve["source_sha256"] != case["inp_sha256"] or rve.get("state") != "completed_and_extracted"
            or not rve.get("read_only") or not rve.get("source_input_hash_verified") or not result.get("read_only")
            or not result.get("source_input_hash_verified") or not result.get("retained_solver_input_hash_verified")
            or result.get("state") != "completed_and_extracted"):
        raise ValueError("Unloading result source or completion is unverified")
    section = result["loading_section_check"]
    peak = result["loading_peak_state"]
    area = result["work_partition"]["nonrecovered_work_Nmm"]
    modulus = beam_conversion_modulus(rve["C_MPa"])
    geom, ki = geometry_and_ki(case, peak["force_N"])
    metric = eta_energy_metrics(ki, modulus, 2+.522*(geom.W-geom.ai[0])/geom.W,
                                geom.B, geom.W-geom.ai[0], area)
    channels = result["maximum_abs_other_energy_channels_Nmm"]
    checks = [result["force_balance_relative_error"], result["release_force_ramp_relative_error"],
              result["free_loading_pin_RF2_relative"], result["relative_cycle_closure_error"],
              result["peak_metrics"]["conversion_modulus_MPa"]]
    if (not result.get("precriterion_peak_verified") or not section["passed_tensile_criterion"]
            or not .109 <= section["maximum_tensile_LE11"] <= .1105
            or not np.isfinite(checks).all() or min(checks) < 0 or max(checks[:3]) > 1e-4
            or result["relative_cycle_closure_error"] > 1e-3
            or any(channels.get(name, float("inf")) != 0 for name in ("ALLAE", "ALLCD", "ALLDMD", "ALLSD", "ALLVD"))
            or not np.isfinite([area, ki, result["relative_cycle_closure_error"]]).all() or area < 0):
        raise ValueError("Unloading result failed criterion/equilibrium/energy eligibility")
    if abs(result["peak_metrics"]["conversion_modulus_MPa"]-modulus) > 1e-6*modulus:
        raise ValueError("Unloading conversion modulus differs from source RVE")
    return {"case_id": case["case_id"], "topology": case["topology"], "l_mm": float(case["l_mm"]),
            "relative_density": float(case["nominal_relative_density"]), "nnx": int(case["nnx"]),
            "endpoint": "QS_verified_precriterion_peak", "method": "qs_measured_recovery_eta",
            "loading_state": "counterfactual_damage_off_QS", "source_work_convention": "measured_nonlinear_recovery",
            "work_partition": "measured_pin_loading_work_minus_matched_nonlinear_recovery",
            "modulus_source": "exact_source_B21_RVE_1_over_S11", "modulus_MPa": modulus,
            "KI_MPa_sqrt_m": ki, "KJ_MPa_sqrt_m": metric["K_eta_MPa_sqrt_m"],
            "value_status": "available_finite_geometry_work_estimate", "qualified_ASTM_J": False,
            "quality_note": "Damage-off QS control at independently verified LE11 criterion, not a correction transferred to Explicit. Nonrecovered work includes residual elastic energy and unload plasticity; not Hsieh Appendix D Apl or qualified ASTM J.",
            "surface_threshold_LE11": .1105, "peak_max_LE11": section["maximum_tensile_LE11"],
            "force_N": peak["force_N"], "displacement_mm": peak["U2_mm"],
            "loading_work_Nmm": result["work_partition"]["loading_work_Nmm"],
            "measured_recovery_Nmm": result["work_partition"]["measured_recovery_Nmm"],
            "nonrecovered_work_Nmm": area, "relative_cycle_closure_error": result["relative_cycle_closure_error"],
            "unload_added_plastic_dissipation_Nmm": result["unload_added_plastic_dissipation_Nmm"],
            "residual_elastic_energy_Nmm": result["residual_elastic_energy_Nmm"],
            "source_inp_sha256": case["inp_sha256"], "source_odb_sha256": result["source_odb_sha256"],
            "modulus_odb_sha256": rve["source_odb_sha256"], "modulus_job": rve["job"],
            "unload_job": result["job"], "matched_load_result_sha256": result["matched_load_result_sha256"]}


def qs_triangle_record(result, case, rve, compliance):
    """Use an actually converged pre-crossing QS sample, independent of unloading."""
    reviewed = result.get("state") == "failed_with_reviewed_converged_prefix" and result.get("reviewed_partial_usable")
    if (result["source_sha256"] != case["inp_sha256"] or not result.get("source_input_hash_verified")
            or not result.get("read_only") or not result.get("retained_solver_input_hash_verified")
            or not result.get("criterion_found")
            or not (result["state"] == "completed_and_extracted" or reviewed)):
        raise ValueError("QS loading control does not have a verified converged criterion prefix")
    point = result["criterion_points"]["last_before_criterion"]
    if not 0 <= point["maximum_tensile_LE11"] < .1105:
        raise ValueError("QS initial-triangle endpoint is not before the criterion")
    _, ki = geometry_and_ki(case, point["force_N"])
    base = {"case_id": case["case_id"], "topology": case["topology"], "l_mm": float(case["l_mm"]),
            "relative_density": float(case["nominal_relative_density"]), "nnx": int(case["nnx"]),
            "endpoint": "QS_criterion_last_intact", "loading_state": "counterfactual_damage_off_QS",
            "force_N": point["force_N"], "displacement_mm": point["U2_mm"],
            "endpoint_time_s": point["time_s"], "peak_max_LE11": point["maximum_tensile_LE11"],
            "loading_work_Nmm": point["integrated_work_Nmm"], "KI_MPa_sqrt_m": ki,
            "modulus_MPa": beam_conversion_modulus(rve["C_MPa"]),
            "modulus_source": "exact_source_B21_RVE_1_over_S11", "qualified_ASTM_J": False,
            "value_status": "available_QS_converged_prefix", "source_inp_sha256": case["inp_sha256"],
            "source_odb_sha256": result["source_odb_sha256"], "load_job": result["job"],
            "load_solver_state": result.get("solver_state"), "reviewed_converged_prefix": bool(reviewed)}
    return triangle_record(base, compliance)


def run(root):
    root = Path(root)
    out = root / "audit/final"; out.mkdir(exist_ok=True)
    events = {row["case_id"]: row for row in read_csv(root / "audit/failure/failure-initiation.csv")}
    rve_path = root / "audit/controls/campaign-rve-results.json"
    rves = {row["case_id"]: row for row in json.loads(rve_path.read_text())}
    attestations = {str(Path(row["source"]).resolve()).lower(): row
                    for row in json.loads((root / "audit/history/stored-energy-summary.json").read_text())}
    compliance_path = root / "audit/controls/compliance-summary.json"
    compliances = ({row["case_id"]: row for row in json.loads(compliance_path.read_text())}
                   if compliance_path.exists() else {})
    cases = [row for row in read_csv(root / "case-index.csv") if row["campaign"] == "Density campaign"
             and row["mode"] == "Fracture" and row["topology"] in ("tri", "kagome", "hex")]
    all_rows, selected, rejected = [], [], []
    for case in cases:
        if sha(root / case["inp"]) != case["inp_sha256"]:
            raise ValueError("Original input changed since indexing")
        verify_native_exports(root, case, attestations)
        rows = [native_record(root, case, events[case["case_id"]], rves[case["case_id"]], endpoint)
                for endpoint in ("criterion_last_intact", "criterion_first_crossed")]
        if case["case_id"] in compliances:
            rows.append(triangle_record(rows[0], compliances[case["case_id"]]))
        all_rows.extend(rows)
        if case["inp"] in SELECTED_INPUTS:
            selected.extend(row for row in rows if row["endpoint"] == "criterion_last_intact")
            if case["case_id"] in compliances:
                for path in sorted((root / "audit/work-partition").glob("QSload*/result.json")):
                    result = json.loads(path.read_text())
                    if result["case_id"] != case["case_id"] or not result.get("criterion_found"):
                        continue
                    qs = qs_triangle_record(result, case, rves[case["case_id"]], compliances[case["case_id"]])
                    qs["loading_result_sha256"] = sha(path)
                    selected.append(qs)
            for path in sorted((root / "audit/work-partition").glob("QSunload*/result.json")):
                result = json.loads(path.read_text())
                if result["case_id"] != case["case_id"]:
                    continue
                try:
                    snapshot = matched_loading_snapshot(result)
                    measured = measured_record(result, case, rves[case["case_id"]])
                except ValueError as error:
                    rejected.append({"job": result["job"], "case_id": case["case_id"], "reason": str(error)})
                    continue
                measured["unloading_result_sha256"] = sha(path)
                measured["matched_loading_evidence_file"] = snapshot
                selected.append(measured)
    write_csv(out / "all-case-results.csv", all_rows)
    write_csv(out / "validation-results.csv", selected)
    receipt = {"native_cases": len(cases), "all_native_method_rows": len(all_rows),
               "selected_cases": len({row["case_id"] for row in selected}), "selected_method_rows": len(selected),
               "method_counts": {name: sum(row["method"] == name for row in selected) for name in sorted({row["method"] for row in selected})},
               "selection": "Largest existing indexed model per topology/length/density in the published .13/.16/.20 domain; no selection by agreement",
               "rule": "Native last recorded undamaged tensile LE11<.1105 bracket and separately verified QS precriterion peaks; exact-source RVE conversion. Initial-triangle source convention and finite-geometry energy estimates remain distinct.",
               "native_PD_is_derived_not_measured": True, "ASTM_Apl_not_claimed": True,
               "completion": "Recalculated available data only; native and counterfactual QS work estimates coexist and never replace each other's histories. Missing/invalid cases remain explicit.",
               "unavailable_unloading_controls": rejected,
               "compliance_results_sha256": sha(compliance_path) if compliance_path.exists() else None,
               "code_sha256": sha(__file__), "rve_results_sha256": sha(rve_path),
               "event_csv_sha256": sha(root / "audit/failure/failure-initiation.csv")}
    (out / "calculation-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    args = parser.parse_args()
    run(args.root)
