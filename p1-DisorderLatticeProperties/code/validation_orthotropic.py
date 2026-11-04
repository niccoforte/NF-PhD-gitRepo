"""Source-bound SiC orthotropic validation replay; --extract uses Abaqus Python.

Reuse native history exports. Material onset, whole-element deletion and global
force maximum remain distinct. No solver, ODB upgrade or original-data write.
"""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import validation_history_audit as history
import validation_failure_audit as failure


ROOT = Path("Z:/p1/sims/SiC/Validation")
EVENTS = {"DUCTCRT_ge_1": ("DUCTCRT", 1.),
          "SDEG_positive": ("SDEG", 0.),
          "SDEG_near_1": ("SDEG", 1.-1e-4),
          "global_STATUS_zero": ("STATUS", 0.)}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def bracket_history(curve, lower, upper, tolerance=1e-8):
    """Field and history grids differ slightly in recorded float precision.

    A point before the first positive field may already lie inside the uncertain
    onset interval; keep it separate from the point at the verified lower bound.
    """
    before = [i for i, row in enumerate(curve) if row["time_s"] < upper-tolerance]
    intact = [i for i, row in enumerate(curve) if lower is not None and row["time_s"] <= lower+tolerance]
    def point(indices):
        return {"index": indices[-1], **curve[indices[-1]]} if indices else None
    return {"last_history_before_positive_field": point(before),
            "last_history_at_or_before_last_nontriggering_field": point(intact),
            "time_matching_tolerance_s": tolerance,
            "limit": "The before-positive point can lie inside the onset bracket; no crossing time is interpolated."}


def triggered(name, minimum, maximum_value):
    if minimum is None or maximum_value is None:
        return False
    if name == "global_STATUS_zero":
        return minimum == 0.
    if name == "SDEG_positive":
        return maximum_value > 0.
    return maximum_value >= EVENTS[name][1]


def locations(field, name, limit=12):
    found = []
    for value in field.values:
        data = np.asarray(value.data, dtype=float).reshape(-1)
        if not data.size or not np.isfinite(data).all():
            continue
        if triggered(name, float(data.min()), float(data.max())):
            found.append({"instance": value.instance.name,
                          "element": int(value.elementLabel),
                          "section_point": getattr(value.sectionPoint, "number", None),
                          "values": data.tolist()})
            if len(found) >= limit:
                break
    return found


def field_replay(path, case_out):
    from odbAccess import openOdb
    curve = read_rows(case_out/"native_curve.csv")
    energy_rows = read_rows(case_out/"native_energy.csv")
    energy = [(r["time_s"], r["ALLKE_Nmm"], r["ALLIE_Nmm"], r["KE_over_IE"]) for r in energy_rows]
    previous, events, rows, errors = {}, {}, [], []
    odb = openOdb(path=str(path), readOnly=True)
    try:
        for index, frame in enumerate(odb.steps["Step-1"].frames):
            row = {"frame": index, "time_s": float(frame.frameValue)}
            for variable in ("DUCTCRT", "SDEG", "STATUS"):
                if variable not in frame.fieldOutputs:
                    row[variable] = None
                    continue
                field = frame.fieldOutputs[variable]
                try:
                    stats = history.field_summary(field)
                    row[variable] = stats
                    for name, (wanted, threshold) in EVENTS.items():
                        if wanted != variable or name in events:
                            continue
                        if triggered(name, stats["min"], stats["max"]):
                            lower = previous.get(name)
                            event = {"variable": variable, "threshold": threshold,
                                     "first_frame": index, "first_positive_field_time_s": float(frame.frameValue),
                                     "last_nontriggering_field_time_s": lower,
                                     "locations_first_12": locations(field, name),
                                     **bracket_history(curve, lower, float(frame.frameValue))}
                            intact = event["last_history_at_or_before_last_nontriggering_field"]
                            event["energy_through_lower_bracket_history"] = history.energy_window(
                                energy, intact["time_s"] if intact else None, True, "through_lower_bracket_history")
                            events[name] = event
                        elif stats["min"] is not None and stats["nonfinite"] == 0:
                            previous[name] = float(frame.frameValue)
                except Exception as error:
                    row[variable] = None
                    errors.append({"frame": index, "variable": variable, "error": str(error)})
            if "S" in frame.fieldOutputs:
                field = frame.fieldOutputs["S"]
                try:
                    component = list(field.componentLabels).index("S11")
                    row["S11_tensile_max"] = failure.maximum(field, component)
                except Exception as error:
                    row["S11_tensile_max"] = None
                    errors.append({"frame": index, "variable": "S11", "error": str(error)})
            else:
                row["S11_tensile_max"] = None
            rows.append(row)
    finally:
        odb.close()
    selected = "DUCTCRT_ge_1" if "DUCTCRT_ge_1" in events else None
    result = {"source": str(path), "events": events, "field_errors": errors,
              "selected_material_onset": selected,
              "near_complete_SDEG_threshold_is_diagnostic": True,
              "interpretation": "First local damage criterion and whole-element deletion are separate; no temporal force-maximum initiation rule.",
              "field_frames": rows}
    history.json_dump(case_out/"orthotropic_events.json", result)
    return result


def extract(root=ROOT):
    root = Path(root)
    audit = root/"audit"
    audit.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.glob("Fracture-*square-20-*-per-1.odb"))
    if len(paths) != 8:
        raise ValueError("Expected exactly eight square/45square source ODBs")
    sources = paths + [p.with_suffix(".inp") for p in paths]
    sources += sorted((root/"transfer").glob("OUT-Fracture-*square-20-*-per-1.csv"))
    if len(sources) != 24 or not all(p.is_file() for p in sources):
        raise ValueError("Expected eight ODB, INP and original transfer triplets")
    print("Hashing eight original ODB/INP/export triplets", flush=True)
    before = {str(path): history.fingerprint(path) for path in sources}
    receipt = audit/"orthotropic-extraction.json"
    code = {str(Path(__file__).resolve()): history.fingerprint(Path(__file__))["sha256"],
            str(Path(history.__file__).resolve()): history.fingerprint(Path(history.__file__))["sha256"],
            str(Path(failure.__file__).resolve()): history.fingerprint(Path(failure.__file__))["sha256"]}
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old["sources_before"] != before:
            raise ValueError("Original source identities changed since the retained replay")
        if old.get("state") == "complete" and old.get("extractor_code") == code:
            if any(history.fingerprint(Path(path))["sha256"] != digest
                   for path, digest in old.get("output_sha256", {}).items()):
                raise ValueError("Retained native replay output changed since extraction")
            print("Existing complete replay matches sources, code and outputs; no duplicate export", flush=True)
            return old
    records = []
    for path in paths:
        print("Replaying " + path.name, flush=True)
        rec = history.audit_case(path, audit/"history", histories_only=True)
        if not rec["source_unchanged"] or not rec["load_time_grids_equal"] or not rec["status_load_time_grids_equal"]:
            raise ValueError("Unchanged aligned native source required: " + path.name)
        case_out = audit/"history"/path.parent.parent.name/path.parent.name/path.stem
        replay = field_replay(path, case_out)
        rec["orthotropic_events"] = str(case_out/"orthotropic_events.json")
        rec["selected_material_onset"] = replay["selected_material_onset"]
        rec["field_error_count"] = len(replay["field_errors"])
        records.append(rec)
        history.json_dump(receipt, {"state": "extracting", "sources_before": before,
                                   "extractor_code": code, "cases": records})
    print("Verifying original source hashes after the full batch", flush=True)
    after = {str(path): history.fingerprint(path) for path in sources}
    if before != after:
        raise ValueError("Original source changed during complete batch replay")
    outputs = {str(path): history.fingerprint(path)["sha256"]
               for path in sorted((audit/"history").rglob("*")) if path.is_file()}
    result = {"state": "complete", "sources_before": before, "sources_after": after,
              "source_unchanged": before == after, "extractor_code": code,
              "output_sha256": outputs, "cases": records,
              "scope": "Read-only native SiC replay, no FEA or ODB upgrade; source transfer CSVs are unchanged.",
              "pending": "Source geometry/tensor/compliance checks and elastic reference comparison; no accepted new toughness values yet."}
    history.json_dump(receipt, result)
    print("Completed eight source-bound SiC native replays", flush=True)
    return result


def orthotropic_elastic(C):
    """Use an already-mapped tensor, retaining its numerical symmetry cleanup.

    Source coordinates and Abaqus single-precision reactions can leave tiny
    symmetry/coupling residuals. Project only residuals below 1e-6 of max(C),
    retaining raw C and the full correction; larger couplings need a new model.
    """
    from resources.lattices import prismatic_plane_strain

    raw = np.asarray(C, dtype=float)
    if raw.shape != (3, 3) or not np.isfinite(raw).all():
        raise ValueError("Finite 3-by-3 engineering-shear C is required")
    scale = float(np.max(np.abs(raw)))
    if scale <= 0:
        raise ValueError("Nonzero positive-definite tensor required")
    aligned = (raw + raw.T)/2
    aligned[:2, 2] = 0.
    aligned[2, :2] = 0.
    correction = float(np.max(np.abs(aligned-raw))/scale)
    if correction > 1e-6:
        raise ValueError("Tensor is not numerically aligned orthotropic")
    reduced = prismatic_plane_strain(aligned, C_state="plane_strain", crack_axis=0)
    lam, chi = float(reduced["lambda_pe"]), float(reduced["chi_pe"])
    return {"C_raw_MPa": raw.tolist(), "C_aligned_MPa": aligned.tolist(),
            "alignment_max_relative_correction": correction,
            "alignment_rule": "Symmetrise numerical reactions and zero only normal/shear residuals below1e-6*max(abs(C)); retain unequal axial constants.",
            "C_state": "plane_strain", "crack_axis": "global x", "opening_axis": "global y",
            "S_pe_MPa_inverse": reduced["S_pe"].tolist(),
            "M_I_pe_MPa": float(reduced["M_I_pe"]),
            "lambda_Ortega": lam, "chi_Ortega": chi,
            "Ortega_fit_parameter_domain": {"lambda": [.03162, 10.], "chi": [.1, 10.]},
            "within_Ortega_fitted_parameter_domain": bool(.03162 <= lam <= 10 and .1 <= chi <= 10),
            "modulus_rule": "Already-mapped B21 homogenisation used directly, without a second plane-strain reduction."}


def extract_controls(root=ROOT, queue_root=None):
    """Gather completed serial deliveries through the existing RVE extractor."""
    from odbAccess import openOdb
    from validation_controls import compliance_values
    from validation_diagnostics import extract as extract_rves, sha, table

    root = Path(root).resolve()
    out = root/"audit/controls"
    queue = Path(queue_root).resolve() if queue_root else out/"queue"
    cases = {row["case_id"]: row for row in table(root/"audit/case-index.csv")}
    queue_paths = list(dict.fromkeys((queue, out/"queue-compliance", out/"queue-tail")))
    states, completed, queue_records = [], {}, {}
    for selected_queue in queue_paths:
        state_path = selected_queue/"state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        states.append({"queue": str(selected_queue), "phase": state["phase"],
                       "state_sha256": sha(state_path)})
        queue_records[str(selected_queue)] = state
        for row in state["jobs"]:
            if row["status"] != "completed":
                continue
            if row["job"] in completed:
                raise ValueError("A completed job appears in more than one queue")
            completed[row["job"]] = selected_queue
    accepted = {}
    review_path = out/"reviewed-wrapper-failure.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        reviewed_queue = Path(review["queue_root"])
        records = queue_records.get(str(reviewed_queue), {}).get("jobs", [])
        recorded = [row for row in records if row["job"] == review["job"]]
        if (review.get("state") != "accepted_control_only" or not review.get("accepted_for_calculation")
                or len(recorded) != 1 or recorded[0]["status"] != "failed"
                or sha(reviewed_queue/"state.json") != review["queue_state_sha256"]
                or sha(reviewed_queue/"runs"/review["job"]/"delivery.json") != review["delivery_sha256"]):
            raise ValueError("Reviewed wrapper-failure identity/status changed")
        if review["job"] in completed:
            raise ValueError("Reviewed failed job is also marked completed")
        for name, digest in review["reviewed_file_sha256"].items():
            if sha(reviewed_queue/"runs"/review["job"]/name) != digest:
                raise ValueError("Reviewed wrapper-failure artifact changed")
        completed[review["job"]] = reviewed_queue
        accepted[review["job"]] = review
    if not states:
        return {"state": "not_started", "queue": str(queue), "cases": []}
    verified, missing = [], []
    for metadata_path in sorted(out.glob("*/source.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata["job"] not in completed:
            missing.append(metadata["job"])
            continue
        case = cases[metadata["case_id"]]
        source = (root/case["inp"]).resolve()
        proposed = metadata_path.parent/(metadata["job"]+".inp")
        if (Path(metadata["source_inp"]).resolve() != source
                or metadata.get("source_sha256", metadata.get("source_inp_sha256")) != case["inp_sha256"]
                or sha(source) != case["inp_sha256"] or sha(proposed) != metadata["input_sha256"]):
            raise ValueError("Original/prepared control source identity differs")
        run = completed[metadata["job"]]/"runs"/metadata["job"]
        delivery = json.loads((run/"delivery.json").read_text())
        entries = {row["file"]: row for row in delivery["files"]}
        if delivery["job"] != metadata["job"] or delivery.get("required_outputs_missing"):
            raise ValueError("Control delivery is incomplete")
        for suffix in (".inp", ".odb", ".dat"):
            name = metadata["job"]+suffix
            if name not in entries or sha(run/name) != entries[name]["sha256"]:
                raise ValueError("Delivered control differs from its receipt")
        if entries[proposed.name]["sha256"] != metadata["input_sha256"]:
            raise ValueError("Solved and prepared control inputs differ")
        verified.append((metadata_path, metadata, run, entries))
    extract_rves(SimpleNamespace(root=root, out=out, queue_root=queue))
    rves = {row["job"]: row for row in json.loads((out/"abaqus-rve-results.json").read_text())}
    results = []
    for metadata_path, metadata, run, entries in verified:
        odb_path = run/(metadata["job"]+".odb")
        before = entries[odb_path.name]["sha256"]
        if metadata["analysis"] == "linear_static_periodic_rve":
            result = rves[metadata["job"]]
            if result["source_odb_sha256"] != before:
                raise ValueError("RVE extraction/delivery identity differs")
            result = dict(result, **orthotropic_elastic(result["C_MPa"]))
        elif metadata["analysis"] == "linear_static" and metadata.get("control") == "unit_load":
            odb = openOdb(str(odb_path), readOnly=True)
            try:
                frame = odb.steps["Compliance"].frames[-1]
                load = frame.fieldOutputs["U"].getSubset(region=odb.rootAssembly.nodeSets["LOAD"]).values
                fixed = frame.fieldOutputs["RF"].getSubset(region=odb.rootAssembly.nodeSets["FIXITY"]).values
                if len(load) != 1 or len(fixed) != 1:
                    raise ValueError("Expected one loading/fixed reference point")
                values = compliance_values(float(load[0].data[1]), float(fixed[0].data[1]))
                if metadata["job"] in accepted:
                    reviewed = accepted[metadata["job"]]
                    if (abs(float(frame.frameValue)-1.) > 1e-12
                            or not np.isclose(values["compliance_mm_N"], reviewed["native_validation"]["compliance_mm_N"], rtol=1e-12)
                            or before != reviewed["source_odb_sha256"]):
                        raise ValueError("Accepted wrapper-failure native result changed")
            finally:
                odb.close()
            result = dict(metadata, **values, source_odb=str(odb_path), source_odb_sha256=before)
        else:
            raise ValueError("Unexpected orthotropic control kind")
        if sha(odb_path) != before or sha(metadata["source_inp"]) != cases[metadata["case_id"]]["inp_sha256"]:
            raise ValueError("Control/source changed during read-only extraction")
        result.update(state="completed_and_extracted", read_only=True,
                      source_input_hash_verified=True, delivery_sha256=sha(run/"delivery.json"),
                      queue_recorded_status="failed" if metadata["job"] in accepted else "completed",
                      calculation_usable=True)
        if metadata["job"] in accepted:
            result.update(wrapper_failure_review=str(review_path), wrapper_failure_review_sha256=sha(review_path),
                          wrapper_failure_acceptance_reason=accepted[metadata["job"]]["reason"])
        results.append(result)
    result = {"state": "complete" if not missing else "partial", "queue_states": states,
              "case_index_sha256": sha(root/"audit/case-index.csv"),
              "pending_jobs": missing, "controls": results,
              "scope": "Source/delivery-bound completed or individually reviewed RVE/unit-load controls; queue failures retain their recorded status; pending jobs produce no values.",
              "extractor_sha256": sha(__file__),
              "extractor_function_sha256": hashlib.sha256(inspect.getsource(extract_controls).encode("utf-8")).hexdigest()}
    history.json_dump(out/"control-results.json", result)
    print("Extracted {} completed controls; {} pending".format(len(results), len(missing)), flush=True)
    return result


def analyse(root=ROOT):
    """Replay the elastic comparison with independently calibrated factors.

    No value is accepted merely because it agrees with a literature curve.
    The plotted point is the last history sample at the undamaged field bound.
    """
    from resources.calculations import calc_FaW_aniso, calc_FaW
    from validation_diagnostics import sha, table, write_table

    root = Path(root)
    audit = root/'audit'
    native = json.loads((audit/'orthotropic-extraction.json').read_text())
    controls = json.loads((audit/'controls/control-results.json').read_text())
    calibration_path=audit/'elastic-calibration/calibration-summary.json'
    calibration=json.loads(calibration_path.read_text())
    reference_path = audit/'reference-workbook-review.json'
    reference = json.loads(reference_path.read_text())
    if (native['state'] != 'complete' or controls['state'] != 'complete'
            or calibration['state']!='complete'):
        raise ValueError('Complete native replay and sixteen usable controls are required')
    factors={r['case_id']:r for r in calibration['factors']}
    if len(factors)!=8:raise ValueError('One directly calibrated factor per case required')
    for path, digest in native['output_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Native export changed: '+path)
    if sha(reference['workbook']) != reference['workbook_sha256']:
        raise ValueError('Reference workbook changed since its source review')
    geom = {r['job']: r for r in json.loads((audit/'ortho-geometry-controls.json').read_text())['cases']}
    cases = {r['case_id']: r for r in table(audit/'case-index.csv')}
    baseline = {(r['topology'], r['case']): r for r in reference['case_mapping']}
    cell = {r['cell']: r['cached'] for row in reference['rows'] for r in row}
    source_points = {}
    for topology, xcol, ycol, last in [('45square','A','B',25),('square','I','J',44)]:
        xy = [(float(cell[f'{xcol}{i}']), float(cell[f'{ycol}{i}'])) for i in range(3,last+1)]
        source_points[topology] = sorted(xy)
    result = []
    for rec in native['cases']:
        name = Path(rec['source']).stem
        case, geometry = cases[name], geom[name]
        if sha(root/case['inp']) != case['inp_sha256']:
            raise ValueError('Source input changed before calculation')
        rve = [r for r in controls['controls'] if r['case_id']==name and r['analysis']=='linear_static_periodic_rve']
        c0 = [r for r in controls['controls'] if r['case_id']==name and r['analysis']=='linear_static']
        if len(rve)!=1 or len(c0)!=1:
            raise ValueError('Expected one authoritative tensor and compliance per case')
        tensor, compliance = rve[0], c0[0]
        event_path = Path(rec['orthotropic_events'])
        event_data = json.loads(event_path.read_text())
        event = event_data['events']['DUCTCRT_ge_1']
        point = event['last_history_at_or_before_last_nontriggering_field']
        if point is None:
            raise ValueError('No observed undamaged lower history bound')
        W, B, length = geometry['W_pin_to_back_mm'], geometry['B_mm'], float(case['l_mm'])
        a = .5*W
        extrapolated_factor = calc_FaW_aniso(a,W,C=np.asarray(tensor['C_aligned_MPa']))
        factor = factors[name]['factor']
        conversion = factor/(B*np.sqrt(W))
        load = point['RF2_N']
        K = load*conversion
        normaliser = geometry['solid_tensile_strength_MPa']*np.sqrt(length)
        curve_path = event_path.parent/'native_curve.csv'
        curve = read_rows(curve_path)
        bracket_load = [r['RF2_N'] for r in curve
                        if point['time_s']-1e-8 <= r['time_s'] <= event['first_positive_field_time_s']+1e-8]
        prefix = curve[:point['index']+1]
        work = float(np.trapz([r['RF2_N'] for r in prefix],[r['U2_mm'] for r in prefix]))
        initial_triangle = load**2/(2*compliance['static_stiffness_N_mm'])
        energy = event['energy_through_lower_bracket_history']
        ref = baseline[(case['topology'],case['batch'])]
        xx,yy = np.array(source_points[case['topology']]).T
        rho = float(case['nominal_relative_density'])
        if not xx.min() <= rho <= xx.max():
            raise ValueError('Literature interpolation requires observed density support')
        yref = float(np.exp(np.interp(np.log(rho),np.log(xx),np.log(yy))))
        result.append(dict(case_id=name,topology=case['topology'],batch=case['batch'],
            relative_density=rho,l_mm=length,W_mm=W,B_mm=B,a_mm=a,a_over_W=.5,
            crack_convention='Nominal generator cut x=.6L, pin x=.2L, W=.8L; first intact member is not substituted for the nominal cut.',
            event='first_local_DUCTCRT_ge_1',event_lower_s=point['time_s'],
            event_upper_s=event['first_positive_field_time_s'],history_index=point['index'],
            load_N=load,opening_mm=point['U2_mm'],
            lambda_Ortega=tensor['lambda_Ortega'],chi_Ortega=tensor['chi_Ortega'],
            Ortega_in_fit_domain=tensor['within_Ortega_fitted_parameter_domain'],
            elastic_factor=factor,M_MPa=tensor['M_I_pe_MPa'],k0_N_mm=compliance['static_stiffness_N_mm'],
            elastic_calibration=factors[name]['method'],
            calibration_odb_sha256=factors[name]['selected_odb_sha256'],
            extrapolated_Ortega_fit_diagnostic_only=extrapolated_factor,
            K_I_MPa_sqrt_mm=K,K_I_MPa_sqrt_m=K/np.sqrt(1000),
            K_normalised=K/normaliser,
            recorded_bracket_K_normalised_min=min(bracket_load)*conversion/normaliser,
            recorded_bracket_K_normalised_max=max(bracket_load)*conversion/normaliser,
            reference_normalised=yref,relative_difference_percent=100*(K/normaliser/yref-1),
            legacy_workbook_normalised=ref['old_plotted_Knorm'],
            isotropic_factor_only_diagnostic=load*calc_FaW(a,W)/(B*np.sqrt(W)*normaliser),
            J_el_N_per_mm=K**2/tensor['M_I_pe_MPa'],native_work_N_mm=work,
            initial_triangle_N_mm=initial_triangle,initial_triangle_residual_N_mm=work-initial_triangle,
            KE_IE_at_lower_bound=energy['positive_IE']['last'][3],
            KE_IE_max_startup_sensitivity=energy['startup_sensitivity_t_ge_0p1_event_IE_gt_1pct_max']['ke_ie_max'],
            calculation_status='source_bound_direct_elastic_calibration',
            original_history_samples=rec['load_histories']['U2']['count'],
            source_odb=rec['source'],source_odb_sha256=rec['before']['sha256'],
            source_inp_sha256=case['inp_sha256'],native_curve_sha256=sha(curve_path),
            tensor_odb_sha256=tensor['source_odb_sha256'],compliance_odb_sha256=compliance['source_odb_sha256']))
    out = audit/'comparison'
    out.mkdir(exist_ok=True)
    write_table(out/'points.csv', result)
    history.json_dump(out/'comparison.json', {'points':result,'reference_points':source_points,
        'reference_review_sha256':sha(reference_path),
        'control_results_sha256':sha(audit/'controls/control-results.json'),
        'calibration_sha256':sha(calibration_path),
        'reference_field_limits_sha256':sha(audit/'reference-field-limits.json'),
        'native_replay_sha256':sha(audit/'orthotropic-extraction.json'),
        'source':reference['benchmark'],
        'reference_comparison_limit':'Reference600l K-field boundary layer versus finite CT ligaments8l/11.3l; long square shear-lag lengths are a source-supported possible contributor, not a fitted correction.',
        'status':'Eight source-bound elastic comparisons use directly calibrated factors, with benchmark residuals retained.',
        'selected_event':'First local tensile material initiation; use last history at or before last undamaged field. No peak-force selection.',
        'next_gate':'Report the remaining benchmark differences without tuning calibration to the literature; proceed to FCC and shared population definitions.'})
    print('Calculated eight source-bound elastic comparisons using directly calibrated factors.',flush=True)
    return result


def plot(root=ROOT):
    """Production formatting; caption carries the calibration qualification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out=Path(root)/'audit/comparison'
    data=json.loads((out/'comparison.json').read_text())
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(8,3.7))
        for ax,topology,title in zip(axes,('square','45square'),(r'(a) $0/90^\circ$ square',r'(b) $\pm45^\circ$ square')):
            xx,yy=np.array(data['reference_points'][topology]).T
            ax.plot(xx,yy,color='0.25',lw=1.2)
            for row in data['points']:
                if row['topology']!=topology: continue
                y=row['K_normalised']
                low=min(y,row['recorded_bracket_K_normalised_min'])
                high=max(y,row['recorded_bracket_K_normalised_max'])
                ax.errorbar(row['relative_density'],y,yerr=[[y-low],[high-y]],fmt='o',
                            ms=5,color='#3067a3',capsize=3,lw=1.1)
            ax.set_title(title)
            ax.set_xlabel(r'Relative density, $\bar\rho$')
            ax.set_xlim(.15,.345)
            ax.set_ylim(bottom=.025)
            ax.grid(alpha=.2)
        axes[0].set_ylabel(r'$K_{I,\mathrm{init}}/(\sigma_f\sqrt{\ell})$')
        fig.legend(handles=[Line2D([],[],color='0.25',lw=1.2,label='Romijn & Fleck'),
                            Line2D([],[],marker='o',ls='none',color='#3067a3',label='Current study')],
                   loc='lower center',ncol=2,frameon=False)
        fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.24,wspace=.27)
        for ext in ('png','pdf','svg'):
            fig.savefig(out/('orthotropic-validation.'+ext),dpi=250)
        plt.close(fig)
    (out/'caption.txt').write_text(
        'Elastic initiation comparison with original digitised Romijn & Fleck (2007) Fig.8a. '
        'Densities use the source slender-beam convention 2t/l, with solid tensile strength 550 MPa. '
        'Current study points use native paired load/opening at the last saved history at the last undamaged field. '
        'Bars show the recorded load range across the local DUCTCRT=1 field bracket, not statistical uncertainty. '
        'The mapped B21 tensors are already constrained and are used once. All eight chi values lie outside '
        'Ortega\'s fitted parameter range, so the factors are obtained directly from linear C(T) compliance '
        'using the measured reduced tensors. Independent isotropic, mesh, crack-step and energy checks '
        'are recorded in calibration-summary.json. Differences from the boundary-layer reference remain '
        'reported, without a fitted correction. No source simulations '
        'or historical workbook values have been overwritten.\n',encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--extract", action="store_true")
    action.add_argument("--extract-controls", action="store_true")
    action.add_argument("--analyse", action="store_true")
    action.add_argument("--plot", action="store_true")
    parser.add_argument("--queue-root", type=Path)
    args = parser.parse_args()
    if args.extract_controls:
        extract_controls(args.root, args.queue_root)
    elif args.analyse:
        analyse(args.root)
    elif args.plot:
        plot(args.root)
    else:
        extract(args.root)
