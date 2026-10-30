"""Extend exact-source B21 conversion and initial compliance controls to all cases.

Preparation only writes derived inputs. Submission is an explicit serial-queue
action. Existing completed controls are referenced, never copied or overwritten.
Compliance preparation reuses the modal builder with unit_load only; gathering
uses completed serial-queue deliveries and reads their ODBs without upgrading.
"""
import argparse
import hashlib
import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from validation_diagnostics import blocks, key, rows, periodic_patch, prepare_rve, sha, source_mesh, table, extract
from validation_modal import prepare as prepare_modal, job_directory


def campaign_cases(root):
    return [row for row in table(Path(root) / "case-index.csv") if row["campaign"] == "Density campaign"
            and row["mode"] == "Fracture" and row["topology"] in ("tri", "kagome", "hex")
            and row["storage_status"] == "retained_odb"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def compliance_values(displacement, fixed_reaction, load=1.):
    if not all(math.isfinite(value) for value in (displacement, fixed_reaction, load)) or displacement <= 0 or load <= 0:
        raise ValueError("Positive finite unit-load compliance required")
    balance = abs(fixed_reaction + load) / load
    if balance > 1e-5:
        raise ValueError("Unit-load fixed reaction does not balance the applied force")
    return {"load_N": load, "compliance_mm_N": displacement / load,
            "static_stiffness_N_mm": load / displacement, "fixed_pin_reaction_y_N": fixed_reaction,
            "force_balance_relative": balance}


def similarity_controls(root, case, generation_record, *, parent_cases=None, control_paths=None):
    """Derive linear controls only after proving uniform source-model similarity.

    Under x,B,t -> s*(x,B,t), beam area and inertia scale as s² and s⁴.
    B21's shear compensation depends on l²*A/I and is therefore unchanged.
    For D=diag(s,s,1) per beam node, Knew=s³*D^-T*Kparent*D^-1.
    Thus homogenized C is unchanged and unit-force specimen stiffness is s*k.
    This says nothing about nonlinear fracture histories or dynamic equivalence.
    Parent ODB identities remain explicit; no new control solve is claimed.
    Optional parent_cases/control_paths support separately indexed new sources;
    callers must provide a fresh similarity relation, not rewrite generation data.
    """
    from validation_calculations import beam_conversion_modulus

    root = Path(root).resolve()
    record = generation_record
    parent_path, new_path = Path(record["parent_input"]).resolve(), (root / case["inp"]).resolve()
    parent_path.relative_to(root)
    new_path.relative_to(root)
    if (new_path != Path(record["input"]).resolve() or case["inp_sha256"] != record["input_sha256"]
            or sha(new_path) != case["inp_sha256"] or sha(parent_path) != record["parent_input_sha256"]):
        raise ValueError("Similarity source/input hash or identity differs")
    parent_index = root / "case-index.csv" if parent_cases is None else parent_cases
    parent_rows = table(parent_index) if isinstance(parent_index, (str, Path)) else parent_index
    parents = [row for row in parent_rows
               if (root / row["inp"]).resolve() == parent_path]
    if len(parents) != 1:
        raise ValueError("Similarity requires one indexed parent")
    parent = parents[0]
    scale = float(case["l_mm"]) / float(parent["l_mm"])
    if (not math.isfinite(scale) or scale <= 0 or not math.isclose(scale, record["length_scale"], rel_tol=1e-13)
            or record["thickness_density_scale"] != 1
            or float(case["nominal_relative_density"]) != float(parent["nominal_relative_density"])
            or case["topology"] != parent["topology"] or int(case["nnx"]) != int(parent["nnx"])
            or parent["inp_sha256"] != record["parent_input_sha256"]
            or record["topology"] != case["topology"] or int(record["nnx"]) != int(case["nnx"])
            or float(record["l_mm"]) != float(case["l_mm"])
            or float(record["relative_density"]) != float(case["nominal_relative_density"])):
        raise ValueError("Uniform length similarity requires identical topology, mesh and relative density")
    old_text, nodes, elements, section, elastic = source_mesh(parent_path)
    new_text, new_nodes, new_elements, new_section, new_elastic = source_mesh(new_path)
    if list(nodes) != list(new_nodes) or elements != new_elements or elastic != new_elastic:
        raise ValueError("Similarity mesh/connectivity or elastic material differs")
    old_blocks, new_blocks = blocks(old_text), blocks(new_text)
    if list(map(key, old_blocks)) != list(map(key, new_blocks)):
        raise ValueError("Similarity keyword inventory differs")
    # All initial model mechanics, section orientation, sets, pin couplings and
    # boundary cards must be identical. Check every node block, including RPs.
    transformed = {"*node", "*beam section", "*amplitude", "*damage evolution"}
    output = {"*output", "*energy output"}
    for old, new in zip(old_blocks, new_blocks):
        kind = key(old)
        if kind in output:
            continue  # Output resolution/components do not enter the linear controls.
        if old.splitlines()[0] != new.splitlines()[0]:
            raise ValueError("Similarity keyword parameters differ: " + kind)
        if kind not in transformed:
            if rows(old) != rows(new):
                raise ValueError("Similarity mechanics/material/boundary differs: " + kind)
            continue
        a, b = rows(old), rows(new)
        if len(a) != len(b):
            raise ValueError("Similarity transformed row count differs")
        for i, (first, second) in enumerate(zip(a, b)):
            if kind == "*beam section" and i > 0:
                if first != second:
                    raise ValueError("Similarity section orientation differs")
                continue
            x = np.array([float(v) for v in first.split(",") if v.strip()])
            y = np.array([float(v) for v in second.split(",") if v.strip()])
            target = x.copy()
            if kind == "*node":
                target[1:] *= scale
            elif kind == "*beam section":
                target *= scale
            elif kind == "*amplitude":
                target[1::2] *= scale
            else:
                target[1] *= scale
            if x.shape != y.shape or not np.isfinite(y).all() or not np.allclose(target, y, rtol=1e-12, atol=1e-12):
                raise ValueError("Similarity coordinates/section/load/damage scaling differs")
    x, edges, classes, periods, origin, _ = periodic_patch(nodes, elements, parent["topology"], float(parent["l_mm"]))
    y, new_edges, new_classes, new_periods, new_origin, _ = periodic_patch(
        new_nodes, new_elements, case["topology"], float(case["l_mm"]))
    if (not np.array_equal(edges, new_edges) or not np.array_equal(classes, new_classes)
            or not np.allclose(y, scale*x, rtol=1e-12, atol=1e-10)
            or not np.allclose(new_periods, scale*periods, rtol=1e-13)
            or not np.allclose(new_origin, scale*origin, rtol=1e-13, atol=1e-10)):
        raise ValueError("Similarity periodic patch/constraints differ")
    paths = ([root / "audit/controls/campaign-rve-results.json", root / "audit/controls/compliance-summary.json"]
             if control_paths is None else [Path(path) for path in control_paths])
    if len(paths) != 2:
        raise ValueError("Similarity needs one RVE and one compliance summary")
    controls = []
    for path in paths:
        matching = [row for row in json.loads(path.read_text()) if row["case_id"] == parent["case_id"]]
        if len(matching) != 1:
            raise ValueError("Similarity requires one completed parent control per kind")
        control = matching[0]
        if (control.get("state") != "completed_and_extracted" or not control.get("read_only")
                or not control.get("source_input_hash_verified") or control["source_sha256"] != parent["inp_sha256"]
                or Path(control["source_inp"]).resolve() != parent_path
                or sha(control["input"]) != control["input_sha256"]
                or sha(control["source_odb"]) != control["source_odb_sha256"]):
            raise ValueError("Similarity parent control completion/provenance differs")
        controls.append(control)
    rve, compliance = controls
    if (rve.get("analysis") != "linear_static_periodic_rve" or compliance.get("analysis") != "linear_static"
            or compliance.get("control") != "unit_load" or rve["elastic_MPa_nu"] != elastic
            or rve["section_B_t_mm"] != section):
        raise ValueError("Similarity parent control is not the matching linear problem")
    modulus = beam_conversion_modulus(rve["C_MPa"])
    if not math.isclose(modulus, rve["conversion_modulus_MPa"], rel_tol=1e-12):
        raise ValueError("Similarity parent conversion modulus differs from its tensor")
    values = compliance_values(compliance["compliance_mm_N"]*compliance["load_N"],
                               compliance["fixed_pin_reaction_y_N"], compliance["load_N"])
    if not math.isclose(values["static_stiffness_N_mm"], compliance["static_stiffness_N_mm"], rel_tol=1e-12):
        raise ValueError("Similarity parent stiffness/compliance mismatch")
    proof = {"parent_case_id": parent["case_id"], "new_case_id": case["case_id"],
             "parent_input_sha256": parent["inp_sha256"], "new_input_sha256": case["inp_sha256"],
             "length_scale": scale, "all_source_node_blocks_scaled": True,
             "connectivity_material_orientation_sets_pins_BC_verified": True,
             "uniform_section_scaling_verified": True, "periodic_patch_connectivity_classes_verified": True,
             "periodic_patch_coordinate_error_mm": float(np.max(np.abs(y-scale*x))),
             "parent_index_sha256": (sha(parent_index) if isinstance(parent_index, (str, Path)) else
                                      hashlib.sha256(json.dumps(parent_rows, sort_keys=True).encode()).hexdigest()),
             "parent_rve_summary_sha256": sha(paths[0]), "parent_compliance_summary_sha256": sha(paths[1]),
             "parent_control_inputs_sha256": [c["input_sha256"] for c in controls],
             "parent_control_odbs_sha256": [c["source_odb_sha256"] for c in controls],
             "derivation": "A->s^2 A; I->s^4 I; B21 l^2 A/I invariant; Knew=s^3 D^-T K D^-1; Cnew=Cparent; knew=s*kparent",
             "limit": "Linear elastic controls only; no nonlinear/dynamic/fracture result transfer"}
    proof_sha = hashlib.sha256(json.dumps(proof, sort_keys=True, allow_nan=False).encode()).hexdigest()
    mapped = []
    for control in controls:
        item = dict(control)
        item.update(case_id=case["case_id"], source_inp=str(new_path), source_sha256=case["inp_sha256"],
                    source_inp_sha256=case["inp_sha256"], analysis="verified_geometric_similarity",
                    source_kind="verified_geometric_similarity", similarity_proof=proof, similarity_proof_sha256=proof_sha,
                    parent_control_analysis=control["analysis"], parent_control_case_id=parent["case_id"],
                    parent_control_source_inp=str(parent_path), parent_control_source_inp_sha256=parent["inp_sha256"],
                    parent_control_input=str(control["input"]), parent_control_input_sha256=control["input_sha256"],
                    parent_control_odb=str(control["source_odb"]), parent_control_odb_sha256=control["source_odb_sha256"],
                    length_scale=scale, new_control_solve=False, reused=True,
                    reuse_basis="Verified complete source-model geometric similarity; parent control ODB retained explicitly")
        mapped.append(item)
    mapped[0].update(section_B_t_mm=new_section, periods_mm=new_periods.tolist(), volume_mm3=rve["volume_mm3"]*scale**3)
    mapped[1].update(compliance_mm_N=compliance["compliance_mm_N"]/scale,
                     static_stiffness_N_mm=compliance["static_stiffness_N_mm"]*scale)
    return tuple(mapped)


def completion_controls(root, case, generation_record):
    """Resolve original-parent similarity or the fresh13% hexagonal controls."""
    source = (Path(root) / case["inp"]).resolve()
    if (source != Path(generation_record["input"]).resolve()
            or case["inp_sha256"] != generation_record["input_sha256"]
            or sha(source) != case["inp_sha256"]):
        raise ValueError("Completion source differs from its generation record")
    if generation_record["thickness_density_scale"] == 1:
        return similarity_controls(root, case, generation_record)
    root = Path(root).resolve()
    if case["topology"] != "hex" or float(case["nominal_relative_density"]) != .13:
        raise ValueError("Changed-density completion case needs its own reviewed linear controls")
    base = root / "audit/new-validation"
    prepared = table(base / "prepared-case-index.csv")
    parents = [row for row in prepared if row["topology"] == "hex"
               and float(row["nominal_relative_density"]) == .13 and float(row["l_mm"]) == 3]
    if len(parents) != 1:
        raise ValueError("Expected one fresh13percent hexagonal parent")
    parent = parents[0]
    paths = (base / "controls/supplemental-rve-results.json",
             base / "controls/supplemental-compliance-results.json")
    if float(case["l_mm"]) == 3:
        controls = []
        for path in paths:
            records = [row for row in json.loads(path.read_text()) if row["case_id"] == case["case_id"]]
            if len(records) != 1:
                raise ValueError("Fresh completion control is missing/ambiguous")
            item = records[0]
            if (item["source_sha256"] != case["inp_sha256"] or sha(root/case["inp"]) != case["inp_sha256"]
                    or item.get("state") != "completed_and_extracted" or not item.get("read_only")
                    or not item.get("source_input_hash_verified")
                    or sha(item["input"]) != item["input_sha256"]
                    or sha(item["source_odb"]) != item["source_odb_sha256"]):
                raise ValueError("Fresh completion control source or evidence changed")
            controls.append(item)
        return tuple(controls)
    # This relation is between the two NEW13% inputs. The original generation
    # record (whose parent had16% density) is retained and never rewritten.
    relation = dict(generation_record, parent_input=str(root/parent["inp"]),
                    parent_input_sha256=parent["inp_sha256"], length_scale=float(case["l_mm"])/3,
                    thickness_density_scale=1)
    return similarity_controls(root, case, relation, parent_cases=parents, control_paths=paths)


def prepare_completion_controls(root):
    """Prepare only the new hexagonal13% RVE and C0; never submit jobs."""
    root = Path(root).resolve()
    base = root / "audit/new-validation"
    index = base / "prepared-case-index.csv"
    cases = [row for row in table(index) if row["topology"] == "hex" and float(row["l_mm"]) == 3
             and float(row["nominal_relative_density"]) == .13]
    if len(cases) != 1:
        raise ValueError("Expected one prepared3mm hexagonal13percent source")
    case = cases[0]
    source = root / case["inp"]
    if sha(source) != case["inp_sha256"]:
        raise ValueError("Prepared hexagonal source changed")
    out = base / "controls"
    out.mkdir(exist_ok=True)
    _, nodes, elements, section, elastic = source_mesh(source)
    coords, edges, classes, periods, _, _ = periodic_patch(nodes, elements, "hex", 3.)
    rve = prepare_rve(out, case, source, coords, edges, classes, periods, section, elastic)
    prepare_modal(SimpleNamespace(root=root, out=out, case_index=index, inp=[case["inp"].replace("\\", "/")],
                                  modes=10, stiffness=False, controls=("unit_load",)))
    job = "stiffness_hex_3_rho0p13_nx{}_unit_load".format(case["nnx"])
    directory = job_directory(out, job)
    path = directory / "source.json"
    compliance = json.loads(path.read_text())
    compliance.update(input=str(directory / (job + ".inp")), source_sha256=case["inp_sha256"],
                      state="prepared_not_submitted", reason="New13percent beam thickness requires a fresh initial elastic C0")
    write_json(path, compliance)
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out / "queue-hex13"),
                "jobs": [{"job": item["job"], "input": item["input"], "input_sha256": item["input_sha256"],
                          "cpus": 8, "allow_solver_failure": False, "required_outputs": [".odb", ".dat"],
                          "kind": item["analysis"], "purpose": "Fresh13percent hexagonal linear control;10mm counterpart may reuse only after source similarity proof"}
                         for item in (rve, compliance)]}
    proposal = out / "serial-hex13-controls-proposal.json"
    if proposal.exists() and json.loads(proposal.read_text()) != manifest:
        raise ValueError("Refuse to replace a different completion-control proposal")
    write_json(proposal, manifest)
    return {"manifest": str(proposal), "rve": rve, "compliance": compliance}


def gather_completion_controls(root, queue_root=None):
    """Read the two delivered new13% controls in Abaqus Python; no solver call.

    Unlike the original campaign gather, this uses the separate prepared case
    index and writes only supplemental summaries beneath new-validation.
    """
    from odbAccess import openOdb
    from validation_calculations import beam_conversion_modulus

    root = Path(root).resolve()
    base = root / "audit/new-validation"
    out = base / "controls"
    queue = Path(queue_root).resolve() if queue_root is not None else base / "queue-resume-20261030"
    cases = {row["case_id"]: row for row in table(base / "prepared-case-index.csv")}
    state = json.loads((queue / "state.json").read_text())
    completed = {item["job"] for item in state["jobs"] if item["status"] == "completed"}
    metadata_paths = list(out.glob("rve_*/source.json")) + list(out.glob("stiffness_*/source.json"))
    if len(metadata_paths) != 2:
        raise ValueError("Expected exactly two supplemental hexagonal controls")
    verified = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        if metadata["job"] not in completed:
            raise ValueError("Supplemental control is not completed: " + metadata["job"])
        case = cases[metadata["case_id"]]
        source = (root / case["inp"]).resolve()
        inp = metadata_path.parent / (metadata["job"] + ".inp")
        if (case["topology"] != "hex" or float(case["l_mm"]) != 3 or float(case["nominal_relative_density"]) != .13
                or metadata["source_sha256"] != case["inp_sha256"]
                or Path(metadata["source_inp"]).resolve() != source or sha(source) != case["inp_sha256"]
                or Path(metadata["input"]).resolve() != inp.resolve() or sha(inp) != metadata["input_sha256"]):
            raise ValueError("Supplemental control source/prepared identity differs")
        run = job_directory(queue / "runs", metadata["job"])
        delivery = json.loads((run / "delivery.json").read_text())
        entries = {entry["file"]: entry for entry in delivery["files"]}
        if delivery["job"] != metadata["job"] or delivery.get("required_outputs_missing"):
            raise ValueError("Supplemental control delivery is incomplete")
        for suffix in (".inp", ".odb", ".dat"):
            name = metadata["job"] + suffix
            if name not in entries or sha(run / name) != entries[name]["sha256"]:
                raise ValueError("Supplemental control differs from delivery receipt")
        if entries[metadata["job"] + ".inp"]["sha256"] != metadata["input_sha256"]:
            raise ValueError("Supplemental solved/prepared input differs")
        verified.append((metadata_path, metadata, run, entries))
    extract(SimpleNamespace(root=root, out=out, queue_root=queue))
    rves = json.loads((out / "abaqus-rve-results.json").read_text())
    if len(rves) != 1:
        raise ValueError("Expected one extracted supplemental RVE")
    for result in rves:
        row = next(item for item in verified if item[1]["job"] == result["job"])
        if result["source_odb_sha256"] != row[3][result["job"] + ".odb"]["sha256"]:
            raise ValueError("Supplemental RVE extraction/delivery differs")
        result.update(conversion_modulus_MPa=beam_conversion_modulus(result["C_MPa"]),
                      source_kind="exact_source_B21_control", delivery_sha256=sha(row[2] / "delivery.json"),
                      conversion_rule="1/S11 from modified-material2D B21 tensor; no second plane-strain correction")
    compliance = []
    for metadata_path, metadata, run, entries in verified:
        if metadata.get("analysis") == "linear_static_periodic_rve":
            continue
        if metadata.get("analysis") != "linear_static" or metadata.get("control") != "unit_load":
            raise ValueError("Supplemental compliance is not a unit-load linear control")
        odb_path = run / (metadata["job"] + ".odb")
        before = entries[odb_path.name]["sha256"]
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            frame = odb.steps["Compliance"].frames[-1]
            load = frame.fieldOutputs["U"].getSubset(region=odb.rootAssembly.nodeSets["LOAD"]).values
            fixed = frame.fieldOutputs["RF"].getSubset(region=odb.rootAssembly.nodeSets["FIXITY"]).values
            if len(load) != 1 or len(fixed) != 1:
                raise ValueError("Expected one supplemental loading/fixed reference point")
            values = compliance_values(float(load[0].data[1]), float(fixed[0].data[1]))
        finally:
            odb.close()
        if sha(odb_path) != before or sha(metadata["source_inp"]) != metadata["source_sha256"]:
            raise ValueError("Supplemental control changed during read-only extraction")
        result = dict(metadata, **values, source_odb=str(odb_path), source_odb_sha256=before,
                      source_input_hash_verified=True, read_only=True, state="completed_and_extracted",
                      source_kind="exact_source_B21_control", reused=False, delivery_sha256=sha(run / "delivery.json"))
        write_json(metadata_path.parent / "compliance-result.json", result)
        compliance.append(result)
    if len(compliance) != 1:
        raise ValueError("Expected one extracted supplemental compliance")
    rve_path, compliance_path = out / "supplemental-rve-results.json", out / "supplemental-compliance-results.json"
    write_json(rve_path, rves)
    write_json(compliance_path, compliance)
    return {"rve_results": str(rve_path), "compliance_results": str(compliance_path),
            "completed_controls": 2, "original_campaign_summaries_changed": False}


def reused_compliance(root):
    """Normalize the original four already extracted elastic controls; do not copy them."""
    root = Path(root)
    cases = {row["case_id"]: row for row in campaign_cases(root)}
    summary_path = root / "audit/modal/static-summary.json"
    records = []
    for old in json.loads(summary_path.read_text(encoding="utf-8")):
        case = cases[old["case_id"]]
        directory = job_directory(root / "audit/modal", old["job"])
        metadata_path = directory / "source.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        inp = directory / (old["job"] + ".inp")
        odb = directory / (old["job"] + ".odb")
        if (old["source_inp_sha256"] != case["inp_sha256"] or sha(root / case["inp"]) != case["inp_sha256"]
                or metadata["case_id"] != case["case_id"] or metadata["control"] != "unit_load"
                or metadata["analysis"] != "linear_static" or metadata["source_inp_sha256"] != case["inp_sha256"]
                or metadata["input_sha256"] != old["input_sha256"] or sha(inp) != old["input_sha256"]):
            raise ValueError("Existing compliance control source/input mismatch")
        if not odb.is_file() or "COMPLETED" not in (directory / "launcher.txt").read_text(encoding="utf-8"):
            raise ValueError("Existing compliance control is not completed")
        values = compliance_values(float(old["compliance_mm_N"])*float(old["load_N"]),
                                   float(old["fixed_pin_reaction_y_N"]), float(old["load_N"]))
        if not math.isclose(values["static_stiffness_N_mm"], float(old["static_stiffness_N_mm"]), rel_tol=1e-12):
            raise ValueError("Existing compliance/stiffness values disagree")
        records.append(dict(old, **values, source_sha256=case["inp_sha256"],
            source_input_hash_verified=True, read_only=True, state="completed_and_extracted",
            native_source_odb=old["source_odb"], source_odb=str(odb), source_odb_sha256=sha(odb),
            input=str(inp), reused=True, reuse_basis="Original validation_modal.extract opened the completed control ODB readOnly=True; existing values reused after source/input/completion checks",
            reused_summary_sha256=sha(summary_path), source_metadata_sha256=sha(metadata_path)))
    return records


def prepare_compliance(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = campaign_cases(root)
    reused = reused_compliance(root)
    reused_ids = {row["case_id"] for row in reused}
    proposed = []
    for case in cases:
        if case["case_id"] in reused_ids:
            continue
        prepare_modal(SimpleNamespace(root=root, out=out, inp=[case["inp"]], modes=10,
                                      stiffness=False, controls=("unit_load",)))
        name = "stiffness_{}_{}_rho{}_nx{}_unit_load".format(case["topology"], case["l_mm"],
                    case["nominal_relative_density"].replace(".", "p"), case["nnx"])
        directory = job_directory(out, name)
        path = directory / "source.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(input=str(directory / (name + ".inp")), source_sha256=case["inp_sha256"],
                        state="prepared_not_submitted", reason="Exact-source initial elastic C0 for the Hsieh initial-compliance triangle; not nonlinear unloading compliance")
        write_json(path, metadata)
        proposed.append(metadata)
    manifest = {"jobs": [{"job": row["job"], "input": row["input"], "input_sha256": row["input_sha256"],
                          "kind": "linear_static_compliance", "purpose": row["reason"], "cpus": 8,
                          "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]}
                         for row in proposed], "queue_root": str(out / "queue-compliance-campaign-20261028"),
                "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    proposal_path = out / "serial-compliance-proposal.json"
    if proposal_path.exists() and json.loads(proposal_path.read_text()) != manifest:
        raise ValueError("Refuse to replace a different compliance proposal")
    write_json(proposal_path, manifest)
    write_json(out / "reused-compliance-results.json", reused)
    prior_path = out / "compliance-summary.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else []
    combined = {row["case_id"]: row for row in prior}
    combined.update({row["case_id"]: row for row in reused})
    write_json(prior_path, list(combined.values()))
    print(json.dumps({"proposed_jobs": len(proposed), "reused_completed": len(reused),
                      "manifest": str(proposal_path), "summary": str(prior_path), "no_jobs_submitted": True}))


def gather_compliance(root, out):
    """Run with Abaqus Python; extract completed retained queue ODBs read-only."""
    from odbAccess import openOdb
    root, out = Path(root).resolve(), Path(out).resolve()
    queue = out / "queue-compliance-campaign-20261028"
    state = json.loads((queue / "state.json").read_text(encoding="utf-8"))
    completed = {row["job"] for row in state["jobs"] if row["status"] == "completed"}
    cases = {row["case_id"]: row for row in campaign_cases(root)}
    records = reused_compliance(root)
    for metadata_path in sorted(out.glob("stiffness_*/source.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["job"] not in completed:
            continue
        if metadata.get("control") != "unit_load" or metadata.get("analysis") != "linear_static":
            raise ValueError("Non-compliance source in completed control list")
        case = cases[metadata["case_id"]]
        if (metadata_path.parent.resolve() != job_directory(out, metadata["job"])
                or Path(metadata["source_inp"]).resolve() != (root / case["inp"]).resolve()
                or metadata["source_sha256"] != case["inp_sha256"]
                or Path(metadata["input"]).resolve() != (metadata_path.parent / (metadata["job"] + ".inp")).resolve()):
            raise ValueError("Compliance metadata differs from the indexed source or prepared directory")
        if sha(metadata["source_inp"]) != metadata["source_sha256"] or sha(metadata["input"]) != metadata["input_sha256"]:
            raise ValueError("Compliance source or prepared input changed")
        run = job_directory(queue / "runs", metadata["job"])
        delivery = json.loads((run / "delivery.json").read_text(encoding="utf-8"))
        delivered = {entry["file"]: entry for entry in delivery["files"]}
        if delivery["job"] != metadata["job"] or delivery.get("required_outputs_missing"):
            raise ValueError("Incomplete compliance delivery")
        for suffix in (".inp", ".odb", ".dat"):
            name = metadata["job"] + suffix
            if name not in delivered or sha(run / name) != delivered[name]["sha256"]:
                raise ValueError("Compliance retained file does not match delivery receipt")
        if delivered[metadata["job"] + ".inp"]["sha256"] != metadata["input_sha256"]:
            raise ValueError("Solved compliance input differs from prepared input")
        odb_path = run / (metadata["job"] + ".odb")
        before = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            frame = odb.steps["Compliance"].frames[-1]
            load = frame.fieldOutputs["U"].getSubset(region=odb.rootAssembly.nodeSets["LOAD"]).values
            fixed = frame.fieldOutputs["RF"].getSubset(region=odb.rootAssembly.nodeSets["FIXITY"]).values
            if len(load) != 1 or len(fixed) != 1:
                raise ValueError("Expected one loading/fixed reference point")
            values = compliance_values(float(load[0].data[1]), float(fixed[0].data[1]))
        finally:
            odb.close()
        if sha(odb_path) != before:
            raise ValueError("Compliance ODB changed during read-only extraction")
        if sha(metadata["source_inp"]) != metadata["source_sha256"]:
            raise ValueError("Original input changed during compliance extraction")
        record = dict(metadata, **values, source_odb=str(odb_path), source_odb_sha256=before,
                      source_input_hash_verified=True, read_only=True, state="completed_and_extracted", reused=False,
                      delivery_sha256=sha(run / "delivery.json"))
        write_json(metadata_path.parent / "compliance-result.json", record)
        records.append(record)
    if len({row["case_id"] for row in records}) != len(records):
        raise ValueError("Duplicate compliance control")
    write_json(out / "compliance-summary.json", records)
    print("Available exact-source initial elastic compliance controls:", len(records))


def prepare(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((root / "audit/diagnostics/abaqus-rve-results.json").read_text())
    completed = {row["case_id"]: row for row in old}
    cases = [row for row in table(root / "case-index.csv") if row["campaign"] == "Density campaign"
             and row["mode"] == "Fracture" and row["topology"] in ("tri", "kagome", "hex")]
    planned, reused = [], []
    for case in cases:
        source = root / case["inp"]
        if sha(source) != case["inp_sha256"]:
            raise ValueError("Original input changed")
        if case["case_id"] in completed:
            record = completed[case["case_id"]]
            if record["source_sha256"] != case["inp_sha256"] or record["state"] != "completed_and_extracted":
                raise ValueError("Mismatched prior conversion control")
            reused.append(record)
            continue
        _, nodes, elements, section, elastic = source_mesh(source)
        coords, edges, classes, periods, origin, mismatch = periodic_patch(
            nodes, elements, case["topology"], float(case["l_mm"]))
        directory = out / ("rve_{}_{}_rho{}_nx{}".format(case["topology"], case["l_mm"],
                            case["nominal_relative_density"].replace(".", "p"), case["nnx"]))
        if directory.exists():
            metadata = json.loads((directory / "source.json").read_text())
            if metadata["source_sha256"] != case["inp_sha256"] or sha(metadata["input"]) != metadata["input_sha256"]:
                raise ValueError("Occupied control directory needs review")
        else:
            metadata = prepare_rve(out, case, source, coords, edges, classes, periods, section, elastic)
        planned.append(metadata)
    manifest = {"jobs": [{"job": row["job"], "input": row["input"], "input_sha256": row["input_sha256"],
                          "kind": row["analysis"], "purpose": row["reason"], "cpus": 8,
                          "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]}
                         for row in planned], "queue_root": str(out / "queue-rve-campaign-20261028"),
                "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    proposed = out / "serial-rve-proposal.json"
    if proposed.exists() and json.loads(proposed.read_text()) != manifest:
        raise ValueError("Refuse to replace a different proposal")
    proposed.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "reused-rve-results.json").write_text(json.dumps(reused, indent=2))
    print(json.dumps({"proposed_jobs": len(planned), "reused_completed": len(reused),
                      "manifest": str(proposed), "original_sources_changed": False}))


def gather(root, out):
    from validation_calculations import beam_conversion_modulus
    root, out = Path(root), Path(out)
    extract(SimpleNamespace(root=root, out=out, queue_root=out / "queue-rve-campaign-20261028"))
    records = json.loads((out / "abaqus-rve-results.json").read_text())
    reused = json.loads((out / "reused-rve-results.json").read_text())
    combined = records + reused
    if len({row["case_id"] for row in combined}) != len(combined):
        raise ValueError("Duplicate conversion control")
    for record in combined:
        record["conversion_modulus_MPa"] = beam_conversion_modulus(record["C_MPa"])
        record["conversion_rule"] = "1/S11 from exact-source modified-material 2D B21 tensor; no second plane-strain correction"
    (out / "campaign-rve-results.json").write_text(json.dumps(combined, indent=2))
    print("Available exact-source conversion controls:", len(combined))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gather", "prepare-compliance", "gather-compliance",
                                           "prepare-completion", "gather-completion"))
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--out", type=Path, default=Path("Z:/p1/sims/Ti/Validation/audit/controls"))
    parser.add_argument("--queue-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare-completion":
        print(json.dumps(prepare_completion_controls(args.root), indent=2))
    elif args.action == "gather-completion":
        print(json.dumps(gather_completion_controls(args.root, args.queue_root), indent=2))
    else:
        globals()[args.action.replace("-", "_")](args.root, args.out)
