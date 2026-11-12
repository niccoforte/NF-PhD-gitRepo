"""Extend exact-source B21 conversion and initial compliance controls to all cases.

Preparation only writes derived inputs. Submission is an explicit serial-queue
action. Existing completed controls are referenced, never copied or overwritten.
Compliance preparation reuses the modal builder with unit_load only; gathering
uses completed serial-queue deliveries and reads their ODBs without upgrading.
"""
import argparse
import csv
import hashlib
import math
import json
import re
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


def verified_control_delivery(meta, queue, completed):
    """Bind an extracted control to unchanged sources and its serial receipt."""
    if sha(meta["source_inp"]) != meta["source_sha256"] or sha(meta["input"]) != meta["input_sha256"]:
        raise ValueError("Control source/input changed")
    if meta["job"] not in completed:
        raise ValueError("Control job has not completed")
    run = job_directory(Path(queue) / "runs", meta["job"])
    receipt_path = run / "delivery.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["job"] != meta["job"] or receipt.get("required_outputs_missing"):
        raise ValueError("Incomplete control delivery")
    delivered = {item["file"]: item for item in receipt["files"]}
    for suffix in (".inp", ".odb", ".dat"):
        filename = meta["job"] + suffix
        if filename not in delivered or sha(run / filename) != delivered[filename]["sha256"]:
            raise ValueError("Control delivery file differs from receipt")
    if sha(run / (meta["job"] + ".inp")) != meta["input_sha256"]:
        raise ValueError("Solved control input differs")
    return run / (meta["job"] + ".odb"), receipt_path


def body_tensor_source(source):
    """Extract the existing BodySet graph and its actual four outer side sets."""
    parsed = blocks(Path(source).read_text(encoding="ascii"))
    nodes, elements, body_sets, sections, materials = {}, {}, [], [], []
    sides = {name: [] for name in ("set-left", "set-right", "set-topbody", "set-bottombody")}
    in_instance, instances = False, 0

    def parameters(item):
        return dict(piece.strip().lower().split("=", 1) for piece in item.splitlines()[0].split(",")[1:]
                    if "=" in piece)

    def labels(item):
        values = [[int(v.strip()) for v in line.split(",") if v.strip()] for line in rows(item)]
        if "generate" in item.splitlines()[0].lower():
            if any(len(row) != 3 or row[2] <= 0 or row[1] < row[0] for row in values):
                raise ValueError("Invalid generated body/set labels")
            return {v for row in values for v in range(row[0], row[1]+1, row[2])}
        return {v for row in values for v in row}

    for item in parsed:
        kind, params = key(item), parameters(item)
        if kind == "*include":
            raise ValueError("Body tensor controls require self-contained sources")
        if kind == "*instance":
            instances += 1
            in_instance = True
            if rows(item):
                raise ValueError("Transformed source instances need separate review")
        elif kind == "*end instance":
            in_instance = False
        elif kind == "*node":
            if not in_instance:
                raise ValueError("Expected the archived mesh inside one instance")
            for line in rows(item):
                values = [v.strip() for v in line.split(",")]
                label = int(values[0])
                if label in nodes:
                    raise ValueError("Duplicate source node")
                if len(values) > 3 and any(float(v) != 0 for v in values[3:] if v):
                    raise ValueError("Only planar B21 source coordinates are supported")
                nodes[label] = [float(values[1]), float(values[2])]
        elif kind == "*element":
            if not in_instance or params.get("type") != "b21":
                raise ValueError("Body tensor controls require exact B21 elements")
            for line in rows(item):
                values = [int(v.strip()) for v in line.split(",")]
                if len(values) != 3 or values[0] in elements:
                    raise ValueError("Invalid/duplicate B21 source element")
                elements[values[0]] = values[1:]
        elif kind == "*elset" and params.get("elset") == "bodyset":
            body_sets.append(labels(item))
        elif kind == "*nset" and params.get("nset") in sides:
            sides[params["nset"]].append(labels(item))
        elif kind == "*beam section" and params.get("elset") == "bodyset":
            if params.get("section") != "rect" or len(rows(item)) != 2:
                raise ValueError("One rectangular BodySet section is required")
            sections.append(item)
        elif kind == "*elastic":
            materials.append(item)
    if instances != 1 or in_instance or len(body_sets) != 1 or len(sections) != 1 or len(materials) != 1:
        raise ValueError("One source instance, BodySet, body section and elastic material are required")
    ids = body_sets[0]
    if not ids or not ids.issubset(elements):
        raise ValueError("BodySet contains unknown/empty elements")
    selected = {label: elements[label] for label in sorted(ids)}
    connected = {node for edge in selected.values() for node in edge}
    if not connected.issubset(nodes):
        raise ValueError("BodySet connectivity contains missing source nodes")
    body_nodes = {label: nodes[label] for label in sorted(connected)}
    boundary_sides = {}
    for name, candidates in sides.items():
        if len(candidates) != 1:
            raise ValueError("One explicit source set required for each body side")
        if not candidates[0].issubset(nodes):
            raise ValueError("Body side set contains unknown source nodes")
        boundary_sides[name] = sorted(candidates[0] & connected)
        if len(boundary_sides[name]) < 2:
            raise ValueError("Each body side must retain at least two connected nodes")
    boundary = sorted(set().union(*(set(v) for v in boundary_sides.values())))
    bounds = np.array(list(body_nodes.values()), dtype=float)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    section = [float(v.strip()) for v in rows(sections[0])[0].split(",")]
    elastic = [float(v.strip()) for v in rows(materials[0])[0].split(",")]
    if len(section) != 2 or min(section) <= 0 or len(elastic) != 2 or elastic[0] <= 0 or not -1 < elastic[1] < .5:
        raise ValueError("Invalid body section or elastic constituent")
    if not np.isfinite(bounds).all() or min(upper-lower) <= 0:
        raise ValueError("Body tensor needs a finite two-dimensional body")
    return {"nodes": body_nodes, "elements": selected, "boundary_sides": boundary_sides,
            "boundary_nodes": boundary, "bounds_mm": [lower.tolist(), upper.tolist()],
            "section_card": sections[0], "section_B_t_mm": section, "elastic_MPa_nu": elastic}


def prepare_body_tensor(source, out, name, macrostrain=1e-4):
    """Three exact-source body-only B21 KUBC solves; preparation never submits."""
    if not math.isfinite(macrostrain) or macrostrain <= 0:
        raise ValueError("Positive finite body macrostrain required")
    source = Path(source).resolve()
    original_hash = sha(source)
    mesh = body_tensor_source(source)
    material = dict(piece.strip().lower().split("=", 1)
                    for piece in mesh["section_card"].splitlines()[0].split(",")[1:] if "=" in piece)["material"]
    contents = ["*Heading\n** Exact archived body-only B21 finite-body KUBC control\n*Node\n"]
    contents += ["%d, %.17g, %.17g\n" % (label, *coord) for label, coord in mesh["nodes"].items()]
    contents += ["*Element, type=B21, elset=BodySet\n"]
    contents += ["%d, %d, %d\n" % (label, *edge) for label, edge in mesh["elements"].items()]
    contents += [mesh["section_card"].rstrip()+"\n", "*Material, name="+material+"\n*Elastic\n",
                 "%.17g, %.17g\n" % tuple(mesh["elastic_MPa_nu"]), "*Nset, nset=BodyBoundary\n"]
    boundary = mesh["boundary_nodes"]
    contents += [", ".join(str(v) for v in boundary[i:i+16])+"\n" for i in range(0, len(boundary), 16)]
    origin = np.mean(np.asarray(mesh["bounds_mm"]), axis=0)
    for column, label in enumerate(("Exx", "Eyy", "Gxy")):
        contents += ["*Step, name=%s, nlgeom=NO\n*Static\n1., 1.\n*Boundary, op=NEW\n" % label]
        for node in boundary:
            x, y = np.asarray(mesh["nodes"][node])-origin
            displacement = ((macrostrain*x, 0.), (0., macrostrain*y),
                            (.5*macrostrain*y, .5*macrostrain*x))[column]
            contents += ["%d, %d, %d, %.17g\n" % (node, dof, dof, displacement[dof-1]) for dof in (1, 2)]
        contents += ["*Output, field\n*Node Output, nset=BodyBoundary\nU, RF\n"
                     "*Output, history\n*Energy Output\nALLSE, ALLWK\n*End Step\n"]
    contents = "".join(contents)
    directory = job_directory(out, name)
    directory.mkdir(parents=True, exist_ok=True)
    inp = directory / (name+".inp")
    if inp.exists() and inp.read_text(encoding="ascii") != contents:
        raise ValueError("Refuse different existing body tensor input")
    inp.write_text(contents, encoding="ascii")
    width, height = np.diff(np.asarray(mesh["bounds_mm"]), axis=0)[0]
    metadata = {"job": name, "input": str(inp), "input_sha256": sha(inp),
                "source_inp": str(source), "source_sha256": original_hash,
                "analysis": "finite_body_B21_KUBC_tensor", "state": "prepared_not_submitted",
                "macrostrain": macrostrain, "elastic_MPa_nu": mesh["elastic_MPa_nu"],
                "section_B_t_mm": mesh["section_B_t_mm"], "source_section_card": mesh["section_card"],
                "body_element_labels": list(mesh["elements"]), "body_node_labels": list(mesh["nodes"]),
                "boundary_nodes": boundary, "boundary_sides": mesh["boundary_sides"],
                "bounds_mm": mesh["bounds_mm"], "origin_mm": origin.tolist(),
                "width_mm": float(width), "height_mm": float(height),
                "volume_mm3": float(width*height*mesh["section_B_t_mm"][0]),
                "volume_definition": "gross body-node bounding-box width*height*source B",
                "boundary_condition": "affine translations on actual four source body-side sets intersected with BodySet connected nodes; all rotations and interior translations free",
                "engineering_shear": "ux=gamma*y/2; uy=gamma*x/2",
                "tensor_order": "xx, yy, engineering xy; raw columns are independent load cases",
                "retains": "original BodySet element/node labels, coordinates, connectivity, rectangular section and mapped constituent E/nu",
                "limits": ["Finite-body kinematic upper-bound sensitivity, not an infinite periodic homogenisation",
                           "No Mohr axial-affine sum; interior bending, shear and rotations relax",
                           "No forced C11=C22, C12=C21 or zero normal/shear coupling",
                           "A coupled tensor does not automatically qualify the aligned orthotropic M formula"]}
    if sha(source) != original_hash:
        raise ValueError("Archived body source changed during preparation")
    write_json(directory / "source.json", metadata)
    return metadata


def prepare_body_tensors(root, out, macrostrain=1e-4):
    """Prepare only the four reviewed hex/kagome periodic/full-ID1 pilots."""
    root, out = Path(root).resolve(), Path(out).resolve()
    uat = root / "audit/controls/uat-initial-response"
    records = []
    for suffix in ("hex_per", "hex_full1", "kag_per", "kag_full1"):
        parent = uat / ("uat_"+suffix) / "source.json"
        previous = json.loads(parent.read_text())
        if sha(previous["source_inp"]) != previous["source_sha256"]:
            raise ValueError("UAT parent source changed before body pilot")
        record = prepare_body_tensor(previous["source_inp"], out, "body_"+suffix, macrostrain)
        record.update(parent_source_json=str(parent), parent_source_json_sha256=sha(parent))
        write_json(Path(record["input"]).parent / "source.json", record)
        records.append(record)
    manifest = {"jobs": [{"job": item["job"], "input": item["input"], "input_sha256": item["input_sha256"],
                          "kind": item["analysis"], "purpose": "Body-tensor disorder/M sensitivity pilot; finite-size KUBC",
                          "cpus": 8, "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]}
                         for item in records], "queue_root": str(out / "queue"),
                "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    proposal = out / "serial-body-tensor-proposal.json"
    if proposal.exists() and json.loads(proposal.read_text()) != manifest:
        raise ValueError("Refuse different existing body pilot proposal")
    write_json(proposal, manifest)
    write_json(out / "preflight.json", {"state": "prepared_not_submitted", "jobs": records,
               "manifest": str(proposal), "source_inputs_changed": False,
               "review_required_before_submission": True})
    return {"prepared_jobs": len(records), "manifest": str(proposal), "preflight": str(out / "preflight.json")}


def body_tensor_values(coordinates, reactions, macrostrain, volume, stored_energies):
    """Recover all raw C entries from boundary virtual work, without projection."""
    xy, forces, energies = (np.asarray(value, dtype=float) for value in (coordinates, reactions, stored_energies))
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or forces.shape != (3, len(xy), 2)
            or energies.shape != (3,) or not all(np.isfinite(v).all() for v in (xy, forces, energies))
            or not math.isfinite(macrostrain) or macrostrain <= 0 or not math.isfinite(volume) or volume <= 0):
        raise ValueError("Invalid body reaction/strain/volume arrays")
    x, y = xy.T
    virtual = np.array([np.column_stack((x, np.zeros_like(x))),
                        np.column_stack((np.zeros_like(y), y)), np.column_stack((y/2, x/2))])
    C = np.einsum("knd,jnd->kj", virtual, forces) / (volume*macrostrain)
    symmetry = float(np.max(np.abs(C-C.T))/max(np.max(np.abs(C)), 1e-30))
    eigenvalues = np.linalg.eigvalsh((C+C.T)/2)
    checks = []
    for column, label in enumerate(("Exx", "Eyy", "Gxy")):
        work = .5*volume*macrostrain**2*C[column, column]
        relative = abs(energies[column]-work)/max(abs(work), 1e-30)
        force_scale = np.linalg.norm(forces[column], axis=1).sum()
        force_balance = float(np.linalg.norm(forces[column].sum(axis=0))/max(force_scale, 1e-30))
        torque = float(np.sum(x*forces[column, :, 1]-y*forces[column, :, 0]))
        torque_balance = abs(torque)/max(force_scale*np.linalg.norm(np.ptp(xy, axis=0)), 1e-30)
        if work <= 0 or relative > 1e-5 or max(force_balance, torque_balance) > 1e-5:
            raise ValueError("Body tensor reaction balance or work-energy check failed")
        checks.append({"load_case": label, "ALLSE_N_mm": float(energies[column]),
                       "macro_work_N_mm": float(work), "relative_error": float(relative),
                       "force_balance_relative": force_balance, "torque_balance_relative": float(torque_balance)})
    if symmetry > 1e-4 or eigenvalues.min() <= 0:
        raise ValueError("Body tensor is asymmetric/nonpositive")
    S = np.linalg.inv(C)
    return {"C_MPa_raw": C.tolist(), "S_per_MPa_raw": S.tolist(), "energy_checks": checks,
            "major_symmetry_relative_error": symmetry, "symmetric_part_eigenvalues_MPa": eigenvalues.tolist(),
            "E_x_MPa": float(1/S[0, 0]), "E_y_MPa": float(1/S[1, 1]),
            "normal_shear_coupling_relative": float(np.max(np.abs(C[:2, 2]))/np.max(np.abs(C))),
            "projection_applied": False, "conversion_M_status": "not selected; retain full tensor and assess coupling/finite-body sensitivity first"}


def gather_body_tensors(out, queue):
    """Receipt-verified, read-only extraction of the four body-tensor pilots."""
    from odbAccess import openOdb
    out, queue = Path(out).resolve(), Path(queue).resolve()
    completed = {item["job"] for item in json.loads((queue / "state.json").read_text())["jobs"]
                 if item["status"] == "completed"}
    records = []
    for path in sorted(out.glob("body_*/source.json")):
        meta = json.loads(path.read_text())
        odb_path, receipt_path = verified_control_delivery(meta, queue, completed)
        mesh = body_tensor_source(meta["source_inp"])
        labels = meta["boundary_nodes"]
        xy = np.array([mesh["nodes"][label] for label in labels])-np.array(meta["origin_mm"])
        before = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        reactions, stored, external = [], [], []
        try:
            for column, label in enumerate(("Exx", "Eyy", "Gxy")):
                step = odb.steps[label]
                frame = step.frames[-1]
                rf = {value.nodeLabel: list(value.data[:2]) for value in frame.fieldOutputs["RF"].values}
                uu = {value.nodeLabel: list(value.data[:2]) for value in frame.fieldOutputs["U"].values}
                if set(rf) != set(labels) or set(uu) != set(labels):
                    raise ValueError("Incomplete or unexpected body boundary fields")
                x, y = xy.T
                target = (np.column_stack((x, np.zeros_like(x))), np.column_stack((np.zeros_like(y), y)),
                          np.column_stack((y/2, x/2)))[column]*meta["macrostrain"]
                if not np.allclose(np.array([uu[node] for node in labels]), target, rtol=1e-6, atol=1e-10):
                    raise ValueError("Body boundary displacement differs from prescribed macrostrain")
                reactions.append([rf[node] for node in labels])
                energies = [region.historyOutputs for region in step.historyRegions.values()
                            if "ALLSE" in region.historyOutputs and "ALLWK" in region.historyOutputs]
                if len(energies) != 1:
                    raise ValueError("Unique whole-model body energy history required")
                se, wk = (float(energies[0][field].data[-1][1]) for field in ("ALLSE", "ALLWK"))
                if abs(se-wk)/max(abs(se), 1e-30) > 1e-5:
                    raise ValueError("Body stored/external work mismatch")
                stored.append(se)
                external.append(wk)
        finally:
            odb.close()
        if sha(odb_path) != before:
            raise ValueError("Body tensor ODB changed during read-only extraction")
        result = dict(meta, **body_tensor_values(xy, reactions, meta["macrostrain"], meta["volume_mm3"], stored),
                      ALLWK_N_mm=external, state="completed_and_extracted", read_only=True,
                      odb=str(odb_path), odb_sha256=before, delivery_sha256=sha(receipt_path))
        write_json(path.parent / "result.json", result)
        records.append(result)
    write_json(out / "body-tensor-summary.json", records)
    return records


def compliance_values(displacement, fixed_reaction, load=1.):
    if not all(math.isfinite(value) for value in (displacement, fixed_reaction, load)) or displacement <= 0 or load <= 0:
        raise ValueError("Positive finite unit-load compliance required")
    balance = abs(fixed_reaction + load) / load
    if balance > 1e-5:
        raise ValueError("Unit-load fixed reaction does not balance the applied force")
    return {"load_N": load, "compliance_mm_N": displacement / load,
            "static_stiffness_N_mm": load / displacement, "fixed_pin_reaction_y_N": fixed_reaction,
            "force_balance_relative": balance}


def prepare_uat_linear(source, out, name, travel=1e-4):
    """Exact archived UAT clamp geometry, undeformed elastic tangent only.

    Prescribed equal clamp travel preserves the original boundary conditions;
    distributing a nominal force across grip nodes would be a different test.
    This calculation supplies a grip-to-grip slope, not a full material tensor.
    """
    if not math.isfinite(travel) or travel <= 0:
        raise ValueError("Positive finite clamp travel required")
    source = Path(source).resolve()
    parsed = blocks(source.read_text(encoding="ascii"))
    starts = [i for i, item in enumerate(parsed) if key(item) == "*step"]
    if len(starts) != 1:
        raise ValueError("Expected one archived UAT step")
    model, step = parsed[:starts[0]], parsed[starts[0]:]
    if any(key(item) in ("*boundary", "*include") for item in model):
        raise ValueError("Initial constraints/includes need separate review")
    boundaries = [line.replace(" ", "").lower() for item in step if key(item) == "*boundary"
                  for line in rows(item)]
    expected = {"set-top,1,1", "set-top,2,2,1.", "set-top,6,6",
                "set-bottom,1,1", "set-bottom,2,2", "set-bottom,6,6"}
    if len(boundaries) != 6 or set(boundaries) != expected:
        raise ValueError("Unfamiliar UAT clamp constraints")
    materials = [rows(item) for item in model if key(item) == "*elastic"]
    if len(materials) != 1 or len(materials[0]) != 1:
        raise ValueError("One homogeneous elastic constituent required")
    sections = [item for item in model if key(item) == "*beam section"]
    if not sections or any("section=rect" not in item.splitlines()[0].lower().replace(" ", "")
                           for item in sections):
        raise ValueError("Only reviewed rectangular B21 sections are supported")
    types = [item.splitlines()[0].lower().replace(" ", "") for item in model if key(item) == "*element"]
    if not types or any("type=b21" not in item for item in types):
        raise ValueError("Only exact B21 mesh controls are supported")
    ignored = {"*amplitude", "*plastic", "*damage initiation", "*damage evolution"}
    prefix = "".join(item for item in model if key(item) not in ignored)
    contents = prefix.rstrip() + ("\n*Step, name=UATLinear, nlgeom=NO\n*Static\n1., 1.\n"
        "*Boundary\nSet-bottom, 1, 2\nSet-bottom, 6, 6\n"
        "Set-top, 1, 1\nSet-top, 6, 6\nSet-top, 2, 2, %.16g\n"
        "*Output, field\n*Node Output\nU, RF\n*Output, history\n"
        "*Energy Output\nALLSE, ALLWK\n*End Step\n") % travel
    directory = job_directory(out, name)
    directory.mkdir(parents=True, exist_ok=True)
    inp = directory / (name + ".inp")
    if inp.exists() and inp.read_text(encoding="ascii") != contents:
        raise ValueError("Refuse different existing UAT linear input")
    inp.write_text(contents, encoding="ascii")
    metadata = {"job": name, "input": str(inp), "input_sha256": sha(inp),
                "source_inp": str(source), "source_sha256": sha(source),
                "analysis": "initial_linear_UAT_clamp_response", "travel_mm": travel,
                "elastic_MPa_nu": [float(v) for v in materials[0][0].split(",")],
                "section_cards": sections, "source_mesh_and_clamp_BCs_retained": True,
                "removed": sorted(ignored), "fit_interval": "none; undeformed linear solve"}
    write_json(directory / "source.json", metadata)
    return metadata


def uat_linear_values(travel, top_force, bottom_force, body_travel, height, width, thickness):
    """Convert a qualified clamp solution using the original gross-area gauge."""
    values = [travel, top_force, body_travel, height, width, thickness]
    if not all(math.isfinite(value) and value > 0 for value in values) or not math.isfinite(bottom_force):
        raise ValueError("Positive finite UAT displacement/force/gauge required")
    balance = abs(top_force + bottom_force) / top_force
    if balance > 1e-5 or body_travel > travel * (1 + 1e-8):
        raise ValueError("UAT balance or body/clamp travel inconsistent")
    area = width * thickness
    return {"force_N": top_force, "bottom_force_N": bottom_force,
            "force_balance_relative": balance, "body_travel_mm": body_travel,
            "body_height_mm": height, "gross_width_mm": width, "B_mm": thickness,
            "gross_area_mm2": area, "clamp_stiffness_N_mm": top_force / travel,
            "E_apparent_MPa": height * top_force / (area * travel),
            "E_body_gauge_MPa": height * top_force / (area * body_travel),
            "body_to_clamp_travel": body_travel / travel}


def gather_uat_linear(out, queue):
    """Read-only, receipt-verified extraction; original source INPs are untouched."""
    from odbAccess import openOdb
    out, queue = Path(out).resolve(), Path(queue).resolve()
    state = json.loads((queue / "state.json").read_text())
    completed = {item["job"] for item in state["jobs"] if item["status"] == "completed"}
    records = []
    for path in sorted(out.glob("uat_*/source.json")):
        meta = json.loads(path.read_text())
        odb_path, receipt_path = verified_control_delivery(meta, queue, completed)
        before = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            frame = odb.steps["UATLinear"].frames[-1]
            assembly = odb.rootAssembly
            def field(component, set_name):
                return [float(v.data[component]) for v in frame.fieldOutputs[
                    "RF" if set_name.endswith("_RF") else "U"].getSubset(
                        region=assembly.nodeSets[set_name.removesuffix("_RF")]).values]
            top_u = field(1, "SET-TOP")
            if not np.allclose(top_u, meta["travel_mm"], rtol=1e-6, atol=1e-12):
                raise ValueError("Clamp travel differs from prescribed control")
            body_delta = float(np.mean(field(1, "SET-TOPBODY")) - np.mean(field(1, "SET-BOTTOMBODY")))
            def node_coords(name):
                groups = assembly.nodeSets[name].nodes
                return [node.coordinates for group in groups for node in group]
            top, bottom = node_coords("SET-TOPBODY"), node_coords("SET-BOTTOMBODY")
            left, right = node_coords("SET-LEFT"), node_coords("SET-RIGHT")
            height = float(np.mean([v[1] for v in top]) - np.mean([v[1] for v in bottom]))
            width = float(np.mean([v[0] for v in right]) - np.mean([v[0] for v in left]))
            thicknesses = [float(rows(card)[0].split(",")[0]) for card in meta["section_cards"]]
            if not np.allclose(thicknesses, thicknesses[0], rtol=0, atol=0):
                raise ValueError("Nonuniform out-of-plane thickness")
            result = uat_linear_values(meta["travel_mm"], sum(field(1, "SET-TOP_RF")),
                sum(field(1, "SET-BOTTOM_RF")), body_delta, height, width, thicknesses[0])
            histories = [region.historyOutputs for region in odb.steps["UATLinear"].historyRegions.values()
                         if "ALLSE" in region.historyOutputs and "ALLWK" in region.historyOutputs]
            if len(histories) != 1:
                raise ValueError("Unique whole-model work/energy history required")
            stored = float(histories[0]["ALLSE"].data[-1][1])
            work = float(histories[0]["ALLWK"].data[-1][1])
            result["work_energy_relative"] = abs(stored-work) / max(abs(work), 1e-30)
            result["clamp_work_energy_relative"] = abs(stored-.5*result["force_N"]*meta["travel_mm"]) / max(abs(stored), 1e-30)
            if max(result["work_energy_relative"], result["clamp_work_energy_relative"]) > 1e-5:
                raise ValueError("UAT linear energy mismatch")
        finally:
            odb.close()
        if before != sha(odb_path):
            raise ValueError("UAT control ODB changed during extraction")
        records.append(dict(meta, **result, state="completed_and_extracted", read_only=True,
                            odb=str(odb_path), odb_sha256=before, delivery_sha256=sha(receipt_path)))
    write_json(out / "uat-linear-summary.json", records)
    return records


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


def output_filter_replay_contents(text, task, physical_cutoff_Hz,
                                  field_interval=.0005, raw_interval=.00001,
                                  saved_interval=.0005):
    """Replace only output requests; retain the full source Explicit loading.

    Named filters run before output decimation. A physical cutoff above saved
    Nyquist is valid here because the filter samples the solver increments.
    Material strain/damage/STATUS fields always remain unfiltered.
    """
    if task not in ("FT", "UT") or not math.isfinite(physical_cutoff_Hz) or physical_cutoff_Hz <= 0:
        raise ValueError("FT/UT and a positive periodic physical cutoff are required")
    if not (0 < raw_interval <= saved_interval and 0 < field_interval):
        raise ValueError("Positive dense/raw and saved output intervals required")
    parsed = blocks(text)
    if (sum(key(b) == "*step" for b in parsed) != 1
            or sum(key(b) == "*end step" for b in parsed) != 1
            or any(key(b) in ("*include", "*filter") for b in parsed)
            or any(key(b) == "*output" and "filter=" in b.splitlines()[0].replace(" ", "").lower()
                   for b in parsed)):
        raise ValueError("One self-contained, previously unfiltered source step is required")
    split = next(i for i, b in enumerate(parsed) if key(b) == "*step")
    model, step = parsed[:split], parsed[split:]
    dynamics = [b for b in step if key(b) == "*dynamic"]
    if len(dynamics) != 1 or "explicit" not in dynamics[0].splitlines()[0].lower():
        raise ValueError("One original Explicit dynamic step is required")
    period = float(rows(dynamics[0])[0].split(",")[1])
    if not 0 < max(field_interval, saved_interval) <= period:
        raise ValueError("Output intervals must resolve the complete original step")
    grip = "load" if task == "FT" else "Set-top"
    if not any(key(b) == "*nset" and re.search(r"nset\s*=\s*"+grip+r"\s*(?:,|$)",
                   b.splitlines()[0], flags=re.I) for b in model):
        raise ValueError("Original grip node set is missing")
    outputs = {"*output", "*node output", "*element output", "*energy output"}
    kept = [b for b in step if key(b) not in outputs and key(b) != "*end step"]
    ending = next(b for b in step if key(b) == "*end step")
    cutoffs = {"PHYSICAL2FN": float(physical_cutoff_Hz),
               "SELECTED": min(float(physical_cutoff_Hz), 1/(6*saved_interval)),
               "ENVELOPE100": 100., "ENVELOPE50": 50.}
    filters = {name: {"cutoff_Hz": cutoff, "order": 2, "startup": "DC"}
               for name, cutoff in cutoffs.items()}
    definitions = "".join("*Filter, name=%s, type=BUTTERWORTH, start condition=DC\n%.12g, 2\n"
                          % (name, cutoff) for name, cutoff in cutoffs.items())
    requested = ("*Output, field, time interval=%.12g\n*Node Output\nU, RF\n"
                 "*Element Output, directions=YES\nLE, S, PEEQ, SDEG, STATUS\n"
                 "*Output, history, time interval=%.12g\n*Energy Output\n"
                 "ALLIE, ALLKE, ALLSE, ALLWK, ALLAE, ALLVD, ALLPD, ALLFD, ALLDMD, ETOTAL\n"
                 "*Output, history, time interval=%.12g\n*Node Output, nset=%s\nU2, RF2\n"
                 % (field_interval, saved_interval, raw_interval, grip))
    requested += "".join("*Output, history, filter=%s, time interval=%.12g\n"
                         "*Node Output, nset=%s\nU2, RF2\n" % (name, saved_interval, grip)
                         for name in filters)
    contents = "".join(model)+definitions+"".join(kept)+requested+ending
    physics = "".join(model+kept+[ending])
    return contents, {"source_period_s": period, "replay_period_s": period,
                      "grip_nset": grip, "filters": filters,
                      "field_interval_s": field_interval, "raw_history_interval_s": raw_interval,
                      "filtered_history_interval_s": saved_interval,
                      "physics_without_output_sha256": hashlib.sha256(physics.encode("ascii")).hexdigest(),
                      "model_prefix_sha256": hashlib.sha256("".join(model).encode("ascii")).hexdigest(),
                      "source_physics_and_full_loading_retained": True,
                      "events_filtered": False}


def prepare_output_filter_replays(root, out):
    """Prepare the two reviewed hex ID336 output-only replays, FT then UT.

    Preparation never submits or accepts a failed prefix. The queue can retain
    solver-failure evidence and continue, while failed job status stays failed.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    index = root/"audit/curve-processing/paired-examples/native-index.json"
    selected = [r for r in json.loads(index.read_text())
                if r["topology"] == "hex" and r["design"] == "full" and r["id"] == 336]
    registry_path = root/"audit/curve-processing/periodic-cutoffs.json"
    frequencies = ({r["task"]: r for r in json.loads(registry_path.read_text())["families"]
                    if r["topology"] == "hex"} if registry_path.exists() else {})
    targets, records, jobs = {}, [], []
    for task, cutoff in (("FT", 318.16), ("UT", 1836.34)):
        matches = [r for r in selected if r["task"] == task]
        if len(matches) != 1:
            raise ValueError("Exactly one reviewed full-disorder hex ID336 source per task is required")
        row = matches[0]
        source, odb = Path(row["source_inp"]).resolve(), Path(row["source_odb"]).resolve()
        if sha(source) != row["source_inp_sha256"]:
            raise ValueError("Reviewed source input changed")
        if task in frequencies:
            control = frequencies[task]
            if control["status"] != "accepted_physical_mode" or not math.isclose(
                    2*control["natural_frequency_Hz"], cutoff, rel_tol=1e-10):
                raise ValueError("Hex periodic physical frequency differs from the reviewed replay")
        contents, info = output_filter_replay_contents(source.read_text(encoding="ascii"), task, cutoff)
        name = "of_hex336_"+task.lower()
        directory = job_directory(out, name)
        inp = directory/(name+".inp")
        digest = hashlib.sha256(contents.encode("ascii")).hexdigest()
        meta = {"job": name, "input": str(inp), "input_sha256": digest,
                "task": task, "topology": "hex", "design": "full", "id": 336,
                "source_job": source.stem, "source_inp": str(source),
                "source_sha256": row["source_inp_sha256"], "source_odb": str(odb),
                "source_odb_sha256": sha(odb), "selection_index": str(index),
                "selection_index_sha256": sha(index),
                "periodic_registry_sha256": sha(registry_path) if frequencies else None,
                "analysis": "source_Explicit_output_filter_replay", "state": "prepared_not_submitted",
                "unchanged": "Original mesh, material, damage, clamps/pins, amplitude, loading rate and full step duration",
                "changes": "Output requests only: unfiltered dense U/RF, native fields/whole energies and named in-solver filters",
                "failure_handling": "Failed status remains failed; no automatic usable-prefix acceptance",
                "history_identity_check_required": "Verify raw U2/RF2 and all four named filtered U2/RF2 channels per original grip node",
                **info}
        targets[inp] = contents.encode("ascii")
        targets[directory/"source.json"] = (json.dumps(meta, indent=2, allow_nan=False)+"\n").encode("utf-8")
        records.append(meta)
        jobs.append({"job": name, "input": str(inp), "input_sha256": digest, "cpus": 8,
                     "allow_solver_failure": True, "required_outputs": [".odb", ".dat", ".sta"],
                     "purpose": "Output-filter qualification on unchanged full-duration source; failed prefixes need separate review"})
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out/"queue"), "jobs": jobs}
    manifest_path = out/"manifest.json"
    targets[manifest_path] = (json.dumps(manifest, indent=2)+"\n").encode("utf-8")
    for path, data in targets.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Prepared output-filter replay is immutable: "+str(path))
    for path, data in targets.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return {"manifest": str(manifest_path), "prepared_jobs": [r["job"] for r in records],
            "source_unchanged": True, "submission_performed": False}


def prepare_output_filter_precision_replay(parent, out, name="of_hex336_ft_dp"):
    """Copy the exact FT replay input; change execution precision only."""
    parent, out = Path(parent).resolve(), Path(out).resolve()
    parent_path = parent/"source.json"
    original = json.loads(parent_path.read_text())
    if original["task"] != "FT":
        raise ValueError("The reviewed precision follow-up is FT only")
    for path, digest in ((original["input"], original["input_sha256"]),
                         (original["source_inp"], original["source_sha256"]),
                         (original["source_odb"], original["source_odb_sha256"])):
        if sha(path) != digest:
            raise ValueError("Precision replay parent/source bytes changed")
    contents = Path(original["input"]).read_bytes()
    directory = job_directory(out, name)
    inp = directory/(name+".inp")
    options = ["double=both", "output_precision=full"]
    meta = {**original, "job": name, "input": str(inp), "state": "prepared_not_submitted",
            "parent_job": original["job"], "parent_input": original["input"],
            "parent_input_sha256": original["input_sha256"], "parent_source_json": str(parent_path),
            "parent_source_json_sha256": sha(parent_path), "execution_options": options,
            "precision_comparison": "Identical input bytes and original full loading; execution precision only"}
    if (parent/"extraction.json").exists():
        meta.update(parent_extraction=str(parent/"extraction.json"),
                    parent_extraction_sha256=sha(parent/"extraction.json"))
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out/"queue"),
                "jobs": [{"job": name, "input": str(inp), "input_sha256": original["input_sha256"],
                          "cpus": 8, "execution_options": options, "allow_solver_failure": True,
                          "required_outputs": [".odb", ".dat", ".sta"],
                          "purpose": "Isolate execution precision for named-filter discrepancy; same full-duration FT source"}]}
    targets = {inp: contents, directory/"source.json": (json.dumps(meta, indent=2)+"\n").encode(),
               out/"manifest.json": (json.dumps(manifest, indent=2)+"\n").encode()}
    for path, data in targets.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("Prepared precision replay is immutable: "+str(path))
    for path, data in targets.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return {"manifest": str(out/"manifest.json"), "job": name,
            "input_bytes_identical_to_parent": True, "submission_performed": False}


def output_filter_history_data(odb, meta, include_filters=True):
    """Verify each original grip node/channel and aggregate paired native units."""
    step = odb.steps["Step-1"]
    groups = {"RAW": ""}
    if include_filters:
        groups.update({name: "_"+name for name in meta["filters"]})
    node_set = odb.rootAssembly.nodeSets[meta["grip_nset"].upper()]
    labels = sorted(node.label for group in node_set.nodes for node in group)
    if not labels or len(labels) != len(set(labels)) or (meta["task"] == "FT" and len(labels) != 1):
        raise ValueError("Original grip node identities are invalid")
    instance = "ASSEMBLY" if meta["task"] == "FT" else "PART-1-1"
    collected, ledger = {name: [] for name in groups}, []
    for label in labels:
        region_name = "Node %s.%d" % (instance, label)
        if region_name not in step.historyRegions:
            raise ValueError("Original grip node history missing: "+region_name)
        history = step.historyRegions[region_name].historyOutputs
        node_proof = {"region": region_name, "node": label, "channels": {}}
        for name, suffix in groups.items():
            channels = [variable+suffix for variable in ("U2", "RF2")]
            if any(channel not in history for channel in channels):
                raise ValueError("Required raw/filtered channel missing at "+region_name+": "+repr(channels))
            pairs = [np.asarray(history[channel].data, dtype=float) for channel in channels]
            for values in pairs:
                if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 2
                        or not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0)):
                    raise ValueError("Need finite strictly increasing paired history observations")
            if not np.array_equal(pairs[0][:, 0], pairs[1][:, 0]):
                raise ValueError("U2/RF2 output times are not aligned")
            interval = meta["raw_history_interval_s"] if name == "RAW" else meta["filtered_history_interval_s"]
            if not include_filters:
                interval = float(np.median(np.diff(pairs[0][:, 0])))
            if pairs[0][0, 0] != 0 or np.max(np.diff(pairs[0][:, 0])) > 1.5*interval:
                raise ValueError("Native history starts late or contains an output gap")
            collected[name].append(np.column_stack((pairs[0][:, 0], pairs[0][:, 1], pairs[1][:, 1])))
            node_proof["channels"][name] = channels
        ledger.append(node_proof)
    data = {}
    for name, arrays in collected.items():
        if any(not np.array_equal(a[:, 0], arrays[0][:, 0]) for a in arrays[1:]):
            raise ValueError("Grip-node output times are not aligned")
        data[name] = np.column_stack((arrays[0][:, 0], np.mean([a[:, 1] for a in arrays], axis=0),
                                     np.sum([a[:, 2] for a in arrays], axis=0)))
    filtered = [a[:, 0] for name, a in data.items() if name != "RAW"]
    if filtered and any(not np.array_equal(t, filtered[0]) for t in filtered[1:]):
        raise ValueError("Named filtered output times are not aligned")
    return data, ledger


def output_filter_replay_delivery(meta, state, queue):
    """Verify retained bytes without converting a failed solver into completion."""
    if state["status"] not in ("completed", "failed"):
        raise ValueError("Only settled retained deliveries may be extracted")
    for path, expected in ((meta["source_inp"], meta["source_sha256"]),
                           (meta["source_odb"], meta["source_odb_sha256"]),
                           (meta["input"], meta["input_sha256"])):
        if sha(path) != expected:
            raise ValueError("Replay source/prepared input changed")
    run = job_directory(Path(queue)/"runs", meta["job"])
    receipt_path = run/"delivery.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["job"] != meta["job"] or receipt.get("required_outputs_missing"):
        raise ValueError("Incomplete output-filter delivery")
    entries = {r["file"]: r for r in receipt["files"]}
    if any(meta["job"]+suffix not in entries for suffix in (".inp", ".odb", ".dat", ".sta")):
        raise ValueError("Required output-filter delivery file missing")
    for filename, record in entries.items():
        target = (run/filename).resolve()
        if target.parent != run or sha(target) != record["sha256"]:
            raise ValueError("Retained delivery differs from its receipt")
    if entries[meta["job"]+".inp"]["sha256"] != meta["input_sha256"]:
        raise ValueError("Retained delivery input differs from prepared model")
    if state["status"] == "completed":
        launcher = (run/"launcher.txt").read_text(errors="replace")
        if state.get("solver_returncode") != 0 or not re.search(
                r"Abaqus\s+JOB\s+"+re.escape(meta["job"])+r"\s+COMPLETED", launcher, re.I):
            raise ValueError("Completed queue status lacks solver completion evidence")
    return {"solver_status": state["status"], "solver_returncode": state.get("solver_returncode"),
            "prefix_qualified": False, "delivery": str(receipt_path),
            "delivery_sha256": sha(receipt_path), "odb": str(run/(meta["job"]+".odb")),
            "odb_sha256": entries[meta["job"]+".odb"]["sha256"]}


def gather_output_filter_replays(out, queue):
    """Extract complete channel evidence from settled jobs; never qualify a prefix."""
    from odbAccess import openOdb
    from validation_failure_audit import maximum
    out, queue = Path(out).resolve(), Path(queue).resolve()
    manifest_path = out/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state = json.loads((queue/"state.json").read_text())
    if state.get("manifest_sha256") != sha(manifest_path):
        raise ValueError("Output-filter queue differs from the immutable manifest")
    statuses = {r["job"]: r for r in state["jobs"]}
    results, pending = [], []

    def curve_metrics(values):
        return {"samples": len(values), "last_time_s": float(values[-1, 0]),
                "force_N": float(values[-1, 2]), "opening_mm": float(values[-1, 1]),
                "work_Nmm": float(np.trapz(values[:, 2], values[:, 1])),
                "peak_force_N": float(np.max(values[:, 2])),
                "min_opening_mm": float(np.min(values[:, 1])), "max_opening_mm": float(np.max(values[:, 1]))}

    def csv_array(path, headings, values):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(headings); writer.writerows(values)
        return {"path": str(path), "sha256": sha(path)}

    for job in manifest["jobs"]:
        if statuses[job["job"]]["status"] not in ("completed", "failed"):
            pending.append({"job": job["job"], "solver_status": statuses[job["job"]]["status"]})
            continue
        directory = job_directory(out, job["job"])
        meta = json.loads((directory/"source.json").read_text())
        if meta["input_sha256"] != job["input_sha256"]:
            raise ValueError("Prepared replay identity differs from immutable manifest")
        proof = output_filter_replay_delivery(meta, statuses[job["job"]], queue)
        odb = openOdb(proof["odb"], readOnly=True)
        fields, energy, event = [], {}, None
        try:
            data, node_ledger = output_filter_history_data(odb, meta)
            step = odb.steps["Step-1"]
            wanted = ("ALLIE", "ALLKE", "ALLSE", "ALLWK", "ALLAE", "ALLVD", "ALLPD", "ALLFD", "ALLDMD", "ETOTAL")
            regions = [r for r in step.historyRegions.values() if all(k in r.historyOutputs for k in wanted)]
            if len(regions) != 1:
                raise ValueError("One whole-model energy history is required")
            for name in wanted:
                values = np.asarray(regions[0].historyOutputs[name].data, dtype=float)
                if (values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all()
                        or np.any(np.diff(values[:, 0]) <= 0)):
                    raise ValueError("Invalid native energy observations")
                energy[name] = values
            et = energy[wanted[0]][:, 0]
            if any(not np.array_equal(v[:, 0], et) for v in energy.values()):
                raise ValueError("Native energy channels are not time aligned")
            if not np.array_equal(et, data["SELECTED"][:, 0]):
                raise ValueError("Energy and saved filtered histories are not time aligned")
            for index, frame in enumerate(step.frames):
                required = {"LE", "S", "PEEQ", "SDEG", "STATUS", "U", "RF"}
                if not required.issubset(frame.fieldOutputs.keys()):
                    raise ValueError("Native strain/damage/nodal field output missing")
                le = frame.fieldOutputs["LE"]
                top = maximum(le, list(le.componentLabels).index("LE11"))
                row = {"field_index": index, "time_s": float(frame.frameValue),
                       "max_LE11": top["value"] if top else None, "critical_point": top}
                for variable in ("SDEG", "STATUS", "PEEQ"):
                    arrays = [np.asarray(b.data, dtype=float).ravel() for b in frame.fieldOutputs[variable].bulkDataBlocks]
                    values = np.concatenate(arrays)
                    finite = values[np.isfinite(values)]
                    row[variable+"_nonfinite_values"] = int(len(values)-len(finite))
                    row[variable+"_min"] = float(np.min(finite)) if len(finite) else None
                    row[variable+"_max"] = float(np.max(finite)) if len(finite) else None
                fields.append(row)
                if event is None and top and top["value"] >= .1105:
                    event = {"previous_field": fields[-2] if len(fields) > 1 else None,
                             "crossing_field": row,
                             "scope": "Global recorded surface LE11 diagnostic, critical location retained; body/section eligibility not qualified"}
        finally:
            odb.close()
        original = openOdb(meta["source_odb"], readOnly=True)
        try:
            source_data, source_nodes = output_filter_history_data(original, meta, include_filters=False)
        finally:
            original.close()
        if [r["region"] for r in node_ledger] != [r["region"] for r in source_nodes]:
            raise ValueError("Source/replay original grip node identities differ")
        raw = data["RAW"]
        source = source_data["RAW"]
        common = source[(source[:, 0] >= raw[0, 0]) & (source[:, 0] <= raw[-1, 0])]
        rx, rp = [np.interp(common[:, 0], raw[:, 0], raw[:, c]) for c in (1, 2)]
        comparison = np.column_stack((common, rx, rp, rx-common[:, 1], rp-common[:, 2]))
        def difference(mask):
            a = comparison[mask]
            if not len(a):
                return {"samples": 0}
            peak = max(float(np.max(np.abs(a[:, 2]))), 1e-30)
            return {"samples": len(a), "last_time_s": float(a[-1, 0]),
                    "max_opening_difference_mm": float(np.max(np.abs(a[:, 5]))),
                    "max_force_difference_percent_source_peak": 100*float(np.max(np.abs(a[:, 6])))/peak,
                    "RMS_force_difference_percent_source_peak": 100*float(np.sqrt(np.mean(a[:, 6]**2)))/peak}
        source_comparison = {"all_shared_observations": difference(np.ones(len(comparison), dtype=bool)),
                             "interpolation": "Dense continuous response evaluated only at observed original-source times; no material-event interpolation",
                             "qualification": "Diagnostic only; no automatic source-equivalence or failed-prefix acceptance"}
        if event and event["previous_field"]:
            stop = event["previous_field"]["time_s"]
            source_comparison["before_global_recorded_strain_bracket"] = difference(comparison[:, 0] <= stop)
        exports = {name: csv_array(directory/("raw-curve.csv" if name == "RAW" else "filter-"+name+".csv"),
                                   ("time_s", "U2_mm", "RF2_N"), values) for name, values in data.items()}
        exports["energies"] = csv_array(directory/"native-energies.csv", ("time_s", *wanted),
                                        np.column_stack((et, *[energy[k][:, 1] for k in wanted])))
        exports["source_comparison"] = csv_array(directory/"source-prefix-comparison.csv",
            ("time_s", "source_U2_mm", "source_RF2_N", "replay_U2_mm", "replay_RF2_N", "difference_U2_mm", "difference_RF2_N"), comparison)
        native_path = directory/"native-fields.json"
        write_json(native_path, fields)
        exports["native_fields"] = {"path": str(native_path), "sha256": sha(native_path)}
        windows = {"all_retained_observations": {name: curve_metrics(v) for name, v in data.items()}}
        if event and event["previous_field"]:
            stop = event["previous_field"]["time_s"]
            windows["before_global_recorded_strain_bracket"] = {
                name: curve_metrics(v[v[:, 0] <= stop]) for name, v in data.items()}
        # Bracket the entire read-only extraction, including original-source reads.
        if sha(proof["odb"]) != proof["odb_sha256"] or sha(meta["source_odb"]) != meta["source_odb_sha256"]:
            raise ValueError("Read-only output-filter extraction changed ODB bytes")
        result = {**meta, **proof, "state": proof["solver_status"], "read_only": True,
                  "extraction_state": "retained_channels_extracted", "source_unchanged": True,
                  "full_requested_time_observed": bool(raw[-1, 0] >= meta["source_period_s"]-1e-7),
                  "failure_interpretation": "Solver status retained; a finite retained prefix is not automatically invalid or qualified",
                  "node_channel_ledger": node_ledger, "source_prefix_comparison": source_comparison,
                  "global_recorded_strain_bracket": event, "native_field_frames": len(fields),
                  "native_energy_samples": len(et), "window_metrics": windows, "exports": exports,
                  "code_sha256": sha(__file__)}
        write_json(directory/"extraction.json", result)
        results.append(result)
    summary = {"results": results, "pending_jobs": pending, "read_only": True,
               "manifest_sha256": sha(manifest_path), "automatic_prefix_qualification": False}
    write_json(out/"extraction-summary.json", summary)
    return summary


def prepare_event_resolution(source, out, name="fcc_init_dense", stop_time=.045,
                             field_interval=.0005, history_interval=.00001):
    """Retain Explicit physics/amplitude; resolve only a short initiation prefix."""
    source, out = Path(source).resolve(), Path(out).resolve()
    original_hash = sha(source)
    text = source.read_text(encoding="ascii")
    dynamic = re.compile(r"(?im)^(\*Dynamic,\s*Explicit[^\n]*\n)\s*,\s*([\d.eE+-]+)[^\n]*")
    matches = list(dynamic.finditer(text))
    if len(matches) != 1:
        raise ValueError("One original Explicit step is required")
    period = float(matches[0].group(2))
    if not (0 < history_interval <= field_interval <= stop_time <= period):
        raise ValueError("Positive resolved intervals and an original-step prefix are required")
    model = text[:re.search(r"(?im)^\*Step\b", text).start()]
    contents = dynamic.sub(lambda m: m.group(1)+", %.12g" % stop_time, text, count=1)
    contents, nfield = re.subn(r"(?im)^\*Output,\s*field[^\n]*",
                              "*Output, field, time interval=%.12g" % field_interval, contents)
    contents, nhistory = re.subn(r"(?im)^\*Output,\s*history[^\n]*",
                                "*Output, history, time interval=%.12g" % field_interval, contents)
    contents, nload = re.subn(r"(?im)^(\*Node Output,\s*nset=load\s*)$",
                             lambda m: "*Output, history, time interval=%.12g\n" % history_interval+m.group(1), contents)
    if (nfield, nhistory, nload) != (1, 1, 1):
        raise ValueError("Expected one field, one history and one load-pin request")
    if not contents.startswith(model):
        raise ValueError("Source model/material/amplitude changed")
    directory = job_directory(out, name)
    directory.mkdir(parents=True, exist_ok=True)
    inp = directory/(name+".inp")
    if inp.exists() and inp.read_text(encoding="ascii") != contents:
        raise ValueError("Refuse to change an existing resolution control")
    inp.write_text(contents, encoding="ascii")
    meta = {"job": name, "input": str(inp), "input_sha256": sha(inp),
            "source_inp": str(source), "source_sha256": original_hash,
            "analysis": "source_Explicit_initiation_resolution", "source_period_s": period,
            "stop_time_s": stop_time, "field_interval_s": field_interval,
            "load_history_interval_s": history_interval,
            "model_prefix_sha256": hashlib.sha256(model.encode("ascii")).hexdigest(),
            "unchanged": "All nodes/elements/sections/material/damage, amplitude, BCs and original loading rate",
            "changes": "Shorten step only; refine unfiltered field/status/energy times and load-pin histories",
            "state": "prepared_not_submitted"}
    if source.with_suffix(".odb").exists():
        meta.update(source_odb=str(source.with_suffix(".odb")),
                    source_odb_sha256=sha(source.with_suffix(".odb")))
    if sha(source) != original_hash:
        raise ValueError("Original source changed")
    write_json(directory/"source.json", meta)
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out/"queue"),
                "jobs": [{"job": name, "input": str(inp), "input_sha256": sha(inp), "cpus": 8,
                          "required_outputs": [".odb", ".dat", ".sta"],
                          "purpose": "Resolve local initiation and pre-decimation filtering; same source physics"}]}
    path = out/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refuse different existing event-resolution manifest")
    write_json(path, manifest)
    return {"manifest": str(path), "source_unchanged": True, **meta}


def gather_event_resolution(out, queue):
    """Resolve the material event in a verified output-refined source replay."""
    from odbAccess import openOdb
    from validation_failure_audit import maximum
    out, queue = Path(out), Path(queue)
    state = json.loads((queue/"state.json").read_text())
    completed = {row["job"] for row in state["jobs"] if row["status"] == "completed"}
    records = []
    for source_json in out.glob("*/source.json"):
        meta = json.loads(source_json.read_text())
        if meta.get("analysis") != "source_Explicit_initiation_resolution":
            continue
        odb_path, delivery = verified_control_delivery(meta, queue, completed)
        digest = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        field_rows, bracket = [], None
        try:
            step = odb.steps["Step-1"]
            candidates = [name for name, region in step.historyRegions.items()
                          if "U2" in region.historyOutputs and "RF2" in region.historyOutputs]
            if len(candidates) != 1:
                raise ValueError("Need one load-pin history")
            load = step.historyRegions[candidates[0]]
            u, p = (np.asarray(load.historyOutputs[key].data, dtype=float) for key in ("U2", "RF2"))
            if not np.array_equal(u[:, 0], p[:, 0]) or not np.isfinite(u).all() or not np.isfinite(p).all():
                raise ValueError("Dense paired history is not finite/time aligned")
            for index, frame in enumerate(step.frames):
                if "LE" not in frame.fieldOutputs:
                    raise ValueError("Unfiltered strain output missing")
                field = frame.fieldOutputs["LE"]
                component = list(field.componentLabels).index("LE11")
                top = maximum(field, component)
                if top is None:
                    raise ValueError("No finite axial strain")
                zeros = sum(int(np.count_nonzero(np.asarray(block.data) == 0))
                            for block in frame.fieldOutputs["STATUS"].bulkDataBlocks)
                row = {"field_index": index, "time_s": float(frame.frameValue),
                       "max_LE11": top["value"], "zero_STATUS": zeros}
                field_rows.append(row)
                if bracket is None and top["value"] >= .1105:
                    if index == 0:
                        raise ValueError("Material event already crossed at initial frame")
                    bracket = {"previous_field": dict(field_rows[-2]), "crossing_field": dict(row),
                               "critical_point": top}
            if bracket is None:
                raise ValueError("Resolution prefix did not reach the material threshold")
        finally:
            odb.close()
        original = Path(meta["source_odb"])
        if sha(original) != meta["source_odb_sha256"]:
            raise ValueError("Original native ODB changed")
        original_odb = openOdb(str(original), readOnly=True)
        try:
            region = original_odb.steps["Step-1"].historyRegions[candidates[0]]
            original_u = np.asarray(region.historyOutputs["U2"].data, dtype=float)
            original_p = np.asarray(region.historyOutputs["RF2"].data, dtype=float)
        finally:
            original_odb.close()
        for label in ("previous_field", "crossing_field"):
            point = bracket[label]
            point.update(opening_mm=float(np.interp(point["time_s"], u[:, 0], u[:, 1])),
                         force_N=float(np.interp(point["time_s"], p[:, 0], p[:, 1])))
        keep = original_u[:, 0] <= bracket["previous_field"]["time_s"]
        times = original_u[keep, 0]
        du = np.interp(times, u[:, 0], u[:, 1])-original_u[keep, 1]
        dp = np.interp(times, p[:, 0], p[:, 1])-original_p[keep, 1]
        scale = max(float(np.max(np.abs(original_p[keep, 1]))), 1.)
        equivalence = {"samples": len(times), "max_opening_error_mm": float(np.max(np.abs(du))),
                       "max_force_error_over_peak": float(np.max(np.abs(dp))/scale),
                       "rms_force_error_over_peak": float(np.sqrt(np.mean(dp**2))/scale)}
        equivalence["passed"] = (len(times) >= 20 and equivalence["max_opening_error_mm"] < 1e-4
                                 and equivalence["max_force_error_over_peak"] < .01)
        if not equivalence["passed"]:
            raise ValueError("Output-refined replay differs materially from original prefix: "+repr(equivalence))
        if sha(odb_path) != digest or sha(original) != meta["source_odb_sha256"]:
            raise ValueError("Read-only ODB changed")
        curve = source_json.parent/"dense-native-curve.csv"
        with curve.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(("time_s", "U2_mm", "RF2_N"))
            writer.writerows(zip(u[:, 0], u[:, 1], p[:, 1]))
        with (source_json.parent/"resolved-fields.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(field_rows[0]));writer.writeheader();writer.writerows(field_rows)
        result = {**meta, "state": "completed_and_extracted", "read_only": True,
                  "source": str(original), "source_sha256": meta["source_odb_sha256"],
                  "original_source_inp": meta["source_inp"], "original_source_inp_sha256": meta["source_sha256"],
                  "original_source_odb": str(original), "original_source_odb_sha256": meta["source_odb_sha256"],
                  "event_source_odb": str(odb_path), "event_source_odb_sha256": digest,
                  "event_source_inp_sha256": meta["input_sha256"], "delivery_sha256": sha(delivery),
                  "criterion": "Last recorded field before maximum tensile LE11 reaches0.1105; events unfiltered",
                  "event": bracket, "source_equivalence": equivalence,
                  "dense_curve_csv": str(curve), "dense_curve_sha256": sha(curve),
                  "dense_sample_count": len(u), "field_sample_count": len(field_rows)}
        write_json(source_json.parent/"resolved-event.json", result)
        records.append(result)
    write_json(out/"resolved-event-summary.json", records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gather", "prepare-compliance", "gather-compliance",
                                           "prepare-completion", "gather-completion",
                                           "prepare-body-tensors", "gather-body-tensors", "prepare-event-resolution", "gather-event-resolution",
                                           "prepare-output-filter-replays", "gather-output-filter-replays",
                                           "prepare-output-filter-double"))
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--out", type=Path, default=Path("Z:/p1/sims/Ti/Validation/audit/controls"))
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--macrostrain", type=float, default=1e-4)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--stop-time", type=float, default=.045)
    args = parser.parse_args()
    if args.action == "prepare-output-filter-double":
        if args.source is None:
            parser.error("prepare-output-filter-double requires --source parent replay directory")
        print(json.dumps(prepare_output_filter_precision_replay(args.source, args.out), indent=2))
    elif args.action == "gather-output-filter-replays":
        result = gather_output_filter_replays(args.out, args.queue_root or args.out/"queue")
        print(json.dumps({"jobs": [{k: r[k] for k in ("job", "solver_status", "native_field_frames", "native_energy_samples")}
                                     for r in result["results"]], "pending_jobs": result["pending_jobs"]}, indent=2))
    elif args.action == "prepare-output-filter-replays":
        print(json.dumps(prepare_output_filter_replays(args.root, args.out), indent=2))
    elif args.action == "gather-event-resolution":
        print(json.dumps([{key: row[key] for key in ("job", "event", "source_equivalence", "dense_sample_count")}
                          for row in gather_event_resolution(args.out, args.queue_root or args.out/"queue")], indent=2))
    elif args.action == "prepare-event-resolution":
        if args.source is None:
            parser.error("prepare-event-resolution requires --source")
        print(json.dumps(prepare_event_resolution(args.source, args.out, stop_time=args.stop_time), indent=2))
    elif args.action == "prepare-body-tensors":
        print(json.dumps(prepare_body_tensors(args.root, args.out, args.macrostrain), indent=2))
    elif args.action == "gather-body-tensors":
        if args.queue_root is None:
            parser.error("gather-body-tensors requires --queue-root")
        records = gather_body_tensors(args.out, args.queue_root)
        print(json.dumps([{key: row[key] for key in ("job", "state", "E_x_MPa", "E_y_MPa",
            "normal_shear_coupling_relative", "major_symmetry_relative_error")}
            for row in records], indent=2))
    elif args.action == "prepare-completion":
        print(json.dumps(prepare_completion_controls(args.root), indent=2))
    elif args.action == "gather-completion":
        print(json.dumps(gather_completion_controls(args.root, args.queue_root), indent=2))
    else:
        globals()[args.action.replace("-", "_")](args.root, args.out)
