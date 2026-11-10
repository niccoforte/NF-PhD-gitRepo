"""Exact-source FCC beam C(T) elastic/plastic feasibility controls.

Preparation never submits. The serial validation queue owns execution. The
potential-material result is an independent crack-energy reference only after
constituent, section and lattice equivalence have been demonstrated; incremental
plastic work is never differentiated as a substitute for J.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from validation_controls import verified_control_delivery, write_json, compliance_values
from validation_diagnostics import blocks, key, rows, sha, source_mesh, periodic_patch, prepare_rve
from validation_modal import job_directory


SOURCE = Path("Z:/p1/HPCarchive/Ti/data/per/0.0/FCC/6538532/zip/Fracture-FCC-20-per-1.inp")
OUT = Path("Z:/p1/sims/Ti/Validation/audit/fcc-calibration")


def immutable_json(path, value):
    """A prepared identity must not change after a queue might have consumed it."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Refusing to change prepared identity: " + str(path))
    else:
        write_json(path, value)


def potential_stress(strain, E, yield_stress, alpha, exponent):
    """Invert Abaqus deformation-plasticity uniaxial Ramberg--Osgood law."""
    from scipy.optimize import brentq
    if E <= 0 or yield_stress <= 0 or alpha <= 0 or exponent <= 1:
        raise ValueError("Invalid deformation-plasticity constants")
    values = np.asarray(strain, dtype=float)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Fit uses finite nonnegative monotonic tensile strains")
    def solve(value):
        if value == 0:
            return 0.
        return brentq(lambda stress: stress/E + alpha*yield_stress/E*(stress/yield_stress)**exponent-value,
                      0., E*value)
    return np.array([solve(value) for value in values.flat]).reshape(values.shape)


def fit_potential(E, nu, plastic):
    """Fit stress, including the elastic-to-yield interval, using only source data."""
    from scipy.optimize import least_squares
    plastic = np.asarray(plastic, dtype=float)
    stress, plastic_strain = plastic.T
    if plastic.shape[1] != 2 or plastic_strain[0] != 0 or np.any(np.diff(stress) <= 0):
        raise ValueError("Expected monotonic source hardening beginning at zero plastic strain")
    yield_stress = stress[0]
    elastic_strain = np.linspace(0, yield_stress/E, 11)[1:-1]
    total = np.r_[elastic_strain, stress/E+plastic_strain]
    source_stress = np.r_[elastic_strain*E, stress]
    result = least_squares(lambda logs: (potential_stress(total, E, yield_stress, *np.exp(logs))-source_stress)/yield_stress,
                           np.log([.2, 20.]), bounds=([-20., np.log(1.001)], [10., np.log(200.)]))
    alpha, exponent = np.exp(result.x)
    fitted = potential_stress(total, E, yield_stress, alpha, exponent)
    error = (fitted-source_stress)/yield_stress
    return {"E_MPa": E, "nu": nu, "yield_stress_MPa": float(yield_stress),
            "alpha": float(alpha), "exponent": float(exponent),
            "source_total_strain": total.tolist(), "source_stress_MPa": source_stress.tolist(),
            "fitted_stress_MPa": fitted.tolist(), "max_stress_error_over_yield": float(max(abs(error))),
            "rms_stress_error_over_yield": float(np.sqrt(np.mean(error**2))),
            "constituent_3pct_screen_passed": bool(max(abs(error)) <= .03),
            "scope": "Two-parameter Ramberg-Osgood approximation; no toughness or eta fit"}


def crack_extension(nodes, elements, crack_y, tip_x, extension):
    """Remove complete original straight struts intersected by the new notch.

    Degree-two collinear mesh vertices are internal subdivisions; true joints
    and corners stop traversal. Crack intervals are half open, preserving the
    original lattice phase at their advanced tip.
    """
    edges = {label: (i, j) for label, i, j in elements}
    adjacency = {node: [] for node in nodes}
    for label, (i, j) in edges.items():
        adjacency[i].append(label)
        adjacency[j].append(label)
    seeds = []
    for label, (i, j) in edges.items():
        p, q = np.array(nodes[i]), np.array(nodes[j])
        if abs(p[1]-q[1]) < 1e-9:
            if abs(p[1]-crack_y) < 1e-9 and max(p[0], q[0]) > tip_x and min(p[0], q[0]) < tip_x+extension:
                raise ValueError("A collinear crack-plane strut needs explicit physical review")
            continue
        if min(p[1], q[1])-1e-9 <= crack_y <= max(p[1], q[1])+1e-9:
            x = p[0]+(crack_y-p[1])*(q[0]-p[0])/(q[1]-p[1])
            if tip_x-1e-8 <= x < tip_x+extension-1e-8:
                seeds.append(label)
    removed = set(seeds)
    pending = list(seeds)
    while pending:
        label = pending.pop()
        for node in edges[label]:
            if len(adjacency[node]) != 2:
                continue
            first, second = adjacency[node]
            neighbors = [next(other for other in edges[item] if other != node) for item in (first, second)]
            u, v = (np.array(nodes[other])-np.array(nodes[node]) for other in neighbors)
            if abs(u[0]*v[1]-u[1]*v[0]) > 1e-8*np.linalg.norm(u)*np.linalg.norm(v) or np.dot(u, v) >= 0:
                continue
            for item in (first, second):
                if item not in removed:
                    removed.add(item)
                    pending.append(item)
    return removed


def label_rows(labels):
    labels = sorted(labels)
    return "\n".join(", ".join(map(str, labels[start:start+16])) for start in range(0, len(labels), 16))+"\n"


def source_geometry(source):
    text, nodes, elements, section, elastic = source_mesh(source)
    parsed = blocks(text)
    steps = next(i for i, block in enumerate(parsed) if key(block) == "*step")
    prefix = parsed[:steps]
    # Source assembly reference points and uniform section define actual dimensions.
    refs = [np.array([float(v) for v in rows(block)[0].split(",")[1:3]])
            for block in prefix if key(block) == "*node" and len(rows(block)) == 1]
    if len(refs) != 2 or abs(refs[0][0]-refs[1][0]) > 1e-8:
        raise ValueError("Expected source upper/lower pin reference points")
    crack_y = float((refs[0][1]+refs[1][1])/2)
    crossing = []
    for _, i, j in elements:
        p, q = np.array(nodes[i]), np.array(nodes[j])
        if min(p[1], q[1]) <= crack_y <= max(p[1], q[1]) and abs(p[1]-q[1]) > 1e-8:
            crossing.append(p[0]+(crack_y-p[1])*(q[0]-p[0])/(q[1]-p[1]))
    tip = float(min(crossing))
    W = float(max(point[0] for point in nodes.values())-refs[0][0])
    plastic = [[float(v) for v in line.split(",")] for block in prefix if key(block) == "*plastic" for line in rows(block)]
    if not plastic or section[0] <= 0:
        raise ValueError("Source material or thickness missing")
    return prefix, nodes, elements, {"W_mm": W, "a_mm": tip-float(refs[0][0]), "B_mm": section[0],
                                    "tip_x_mm": tip, "crack_y_mm": crack_y, "pin_x_mm": float(refs[0][0])}, elastic, plastic


def model_prefix(parsed, nodes, elements, removed, law, fit):
    """Retain exact mesh/sections/couplings; remove only reviewed notch struts."""
    surviving = {label: (i, j) for label, i, j in elements if label not in removed}
    used_nodes = {node for edge in surviving.values() for node in edge}
    prefix, in_instance = [], False
    for block in parsed:
        kind, header = key(block), block.splitlines()[0]
        if kind in ("*damage initiation", "*damage evolution", "*amplitude"):
            continue
        if kind == "*instance":
            in_instance = True
        elif kind == "*end instance":
            in_instance = False
        if kind == "*node" and in_instance:
            block = header+"\n"+"\n".join(line for line in rows(block) if int(line.split(",")[0]) in used_nodes)+"\n"
        elif kind == "*element":
            block = header+"\n"+"\n".join(line for line in rows(block) if int(line.split(",")[0]) in surviving)+"\n"
        elif kind in ("*elset", "*nset") and (in_instance or "instance=" in header.lower()):
            tokens = [int(value.strip()) for line in rows(block) for value in line.split(",") if value.strip()]
            if "generate" in header.lower():
                tokens = [value for start, stop, increment in zip(tokens[::3], tokens[1::3], tokens[2::3])
                          for value in range(start, stop+1, increment)]
                header = re.sub(r",\s*generate\b", "", header, flags=re.I)
            retained = set(tokens) & (set(surviving) if kind == "*elset" else used_nodes)
            if not retained:
                raise ValueError("Notch removes a required source set: " + header)
            block = header+"\n"+label_rows(retained)
        elif kind == "*plastic" and law != "incremental":
            continue
        elif kind == "*elastic" and law == "potential":
            block = "*Deformation Plasticity\n{E_MPa:.12g}, {nu:.12g}, {yield_stress_MPa:.12g}, {exponent:.12g}, {alpha:.12g}\n".format(**fit)
        prefix.append(block)
    return "".join(prefix).rstrip()+"\n", len(surviving), len(used_nodes)


def prepare(source, out, delta, h=10., endpoint_receipt=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if not np.isfinite(delta) or delta <= 0 or h <= 0:
        raise ValueError("Positive source-supported loading displacement and cell increment required")
    if endpoint_receipt is None:
        raise ValueError("A native initiation endpoint receipt is required")
    endpoint_receipt = Path(endpoint_receipt).resolve()
    endpoint = json.loads(endpoint_receipt.read_text())
    if Path(endpoint["source"]).resolve() != source.with_suffix(".odb"):
        raise ValueError("Native endpoint does not belong to this source model")
    field_cap = endpoint["events"]["LE"]["last_field_confirmed_delta_LL_mm"]
    if delta > field_cap*(1+1e-10):
        raise ValueError("Loading exceeds the field-confirmed intact source bracket")
    parsed, nodes, elements, geometry, elastic, plastic = source_geometry(source)
    fit = fit_potential(*elastic, plastic)
    out.mkdir(parents=True, exist_ok=True)
    immutable_json(out / "material-fit.json", dict(fit, source_inp=str(source), source_sha256=sha(source)))
    records = []
    for index in range(3):
        extension = h*index
        removed = crack_extension(nodes, elements, geometry["crack_y_mm"], geometry["tip_x_mm"], extension)
        if bool(removed) != bool(index):
            raise ValueError("Crack increment did not remove source struts")
        for law in ("elastic", "incremental", "potential"):
            name = "fcc_{}_a{}".format({"elastic": "el", "incremental": "inc", "potential": "pot"}[law], index)
            directory = job_directory(out, name)
            directory.mkdir(exist_ok=True)
            prefix, nel, nnode = model_prefix(parsed, nodes, elements, removed, law, fit)
            if law == "elastic":
                step = ("*Step,name=Compliance,nlgeom=NO\n*Static\n1.,1.\n*Boundary\nfixity,1,2\nload,1,1\n"
                        "*Cload\nload,2,1.\n*Output,field\n*Node Output\nU,RF\n*Output,history\n*Energy Output\nALLSE,ALLWK\n*End Step\n")
            else:
                step = ("*Step,name=Loading,nlgeom=YES,inc=10000\n*Static\n0.005,1.,1.e-9,0.005\n"
                        "*Boundary\nfixity,1,2\nload,1,1\nload,2,2,{:.12g}\n"
                        "*Output,field,time interval=0.025\n*Node Output\nU,RF\n"
                        "*Element Output,directions=YES\nS,E,SF,SM\n"
                        "*Output,history,frequency=1\n*Node Output,nset=load\nU2,RF2\n"
                        "*Node Output,nset=fixity\nU2,RF2\n*Energy Output\nALLIE,ALLSE,ALLPD,ALLWK\n*End Step\n").format(delta)
            contents = prefix+step
            inp = directory/(name+".inp")
            if inp.exists() and inp.read_text(encoding="ascii") != contents:
                raise ValueError("Refusing to overwrite prepared input: " + str(inp))
            inp.write_text(contents, encoding="ascii")
            meta = dict(geometry, job=name, input=str(inp), input_sha256=sha(inp), source_inp=str(source),
                        source_sha256=sha(source), source_odb=str(source.with_suffix(".odb")), law=law,
                        a_mm=geometry["a_mm"]+extension, h_mm=h, crack_index=index,
                        removed_source_element_labels=sorted(removed), retained_elements=nel, retained_nodes=nnode,
                        loading_displacement_mm=delta, endpoint_receipt=str(endpoint_receipt),
                        endpoint_receipt_sha256=sha(endpoint_receipt),
                        endpoint_definition=endpoint.get("criterion", "source-native initiation receipt"),
                        material_fit_sha256=sha(out/"material-fit.json"),
                        plane_strain_state="exact source mapped constituent; no additional reduction",
                        transformations=["Exact original B21 mesh, sections, material and pin couplings",
                                         "Whole source struts removed only for admissible +h/+2h crack extensions",
                                         "Damage/deletion and Explicit amplitude suppressed; stationary monotonic Standard solve",
                                         "Elastic controls remove plasticity; potential controls replace elastic/plastic law with fitted deformation plasticity"])
            immutable_json(directory/"source.json", meta)
            records.append(meta)
    def manifest(selected, name):
        value = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out/"queue"),
                 "jobs": [{"job": row["job"], "input": row["input"], "input_sha256": row["input_sha256"],
                           "cpus": 8, "required_outputs": [".odb", ".dat", ".sta"], "kind": "fcc_"+row["law"],
                           "purpose": "Source-exact FCC lattice feasibility; not completed plastic validation"} for row in selected]}
        immutable_json(out/name, value)
    manifest([row for row in records if row["law"] == "elastic"], "elastic-manifest.json")
    manifest([row for row in records if row["law"] != "elastic"], "plastic-manifest.json")
    manifest([row for row in records if row["law"] != "elastic" and row["crack_index"] == 0], "pilot-manifest.json")
    return records


def prepare_tensor(source, out):
    """Use the verified square *period*, retaining the actual FCC crossed graph.

    This does not call a square-lattice constitutive formula. The shared clipper
    merely uses a10x10mm translation window and checks opposite periodic classes.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    _, nodes, elements, section, elastic = source_mesh(source)
    coordinates, edges, classes, periods, origin, mismatch = periodic_patch(nodes, elements, "square", 10.)
    # Actual FCC contains axial and both diagonal families, unlike plain square.
    directions = np.array([coordinates[j]-coordinates[i] for i, j in edges])
    axial = np.any(np.isclose(directions, 0, atol=1e-8), axis=1)
    slopes = directions[:, 0]*directions[:, 1]
    if not (np.any(axial) and np.any(slopes > 1e-8) and np.any(slopes < -1e-8)):
        raise ValueError("FCC source periodic graph lacks axial/cross-brace families")
    out.mkdir(parents=True, exist_ok=True)
    case = {"topology": "FCC", "l_mm": "10", "nominal_relative_density": "0.2", "nnx": "20", "case_id": "FCC-source-CT-mapped"}
    existing = out/"rve_FCC_10_rho0p2_nx20"/"source.json"
    if existing.exists():
        meta = json.loads(existing.read_text())
        if sha(meta["input"]) != meta["input_sha256"] or sha(source) != meta["source_sha256"]:
            raise ValueError("Prepared FCC tensor identity changed")
        return [meta]
    meta = prepare_rve(out, case, source, coordinates, edges, classes, periods, section, elastic)
    meta.update(source_topology="FCC", origin_mm=origin.tolist(), opposite_boundary_mismatch_mm=mismatch,
                periodicity="10x10mm actual FCC source translations; square-period clipper, no square constitutive law",
                material_state="source mapped beam constituent; use resulting compliance directly without second plane-strain reduction")
    # The shared builder writes its initial receipt; freeze the final FCC-specific identity.
    path = Path(meta["input"]).parent/"source.json"
    write_json(path, meta)
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out.parent/"queue"), "jobs": [
        {"job": meta["job"], "input": meta["input"], "input_sha256": meta["input_sha256"], "cpus": 8,
         "required_outputs": [".odb", ".dat", ".sta"], "kind": "fcc_source_tensor",
         "purpose": "Actual source FCC CT section/mapped material periodic tensor; three independent perturbations"}]}
    immutable_json(out/"manifest.json", manifest)
    return [meta]


def prepare_continuation(out):
    """Select the four already-frozen remaining models, without resubmitting a0."""
    out = Path(out).resolve()
    records = [json.loads(path.read_text()) for path in sorted(out.glob("fcc_*/source.json"))]
    selected = [row for row in records if row["law"] in ("incremental", "potential") and row["crack_index"] in (1, 2)]
    if len(selected) != 4:
        raise ValueError("Exactly four remaining nonlinear source models required")
    for row in selected:
        if sha(row["input"]) != row["input_sha256"] or sha(row["source_inp"]) != row["source_sha256"]:
            raise ValueError("Prepared continuation identity changed")
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out/"nonlinear-continuation-queue"),
                "jobs": [{"job": row["job"], "input": row["input"], "input_sha256": row["input_sha256"], "cpus": 8,
                          "required_outputs": [".odb", ".dat", ".sta"], "kind": "fcc_"+row["law"],
                          "purpose": "Bounded source/potential crack-increment sensitivity; no automatic original-model eta qualification"}
                         for row in sorted(selected, key=lambda item: (item["crack_index"], item["law"]))]}
    immutable_json(out/"nonlinear-continuation-manifest.json", manifest)
    return selected


def gather_tensor(out, queue):
    """Receipt-bound three-strain periodic tensor with energy/reaction checks."""
    from odbAccess import openOdb
    out, queue = Path(out).resolve(), Path(queue).resolve()
    completed = {row["job"] for row in json.loads((queue/"state.json").read_text())["jobs"] if row["status"] == "completed"}
    results = []
    for path in sorted(out.glob("rve_FCC_*/source.json")):
        meta = json.loads(path.read_text())
        if meta["job"] not in completed:
            continue
        odb_path, receipt = verified_control_delivery(meta, queue, completed)
        before = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        columns, energies = [], []
        try:
            for index, label in enumerate(("Exx", "Eyy", "Gxy")):
                step = odb.steps[label]
                frame = step.frames[-1]
                rf = {value.nodeLabel: float(value.data[0]) for value in frame.fieldOutputs["RF"].values}
                reaction = np.array([rf[node] for node in meta["macro_nodes"]])
                columns.append(reaction/(meta["volume_mm3"]*meta["macrostrain"]))
                stored = [region.historyOutputs["ALLSE"].data[-1][1] for region in step.historyRegions.values()
                          if "ALLSE" in region.historyOutputs]
                if len(stored) != 1:
                    raise ValueError("Unique periodic stored energy required")
                predicted = .5*reaction[index]*meta["macrostrain"]
                if abs(stored[0]-predicted)/max(abs(predicted), 1e-30) > 1e-5:
                    raise ValueError("Periodic macro-reaction/stored-energy check failed")
                energies.append(float(stored[0]))
        finally:
            odb.close()
        C = np.column_stack(columns)
        symmetry_error = float(np.max(abs(C-C.T))/np.max(abs(C)))
        if symmetry_error > 1e-5 or np.linalg.eigvalsh((C+C.T)/2).min() <= 0:
            raise ValueError("Periodic tensor lacks symmetry/positive definiteness")
        S = np.linalg.inv(C)
        lam = float(S[0, 0]/S[1, 1])
        chi = float((2*S[0, 1]+S[2, 2])/(2*np.sqrt(S[0, 0]*S[1, 1])))
        M = float(np.sqrt(2*lam**1.5/(1+chi))/S[0, 0])
        if sha(odb_path) != before:
            raise ValueError("Periodic ODB changed during read-only extraction")
        result = dict(meta, C_MPa=C.tolist(), S_MPa_inv=S.tolist(), lambda_crack_axis=lam,
                      chi=chi, M_crack_axis_MPa=M, symmetry_relative_error=symmetry_error,
                      ALLSE_N_mm=energies, odb=str(odb_path), odb_sha256=before,
                      source_odb_sha256=before, source_input_hash_verified=True,
                      preparation_state=meta["state"], state="completed_and_extracted",
                      delivery_sha256=sha(receipt), read_only=True)
        write_json(path.parent/"result.json", result)
        results.append(result)
    write_json(out/"tensor-results.json", results)
    return results


def gather(out, queue):
    """Run in Abaqus Python: receipt-verified read-only native histories/sections."""
    from odbAccess import openOdb
    out = Path(out).resolve()
    queues = ([Path(queue).resolve()] if queue else
              [out/name for name in ("queue", "elastic-queue", "nonlinear-continuation-queue") if (out/name/"state.json").exists()])
    owners = {}
    for candidate in queues:
        for row in json.loads((candidate/"state.json").read_text())["jobs"]:
            if row["status"] == "completed":
                if row["job"] in owners:
                    raise ValueError("Duplicate completed job across FCC queues")
                owners[row["job"]] = candidate
    results = []
    for path in sorted(out.glob("fcc_*/source.json")):
        meta = json.loads(path.read_text())
        if meta["job"] not in owners:
            continue
        if sha(meta["endpoint_receipt"]) != meta["endpoint_receipt_sha256"]:
            raise ValueError("Source endpoint receipt changed")
        native_endpoint = json.loads(Path(meta["endpoint_receipt"]).read_text())
        if sha(native_endpoint["source"]) != native_endpoint["source_sha256"]:
            raise ValueError("Original native source ODB changed")
        odb_path, receipt = verified_control_delivery(meta, owners[meta["job"]], set(owners))
        before = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            if meta["law"] == "elastic":
                frame = odb.steps["Compliance"].frames[-1]
                upper = frame.fieldOutputs["U"].getSubset(region=odb.rootAssembly.nodeSets["LOAD"]).values
                fixed = frame.fieldOutputs["RF"].getSubset(region=odb.rootAssembly.nodeSets["FIXITY"]).values
                values = compliance_values(float(upper[0].data[1]), float(fixed[0].data[1]))
            else:
                step = odb.steps["Loading"]
                # Pin labels are assembly nodes, distinct from instance mesh nodes.
                load_label = odb.rootAssembly.nodeSets["LOAD"].nodes[0][0].label
                pin = [region.historyOutputs for name, region in step.historyRegions.items()
                       if name.upper() == "NODE ASSEMBLY."+str(load_label) and "RF2" in region.historyOutputs]
                if len(pin) != 1:
                    raise ValueError("Unique assembly loading-pin history required")
                u, rf = pin[0]["U2"].data, pin[0]["RF2"].data
                if len(u) != len(rf) or any(abs(a[0]-b[0]) > 1e-10 for a, b in zip(u, rf)):
                    raise ValueError("Native paired load/displacement timestamps differ")
                displacement, force = np.array([row[1] for row in u]), np.array([row[1] for row in rf])
                if np.any(np.diff(displacement) < -1e-9) or force[-1] <= 0:
                    raise ValueError("Expected monotonic positive pin loading")
                work = np.r_[0., np.cumsum(.5*(force[1:]+force[:-1])*np.diff(displacement))]
                energy = [region.historyOutputs for region in step.historyRegions.values() if "ALLWK" in region.historyOutputs]
                if len(energy) != 1:
                    raise ValueError("Unique whole-model energy history required")
                energy_values = {name: [[float(t), float(v)] for t, v in item.data]
                                 for name, item in energy[0].items() if name in ("ALLIE", "ALLSE", "ALLPD", "ALLWK")}
                section = {}
                selected = sorted({min(range(len(step.frames)), key=lambda i: abs(step.frames[i].frameValue-target))
                                   for target in (0., .1, .25, .5, .75, 1.)})
                for number in selected:
                    frame = step.frames[number]
                    section["time_{}".format(number)] = np.array([float(frame.frameValue)])
                    for name in ("S", "E", "SF", "SM"):
                        if name in frame.fieldOutputs:
                            field = frame.fieldOutputs[name]
                            labels, data = [], []
                            for value in field.values:
                                labels.append([int(value.elementLabel), int(value.integrationPoint),
                                               int(value.sectionPoint.number) if value.sectionPoint else 0])
                                data.append(list(map(float, value.data)))
                            section["{}_{}_labels".format(name, number)] = np.asarray(labels)
                            section["{}_{}_data".format(name, number)] = np.asarray(data)
                np.savez_compressed(str(path.parent/"sections.npz"), **section)
                values = {"time": [float(row[0]) for row in u], "opening_mm": displacement.tolist(),
                          "force_N": force.tolist(), "work_N_mm": work.tolist(), "energy": energy_values,
                          "sections": str(path.parent/"sections.npz"), "sections_sha256": sha(path.parent/"sections.npz")}
        finally:
            odb.close()
        if sha(odb_path) != before:
            raise ValueError("Source ODB changed during read-only extraction")
        record = dict(meta, **values, raw_source_odb=meta["source_odb"], raw_source_odb_sha256=native_endpoint["source_sha256"],
                      source_odb=str(odb_path), source_odb_sha256=before,
                      state="completed_and_extracted", source_input_hash_verified=True,
                      odb=str(odb_path), odb_sha256=before, delivery_sha256=sha(receipt), read_only=True)
        write_json(path.parent/"result.json", record)
        results.append(record)
    write_json(out/"results.json", results)
    return results


def matched_section_frames(original, potential, tolerance=1e-7):
    """Join section snapshots by saved loading time, never by ODB frame index."""
    def frames(archive):
        return [(float(archive[name][0]), name.split("_")[1])
                for name in archive.files if name.startswith("time_")]
    pairs = []
    available = frames(potential)
    for time, index in sorted(frames(original)):
        matched = [(other_time, other_index) for other_time, other_index in available
                   if abs(other_time-time) <= tolerance]
        if len(matched) != 1:
            raise ValueError("Unique potential section frame at original loading time required")
        other_time, other_index = matched[0]
        pairs.append((time, index, other_index))
        available.remove((other_time, other_index))
    if available:
        raise ValueError("Original/potential saved section loading times differ")
    return pairs


def compare_pilot(out):
    """Write measured global/local feasibility evidence, without promoting eta."""
    out = Path(out).resolve()
    original = json.loads((out/"fcc_inc_a0/result.json").read_text())
    potential = json.loads((out/"fcc_pot_a0/result.json").read_text())
    fit = json.loads((out/"material-fit.json").read_text())
    opening = np.linspace(0., min(original["opening_mm"][-1], potential["opening_mm"][-1]), 201)
    forces = [np.interp(opening, row["opening_mm"], row["force_N"]) for row in (original, potential)]
    works = [float(np.trapz(force, opening)) for force in forces]
    native = np.load(original["sections"])
    counterpart = np.load(potential["sections"])
    local = []
    frames = matched_section_frames(native, counterpart)
    for time, frame, other_frame in frames:
        label, other_label = "S_{}_data".format(frame), "S_{}_data".format(other_frame)
        ids, other_ids = label.replace("_data", "_labels"), other_label.replace("_data", "_labels")
        if not np.array_equal(native[ids], counterpart[other_ids]):
            raise ValueError("Potential/source section output locations differ")
        difference = counterpart[other_label]-native[label]
        local.append({"frame": int(frame), "potential_frame": int(other_frame), "time": time,
                      "max_stress_error_over_yield": float(np.max(abs(difference))/fit["yield_stress_MPa"]),
                      "rms_stress_error_over_yield": float(np.sqrt(np.mean(difference**2))/fit["yield_stress_MPa"])})
    _, final_frame, other_final_frame = frames[-1]
    final_key, other_final_key = "S_{}_data".format(final_frame), "S_{}_data".format(other_final_frame)
    error = abs(counterpart[other_final_key]-native[final_key])
    row, component = np.unravel_index(np.argmax(error), error.shape)
    point = native[final_key.replace("_data", "_labels")][row].astype(int).tolist()
    local_curve = []
    for sample in local:
        frame, other_frame = sample["frame"], sample["potential_frame"]
        labels = native["S_{}_labels".format(frame)]
        matched = np.flatnonzero(np.all(labels == point, axis=1))
        if len(matched) != 1:
            raise ValueError("Section point not unique across feasibility frames")
        item = int(matched[0])
        local_curve.append([sample["time"], float(native["S_{}_data".format(frame)][item, component]),
                            float(counterpart["S_{}_data".format(other_frame)][item, component])])
    _, nodes, elements, geometry, _, _ = source_geometry(original["source_inp"])
    distances = {label: float(np.linalg.norm((np.array(nodes[i])+np.array(nodes[j]))/2-
                                           [geometry["tip_x_mm"], geometry["crack_y_mm"]]))
                 for label, i, j in elements}
    section_labels = native[final_key.replace("_data", "_labels")]
    near_tip = np.array([distances[int(label[0])] <= 20. for label in section_labels])
    report = {"source_sha256": original["source_sha256"], "raw_source_odb_sha256": original["raw_source_odb_sha256"],
              "original_result_sha256": sha(out/"fcc_inc_a0/result.json"),
              "potential_result_sha256": sha(out/"fcc_pot_a0/result.json"),
              "material_fit_sha256": sha(out/"material-fit.json"),
              "opening_mm": float(opening[-1]), "force_N": [float(force[-1]) for force in forces],
              "work_N_mm": works, "relative_final_force_difference": float(forces[1][-1]/forces[0][-1]-1),
              "max_force_difference_over_original_peak": float(np.max(abs(forces[1]-forces[0]))/np.max(abs(forces[0]))),
              "relative_work_difference": works[1]/works[0]-1,
              "section_stress_comparison": local, "worst_final_section_element_ip_sp": point,
              "worst_final_section_component_index": int(component), "worst_point_stress_history": local_curve,
              "near_tip_radius_mm": 20., "near_tip_location_definition": "Source beam element midpoint within radius of original notch tip",
              "near_tip_final_max_stress_error_over_yield": float(np.max(error[near_tip])/fit["yield_stress_MPa"]),
              "worst_final_point_distance_from_tip_mm": distances[point[0]],
              "constituent_3pct_screen_passed": fit["constituent_3pct_screen_passed"],
              "potential_original_equivalence_accepted": False,
              "eta_qualified": False, "eta_plastic": None,
              "decision": "Bounded six-run feasibility completed. Excellent global response does not remove constituent/local unloading mismatch; no source-model eta qualified.",
              "ASTM_eta": 2.+.522*(original["W_mm"]-original["a_mm"])/original["W_mm"],
              "scope": "Actual source-lattice monotonic Standard feasibility; source intact-field cap, no simulated initiation claim"}
    write_json(out/"pilot-equivalence.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].plot(np.array(fit["source_total_strain"])*100, fit["source_stress_MPa"], "k.-", label="Original Ti law")
    axes[0].plot(np.array(fit["source_total_strain"])*100, fit["fitted_stress_MPa"], color="tab:orange", label="Potential approximation")
    axes[0].set(xlabel="Constituent tensile strain (%)", ylabel="Stress (MPa)")
    axes[0].legend(fontsize=8)
    axes[0].text(.03, .05, "Maximum mismatch: {:.2f}% of yield stress".format(100*fit["max_stress_error_over_yield"]), transform=axes[0].transAxes, fontsize=8)
    for force, style, name in zip(forces, ("-", "--"), ("Original incremental plasticity", "Potential approximation")):
        axes[1].plot(opening, force/1000, style, color="tab:blue" if style == "-" else "tab:orange", label=name)
    axes[1].set(xlabel="Load-line opening (mm)", ylabel="Force (kN)")
    axes[1].legend(fontsize=8)
    axes[1].text(.03, .05, "Final force: {:+.2f}%\nIntegrated work: {:+.2f}%".format(100*report["relative_final_force_difference"], 100*report["relative_work_difference"]), transform=axes[1].transAxes, fontsize=8)
    values = np.array(local_curve)
    axes[2].plot(values[:, 0]*opening[-1], values[:, 1], "o-", color="tab:blue", markersize=4, label="Original incremental plasticity")
    axes[2].plot(values[:, 0]*opening[-1], values[:, 2], "s--", color="tab:orange", markersize=4, label="Potential approximation")
    axes[2].set(xlabel="Load-line opening (mm)", ylabel="Section stress (MPa)", title="Element {}, section point {}".format(point[0], point[2]))
    axes[2].legend(fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(out/("plastic-feasibility."+suffix), dpi=200)
    plt.close(fig)
    return report


def work_derivative(opening, forces, h, B):
    """Generic fixed-opening work differences, with no claim that they are J."""
    opening, forces = np.asarray(opening), np.asarray(forces)
    if (forces.shape != (3, len(opening)) or h <= 0 or B <= 0 or len(opening) < 2 or
            not np.all(np.isfinite(forces)) or not np.all(np.isfinite(opening)) or
            np.any(np.diff(opening) <= 0) or opening[0] != 0):
        raise ValueError("Three finite aligned responses from zero on an increasing common grid required")
    work = np.column_stack((np.zeros(3), np.cumsum(.5*(forces[:, 1:]+forces[:, :-1])*np.diff(opening), axis=1)))
    return {"work_N_mm": work.tolist(),
            "h": ((work[0]-work[1])/(B*h)).tolist(),
            "next_h": ((work[1]-work[2])/(B*h)).tolist(),
            "2h": ((work[0]-work[2])/(2*B*h)).tolist(),
            "tip_second_order": ((3*work[0]-4*work[1]+work[2])/(2*B*h)).tolist()}


def energy_reference(opening, potential_forces, h, B, *, law):
    """Potential-model energy differences; reject incremental-plasticity inputs."""
    if law != "potential":
        raise ValueError("Only a potential constitutive law supplies this conditional energy reference")
    values = work_derivative(opening, potential_forces, h, B)
    return dict({"J_{}_N_mm".format(name): values[name] for name in ("h", "next_h", "2h", "tip_second_order")},
                scope="Potential-model energy derivative only; finite lattice crack increments, equivalence not implied")


def compare_elastic(out):
    """Actual lattice compliance versus the same-interval Ortega elastic J."""
    from scipy.integrate import quad
    from resources.calculations import calc_FaW_aniso
    out = Path(out).resolve()
    controls = [json.loads((out/("fcc_el_a"+str(i))/"result.json").read_text()) for i in range(3)]
    tensor = json.loads((out/"periodic/tensor-results.json").read_text())[0]
    C = np.array(tensor["C_MPa"])
    M = tensor["M_crack_axis_MPa"]
    # This FCC tensor has equal normal coefficients, so the axis swap leaves
    # lambda=1. Still reorder explicitly to Ortega's loading-axis-first order.
    source_C = C[np.ix_([1, 0, 2], [1, 0, 2])]
    B, W, a, h = (controls[0][name] for name in ("B_mm", "W_mm", "a_mm", "h_mm"))
    if abs(C[0, 0]/C[1, 1]-1) > 1e-5:
        raise ValueError("FCC equal-axis verification cannot silently reuse a crack-axis M for unequal axes")
    compliance = np.array([control["compliance_mm_N"] for control in controls])
    intervals = []
    factor = lambda crack: calc_FaW_aniso(crack, W, C=source_C)**2/(B*B*W*M)
    for first, second in ((0, 1), (1, 2), (0, 2)):
        increment = (second-first)*h
        lattice = (compliance[second]-compliance[first])/(2*B*increment)
        ortega = quad(factor, a+first*h, a+second*h, epsabs=1e-16, epsrel=1e-10)[0]/increment
        intervals.append({"a1_mm": a+first*h, "a2_mm": a+second*h,
                          "lattice_G_over_P2_Ninv_mminv": float(lattice),
                          "Ortega_J_over_P2_Ninv_mminv": float(ortega),
                          "relative_J_difference": float(ortega/lattice-1),
                          "equivalent_K_difference": float(np.sqrt(ortega/lattice)-1)})
    report = {"source_inp": controls[0]["source_inp"], "source_sha256": controls[0]["source_sha256"],
              "compliance_results_sha256": [sha(out/("fcc_el_a"+str(i))/"result.json") for i in range(3)],
              "tensor_result_sha256": sha(out/"periodic/tensor-results.json"),
              "M_MPa": M, "lambda": tensor["lambda_crack_axis"], "chi": tensor["chi"],
              "W_mm": W, "B_mm": B, "a_mm": a, "h_mm": h,
              "compliance_mm_N": compliance.tolist(), "intervals": intervals,
              "scope": "Actual original B21 lattice; independent unit-load compliance derivative versus interval-averaged Ortega. This is numerical elastic verification, not published initiation data or plastic eta validation.",
              "further_gate": "Single source mesh/cell increment pilot; local lattice phase and discretisation limits remain."}
    write_json(out/"elastic-comparison.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    _, nodes, elements, _, _ = source_mesh(controls[0]["source_inp"])
    geometry = source_geometry(controls[0]["source_inp"])[3]
    all_segments = [[nodes[i], nodes[j]] for _, i, j in elements]
    axes[0].add_collection(LineCollection(all_segments, colors=".65", linewidths=.35))
    previous = set()
    for index, color in ((1, "tab:red"), (2, "tab:green")):
        removed = set(controls[index]["removed_source_element_labels"])
        segments = [[nodes[i], nodes[j]] for label, i, j in elements if label in removed-previous]
        axes[0].add_collection(LineCollection(segments, colors=color, linewidths=1.4, label="+{} mm crack extension".format(int(h*index))))
        previous = removed
    axes[0].plot([geometry["tip_x_mm"]+i*h for i in range(3)], [geometry["crack_y_mm"]]*3, "o", color="tab:blue", markersize=4)
    axes[0].autoscale()
    axes[0].set(aspect="equal", xlabel="x (mm)", ylabel="y (mm)", title="Original FCC beam mesh")
    axes[0].legend(fontsize=8)
    positions = np.arange(3)
    axes[1].plot(positions, [row["lattice_G_over_P2_Ninv_mminv"]*1e8 for row in intervals], "o", color="tab:blue", label="Lattice compliance energy")
    axes[1].plot(positions, [row["Ortega_J_over_P2_Ninv_mminv"]*1e8 for row in intervals], "^", mfc="none", color="k", label="Ortega interval average")
    axes[1].set(xticks=positions, xticklabels=["{}–{}".format(int(row["a1_mm"]), int(row["a2_mm"])) for row in intervals],
                xlabel="Crack-length interval (mm)", ylabel=r"$G/P^2$ ($10^{-8}$ N$^{-1}$ mm$^{-1}$)")
    axes[1].legend(fontsize=8)
    for pos, row in zip(positions, intervals):
        axes[1].annotate("{:+.2f}% J".format(row["relative_J_difference"]*100), (pos, row["Ortega_J_over_P2_Ninv_mminv"]*1e8), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axes[1].margins(x=.18, y=.18)
    for suffix in ("png", "svg"):
        fig.savefig(out/("elastic-lattice-verification."+suffix), dpi=200)
    plt.close(fig)
    return report


def conditional_plastic(out):
    """Six-run sensitivity, explicitly conditional on the potential approximation.

    The incremental-work derivative is retained only to quantify sensitivity;
    it is not called J. No scalar is promoted to a validated source-model eta.
    """
    from resources.calculations import calc_FaW_aniso
    out = Path(out).resolve()
    all_rows = {}
    for law, short in (("incremental", "inc"), ("potential", "pot")):
        all_rows[law] = [json.loads((out/("fcc_{}_a{}".format(short, i))/"result.json").read_text()) for i in range(3)]
    baseline = all_rows["potential"][0]
    elastic = [json.loads((out/("fcc_el_a"+str(i))/"result.json").read_text()) for i in range(3)]
    tensor = json.loads((out/"periodic/tensor-results.json").read_text())[0]
    cap = min(row["opening_mm"][-1] for values in all_rows.values() for row in values)
    opening = np.linspace(0., cap, 201)
    B, W, a, h = (baseline[name] for name in ("B_mm", "W_mm", "a_mm", "h_mm"))
    M = tensor["M_crack_axis_MPa"]
    C = np.array(tensor["C_MPa"])[np.ix_([1, 0, 2], [1, 0, 2])]
    f = calc_FaW_aniso(a, W, C=C)
    forces, work_differences = {}, {}
    equivalence = []
    for law, values in all_rows.items():
        if any(row["law"] != law or row["source_sha256"] != baseline["source_sha256"] or
               not row["read_only"] or not row["source_input_hash_verified"] or
               row["a_mm"] != a+i*h or row["W_mm"] != W or row["B_mm"] != B
               for i, row in enumerate(values)):
            raise ValueError("Source-qualified, geometrically matched three-crack results required")
        forces[law] = np.array([np.interp(opening, row["opening_mm"], row["force_N"]) for row in values])
        work_differences[law] = work_derivative(opening, forces[law], h, B)
    derivatives = energy_reference(opening, forces["potential"], h, B, law="potential")
    for index in range(3):
        first, second = forces["incremental"][index], forces["potential"][index]
        first_work, second_work = np.trapz(first, opening), np.trapz(second, opening)
        equivalence.append({"a_mm": a+index*h, "relative_final_force_difference": float(second[-1]/first[-1]-1),
                            "relative_work_difference": float(second_work/first_work-1),
                            "max_force_difference_over_original_peak": float(max(abs(second-first))/max(abs(first)))})
    compliance = np.array([row["compliance_mm_N"] for row in elastic])
    k = np.array([row["static_stiffness_N_mm"] for row in elastic])
    coefficients = []
    coefficient_arrays = {}
    # Forward intervals and original-notch three-point derivative. The same
    # stencil acts on potential work and the independently solved compliance.
    stencils = (("h", 0, [-1., 1., 0.], h), ("2h", 0, [-1., 0., 1.], 2*h),
                ("tip_second_order", 0, [-3., 4., -1.], 2*h),
                ("next_h", 1, [0., -1., 1.], h))
    for name, anchor, weights, denominator in stencils:
        weights = np.array(weights)/denominator
        force = forces["potential"][anchor]
        work = np.array(work_differences["potential"]["work_N_mm"])[anchor]
        Apl = work-.5*force**2*compliance[anchor]
        factor = calc_FaW_aniso(a+anchor*h, W, C=C)
        Jel = force**2*factor**2/(B*B*W*M)
        compliance_derivative = float(np.dot(weights, compliance))
        Jel_lattice = force**2*compliance_derivative/(2*B)
        # A finite crack difference at fixed opening differs from P² dC/da;
        # retain its exact linear limit as an additional discretisation check.
        linear_opening_coefficient = float(-np.dot(weights, k)/(2*B))
        Jel_fixed_opening = (force/k[anchor])**2*linear_opening_coefficient
        reference = np.array(derivatives["J_{}_N_mm".format(name)])
        usable = Apl > max(1e-6, abs(work[-1])*1e-5)
        eta, eta_lattice, eta_fixed_opening = [np.full(len(opening), np.nan) for _ in range(3)]
        for values, elastic_reference in ((eta, Jel), (eta_lattice, Jel_lattice),
                                           (eta_fixed_opening, Jel_fixed_opening)):
            values[usable] = B*(W-a-anchor*h)*(reference[usable]-elastic_reference[usable])/Apl[usable]
        coefficient_arrays[name] = (eta, eta_lattice, eta_fixed_opening)
        finite_list = lambda values: [float(value) if np.isfinite(value) else None for value in values]
        levels = []
        for fraction in (.25, .5, .75, 1.):
            index = int(round(fraction*(len(opening)-1)))
            levels.append({"fraction_of_source_intact_field_opening": fraction,
                           "opening_mm": float(opening[index]), "force_N": float(force[index]),
                           "Apl_N_mm": float(Apl[index]), "Apl_over_total_work": float(Apl[index]/work[index]),
                           "Ortega_Jel_N_mm": float(Jel[index]),
                           "lattice_compliance_Jel_N_mm": float(Jel_lattice[index]),
                           "lattice_fixed_opening_linear_J_N_mm": float(Jel_fixed_opening[index]),
                           "potential_J_cell_difference_N_mm": float(reference[index]),
                           "eta_potential_Ortega_conditional": finite_list(eta[index:index+1])[0],
                           "eta_potential_lattice_compliance_conditional": finite_list(eta_lattice[index:index+1])[0],
                           "eta_potential_fixed_opening_linear_conditional": finite_list(eta_fixed_opening[index:index+1])[0],
                           "incremental_work_cell_derivative_NOT_J_N_mm": work_differences["incremental"][name][index]})
        coefficients.append({"stencil": name, "anchor_a_mm": a+anchor*h,
                             "crack_interval_mm": [a+anchor*h, a+(2 if name in ("2h", "tip_second_order", "next_h") else 1)*h],
                             "compliance_derivative_Ninv": compliance_derivative,
                             "linear_fixed_opening_coefficient_N_mm3": linear_opening_coefficient,
                             "ASTM_eta_at_anchor": 2.+.522*(W-a-anchor*h)/W,
                             "levels": levels,
                             "eta_Ortega": finite_list(eta), "eta_lattice_compliance": finite_list(eta_lattice),
                             "eta_fixed_opening_linear": finite_list(eta_fixed_opening),
                             "Apl_N_mm": Apl.tolist(), "J_potential_N_mm": reference.tolist(),
                             "Jel_Ortega_N_mm": Jel.tolist(), "Jel_lattice_compliance_N_mm": Jel_lattice.tolist()})
    tip = coefficients[2]
    eta, eta_lattice, eta_fixed_opening = coefficient_arrays["tip_second_order"]
    ASTM = 2.+.522*(W-a)/W
    lattice_tip_coefficient = (3*k[0]-4*k[1]+k[2])/(4*B*h)
    ortega_tip_coefficient = k[0]**2*f*f/(B*B*W*M)
    energy_accounting = []
    for law, values in all_rows.items():
        for row in values:
            native_work = row["work_N_mm"][-1]
            energy = {name: series[-1][1] for name, series in row["energy"].items()}
            energy_accounting.append({"law": law, "a_mm": row["a_mm"], "native_pin_work_N_mm": native_work,
                                      "whole_model_final_energies_N_mm": energy,
                                      "ALLWK_over_pin_work_minus_one": energy["ALLWK"]/native_work-1,
                                      "ALLIE_over_pin_work_minus_one": energy["ALLIE"]/native_work-1})
    report = {"state": "conditional_potential_reference_only", "eta_source_model_qualified": False,
              "source_sha256": baseline["source_sha256"], "M_MPa": M, "Ortega_f": f,
              "source_inp": baseline["source_inp"], "raw_source_odb": baseline["raw_source_odb"],
              "raw_source_odb_sha256": baseline["raw_source_odb_sha256"],
              "W_mm": W, "B_mm": B, "a_mm": a, "h_mm": h, "ASTM_eta": ASTM,
              "potential_eta_final": float(eta[-1]),
              "potential_eta_lattice_compliance_final": float(eta_lattice[-1]),
              "potential_eta_fixed_opening_linear_final": float(eta_fixed_opening[-1]),
              "relative_final_eta_difference_from_ASTM": float(eta[-1]/ASTM-1),
              "relative_final_lattice_eta_difference_from_ASTM": float(eta_lattice[-1]/ASTM-1),
              "opening_mm": opening.tolist(), "levels": tip["levels"], "crack_stencils": coefficients,
              "global_response_equivalence": equivalence, "energy_accounting": energy_accounting,
              "linear_tip_cell_difference_coefficient_N_mm3": float(lattice_tip_coefficient),
              "linear_Ortega_tip_coefficient_N_mm3": float(ortega_tip_coefficient),
              "linear_cell_difference_vs_Ortega_J_relative": float(lattice_tip_coefficient/ortega_tip_coefficient-1),
              "independent_reference": "Potential-model work integrated from paired monotonic pin histories; one-sided three-cell fixed-opening energy difference at original notch",
              "coefficient_definitions": {"Apl": "Integral P d(delta_LL) minus P² C0/2; original negative values retained",
                                          "eta_Ortega": "B(W-a)(Jpot - P² f²/(B² W M))/Apl",
                                          "eta_lattice_compliance": "B(W-a)(Jpot - P² (same-stencil dC/da)/(2B))/Apl",
                                          "eta_fixed_opening_linear": "Diagnostic B(W-a)(Jpot - (P/k0)² (same-stencil -dk/da)/(2B))/Apl; exact finite-difference linear limit",
                                          "unavailable": "Apl <= max(1e-6 Nmm, final total work times 1e-5); no clipping of negative area",
                                          "opening": "Upper pin U2, lower pin fixed; full specimen, no factor2"},
              "limitations": ["Source incremental law and potential approximation have different local unloading; see pilot-equivalence.json",
                              "h=10mm and2h=20mm are finite lattice intervals, not converged smooth crack derivatives",
                              "Potential/incremental global work derivative difference is sensitivity, not validation of incremental J",
                              "Finite-difference compliance and stiffness derivatives do not commute; same-stencil compliance subtraction does not exactly cancel the fixed-opening linear energy difference",
                              "Near-elastic eta is ill-conditioned because small Apl magnifies elastic-reference offsets",
                              "ALLIE and ALLSE are retained diagnostics, not substituted for the integrated potential-model pin work",
                              "Elastic lattice-to-continuum mismatch remains; no plastic eta promoted to production"],
              "tensor_result_sha256": sha(out/"periodic/tensor-results.json"),
              "elastic_result_sha256": [sha(out/("fcc_el_a"+str(i))/"result.json") for i in range(3)],
              "pilot_equivalence_sha256": sha(out/"pilot-equivalence.json"),
              "result_sha256": {law: [sha(out/("fcc_{}_a{}".format("inc" if law == "incremental" else "pot", i))/"result.json") for i in range(3)] for law in all_rows}}
    write_json(out/"plastic-calibration-conditional.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    labels = {"h": "80–90 mm", "2h": "80–100 mm", "tip_second_order": "80 mm tip (three cracks)"}
    colors = {"h": "tab:orange", "2h": "tab:green", "tip_second_order": "tab:blue"}
    for name in ("h", "2h", "tip_second_order"):
        for ax, curve in zip(axes, coefficient_arrays[name]):
            ax.plot(opening, curve, color=colors[name], label=labels[name])
    for ax in axes:
        ax.axhline(ASTM, color="k", linestyle="--", label="ASTM ({:.3f})".format(ASTM))
        # Zoom only: all load levels, near-elastic singular values and the90mm
        # anchor remain available in the JSON with no value clipping.
        ax.set(xlim=(cap*.5, cap), xlabel="Full load-line opening (mm)", ylabel=r"Conditional $\eta_{\mathrm{pl}}$")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    visible = opening >= cap*.5
    for ax, position in zip(axes, (0, 1, 2)):
        curves = np.concatenate([coefficient_arrays[name][position][visible] for name in labels])
        finite = curves[np.isfinite(curves)]
        lower, upper = min(float(finite.min()), ASTM), max(float(finite.max()), ASTM)
        margin = max(.15, (upper-lower)*.1)
        ax.set_ylim(lower-margin, upper+margin)
    axes[0].set_title("Subtract Ortega elastic J")
    axes[1].set_title("Subtract same-stencil lattice compliance J")
    axes[2].set_title("Subtract exact fixed-opening linear limit\n(discretisation diagnostic)")
    fig.suptitle("FCC potential-model sensitivity — source-model η remains unqualified\nLast half of intact-field loading; full curves and 90–100 mm anchor in JSON", fontsize=11)
    for suffix in ("png", "svg"):
        fig.savefig(out/("plastic-cell-sensitivity."+suffix), dpi=200)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gather", "prepare-tensor", "gather-tensor", "compare-pilot", "prepare-continuation", "compare-elastic", "conditional-plastic"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--h", type=float, default=10.)
    parser.add_argument("--endpoint-receipt", type=Path)
    args = parser.parse_args()
    if args.action == "conditional-plastic":
        result = [conditional_plastic(args.out)]
    elif args.action == "compare-elastic":
        result = [compare_elastic(args.out)]
    elif args.action == "prepare-continuation":
        result = prepare_continuation(args.out)
    elif args.action == "compare-pilot":
        result = [compare_pilot(args.out)]
    elif args.action == "prepare-tensor":
        result = prepare_tensor(args.source, args.out)
    elif args.action == "gather-tensor":
        result = gather_tensor(args.out, args.queue or args.out.parent/"queue")
    elif args.action == "prepare":
        result = prepare(args.source, args.out, args.delta, args.h, args.endpoint_receipt)
    else:
        result = gather(args.out, args.queue)
    print(json.dumps({"prepared" if args.action.startswith("prepare") else "extracted": len(result), "out": str(args.out)}))


if __name__ == "__main__":
    main()
