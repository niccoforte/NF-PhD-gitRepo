"""Independent validation diagnostics; preparation never submits Abaqus jobs.

Periodic elastic controls are clipped from exact archived B21 meshes. A local
frame-energy calculation includes bending, periodic fluctuation relaxation and
optional transverse shear. Abaqus controls, run separately in a serial queue,
test the independent implementation against the source B21 formulation.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PRIORITY = ("10/0.2/Fracture-tri-70-per-1.inp",
            "10/0.2/Fracture-kagome-40-per-1.inp",
            "10/0.2/Fracture-hex-60-per-1.inp",
            "3/0.16/Fracture-hex-80-per-1.inp")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_table(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def blocks(text):
    starts = list(re.finditer(r"(?m)^\*(?!\*)[^\n]*", text))
    return [text[item.start():starts[i + 1].start() if i + 1 < len(starts) else len(text)]
            for i, item in enumerate(starts)]


def key(block):
    return block.splitlines()[0].split(",")[0].lower()


def rows(block):
    return [line.strip() for line in block.splitlines()[1:]
            if line.strip() and not line.startswith("**")]


def source_mesh(path):
    text = Path(path).read_text(encoding="ascii")
    nodes, elements, section, elastic = {}, [], None, None
    in_instance = False
    for block in blocks(text):
        kind = key(block)
        if kind == "*instance":
            if in_instance:
                raise ValueError("Multiple/nested mesh instances need review")
            in_instance = True
        elif kind == "*end instance":
            in_instance = False
        elif in_instance and kind == "*node":
            for line in rows(block):
                values = line.split(",")
                nodes[int(values[0])] = [float(values[1]), float(values[2])]
        elif in_instance and kind == "*element":
            if "type=b21" not in block.splitlines()[0].lower().replace(" ", ""):
                raise ValueError("Only exact B21 source meshes are supported")
            elements.extend([int(v) for v in line.split(",")] for line in rows(block))
        elif kind == "*beam section":
            if section is not None or "section=rect" not in block.splitlines()[0].lower().replace(" ", ""):
                raise ValueError("Expected one uniform rectangular section")
            section = [float(v) for v in rows(block)[0].split(",")]
            if rows(block)[1].replace(" ", "") != "0.,0.,-1.":
                raise ValueError("Unexpected source beam section orientation")
        elif kind == "*elastic":
            elastic = [float(v) for v in rows(block)[0].split(",")]
    if not nodes or not elements or section is None or elastic is None:
        raise ValueError("Incomplete source mesh/material")
    return text, nodes, elements, section, elastic


def clip_segment(a, b, lower, upper):
    direction, enter, leave = b - a, 0., 1.
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if a[axis] < lower[axis] or a[axis] > upper[axis]:
                return None
        else:
            values = sorted(((lower[axis] - a[axis]) / direction[axis],
                             (upper[axis] - a[axis]) / direction[axis]))
            enter, leave = max(enter, values[0]), min(leave, values[1])
    if leave - enter < 1e-8:
        return None
    return a + enter * direction, a + leave * direction


def periodic_patch(nodes, elements, lat, length):
    # Periods follow the actual source beam-angle families. A shifted cut avoids
    # coincident boundary bars, so no thickness/energy half-weighting is needed.
    periods = np.array([2 * length, 2 * np.sqrt(3) * length]) if lat == "kagome" else np.array([
        np.sqrt(3) * length, (3 if lat == "hex" else 1) * length])
    maximum = np.max(list(nodes.values()), axis=0)
    origin = np.floor(maximum * [.7, .2] / periods) * periods + [.137 * length, .193 * length]
    patch, edges = [], []
    tolerance = length * 2e-5

    def point_index(point):
        point = point - origin
        for i, existing in enumerate(patch):
            if np.linalg.norm(point - existing) < tolerance:
                return i
        patch.append(point)
        return len(patch) - 1

    for _, first, second in elements:
        clipped = clip_segment(np.array(nodes[first]), np.array(nodes[second]), origin, origin + periods)
        if clipped is not None:
            a, b = [point_index(point) for point in clipped]
            if a != b:
                edges.append([a, b])
    coords = np.array(patch)
    representatives, classes, mismatches = [], [], []
    reduced = np.mod(coords + tolerance / 10, periods) - tolerance / 10
    for point in reduced:
        matches = [i for i, other in enumerate(representatives) if np.linalg.norm(point - other) < tolerance]
        if matches:
            classes.append(matches[0])
            mismatches.append(float(np.linalg.norm(point - representatives[matches[0]])))
        else:
            classes.append(len(representatives))
            representatives.append(point)
    classes = np.array(classes)
    for axis in (0, 1):
        low = np.flatnonzero(np.isclose(coords[:, axis], 0, atol=tolerance))
        high = np.flatnonzero(np.isclose(coords[:, axis], periods[axis], atol=tolerance))
        if not len(low) or len(low) != len(high) or set(classes[low]) != set(classes[high]):
            raise ValueError("Opposite RVE boundaries do not match source periodicity")
    if len(edges) != len(set(tuple(sorted(edge)) for edge in edges)):
        raise ValueError("Duplicate RVE segment")
    return coords, np.array(edges), classes, periods, origin, max(mismatches, default=0.)


def frame_matrix(first, second, E, nu, B, t, shear):
    vector = second - first
    L = float(np.linalg.norm(vector))
    c, s = vector / L
    area, inertia = B * t, B * t ** 3 / 12
    phi = 0. if not shear else 12 * E * inertia / (.85 * (E / (2 * (1 + nu))) * area * L ** 2)
    a, b = E * area / L, E * inertia / ((1 + phi) * L ** 3)
    local = np.array([[a, 0, 0, -a, 0, 0],
                      [0, 12*b, 6*L*b, 0, -12*b, 6*L*b],
                      [0, 6*L*b, (4+phi)*L*L*b, 0, -6*L*b, (2-phi)*L*L*b],
                      [-a, 0, 0, a, 0, 0],
                      [0, -12*b, -6*L*b, 0, 12*b, -6*L*b],
                      [0, 6*L*b, (2-phi)*L*L*b, 0, -6*L*b, (4+phi)*L*L*b]])
    rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    transform = np.zeros((6, 6)); transform[:3, :3] = rotation; transform[3:, 3:] = rotation
    return transform.T @ local @ transform


def condense(coords, edges, classes, periods, section, elastic, shear):
    size, nclasses = len(coords) * 3, int(classes.max()) + 1
    K = np.zeros((size, size))
    for a, b in edges:
        ids = np.r_[np.arange(3*a, 3*a+3), np.arange(3*b, 3*b+3)]
        K[np.ix_(ids, ids)] += frame_matrix(coords[a], coords[b], *elastic, *section, shear)
    # Unknown periodic fluctuations include all joint rotations. Remove only the
    # two rigid translations; affine macrostrain uses engineering gamma_xy.
    P = np.zeros((size, 3*nclasses))
    A = np.zeros((size, 3))
    for i, ((x, y), group) in enumerate(zip(coords, classes)):
        P[3*i:3*i+3, 3*group:3*group+3] = np.eye(3)
        A[3*i, [0, 2]] = [x, y/2]
        A[3*i+1, [1, 2]] = [y, x/2]
    P = P[:, 2:]
    H = P.T @ K @ P
    eigenvalues = np.linalg.eigvalsh(H)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("Periodic frame stiffness has an unresolved mechanism")
    U = A - P @ np.linalg.solve(H, P.T @ K @ A)
    C = U.T @ K @ U / (float(np.prod(periods)) * section[0])
    S = np.linalg.inv(C)
    anisotropy = float(abs(C[2, 2] - .5*(C[0, 0]-C[0, 1])) / C[2, 2])
    return {"C_MPa": C.tolist(), "E_x_MPa": float(1/S[0, 0]), "E_y_MPa": float(1/S[1, 1]),
            "nu_xy": float(-S[0, 1]/S[0, 0]), "shear_MPa": float(1/S[2, 2]),
            "isotropy_shear_relative_residual": anisotropy,
            "periodic_equilibrium_residual": float(np.max(np.abs(P.T @ K @ U)) / np.max(np.abs(K @ U)))}


def prepare_rve(out, case, source, coords, edges, classes, periods, section, elastic):
    name = "rve_{}_{}_rho{}_nx{}".format(case["topology"], case["l_mm"],
                                             case["nominal_relative_density"].replace(".", "p"), case["nnx"])
    directory = out / name; directory.mkdir(exist_ok=True)
    macro = [len(coords)+i+1 for i in range(3)]
    text = "*Heading\n** Exact archived B21 interior periodic elastic control\n*Node\n"
    text += "".join("{}, {:.12g}, {:.12g}\n".format(i+1, *point) for i, point in enumerate(coords))
    text += "".join("{}, 0., 0.\n".format(node) for node in macro)
    text += "*Element, type=B21, elset=BODY\n"
    text += "".join("{}, {}, {}\n".format(i+1, a+1, b+1) for i, (a, b) in enumerate(edges))
    text += "*Beam Section, elset=BODY, material=SOURCE, section=RECT\n{}, {}\n0.,0.,-1.\n".format(*section)
    text += "*Material, name=SOURCE\n*Elastic\n{}, {}\n".format(*elastic)
    masters = {}
    for i, group in enumerate(classes):
        if group not in masters:
            masters[group] = i
            continue
        master = masters[group]
        dx, dy = coords[i] - coords[master]
        for dof, terms in [(1, [(macro[0], dx), (macro[2], dy/2)]),
                           (2, [(macro[1], dy), (macro[2], dx/2)]), (6, [])]:
            terms = [(node, value) for node, value in terms if abs(value) > 1e-10]
            text += "*Equation\n{}\n{}, {}, 1., {}, {}, -1.".format(2+len(terms), i+1, dof, master+1, dof)
            text += "".join(", {}, 1, {:.12g}".format(node, -value) for node, value in terms) + "\n"
    text += "*Nset, nset=MACRO\n{}\n*Boundary\n{}, 1, 2\n".format(
        ", ".join(str(node) for node in macro), masters[0]+1)
    for column, label in enumerate(("Exx", "Eyy", "Gxy")):
        text += "*Step, name={}, perturbation\n*Static\n*Boundary\n".format(label)
        text += "".join("{}, 1, 1, {}\n".format(node, .001 if i == column else 0.) for i, node in enumerate(macro))
        text += "*Output, field\n*Node Output\nU, RF\n*Output, history\n*Energy Output\nALLSE\n*End Step\n"
    inp = directory / (name + ".inp")
    if inp.exists() and inp.read_text() != text:
        raise ValueError("Refuse to overwrite a different proposed RVE")
    inp.write_text(text, encoding="ascii")
    metadata = {"job": name, "analysis": "linear_static_periodic_rve", "case_id": case["case_id"],
                "source_inp": str(source), "source_sha256": sha(source), "input": str(inp),
                "input_sha256": sha(inp), "CPUs": 8, "memory": "25%", "state": "prepared_not_submitted",
                "macro_nodes": macro, "macrostrain": .001, "volume_mm3": float(np.prod(periods)*section[0]),
                "elastic_MPa_nu": elastic, "section_B_t_mm": section, "periods_mm": periods.tolist(),
                "nodes": len(coords), "elements": len(edges), "estimated_wall_seconds": 30,
                "estimate_basis": "Existing exact-mesh elastic controls took6-9 solver seconds; this RVE has fewer than100nodes;30s is a planning allowance, not measured runtime.",
                "reason": "Independent bending/shear-aware homogenized numerical 2D C and J-K conversion modulus; test axial-affine hex failure.",
                "changes": ["Extract an intact source interior periodic patch; no source model edited", "Retain original B21 connectivity/joints, subdivision, section and modified elastic material", "Replace nonlinear damage/loading by three independent periodic elastic macrostrain perturbations"]}
    (directory / "source.json").write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(directory / "patch.npz", coords=coords, edges=edges, classes=classes)
    return metadata


def prepare_geometric(root, out, cases):
    relative = "3/0.16/Fracture-hex-80-per-1.inp"
    case = next(row for row in cases if row["inp"] == relative)
    source = root / relative
    if sha(source) != case["inp_sha256"]:
        raise ValueError("Geometric control source changed")
    curve = table(root / "audit/history/3/0.16/Fracture-hex-80-per-1/native_curve.csv")
    first = next(i for i, row in enumerate(curve) if float(row["observed_STATUS_zero"]) > 0)
    target = float(curve[first-1]["U2_mm"])
    prefix = []
    for block in blocks(source.read_text()):
        if key(block) == "*step":
            break
        if key(block) not in {"*amplitude", "*plastic", "*damage initiation", "*damage evolution"}:
            prefix.append(block)
    name = "elasticNL_hex_3_rho0p16_nx80"
    directory = out / name; directory.mkdir(exist_ok=True)
    text = "".join(prefix).rstrip() + "\n"
    text += ("*Step, name=ElasticNL, nlgeom=YES, inc=1000\n"
             "*Static\n0.005, 1., 1.e-9, 0.02\n*Boundary\n"
             "fixity, 1, 2\nload, 1, 1\nload, 2, 2, {:.12g}\n"
             "*Output, field, frequency=10\n*Node Output, nset=load\nU, RF\n"
             "*Output, history, frequency=1\n*Node Output, nset=load\nU2, RF2\n"
             "*Energy Output\nALLSE, ALLWK\n*End Step\n").format(target)
    inp = directory / (name + ".inp")
    if inp.exists() and inp.read_text() != text:
        raise ValueError("Refuse to overwrite a different elastic geometric control")
    inp.write_text(text, encoding="ascii")
    sta = source.with_suffix(".sta").read_text()
    elapsed = re.findall(r"WALLCLOCK TIME \(SEC\)\s*=\s*([\d.]+)", sta)
    meta = {"job": name, "analysis": "elastic_geometric_nonlinear_static_CT", "case_id": case["case_id"],
            "source_inp": str(source), "source_sha256": sha(source), "input": str(inp), "input_sha256": sha(inp),
            "CPUs": 8, "memory": "25%", "state": "prepared_not_submitted", "target_U2_mm": target,
            "native_last_intact_index": first-1, "native_last_intact_time_s": float(curve[first-1]["time_s"]),
            "original_explicit_wall_seconds": float(elapsed[-1]) if elapsed else None,
            "original_explicit_CPUs": 12, "estimated_wall_seconds_range": [600, 3600],
            "estimate_basis": "Initial elastic full-mesh static controls took seconds;50+nonlinear increments can take10-60minutes. This is a planning range. Original Explicit50131s at12CPUs is not an extrapolator for Standard.",
            "reason": "Test whether reversible elastic geometry alone gives negative W-P^2/(2k0), before attributing all negative apparent plastic area to force noise.",
            "changes": ["Exact original mesh, pins, sections and E'/nu' retained", "Plasticity/damage removed: diagnostic elastic counterfactual, not fracture prediction", "Quasi-static Standard NLGEOM ramp to the existing last-intact displacement", "History every converged increment; low-volume reference-point field output"],
            "baseline": "Existing exact 1N linear static k0=3601.21825739327N/mm; small-displacement elastic W=P^2/(2k0) analytically."}
    (directory / "source.json").write_text(json.dumps(meta, indent=2))
    return meta


def prepare(args):
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = table(root / "case-index.csv")
    records, queue = [], []
    for relative in PRIORITY:
        matching = [case for case in cases if case["inp"] == relative]
        if len(matching) != 1:
            raise ValueError("Ambiguous indexed source")
        case = matching[0]; source = (root / relative).resolve(); source.relative_to(root)
        if sha(source) != case["inp_sha256"]:
            raise ValueError("Indexed source changed")
        _, nodes, elements, section, elastic = source_mesh(source)
        coords, edges, classes, periods, origin, mismatch = periodic_patch(
            nodes, elements, case["topology"], float(case["l_mm"]))
        result = {"case_id": case["case_id"], "source_inp": str(source), "source_sha256": sha(source),
                  "section_B_t_mm": section, "elastic_MPa_nu": elastic, "origin_mm": origin.tolist(),
                  "periods_mm": periods.tolist(), "nodes": len(coords), "elements": len(edges),
                  "maximum_periodic_pair_mismatch_mm": mismatch,
                  "Euler_Bernoulli": condense(coords, edges, classes, periods, section, elastic, False),
                  "Timoshenko_kappa0p85_no_SCF": condense(coords, edges, classes, periods, section, elastic, True),
                  "limit": "Independent frame theory; the proposed Abaqus B21 control tests default shear/slenderness effects. No extra lattice plane-strain correction applied."}
        records.append(result)
        queue.append(prepare_rve(out, case, source, coords, edges, classes, periods, section, elastic))
    queue.append(prepare_geometric(root, out, cases))
    (out / "beam-homogenization.json").write_text(json.dumps(records, indent=2))
    (out / "queue-manifest.json").write_text(json.dumps(queue, indent=2))
    serial = {"jobs": [{"job": row["job"], "input": row["input"], "input_sha256": row["input_sha256"],
                        "kind": row["analysis"], "purpose": row["reason"], "cpus": 8,
                        "allow_solver_failure": False, "required_outputs": [".odb", ".dat"]}
                       for row in queue], "queue_root": str(out / "queue-20261028"),
              "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    (out / "serial-proposal.json").write_text(json.dumps(serial, indent=2))
    print("Prepared", len(queue), "controls; no jobs submitted")


def extract(args):
    from odbAccess import openOdb
    out = Path(args.out)
    queue_root = Path(args.queue_root) if args.queue_root else out / "queue-20261028"
    state = json.loads((queue_root / "state.json").read_text())
    completed = {job["job"] for job in state["jobs"] if job["status"] == "completed"}
    results = []
    for source in out.glob("rve_*/source.json"):
        metadata = json.loads(source.read_text())
        if metadata["job"] not in completed:
            continue
        odb_path = queue_root / "runs" / metadata["job"] / (metadata["job"] + ".odb")
        before = sha(odb_path)
        if sha(metadata["source_inp"]) != metadata["source_sha256"]:
            raise ValueError("Original RVE source input changed")
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            C = np.zeros((3, 3))
            energies = []
            for column, label in enumerate(("Exx", "Eyy", "Gxy")):
                step = odb.steps[label]
                frame = step.frames[-1]
                rf = {value.nodeLabel: value.data[0] for value in frame.fieldOutputs["RF"].values
                      if value.nodeLabel in metadata["macro_nodes"]}
                if set(rf) != set(metadata["macro_nodes"]):
                    raise ValueError("Incomplete macro reactions")
                C[:, column] = np.array([rf[node] for node in metadata["macro_nodes"]]) / (
                    metadata["volume_mm3"] * metadata["macrostrain"])
                energy_regions = [region for region in step.historyRegions.values() if "ALLSE" in region.historyOutputs]
                if len(energy_regions) != 1:
                    raise ValueError("Expected one global elastic-energy history")
                actual = float(energy_regions[0].historyOutputs["ALLSE"].data[-1][1])
                expected = .5 * metadata["volume_mm3"] * metadata["macrostrain"]**2 * C[column, column]
                relative = abs(actual - expected)/expected
                if actual <= 0 or relative > 1e-4:
                    raise ValueError("Macro work and elastic energy disagree")
                energies.append({"load_case": label, "ALLSE_N_mm": actual,
                                 "macro_work_N_mm": expected, "relative_error": relative})
            symmetry = float(np.max(np.abs(C-C.T))/np.max(np.abs(C)))
            if symmetry > 1e-4 or np.linalg.eigvalsh(C).min() <= 0:
                raise ValueError("RVE constitutive matrix is asymmetric/nonpositive")
            S = np.linalg.inv(C)
            mu = .5*(C[0, 0]-C[0, 1])
            isotropic = np.array([[C[0, 0], C[0, 1], 0.], [C[0, 1], C[0, 0], 0.], [0., 0., mu]])
            result = dict(metadata, C_MPa=C.tolist(), E_x_MPa=float(1/S[0, 0]),
                          E_y_MPa=float(1/S[1, 1]), nu_xy=float(-S[0, 1]/S[0, 0]),
                          major_symmetry_relative_error=symmetry, energy_checks=energies,
                          eigenvalues_MPa=np.linalg.eigvalsh(C).tolist(), source_odb=str(odb_path),
                          source_odb_sha256=before, read_only=True, source_input_hash_verified=True,
                          preparation_state=metadata["state"], state="completed_and_extracted",
                          normalized_isotropic_tensor_error=float(np.max(abs(C-isotropic))/np.max(abs(C))))
            if sha(odb_path) != before:
                raise ValueError("RVE ODB changed during read-only extraction")
            (source.parent / "result.json").write_text(json.dumps(result, indent=2))
            results.append(result)
        finally:
            odb.close()
    (out / "abaqus-rve-results.json").write_text(json.dumps(results, indent=2))
    for source in out.glob("elasticNL_*/source.json"):
        metadata = json.loads(source.read_text())
        if metadata["job"] not in completed:
            continue
        odb_path = queue_root / "runs" / metadata["job"] / (metadata["job"] + ".odb")
        before = sha(odb_path)
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            step = odb.steps["ElasticNL"]
            regions = [region for region in step.historyRegions.values()
                       if "RF2" in region.historyOutputs and "U2" in region.historyOutputs]
            if len(regions) != 1:
                raise ValueError("Expected one loading reference-point history")
            uu = np.array(regions[0].historyOutputs["U2"].data)
            ff = np.array(regions[0].historyOutputs["RF2"].data)
            if not np.array_equal(uu[:, 0], ff[:, 0]) or not np.isfinite(uu).all() or not np.isfinite(ff).all():
                raise ValueError("Invalid static history alignment")
            baseline_path = Path(args.root) / "audit/modal/static-summary.json"
            baseline = next(row for row in json.loads(baseline_path.read_text())
                            if row["case_id"] == metadata["case_id"])
            if baseline["source_inp_sha256"] != metadata["source_sha256"]:
                raise ValueError("Elastic initial stiffness comes from a different source input")
            k0 = baseline["static_stiffness_N_mm"]
            work = np.r_[0., np.cumsum(.5*(ff[1:, 1]+ff[:-1, 1])*np.diff(uu[:, 1]))]
            elastic_subtraction = ff[:, 1]**2/(2*k0)
            output = [{"step_time": float(t), "U2_mm": float(u), "RF2_N": float(f),
                       "work_trapezoid_N_mm": float(w), "linear_k0_energy_N_mm": float(e),
                       "apparent_Apl_N_mm": float(w-e), "secant_stiffness_N_mm": float(f/u) if u else None}
                      for (t, u), (_, f), w, e in zip(uu, ff, work, elastic_subtraction)]
            write_table(source.parent / "elastic-work.csv", output)
            result = dict(metadata, source_odb=str(odb_path), samples=len(output), final=output[-1],
                          negative_apparent_Apl_count=sum(row["apparent_Apl_N_mm"] < -1e-6 for row in output),
                          initial_static_stiffness_N_mm=k0, source_odb_sha256=before, read_only=True,
                          baseline_static_summary=str(baseline_path), baseline_static_summary_sha256=sha(baseline_path),
                          preparation_state=metadata["state"], state="completed_and_extracted",
                          source_input_hash_verified=sha(metadata["source_inp"]) == metadata["source_sha256"])
            energies = [region for region in step.historyRegions.values() if "ALLSE" in region.historyOutputs]
            if len(energies) != 1:
                raise ValueError("Expected one elastic static-energy region")
            energy = float(energies[0].historyOutputs["ALLSE"].data[-1][1])
            result["final_ALLSE_N_mm"] = energy
            result["trapezoid_work_vs_ALLSE_relative_error"] = abs(output[-1]["work_trapezoid_N_mm"]-energy)/energy
            if sha(odb_path) != before or not result["source_input_hash_verified"]:
                raise ValueError("Elastic control source changed during extraction")
            (source.parent / "result.json").write_text(json.dumps(result, indent=2))
        finally:
            odb.close()
    print("Extracted", len(results), "existing RVE results")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from resources.lattices import Geometry, effProperties

    out = Path(args.out)
    directory = out / "elasticNL_hex_3_rho0p16_nx80"
    data = np.genfromtxt(directory / "elastic-work.csv", delimiter=",", names=True)
    result = json.loads((directory / "result.json").read_text())
    u, f = data["U2_mm"], data["RF2_N"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(u, f/1000, label="Elastic NLGEOM static")
    axes[0].plot(u, result["initial_static_stiffness_N_mm"]*u/1000, "--", label="Initial k0 × displacement")
    axes[0].set(xlabel="Loading-pin displacement (mm)", ylabel="Force (kN)", title="Reversible geometric stiffening")
    axes[0].legend(fontsize=8)
    axes[1].plot(u, data["work_trapezoid_N_mm"]/1000, label="Integrated elastic work")
    axes[1].plot(u, data["linear_k0_energy_N_mm"]/1000, "--", label="P² / (2 k0)")
    axes[1].set(xlabel="Displacement (mm)", ylabel="Work (kN mm)", title="Linear compliance over-subtracts energy")
    axes[1].legend(fontsize=8)
    axes[2].plot(u, data["apparent_Apl_N_mm"]/1000, color="tab:red")
    axes[2].axhline(0, color="0.5", lw=.7)
    axes[2].set(xlabel="Displacement (mm)", ylabel="Apparent Apl (kN mm)", title="Negative area without plasticity or dynamics")
    fig.suptitle("Exact hex80, l=3 mm, density .16: elastic counterfactual; no fracture prediction")
    fig.savefig(out / "elastic-geometric-work.png", dpi=180); plt.close(fig)
    records = json.loads((out / "abaqus-rve-results.json").read_text())
    matrices = json.loads((out / "beam-homogenization.json").read_text())
    labels, affine, scaling, rve, eb, timo = [], [], [], [], [], []
    for record in records:
        case = next(row for row in table(Path(args.root)/"case-index.csv") if row["case_id"] == record["case_id"])
        geom = Geometry(case["topology"], float(case["l_mm"]), int(case["nnx"]), float(case["nominal_relative_density"]))
        affine.append(effProperties(case["topology"], geom, E_s=123000., mode="stiff")[2])
        scaling.append(effProperties(case["topology"], geom, E_s=123000., mode="scaling")[2])
        rve.append(record["E_x_MPa"])
        independent = next(row for row in matrices if row["case_id"] == record["case_id"])
        eb.append(independent["Euler_Bernoulli"]["E_x_MPa"])
        timo.append(independent["Timoshenko_kappa0p85_no_SCF"]["E_x_MPa"])
        labels.append("{} {}mm ρ{}".format(case["topology"], case["l_mm"], case["nominal_relative_density"]))
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    y = np.arange(len(records))
    for values, marker, label in [(affine, "x", "Current axial-affine + lattice correction"),
                                  (scaling, "s", "Current literature-scaling + lattice correction"),
                                  (eb, "+", "Independent periodic Euler–Bernoulli"),
                                  (timo, "^", "Independent periodic Timoshenko (no Abaqus SCF)"),
                                  (rve, "o", "Exact-material B21 periodic RVE, 1/S11")]:
        axis.plot(values, y, linestyle="none", marker=marker, label=label, markersize=7)
    axis.set(yticks=y, yticklabels=labels, xscale="log", xlabel="In-plane conversion modulus (MPa)",
             title="Bending-aware homogenization versus previous modulus choices")
    axis.invert_yaxis(); axis.grid(axis="x", alpha=.2); axis.legend(fontsize=8, loc="best")
    fig.savefig(out / "conversion-modulus-comparison.png", dpi=180); plt.close(fig)
    for result_path in out.glob("epUnload*/result.json"):
        result = json.loads(result_path.read_text())
        loaded = np.genfromtxt(result_path.parent / "load-curve.csv", delimiter=",", names=True)
        unloaded = np.genfromtxt(result_path.parent / "unload-curve.csv", delimiter=",", names=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(loaded["U2_mm"], loaded["force_from_fixed_pin_N"]/1000, label="Static loading")
        axes[0].plot(unloaded["U2_mm"], unloaded["force_from_fixed_pin_N"]/1000, label="Release-force unloading")
        axes[0].set(xlabel="Loading-pin displacement (mm)", ylabel="Force (kN)", title="Same source plastic law; damage disabled")
        axes[0].legend(fontsize=8)
        axes[1].bar([0, 1], [result["initial_triangle_apparent_Apl_Nmm"]/1000,
                           result["work_minus_recovered_Nmm"]/1000], color=["tab:red", "tab:green"])
        axes[1].axhline(0, color="0.4", lw=.8)
        axes[1].set(xticks=[0, 1], xticklabels=["Work − initial triangle", "Work − measured recovery"],
                    ylabel="Apparent non-recovered work (kN mm)", title="Unloading changes the work partition")
        check = result["loading_section_check"]
        fig.suptitle("U={:.4f} mm; max PEEQ={:.6f} {} original criterion {:.6f}".format(
            result["target_U2_mm"], check["PEEQ_max"], ">" if check["max_PEEQ_exceeds_original_threshold"] else "≤",
            check["source_PEEQ_initiation_threshold"]))
        fig.savefig(result_path.parent / "unloading-work.png", dpi=180); plt.close(fig)
    endpoint_rows = table(out / "criterion-endpoint-metrics.csv")
    source_rows = table(Path(args.root) / "audit/benchmark/fig10-digitised.csv")
    endpoint_names = ("last_before_section_damage_bracket", "first_after_section_damage_bracket", "last_before_monitored_deletion")
    methods = (("raw", "KI_MPa_sqrt_m", -.12, "x", "tab:blue", "Raw KI"),
               ("raw", "KJ_MPa_sqrt_m", -.04, "o", "tab:blue", "Raw KJ"),
               ("causal_BW2_force_only", "KI_MPa_sqrt_m", .04, "x", "tab:orange", "Causal BW2 KI"),
               ("causal_BW2_force_only", "KJ_MPa_sqrt_m", .12, "o", "tab:orange", "Causal BW2 KJ"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, case_id, density, length in (
            (axes[0], "local:old/10/0.2/Fracture-hex-60-per-1.odb", .2, 10),
            (axes[1], "local:old/3/0.16/Fracture-hex-80-per-1.odb", .16, 3)):
        ref = next(row for row in source_rows if row["topology"] == "hex" and row["stage"] == "initial"
                   and float(row["nominal_relative_density"]) == density)
        scale = np.sqrt(length/float(ref["bar_length_mm"]))
        lower, upper, center = [float(ref[name])*scale for name in ("K_graphical_lower", "K_graphical_upper", "K_MPa_sqrt_m")]
        axis.axhspan(lower, upper, color="0.7", alpha=.35, label="Fig.10 initial KJ band")
        axis.axhline(center, color="0.4", lw=.8)
        values, missing = [lower, upper], []
        for method, column, offset, marker, color, label in methods:
            label_pending = True
            for x, endpoint in enumerate(endpoint_names):
                row = next(row for row in endpoint_rows if row["case_id"] == case_id
                           and row["endpoint"] == endpoint and row["method"] == method)
                valid = row[column] not in ("", None) and (column.startswith("KI_") or row["valid_plastic_area"] == "True")
                if valid:
                    value = float(row[column]); values.append(value)
                    axis.plot(x+offset, value, marker=marker, linestyle="none", color=color,
                              markerfacecolor="none" if marker == "o" else color,
                              label=label if label_pending else None)
                    label_pending = False
                elif column.startswith("KJ_"):
                    missing.append((x, "Raw KJ" if method == "raw" else "Causal KJ", color))
        bottom, top = min(values)*.68, max(values)*1.12
        for x, label, color in missing:
            axis.text(x, bottom+.03*(top-bottom), label+" unavailable\nApl<0", ha="center", color=color, fontsize=7)
        axis.set(xticks=range(3), xticklabels=["Before section\nthreshold bracket", "After section\nthreshold bracket", "Last before\nwhole-element\ndeletion"],
                 ylabel="MPa sqrt(m)", xlim=(-.5, 2.5), ylim=(bottom, top),
                 title="Hex ρ{}, l={} mm\n{}".format(density, length,
                    "Fig.10 3 mm × sqrt(10/3): inferred 10 mm" if length == 10 else "Fig.10: direct 3 mm source"))
        axis.grid(axis="y", alpha=.2)
        axis.tick_params(axis="x", labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8)
    fig.text(.5, .025, "KI and KJ are different quantities. Discrete endpoints are not continuous bounds.\n"
             "Negative-area KJ is unavailable; source bands show graphical sensitivity, not statistical confidence.",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.78, bottom=.23, wspace=.25)
    fig.savefig(out / "hex-criterion-endpoints.png", dpi=180); plt.close(fig)


def analyze(args):
    from resources.lattices import Geometry, effProperties
    from validation_calculations import filter_force, fracture_metrics

    root, out = Path(args.root), Path(args.out)
    cases = table(root / "case-index.csv")
    statics = json.loads((root / "audit/modal/static-summary.json").read_text())
    static = {row["case_id"]: row["static_stiffness_N_mm"] for row in statics}
    modal = {row["case_id"]: row["cutoff_2fn_Hz"] for row in json.loads(
        (root / "audit/modal/summary.json").read_text()) if row["control"] == "released"}
    verified = {row["case_id"]: row["E_x_MPa"] for row in json.loads((out / "abaqus-rve-results.json").read_text())}
    records, endpoints, criterion_metrics = [], [], []
    for case in cases:
        if case["mode"] != "Fracture" or case["topology"] not in ("tri", "kagome", "hex") or case["campaign"] != "Density campaign":
            continue
        directory = root / "audit/history" / Path(case["odb"]).parent / Path(case["odb"]).stem
        if not (directory / "native_curve.csv").exists():
            continue
        curve = np.genfromtxt(directory / "native_curve.csv", delimiter=",", names=True)
        summary = json.loads((directory / "summary.json").read_text())
        first = int(summary["first_monitored_deletion_index"])
        t, u, f = curve["time_s"][:first+1], curve["U2_mm"][:first+1], curve["RF2_N"][:first+1]
        if not np.isfinite(u).all() or not np.isfinite(f).all():
            raise ValueError("Nonfinite prefix before the observed deletion")
        source = root / case["inp"]
        _, _, _, section, elastic = source_mesh(source)
        if sha(source) != case["inp_sha256"]:
            raise ValueError("Indexed diagnostic source changed")
        geom = Geometry(case["topology"], float(case["l_mm"]), int(case["nnx"]), float(case["nominal_relative_density"]))
        middle = f[int(.2*first):max(int(.8*first), int(.2*first)+3)]
        scale = np.max(np.abs(f[:-1]))
        record = {"case_id": case["case_id"], "topology": case["topology"], "l_mm": case["l_mm"],
                  "rho": case["nominal_relative_density"], "nnx": case["nnx"],
                  "source_inp": str(source), "source_input_hash_verified": True,
                  "E_beam_MPa": elastic[0], "nu_beam": elastic[1], "section_B_mm": section[0], "section_t_mm": section[1],
                  "samples": len(curve), "native_fs_Hz": float(1/np.median(np.diff(t))),
                  "deletion_time_s": float(t[first]), "last_intact_U2_mm": float(u[first-1]),
                  "dimensionless_last_intact_U_over_W": float(u[first-1]/geom.W),
                  "median_velocity_mm_s": float(np.median(np.diff(u)/np.diff(t))),
                  "rms_second_force_difference_over_peak": float(np.sqrt(np.mean(np.diff(middle, n=2)**2))/scale),
                  "noise_indicator_limit": "Discrete curvature indicator over20-80%of intact prefix; not a calibrated noise/inertia measure.",
                  "current_affine_lattice_correction_MPa": float(effProperties(case["topology"], geom, E_s=123000., mode="stiff")[2]),
                  "current_scaling_lattice_correction_MPa": float(effProperties(case["topology"], geom, E_s=123000., mode="scaling")[2]),
                  "verified_B21_RVE_conversion_MPa": verified.get(case["case_id"])}
        energy = np.genfromtxt(directory / "native_stored_energy.csv", delimiter=",", names=True)
        ei = int(np.argmin(abs(energy["time_s"]-t[first-1])))
        if abs(energy["time_s"][ei]-t[first-1]) > 1e-7:
            raise ValueError("Last-intact stored energy sample is not aligned")
        work_intact = float(np.sum(.5*(f[1:first]+f[:first-1])*np.diff(u[:first])))
        record.update(last_intact_raw_work_Nmm=work_intact,
                      last_intact_ALLSE_Nmm=float(energy["ALLSE_Nmm"][ei]),
                      last_intact_W_minus_ALLSE_diagnostic_Nmm=work_intact-float(energy["ALLSE_Nmm"][ei]),
                      last_intact_ALLKE_Nmm=float(energy["ALLKE_Nmm"][ei]))
        records.append(record)
        if case["case_id"] not in static:
            continue
        event = summary["field_event_brackets"]["first_section_SDEG_positive"]
        # A coarse first-positive field frame may occur after the finer native
        # deletion sample (tri/kagome). Retain that fact explicitly; it is not an
        # extra intact endpoint and no future samples feed the causal filter.
        t, u, f = curve["time_s"], curve["U2_mm"], curve["RF2_N"]
        selected = [("last_before_section_damage_bracket", int(np.flatnonzero(t<=event["previous_frame_time"]+1e-7)[-1])),
                    ("first_after_section_damage_bracket", int(np.flatnonzero(t>=event["first_frame_time"]-1e-7)[0])),
                    ("last_before_monitored_deletion", first-1)]
        last = max(index for _, index in selected)
        t, u, f = t[:last+1], u[:last+1], f[:last+1]
        if not np.isfinite(u).all() or not np.isfinite(f).all():
            raise ValueError("Nonfinite explicitly selected endpoint prefix")
        variants = {"raw": f, "causal_BW2_force_only": filter_force(t, f, modal[case["case_id"]], 2)[0]}
        for method, force in variants.items():
            work = np.r_[0., np.cumsum(.5*(force[1:]+force[:-1])*np.diff(u))]
            for label, index in selected:
                ei = int(np.argmin(abs(energy["time_s"]-t[index])))
                if abs(energy["time_s"][ei]-t[index]) > 1e-7:
                    raise ValueError("Native stored energy sample is not aligned")
                se = float(energy["ALLSE_Nmm"][ei])
                endpoint = {"case_id": case["case_id"], "endpoint": label, "method": method,
                                  "index": index, "time_s": float(t[index]), "U2_mm": float(u[index]),
                                  "observed_STATUS_zero_at_endpoint": float(curve["observed_STATUS_zero"][index]),
                                  "RF2_N": float(force[index]), "W_Nmm": float(work[index]),
                                  "initial_triangle_Nmm": float(force[index]**2/(2*static[case["case_id"]])),
                                  "apparent_Apl_initial_k0_Nmm": float(work[index]-force[index]**2/(2*static[case["case_id"]])),
                                  "native_ALLSE_Nmm": se, "native_ALLIE_Nmm": float(energy["ALLIE_Nmm"][ei]),
                                  "native_ALLKE_Nmm": float(energy["ALLKE_Nmm"][ei]),
                                  "W_minus_native_ALLSE_Nmm_diagnostic": float(work[index]-se),
                                  "limit": "W-ALLSE is an energy diagnostic including dynamic/dissipative contributions, not a calibrated ASTM Apl or Jpl."}
                endpoints.append(endpoint)
                geom.FTcalc()
                metric = fracture_metrics(u, force, index, geom, verified[case["case_id"]], static[case["case_id"]])
                criterion_metrics.append(dict(endpoint, **{name: value for name, value in metric.items()
                                                          if name not in endpoint},
                                              modulus_source="source-material B21 periodic control; approximate isotropy/mesh limits retained",
                                              compliance_source="existing exact-source initial1N elastic static control"))
    write_table(out / "native-48-diagnostics.csv", records)
    write_table(out / "section-onset-vs-deletion.csv", endpoints)
    write_table(out / "criterion-endpoint-metrics.csv", criterion_metrics)
    existing = []
    for lat, nx in (("tri", 30), ("kagome", 20), ("hex", 20)):
        directory = root.parent / "stiffMatrix/Cmatrix-periodic"
        observations = {}
        for load_case in ("A", "B", "C"):
            for channel in (0, 1):
                path = directory / "transfer" / "OUT-Ductile-{}-{}-Cmatrix{}-per-1-{}.csv".format(lat, nx, load_case, channel)
                if not path.exists():
                    continue
                series = np.loadtxt(path, delimiter=",")
                index = np.arange(len(series))
                use = (index>=max(3, int(.05*len(series)))) & (index<=int(.25*len(series)))
                slope = float(np.polyfit(series[use, 0], series[use, 1], 1)[0])
                observations[load_case+str(channel)] = {"csv": str(path), "sha256": sha(path), "slope": slope}
        inp = directory / "Ductile-{}-{}-CmatrixA-per-1.inp".format(lat, nx)
        elastic = next(rows(block)[0] for block in blocks(inp.read_text()) if key(block) == "*elastic")
        existing.append({"topology": lat, "source_inp": str(inp), "source_sha256": sha(inp),
                         "source_elastic_MPa_nu": elastic, "observations": observations,
                         "limit": "Existing periodic B21 evidence; CSV normalization and directional inconsistencies not certified. Different source material from validation. Do not promote these slopes to conversion moduli."})
    (out / "existing-elastic-evidence.json").write_text(json.dumps(existing, indent=2))
    print("Analysed", len(records), "native cases and", len(endpoints), "explicit endpoint controls")


def section_fields(args):
    from odbAccess import openOdb
    root, out = Path(args.root), Path(args.out)
    cases = table(root / "case-index.csv")
    output = []
    for relative in ("10/0.2/Fracture-hex-60-per-1.inp", "3/0.16/Fracture-hex-80-per-1.inp"):
        case = next(row for row in cases if row["inp"] == relative)
        summary_path = root / "audit/history" / Path(relative).parent / Path(relative).stem / "summary.json"
        event = json.loads(summary_path.read_text())["field_event_brackets"]["first_section_SDEG_positive"]
        odb_path = root / case["odb"]
        before = sha(odb_path)
        if before != case["odb_sha256"]:
            raise ValueError("Original native ODB hash changed")
        odb = openOdb(str(odb_path), readOnly=True)
        try:
            step = odb.steps["Step-1"]
            for endpoint, time in (("last_field_before_section_onset", event["previous_frame_time"]),
                                   ("first_field_with_section_onset", event["first_frame_time"])):
                index = min(range(len(step.frames)), key=lambda i: abs(step.frames[i].frameValue-time))
                frame = step.frames[index]
                if abs(frame.frameValue-time) > 1e-7:
                    raise ValueError("Missing specified native field frame")
                plastic = frame.fieldOutputs["PEEQ"]
                strain = frame.fieldOutputs["LE"]
                component = list(strain.componentLabels).index("LE11")
                def point(value):
                    return (value.instance.name, value.elementLabel, value.sectionPoint.number, value.integrationPoint)
                pp = {point(value): float(value.data) for value in plastic.values}
                ee = {point(value): float(value.data[component]) for value in strain.values}
                cc = {point(value): float(value.data) for value in frame.fieldOutputs["DUCTCRT"].values}
                dd = {point(value): float(value.data) for value in frame.fieldOutputs["SDEG"].values}
                ss = {}
                if "S" in frame.fieldOutputs and "S11" in frame.fieldOutputs["S"].componentLabels:
                    stress = frame.fieldOutputs["S"]
                    si = list(stress.componentLabels).index("S11")
                    ss = {point(value): float(value.data[si]) for value in stress.values}
                if set(pp) != set(ee) or not np.isfinite(list(pp.values())).all() or not np.isfinite(list(ee.values())).all():
                    raise ValueError("Unaligned/nonfinite native section fields")
                maxp, maxe = max(pp, key=pp.get), max(ee, key=ee.get)
                maxc = max(cc, key=cc.get)
                damaged = [{"instance_element_section_IP": str(location), "SDEG": severity,
                            "PEEQ": pp.get(location), "LE11": ee.get(location), "S11_MPa": ss.get(location),
                            "DUCTCRT": cc.get(location)} for location, severity in dd.items() if severity > 0]
                output.append({"case_id": case["case_id"], "endpoint": endpoint,
                               "field_index": index, "time_s": frame.frameValue,
                               "max_PEEQ": pp[maxp], "max_PE_instance_element_section_IP": str(maxp),
                               "LE11_at_max_PEEQ": ee[maxp], "max_LE11": ee[maxe],
                               "S11_MPa_at_max_PEEQ": ss.get(maxp),
                               "max_LE_instance_element_section_IP": str(maxe), "PEEQ_at_max_LE11": pp[maxe],
                               "S11_MPa_at_max_LE11": ss.get(maxe),
                               "min_LE11": min(ee.values()),
                               "max_DUCTCRT": cc[maxc], "max_DUCTCRT_point": str(maxc),
                               "LE11_at_max_DUCTCRT": ee.get(maxc), "S11_MPa_at_max_DUCTCRT": ss.get(maxc),
                               "PEEQ_at_max_DUCTCRT": pp.get(maxc),
                               "max_SDEG": max(dd.values()), "damaged_section_points": json.dumps(damaged),
                               "S11_point_keys_match_PEEQ": set(ss)==set(pp),
                               "source_odb": str(odb_path), "source_odb_sha256": before, "read_only": True,
                               "section_observations": len(pp),
                               "Hsieh_axial_strain_reference": .1105,
                               "limit": "LE11 is the saved logarithmic axial strain; do not assume Hsieh's unspecified strain measure/subroutine is identical."})
        finally:
            odb.close()
        if sha(odb_path) != before:
            raise ValueError("Original native ODB changed during read-only section scan")
    write_table(out / "native-hex-section-fields.csv", output)
    print("Read4native hex section-field frames; original hashes unchanged")


def locate_events(args):
    from resources.lattices import Geometry
    root, out = Path(args.root), Path(args.out)
    cases = table(root / "case-index.csv")
    output = []
    for row in table(out / "native-hex-section-fields.csv"):
        case = next(case for case in cases if case["case_id"] == row["case_id"])
        source = root / case["inp"]
        text, nodes, elements, _, _ = source_mesh(source)
        connectivity = {label: (a, b) for label, a, b in elements}
        assembly_nodes = {}; in_instance = False; reference = {}
        for block in blocks(text):
            kind = key(block)
            if kind == "*instance": in_instance = True
            elif kind == "*end instance": in_instance = False
            elif kind == "*node" and not in_instance:
                for line in rows(block):
                    value = line.split(","); assembly_nodes[int(value[0])] = np.array([float(value[1]), float(value[2])])
            elif kind == "*nset" and not in_instance:
                match = re.search(r"nset=([^,\r\n]+)", block.splitlines()[0], re.IGNORECASE)
                if match and match[1].strip().lower() in ("load", "fixity"):
                    reference[match[1].strip().lower()] = int(rows(block)[0].split(",")[0])
        geom = Geometry("hex", float(case["l_mm"]), int(case["nnx"]), float(case["nominal_relative_density"]))
        tip = np.array([geom.a0, geom.H/2])
        locations = [("max_PEEQ", ast.literal_eval(row["max_PE_instance_element_section_IP"])),
                     ("max_tensile_LE11", ast.literal_eval(row["max_LE_instance_element_section_IP"])),
                     ("max_DUCTCRT", ast.literal_eval(row["max_DUCTCRT_point"]))]
        locations += [("first_observed_damaged_section", ast.literal_eval(point["instance_element_section_IP"]))
                      for point in json.loads(row["damaged_section_points"])]
        for role, (_, element, section, ip) in locations:
            a, b = connectivity[element]
            midpoint = .5*(np.array(nodes[a])+nodes[b])
            load = assembly_nodes[reference["load"]]
            fixed = assembly_nodes[reference["fixity"]]
            output.append({"case_id": row["case_id"], "time_s": row["time_s"], "role": role,
                           "element": element, "section_point": section, "integration_point": ip,
                           "midpoint_x_mm": midpoint[0], "midpoint_y_mm": midpoint[1],
                           "nominal_cracktip_x_mm": tip[0], "nominal_cracktip_y_mm": tip[1],
                           "distance_to_nominal_tip_mm": float(np.linalg.norm(midpoint-tip)),
                           "distance_to_loading_pin_mm": float(np.linalg.norm(midpoint-load)),
                           "distance_to_fixed_pin_mm": float(np.linalg.norm(midpoint-fixed)),
                           "source_inp": str(source), "source_sha256": sha(source),
                           "limit": "Original FE-segment centerline midpoint; nominal crack tip uses pre-FTcalc absolute coordinate, not load-line crack length."})
    write_table(out / "native-hex-event-locations.csv", output)
    print("Located native event elements from original INPs without field rereads")


def prepare_unload(args):
    root, out = Path(args.root), Path(args.out)
    relative = "3/0.16/Fracture-hex-80-per-1.inp"
    case = next(row for row in table(root / "case-index.csv") if row["inp"] == relative)
    source = root / relative
    if sha(source) != case["inp_sha256"]:
        raise ValueError("Unloading source hash changed")
    native = root / "audit/history/3/0.16/Fracture-hex-80-per-1"
    summary = json.loads((native / "summary.json").read_text())
    bracket = summary["field_event_brackets"]["first_section_SDEG_positive"]
    curve = table(native / "native_curve.csv")
    index = max(i for i, row in enumerate(curve) if float(row["time_s"]) <= bracket["previous_frame_time"]+1e-7)
    target = float(curve[index]["U2_mm"])
    if args.unload_target is not None:
        if not np.isfinite(args.unload_target) or not 0 < args.unload_target <= target:
            raise ValueError("Choose a positive target no later than the native pre-section-damage reference")
        target = args.unload_target
        index = min(range(len(curve)), key=lambda i: abs(float(curve[i]["U2_mm"])-target))
    model = []
    for block in blocks(source.read_text()):
        if key(block) == "*step":
            break
        if key(block) not in {"*amplitude", "*damage initiation", "*damage evolution"}:
            model.append(block)
    name = args.unload_job
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("Unsafe unloading job name")
    directory = out / name; directory.mkdir(exist_ok=True)
    text = "".join(model).rstrip() + "\n"
    outputs = ("*Output, field, frequency=10\n*Node Output, nset=load\nU, RF\n"
               "*Node Output, nset=fixity\nU, RF\n*Output, history, frequency=1\n"
               "*Node Output, nset=load\nU1, U2, RF1, RF2\n"
               "*Node Output, nset=fixity\nU1, U2, RF1, RF2\n"
               "*Energy Output\nALLSE, ALLPD, ALLIE, ALLWK, ALLVD, ALLAE, ALLKE, ALLDMD\n*End Step\n")
    loading_outputs = outputs.replace("*Output, history, frequency=1\n",
        "*Output, field, frequency=1000000, op=ADD\n"
        "*Element Output, elset=Part-1-1.AllPartSet\nPEEQ, LE\n*Output, history, frequency=1\n", 1)
    text += ("*Step, name=Load, nlgeom=YES, inc=1000, amplitude=RAMP\n*Static\n"
             "0.005, 1., 1.e-9, 0.02\n*Boundary\nfixity, 1, 2\nload, 1, 1\n"
             "load, 2, 2, {:.12g}\n").format(target) + loading_outputs
    text += ("*Step, name=Unload, nlgeom=YES, inc=1000, amplitude=RAMP\n*Static\n"
             "0.005, 1., 1.e-9, 0.02\n*Boundary, op=NEW\nfixity, 1, 2\nload, 1, 1\n") + outputs
    inp = directory / (name + ".inp")
    input_exists = inp.exists()
    if input_exists and inp.read_text() != text:
        raise ValueError("Refuse to overwrite a different unloading proposal; use a unique job")
    source_record = directory / "source.json"
    if (input_exists and not source_record.exists()) or (not input_exists and any(directory.iterdir())):
        raise ValueError("Unloading job directory is occupied without matching provenance")
    proposed_hash = sha(inp) if input_exists else hashlib.sha256(text.encode("ascii")).hexdigest()
    metadata = {"job": name, "input": str(inp), "input_sha256": proposed_hash, "source_inp": str(source),
                "source_sha256": sha(source), "case_id": case["case_id"], "state": "prepared_not_submitted",
                "target_U2_mm": target, "native_sample_index": index, "native_time_s": float(curve[index]["time_s"]),
                "native_reference_U2_mm": float(curve[index]["U2_mm"]),
                "native_reference_displacement_difference_mm": float(curve[index]["U2_mm"])-target,
                "SDEG_field_bracket_s": bracket, "kind": "elastoplastic_intact_load_unload_static",
                "changes": ["Retain exact original mesh, pin constraints, section, modified elastic material and complete plastic table",
                            "Remove damage initiation/evolution only, plus unused Explicit amplitude/step/output",
                            "Standard NLGEOM static load to the last saved frame before first section damage",
                            "Release loading U2 with OP=NEW; restate fixed pin U1/U2 and loading U1; default replaced reaction ramps to0over unload step"],
                "purpose": "Measure recoverable unloading work with plasticity and geometric nonlinearity before the original section-damage bracket, instead of assuming initial compliance triangle.",
                "release_source": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-boundary.htm",
                "loading_fields": "Final Loading PEEQ/LE at original beam section points; frequency1000000 exceeds max1000increments, and Standard always writes final increment.",
                "force_extraction": "During load use loading RF2 and check against -fixed-pin RF2. During released unload loading RF2 is0bydefinition: use -fixed-pin RF2 and independently verify P_start*(1-step_time).",
                "limit": "Undamaged static counterfactual; verify reaction equilibrium, energy balance and residual plastic change during unloading before treating recovered work as elastic energy.",
                "estimated_wall_seconds_range": [120, 900], "estimate_basis": "Prior same-mesh elasticNLload finished in58seconds; two elastoplastic steps expected2-15minutes, uncertain."}
    if input_exists:
        existing = json.loads(source_record.read_text())
        if any(existing.get(field) != metadata[field] for field in ("job", "source_sha256", "input_sha256", "target_U2_mm")):
            raise ValueError("Existing unloading provenance differs; retain it and choose a unique job")
        metadata = existing
    proposal = {"jobs": [{"job": name, "input": str(inp), "input_sha256": metadata["input_sha256"], "kind": metadata["kind"],
                          "purpose": metadata["purpose"], "cpus": 8, "allow_solver_failure": False,
                          "required_outputs": [".odb", ".dat"]}], "queue_root": str(
                              Path(args.queue_root) if args.queue_root else out / "queue-unload-20261028"),
                "abaqus": "C:/SIMULIA/Commands/abaqus.bat"}
    proposal_path = Path(args.proposal) if args.proposal else out / "serial-unload-proposal.json"
    proposal_text = json.dumps(proposal, indent=2)
    if proposal_path.exists() and proposal_path.read_text() != proposal_text:
        raise ValueError("Refuse to overwrite a different proposal manifest")
    state_path = Path(proposal["queue_root"]) / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not input_exists or not proposal_path.exists() or Path(state["manifest"]).resolve() != proposal_path.resolve() or sha(proposal_path) != state["manifest_sha256"]:
            raise ValueError("Queue root is already pinned to a different proposal")
    if not input_exists:
        inp.write_bytes(text.encode("ascii"))
        source_record.write_text(json.dumps(metadata, indent=2))
    if not proposal_path.exists():
        proposal_path.write_text(proposal_text)
    print("Verified existing" if input_exists else "Prepared new", "unloading control; no submission")


def extract_unload(args):
    from odbAccess import openOdb
    out = Path(args.out)
    directory = out / args.unload_job
    metadata = json.loads((directory / "source.json").read_text())
    queue_root = Path(args.queue_root) if args.queue_root else out / "queue-unload-20261028"
    state = json.loads((queue_root / "state.json").read_text())
    job = next(row for row in state["jobs"] if row["job"] == metadata["job"])
    if job["status"] != "completed":
        print("Unloading control is", job["status"], "; active ODB left unopened")
        return
    odb_path = queue_root / "runs" / metadata["job"] / (metadata["job"] + ".odb")
    before = sha(odb_path)
    odb = openOdb(str(odb_path), readOnly=True)
    try:
        load_step = odb.steps["Load"]
        regions = {name: region for name, region in load_step.historyRegions.items()
                   if "RF2" in region.historyOutputs and "U2" in region.historyOutputs}
        if len(regions) != 2:
            raise ValueError("Expected exactly two reference-point histories")
        load_name = max(regions, key=lambda name: regions[name].historyOutputs["U2"].data[-1][1])
        fixed_name = next(name for name in regions if name != load_name)
        histories = {}
        for label in ("Load", "Unload"):
            step = odb.steps[label]
            moving = step.historyRegions[load_name]
            fixed = step.historyRegions[fixed_name]
            u = np.array(moving.historyOutputs["U2"].data)
            rf = np.array(moving.historyOutputs["RF2"].data)
            support = np.array(fixed.historyOutputs["RF2"].data)
            if not np.array_equal(u[:, 0], rf[:, 0]) or not np.array_equal(u[:, 0], support[:, 0]):
                raise ValueError("Unloading history grids differ")
            energy_regions = [region for region in step.historyRegions.values() if "ALLSE" in region.historyOutputs]
            if len(energy_regions) != 1:
                raise ValueError("Expected one global unloading energy region")
            energies = {name: np.array(channel.data) for name, channel in energy_regions[0].historyOutputs.items()
                        if name.startswith("ALL")}
            if any(not np.array_equal(value[:, 0], u[:, 0]) for value in energies.values()):
                raise ValueError("Unloading energy/reference-point histories differ")
            force = -support[:, 1]
            if not np.isfinite(u).all() or not np.isfinite(force).all():
                raise ValueError("Nonfinite load-unload history")
            records = []
            for i in range(len(u)):
                row = {"step_time": float(u[i, 0]), "U2_mm": float(u[i, 1]),
                       "force_from_fixed_pin_N": float(force[i]), "loading_pin_RF2_N": float(rf[i, 1]),
                       "fixed_pin_RF2_N": float(support[i, 1])}
                row.update({name+"_Nmm": float(value[i, 1]) for name, value in energies.items()})
                records.append(row)
            write_table(directory / (label.lower()+"-curve.csv"), records)
            histories[label] = (u, rf, force, energies)
        u_load, rf_load, p_load, e_load = histories["Load"]
        u_unload, rf_unload, p_unload, e_unload = histories["Unload"]
        p_start = float(p_load[-1])
        expected = p_start * (1-u_unload[:, 0]/u_unload[-1, 0])
        ramp_error = float(np.max(abs(p_unload-expected))/p_start)
        balance_error = float(np.max(abs(rf_load[:, 1]-p_load))/p_start)
        if ramp_error > 1e-4 or balance_error > 1e-4 or abs(u_load[-1, 1]-metadata["target_U2_mm"]) > 1e-5:
            raise ValueError("Boundary release/force equilibrium/target checks failed")
        work = float(np.sum(.5*(p_load[1:]+p_load[:-1])*np.diff(u_load[:, 1])))
        recovered = -float(np.sum(.5*(p_unload[1:]+p_unload[:-1])*np.diff(u_unload[:, 1])))
        baseline_path = Path(args.root) / "audit/modal/static-summary.json"
        baseline = next(row for row in json.loads(baseline_path.read_text()) if row["case_id"] == metadata["case_id"])
        if baseline["source_inp_sha256"] != metadata["source_sha256"]:
            raise ValueError("Unloading initial stiffness comes from a different source input")
        k0 = baseline["static_stiffness_N_mm"]
        se_released = float(e_unload["ALLSE"][0, 1]-e_unload["ALLSE"][-1, 1])
        pd_added = float(e_unload["ALLPD"][-1, 1]-e_unload["ALLPD"][0, 1]) if "ALLPD" in e_unload else None
        result = dict(metadata, state="completed_and_extracted", preparation_state=metadata["state"],
                      source_odb=str(odb_path), source_odb_sha256=before, read_only=True,
                      source_input_hash_verified=sha(metadata["source_inp"]) == metadata["source_sha256"],
                      load_samples=len(u_load), unload_samples=len(u_unload), load_end_force_N=p_start,
                      load_work_Nmm=work, recovered_unloading_work_Nmm=recovered,
                      work_minus_recovered_Nmm=work-recovered,
                      initial_compliance_triangle_Nmm=p_start*p_start/(2*k0),
                      initial_triangle_apparent_Apl_Nmm=work-p_start*p_start/(2*k0),
                      elastic_energy_released_Nmm=se_released, added_unload_plastic_dissipation_Nmm=pd_added,
                      residual_U2_mm=float(u_unload[-1, 1]), residual_ALLSE_Nmm=float(e_unload["ALLSE"][-1, 1]),
                      final_ALLPD_Nmm=float(e_unload["ALLPD"][-1, 1]) if "ALLPD" in e_unload else None,
                      release_ramp_relative_error=ramp_error, loading_force_balance_relative_error=balance_error)
        frame = load_step.frames[-1]
        if "PEEQ" not in frame.fieldOutputs or "LE" not in frame.fieldOutputs:
            raise ValueError("Requested final loading PEEQ/LE snapshot is missing")
        plastic = frame.fieldOutputs["PEEQ"]
        logstrain = frame.fieldOutputs["LE"]
        if "LE11" not in logstrain.componentLabels:
            raise ValueError("Expected axial LE11 beam output")
        component = list(logstrain.componentLabels).index("LE11")
        peeq = np.array([value.data for value in plastic.values], dtype=float)
        axial = np.array([value.data[component] for value in logstrain.values], dtype=float)
        if not np.isfinite(peeq).all() or not np.isfinite(axial).all():
            raise ValueError("Nonfinite final loading section strain")
        initiation = next(float(rows(block)[0].split(",")[0]) for block in blocks(
            Path(metadata["source_inp"]).read_text()) if key(block) == "*damage initiation")
        result["loading_section_check"] = {"PEEQ_max": float(peeq.max()), "LE11_max": float(axial.max()),
                                           "source_PEEQ_initiation_threshold": initiation,
                                           "max_PEEQ_exceeds_original_threshold": bool(peeq.max()>initiation+1e-7),
                                           "Hsieh_total_axial_strain_threshold": .1105,
                                           "section_observations": len(peeq),
                                           "limit": "This is an undamaged static counterfactual; exceeding the original threshold invalidates a claim that this exact endpoint is pre-initiation in the static control."}
        result["unloading_U2_monotone_decreasing"] = bool(np.max(np.diff(u_unload[:, 1])) <= 1e-6)
        result["unloading_force_nonnegative"] = bool(np.min(p_unload) >= -1e-5)
        if "ALLWK" in e_unload:
            result["final_ALLWK_Nmm"] = float(e_unload["ALLWK"][-1, 1])
            result["integrated_cycle_vs_ALLWK_relative_error"] = abs(work-recovered-result["final_ALLWK_Nmm"])/work
        if sha(odb_path) != before or not result["source_input_hash_verified"]:
            raise ValueError("Load-unload source changed during extraction")
        (directory / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        odb.close()
    print("Extracted completed elastoplastic unloading control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract", "plot", "analyze", "section_fields", "locate_events", "prepare_unload", "extract_unload"))
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out", default="Z:/p1/sims/Ti/Validation/audit/diagnostics")
    parser.add_argument("--queue-root", help="Existing completed serial-queue results; never opens active jobs")
    parser.add_argument("--unload-target", type=float, help="Explicit unloading-control peak U2 in mm; default native pre-section reference")
    parser.add_argument("--unload-job", default="epUnload_hex_3_rho0p16_nx80", help="Unique unloading-control job/directory")
    parser.add_argument("--proposal", help="Explicit new unloading-proposal JSON path")
    options = parser.parse_args()
    globals()[options.action](options)
