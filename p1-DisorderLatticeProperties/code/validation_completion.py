"""Prepare the seven absent Hsieh validation cases; never submit from here.

Exact A1-generated meshes are reused through the dimensional scaling already
present in archived 3/10 mm pairs. The two 13% hexagonal cases retain the largest
available hexagonal mesh (nx80); convergence at 13% is not thereby established.
The original collection catalogue and its inputs are immutable.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from resources.lattices import Geometry
from validation_diagnostics import blocks, key, rows, source_mesh


ROOT = Path("Z:/p1/sims/Ti/Validation")
PLANS = (
    ("tri", 3, .16, 80, "10/0.16/Fracture-tri-80-per-1.inp", 10, .16),
    ("kagome", 3, .16, 50, "10/0.16/Fracture-kagome-50-per-1.inp", 10, .16),
    ("tri", 3, .13, 120, "10/0.13/Fracture-tri-120-per-1.inp", 10, .13),
    ("kagome", 3, .13, 80, "10/0.13/Fracture-kagome-80-per-1.inp", 10, .13),
    ("hex", 10, .16, 80, "3/0.16/Fracture-hex-80-per-1.inp", 3, .16),
    ("hex", 3, .13, 80, "3/0.16/Fracture-hex-80-per-1.inp", 3, .16),
    ("hex", 10, .13, 80, "3/0.16/Fracture-hex-80-per-1.inp", 3, .16),
)
ENERGIES = "ALLIE, ALLKE, ALLSE, ALLAE, ALLPD, ALLVD, ALLWK, ALLDMD, ALLCD"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data(block):
    return [[float(v) for v in line.split(",") if v.strip()] for line in rows(block)]


def transform(text, scale, density_ratio):
    """Change reviewed length quantities and output resolution, nothing else."""
    result = []
    for block in blocks(text):
        kind = key(block)
        lines = block.splitlines(keepends=True)
        active_row = 0
        for i, line in enumerate(lines[1:], 1):
            if not line.strip() or line.startswith("**"):
                continue
            values = [v.strip() for v in line.strip().split(",")]
            if kind == "*node":
                values[1:] = [format(float(v)*scale, ".15g") for v in values[1:] if v]
            elif kind == "*beam section" and active_row == 0:
                values = [format(float(values[0])*scale, ".15g"),
                          format(float(values[1])*scale*density_ratio, ".15g")]
            elif kind == "*amplitude":
                for j in range(1, len(values), 2):
                    values[j] = format(float(values[j])*scale, ".15g")
            elif kind == "*damage evolution":
                values[1] = format(float(values[1])*scale, ".15g")
            elif kind == "*energy output":
                values = [ENERGIES]
            else:
                active_row += 1
                continue
            lines[i] = ", ".join(values) + "\n"
            active_row += 1
        if kind == "*energy output":
            # ALLWK is unavailable for an element-set request. This source has
            # one beam part and massless reference-point couplings only.
            lines[0] = "*Energy Output\n"
        if kind == "*output":
            lower = lines[0].lower()
            if "field" in lower:
                lines[0] = "*Output, field, number interval=200\n"
            elif "history" in lower:
                lines[0] = "*Output, history, time interval=0.0001\n"
            else:
                raise ValueError("Unexpected output request")
        result.extend(lines)
    return "".join(result)


def finite_density(path, lat, length, nx):
    _, nodes, elements, section, _ = source_mesh(path)
    total = sum(np.linalg.norm(np.array(nodes[b])-nodes[a]) for _, a, b in elements)
    geom = Geometry(lat, length, nx)
    return float(total * section[1] / (geom.L*geom.H))


def pair_checks(root):
    """Independent existing-case evidence for the dimensional A1 convention."""
    checks = []
    for lat, nx in (("tri", 70), ("kagome", 40)):
        ten = root / f"10/0.2/Fracture-{lat}-{nx}-per-1.inp"
        three = root / f"3/0.2/Fracture-{lat}-{nx}-per-1.inp"
        t10, n10, e10, s10, el10 = source_mesh(ten)
        t3, n3, e3, s3, el3 = source_mesh(three)
        if e10 != e3 or list(n10) != list(n3) or el10 != el3:
            raise ValueError("Existing A1 pair topology/material differs")
        coordinate_error = float(np.max(np.abs(np.array(list(n3.values()))-
                                               .3*np.array(list(n10.values())))))
        section_error = float(np.max(np.abs(np.array(s3)/(.3*np.array(s10))-1)))
        b10, b3 = blocks(t10), blocks(t3)
        evolution10 = next(data(b) for b in b10 if key(b) == "*damage evolution")
        evolution3 = next(data(b) for b in b3 if key(b) == "*damage evolution")
        expected = np.array(evolution10)*[1, .3]
        evolution_error = float(np.max(np.abs(expected-evolution3)))
        for kind in ("*density", "*elastic", "*plastic", "*damage initiation", "*dynamic", "*boundary"):
            if [rows(b) for b in b10 if key(b) == kind] != [rows(b) for b in b3 if key(b) == kind]:
                raise ValueError("Existing pair's material/time/BC differs: " + kind)
        amplitude_ratio = (next(data(b)[0][3] for b in b3 if key(b) == "*amplitude") /
                           next(data(b)[0][3] for b in b10 if key(b) == "*amplitude"))
        if coordinate_error > .0006 or section_error > 1e-5 or evolution_error > 3e-9:
            raise ValueError("Existing pair does not support length scaling")
        checks.append({"topology":lat, "nx":nx, "ten_mm":str(ten), "three_mm":str(three),
                       "ten_mm_sha256":sha(ten), "three_mm_sha256":sha(three),
                       "identical_connectivity":True, "node_coordinate_rounding_error_mm":coordinate_error,
                       "relative_section_rounding_error":section_error,
                       "damage_evolution_rounding_error_mm":evolution_error,
                       "archived_load_amplitude_ratio":amplitude_ratio,
                       "note":"Triangular amplitudes differ; new jobs scale their own parent displacement, not another density's loading."})
    return checks


def prepare(root=ROOT, refresh_unsubmitted=False):
    root = Path(root).resolve()
    out = root / "audit/new-validation"
    out.mkdir(parents=True, exist_ok=True)
    if refresh_unsubmitted and (out / "queue-20261029/state.json").exists():
        raise ValueError("Submitted queue preparation is immutable")
    prior_path = out / "generation-receipt.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {"cases":[]}
    known_inputs = {row["input"]:row["input_sha256"] for row in prior["cases"]}
    pairing = pair_checks(root)
    jobs, records = [], []
    allowed_changes = {"*node", "*beam section", "*amplitude", "*damage evolution", "*output", "*energy output"}
    for lat, length, density, nx, parent, old_length, old_density in PLANS:
        source = root / parent
        before = sha(source)
        text, nodes, elements, section, elastic = source_mesh(source)
        scale, dr = length/old_length, density/old_density
        job = f"complete_{lat}_{length}_rho{str(density).replace('.', 'p')}_nx{nx}"
        destination = out / "inputs" / (job + ".inp")
        destination.parent.mkdir(exist_ok=True)
        generated = transform(text, scale, dr)
        if destination.exists() and destination.read_text() != generated:
            if not refresh_unsubmitted or known_inputs.get(str(destination)) != sha(destination):
                raise ValueError("Refuse to overwrite different/unrecorded generated input")
        destination.write_text(generated, encoding="ascii")
        new_text, new_nodes, new_elements, new_section, new_elastic = source_mesh(destination)
        if elements != new_elements or list(nodes) != list(new_nodes) or elastic != new_elastic:
            raise ValueError("Generated topology/elastic material changed")
        old_blocks, new_blocks = blocks(text), blocks(new_text)
        if Counter(map(key,old_blocks)) != Counter(map(key,new_blocks)):
            raise ValueError("Input keyword inventory changed")
        for old, new in zip(old_blocks,new_blocks):
            if key(old) not in allowed_changes and old != new:
                raise ValueError("Unexpected keyword mutation: " + key(old))
        if not np.allclose(np.array(list(new_nodes.values())), scale*np.array(list(nodes.values())), rtol=1e-13, atol=1e-10):
            raise ValueError("Generated node scaling differs")
        if not np.allclose(new_section, np.array(section)*[scale, scale*dr], rtol=1e-13):
            raise ValueError("Generated beam section differs")
        original_rho = finite_density(source, lat, old_length, nx)
        actual_rho = finite_density(destination, lat, length, nx)
        if abs(actual_rho/density-1) > 1e-5 or abs(original_rho/old_density-1) > 1e-5:
            raise ValueError("Finite specimen density does not match declared nominal density")
        if before != sha(source):
            raise ValueError("Original input changed")
        record = {"job":job, "topology":lat, "l_mm":length, "relative_density":density,
                  "nnx":nx, "parent_input":str(source), "parent_input_sha256":before,
                  "input":str(destination), "input_sha256":sha(destination),
                  "length_scale":scale, "thickness_density_scale":dr,
                  "nodes":len(nodes), "elements":len(elements), "section_B_t_mm":new_section,
                  "finite_specimen_relative_density":actual_rho,
                  "parent_finite_specimen_relative_density":original_rho,
                  "material_density_kg_m3":4430, "material_and_BC_unchanged":True,
                  "field_intervals":200, "history_intervals":1000,
                  "history_time_interval_s":.0001,
                  "energy_region":"whole model, enabling ALLWK; one physical beam part plus massless reference points",
                  "size_basis":("largest_existing_hexagonal_nx80_no_13_percent_convergence_claim"
                                if lat=="hex" and density==.13 else "existing_selected_source_specimen"),
                  "changes":["node/RP coordinates", "rectangular section B and t",
                             "displacement amplitude", "A1 element-length-scaled damage evolution",
                             "output resolution and energy components"]}
        records.append(record)
        jobs.append({"job":job, "input":str(destination), "input_sha256":record["input_sha256"],
                     "cpus":8, "kind":"new_original_Explicit_fracture", "allow_solver_failure":True,
                     "required_outputs":[".odb", ".dat"], "execution_options":["output_precision=full"],
                     "purpose":f"Complete absent Hsieh validation case {lat}, {length}mm, {density:.0%}; original A1-source similarity transform"})
    receipt = {"preparation":"input_only_no_submission", "original_catalogues_changed":False,
               "pair_checks":pairing, "cases":records,
               "solver_precision":"original A1 single Explicit; full nodal output precision is an output-only improvement",
               "material_authority":"Historical parent INP, including its original element-length-scaled damage evolution; current A1 material defaults were not regenerated",
               "hexagonal_13_percent_size_limit":"nx80 anchored to largest existing hexagonal source; no new convergence assertion",
               "helper_sha256":sha(__file__)}
    manifest = {"abaqus":"C:/SIMULIA/Commands/abaqus.bat", "queue_root":str(out/"queue-20261029"), "jobs":jobs}
    for name, value in (("generation-receipt.json", receipt), ("serial-fracture-proposal.json", manifest)):
        target = out / name
        if target.exists() and json.loads(target.read_text()) != value and not refresh_unsubmitted:
            raise ValueError("Refuse to replace a different retained preparation")
        target.write_text(json.dumps(value, indent=2), encoding="utf-8")
    catalogue = []
    for record in records:
        name = record["job"]
        lat, length, density, nx = (record[k] for k in ("topology", "l_mm", "relative_density", "nnx"))
        catalogue.append({"case_id":f"new:{length}/{density}/Fracture-{lat}-{nx}-per-1.odb",
                          "topology":lat, "l_mm":length, "nominal_relative_density":density, "nnx":nx,
                          "mode":"Fracture", "campaign":"New validation completion",
                          "job":name, "inp":str(Path(record["input"]).relative_to(root)),
                          "inp_sha256":record["input_sha256"],
                          "odb":str((out/"queue-20261029/runs"/name/(name+".odb")).relative_to(root)),
                          "parent_input_sha256":record["parent_input_sha256"],
                          "purpose":"Prepared identity-to-expected-output mapping; completion/hash/event qualification required before plotting"})
    with (out/"prepared-case-index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(catalogue[0]))
        writer.writeheader(); writer.writerows(catalogue)
    return {"prepared_jobs":len(jobs), "manifest":str(out/"serial-fracture-proposal.json"),
            "receipt":str(out/"generation-receipt.json")}


def datacheck(root=ROOT):
    """Run one bounded input preflight under the same validation serial lock."""
    import validation_queue as queue
    root = Path(root).resolve()
    out = root / "audit/new-validation"
    manifest, _ = queue.validate_manifest(out / "serial-fracture-proposal.json", root)
    job = dict(manifest["jobs"][0], job="check_"+manifest["jobs"][0]["job"], required_outputs=[".dat"])
    check_root = out / "datacheck"
    old_receipt = out / "datacheck-receipt.json"
    if old_receipt.exists():
        previous = json.loads(old_receipt.read_text())
        previous_job = previous["delivery"]["job"]
        previous_path = check_root/"runs"/previous_job/"preflight.json"
        queue.write_json(previous_path, previous)
    attempt = len(list((check_root/"runs").glob("check_*"))) if (check_root/"runs").exists() else 0
    if attempt:
        job["job"] += "_r" + str(attempt)
    work = check_root / "scratch" / job["job"]
    with queue.worker_lock(root/"audit", "serial-worker.lock"):
        owner = json.loads((root/"audit/serial-owner.json").read_text())
        if owner["phase"] != "settled":
            raise ValueError("Another validation queue remains active")
        work.mkdir(parents=True, exist_ok=False)
        queue.write_json(work/"scratch-owner.json", {"job":job["job"], "input_sha256":job["input_sha256"]})
        shutil.copy2(job["input"], work/(job["job"]+".inp"))
        if sha(work/(job["job"]+".inp")) != job["input_sha256"]:
            raise ValueError("Datacheck staging differs")
        (work/"solver-scratch").mkdir()
        command = [manifest["abaqus"], "job="+job["job"], "input="+job["job"]+".inp",
                   "cpus=8", "datacheck", "interactive", "scratch="+str(work/"solver-scratch")]
        with (work/"launcher.txt").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        delivered = queue.deliver_and_clean(work, check_root/"runs"/job["job"], job)
        destination = check_root / "runs" / job["job"]
        output = (destination/"launcher.txt").read_text(errors="replace")
        dat = (destination/(job["job"]+".dat")).read_text(errors="replace")
        passed = process.returncode == 0 and "COMPLETED" in output and "***ERROR" not in dat
        receipt = {"passed":passed, "returncode":process.returncode, "command":command,
                   "input_sha256":job["input_sha256"], "delivery":delivered,
                   "warnings":[line.strip() for line in dat.splitlines() if "WARNING" in line.upper()],
                   "energy_requests":ENERGIES, "serial_lock_used":True, "completed_at":queue.timestamp()}
        queue.write_json(out/"datacheck-receipt.json", receipt)
        if not passed:
            raise ValueError("Input datacheck failed; inspect retained receipt/logs before submission")
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refresh-unsubmitted", action="store_true")
    parser.add_argument("--datacheck", action="store_true")
    args = parser.parse_args()
    print(json.dumps(datacheck(args.root) if args.datacheck else
                     prepare(args.root, args.refresh_unsubmitted), indent=2))
