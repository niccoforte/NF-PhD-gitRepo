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
import os
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
    if (out / "queue-20261029/state.json").exists():
        raise ValueError("Submitted queue preparation is immutable")
    out.mkdir(parents=True, exist_ok=True)
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
    if (out / "queue-20261029/state.json").exists():
        raise ValueError("Submitted queue preparation is immutable; use retained preflight evidence")
    manifest, _ = queue.validate_manifest(out / "serial-fracture-proposal.json", root)
    job = dict(manifest["jobs"][0], job="check_"+manifest["jobs"][0]["job"], required_outputs=[".dat"])
    check_root = out / "datacheck"
    with queue.worker_lock(root/"audit", "serial-worker.lock"):
        owner_path = root / "audit/serial-owner.json"
        owner = json.loads(owner_path.read_text())
        if owner["phase"] != "settled":
            raise ValueError("Another validation queue remains active")
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
        work.mkdir(parents=True, exist_ok=False)
        queue.write_json(work/"scratch-owner.json", {"job":job["job"], "input_sha256":job["input_sha256"]})
        shutil.copy2(job["input"], work/(job["job"]+".inp"))
        if sha(work/(job["job"]+".inp")) != job["input_sha256"]:
            raise ValueError("Datacheck staging differs")
        (work/"solver-scratch").mkdir()
        command = [manifest["abaqus"], "job="+job["job"], "input="+job["job"]+".inp",
                   "cpus=8", "datacheck", "interactive", "scratch="+str(work/"solver-scratch")]
        preflight_owner = {"phase":"active", "kind":"input_datacheck",
                           "queue_root":str(check_root), "worker_pid":os.getpid(),
                           "command":command, "started_at":queue.timestamp()}
        queue.write_json(owner_path, preflight_owner)
        with (work/"launcher.txt").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        # Only a returned subprocess confirms the solver has exited. Worker
        # interruption leaves the durable gate active for explicit review.
        queue.write_json(owner_path, dict(preflight_owner, phase="settled", finished_at=queue.timestamp()))
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


def merge_completion_attempts(existing, incoming, attempt_history=()):
    """Keep qualified cases; promote a replacement only after qualification."""
    selected = {row["case"]["case_id"]:row for row in existing}
    lineage = list(attempt_history)
    for row in incoming:
        identity = row["case"]["case_id"]
        previous = selected.get(identity)
        if previous is None:
            selected[identity] = row
            continue
        if previous["before"]["sha256"] == row["before"]["sha256"]:
            continue
        replace = not previous["initiation_prefix_qualified"] and row["initiation_prefix_qualified"]
        other = previous if replace else row
        item = {"case_id":identity, "source_odb":other["case"]["source_odb"],
                "source_sha256":other["before"]["sha256"], "state":other["state"],
                "initiation_prefix_qualified":other["initiation_prefix_qualified"],
                "event_json":other["event_json"], "event_json_sha256":other.get("event_json_sha256"),
                "receipt":str(Path(other["history_export_directory"])/"extraction.json"),
                "selection_reason":"Replaced unqualified attempt" if replace else "Retained prior qualified case or rejected an unqualified replacement"}
        if item not in lineage:
            lineage.append(item)
        if replace:
            selected[identity] = row
    return list(selected.values()), lineage


def extract_finished(root=ROOT, include_interrupted=False, queue_root=None):
    """Reuse the native exporters for finished jobs in an additive result tree.

    Existing helpers share a cached source fingerprint inside one extraction;
    actual full-file fingerprints bracket the complete sequence. No source ODB
    is upgraded, copied, edited, or deleted, including a reviewed dead-worker
    partial ODB. The original case catalogue remains untouched.
    """
    import ctypes
    import validation_failure_audit as failure
    import validation_history_audit as history
    from odbAccess import openOdb

    extractor_digest = sha(__file__)

    def active(pid):
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            if ctypes.windll.kernel32.GetLastError() == 5:
                raise PermissionError("Cannot establish worker absence: access denied")
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError("Cannot establish worker process status")
            return code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    root = Path(root).resolve()
    campaign = root / "audit/new-validation"
    queue_root = Path(queue_root).resolve() if queue_root else campaign / "queue-20261029"
    queue_root.relative_to(campaign)
    state = json.loads((queue_root / "state.json").read_text())
    manifest = json.loads(Path(state["manifest"]).read_text())
    definitions = {row["job"]:row for row in manifest["jobs"]}
    prepared = {row["job"]:row for row in csv.DictReader((campaign/"prepared-case-index.csv").open())}
    output = campaign / "results"
    history_out, failure_out = output/"audit/history", output/"audit/failure"
    history_out.mkdir(parents=True, exist_ok=True)
    failure_out.mkdir(parents=True, exist_ok=True)
    previous_path = output/"extraction-summary.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {"cases":[]}
    if previous.get("state", "complete") != "complete":
        raise ValueError("Existing unfinished extraction requires explicit review")
    for name, path in (("event_csv_sha256", failure_out/"failure-initiation.csv"),
                       ("stored_energy_summary_sha256", history_out/"stored-energy-summary.json"),
                       ("case_index_sha256", output/"case-index.csv")):
        if previous.get(name) and sha(path) != previous[name]:
            raise ValueError("Existing extraction aggregate changed before merge: " + name)
    cases, events, energies, statuses = [], [], [], []
    real_history_fingerprint, real_failure_fingerprint = history.fingerprint, failure.fingerprint
    for job in state["jobs"]:
        definition = definitions[job["job"]]
        if definition.get("kind") != "new_original_Explicit_fracture":
            continue
        status = job["status"]
        if status not in ("completed", "failed"):
            if not (status in ("running", "interrupted") and include_interrupted):
                continue
            if active(state["worker_pid"]) or active(job["solver_launcher_pid"]):
                raise ValueError("Refuse to read a live interrupted-job candidate")
        source = queue_root / ("scratch" if status in ("running", "interrupted") else "runs") / job["job"] / (job["job"]+".odb")
        if not source.exists():
            raise ValueError("No usable database exists for finished job: " + job["job"])
        print("Extracting source read-only: " + str(source), flush=True)
        before = real_history_fingerprint(source)
        delivery_path = source.parent / "delivery.json"
        if delivery_path.exists():
            delivery = json.loads(delivery_path.read_text())
            expected = next(row["sha256"] for row in delivery["files"] if row["file"] == source.name)
            if before["sha256"] != expected:
                raise ValueError("Delivered source changed before extraction")
        parent_job = definition.get("parent_job", job["job"])
        case = dict(prepared[parent_job])
        case.update(job=job["job"], parent_job=parent_job, queue_root=str(queue_root))
        if sha(root/case["inp"]) != case["inp_sha256"] or sha(source.with_suffix(".inp")) != case["inp_sha256"]:
            raise ValueError("Prepared/retained solver input differs from its frozen source")
        case.update(source_odb=str(source), source_inp=str(root/case["inp"]),
                    inp=str(root/case["inp"]), odb_sha256=before["sha256"],
                    odb=str(Path(source.parent.parent.name)/source.parent.name/source.name))
        case_out = history_out / Path(case["odb"]).parent / source.stem
        receipt_path = case_out / "extraction.json"
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            if prior["before"] != before or not prior["source_unchanged"]:
                raise ValueError("Retained extraction source differs; explicit review required")
            for item in prior["exports"]:
                if sha(item["path"]) != item["sha256"]:
                    raise ValueError("Retained native export changed")
            event = json.loads(Path(prior["event_json"]).read_text())
            if prior.get("event_json_sha256") and sha(prior["event_json"]) != prior["event_json_sha256"]:
                raise ValueError("Retained initiation event changed")
            cases.append(prior["case"]); events.append(event)
            energies.append(prior["energy_attestation"]); statuses.append(prior)
            continue

        def cached(path):
            return dict(before) if Path(path).resolve() == source.resolve() else real_history_fingerprint(path)

        history.fingerprint = cached
        failure.fingerprint = cached
        try:
            curve_proof = history.audit_case(source, history_out, histories_only=True)
            energy_proof = history.export_stored_energy(source, history_out)
            # Add the new whole-model channels to the same native time grid.
            energy_path = case_out / "native_stored_energy.csv"
            with energy_path.open(newline="") as file:
                reader = csv.DictReader(file)
                fieldnames, energy_rows = list(reader.fieldnames), list(reader)
            odb = openOdb(path=str(source), readOnly=True)
            try:
                region = odb.steps["Step-1"].historyRegions[energy_proof["region"]]
                for variable in ("ALLCD", "ALLWK"):
                    values = list(region.historyOutputs[variable].data) if variable in region.historyOutputs else []
                    lookup = dict(values)
                    if len(lookup) != len(values):
                        raise ValueError("Repeated native energy times need review")
                    column = variable+"_Nmm"; fieldnames.append(column)
                    for row in energy_rows:
                        row[column] = lookup.get(float(row["time_s"]), float("nan"))
                    energy_proof["variables"][variable] = {
                        "available":bool(values), "samples":len(values),
                        "meaning_if_unavailable":"not recorded; unknown, not zero",
                        "values":history.numeric_summary([value for _, value in values])}
                    energy_proof["all_recorded_time_grids_equal"] &= [time for time, _ in values] == [float(row["time_s"]) for row in energy_rows]
            finally:
                odb.close()
            with energy_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader(); writer.writerows(energy_rows)
            energy_proof["csv_sha256"] = sha(energy_path)
            event = failure.initiation_only(source, case, failure_out)
            failure.save_initiations(failure_out, [event])
            failure.damage_prefix_audit(failure_out)
            event = json.loads((failure_out/"failure-initiation.json").read_text())["cases"][0]
        finally:
            history.fingerprint = real_history_fingerprint
            failure.fingerprint = real_failure_fingerprint
        after = real_history_fingerprint(source)
        if before != after:
            raise ValueError("Source changed during native extraction")
        scope = "Actual full-file SHA256 before/after complete native extraction; inner helpers share that batch fingerprint"
        for record in (curve_proof, energy_proof, event, event["damage_prefix"]):
            record.update(before=before, after=after, source_unchanged=True,
                          read_only=True, source_fingerprint_scope=scope)
        event_path = failure_out/Path(case["odb"]).parent/source.stem/"initiation.json"
        failure.dump(event_path, event)
        history.json_dump(case_out/"summary.json", curve_proof)
        valid = bool(event["threshold_found"] and event["recommended_prefix_available"]
                     and event["undamaged_prefix_through_primary_lower_bound_verified"])
        state_name = ("completed_and_extracted" if status == "completed" else
                      "failed_with_usable_preinitiation_prefix" if status == "failed" and valid else
                      "interrupted_with_usable_preinitiation_prefix" if status in ("running", "interrupted") and valid else
                      "interrupted_before_qualified_initiation" if status in ("running", "interrupted") else "failed_without_qualified_prefix")
        exports = [{"path":str(path), "sha256":sha(path)} for path in sorted(case_out.iterdir())
                   if path.is_file() and path.suffix in (".csv", ".npz", ".json") and path != receipt_path]
        receipt = {"case":case, "before":before, "after":after, "source_unchanged":True,
                   "read_only":True, "source_fingerprint_scope":scope,
                   "extractor_sha256":extractor_digest, "source_input_hash_verified":True,
                   "retained_solver_input_hash_verified":True,
                   "queue_solver_status":status, "state":state_name, "initiation_prefix_qualified":valid,
                   "solver_complete":status == "completed", "event_json":str(event_path),
                   "event_json_sha256":sha(event_path),
                   "energy_attestation":energy_proof, "exports":exports,
                   "solver_diagnostics":curve_proof.get("solver_diagnostics"),
                   "history_export_directory":str(case_out)}
        failure.dump(receipt_path, receipt)
        cases.append(case); events.append(event); energies.append(energy_proof); statuses.append(receipt)
        failure.save_initiations(failure_out, events)
        failure.dump(history_out/"stored-energy-summary.json", energies)
        failure.dump(output/"extraction-summary.json", {"cases":statuses, "state":"extracting",
                                                       "extractor_sha256":extractor_digest})
        print(json.dumps({"job":job["job"], "state":state_name,
                          "criterion_found":event["threshold_found"], "usable_prefix":valid}), flush=True)
    statuses, attempt_history = merge_completion_attempts(previous["cases"], statuses, previous.get("attempt_history", []))
    cases = [row["case"] for row in statuses]
    events = [json.loads(Path(row["event_json"]).read_text()) for row in statuses]
    energies = [row["energy_attestation"] for row in statuses]
    failure.save_initiations(failure_out, events)
    failure.dump(history_out/"stored-energy-summary.json", energies)
    with (output/"case-index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in cases for key in row)))
        writer.writeheader(); writer.writerows(cases)
    result = {"cases":statuses, "state":"complete", "extractor_sha256":extractor_digest,
              "attempt_history":attempt_history, "last_extracted_queue":str(queue_root),
              "note":"Additive catalogue for completed/reviewed partial new jobs only; original catalogue and live queue unchanged"}
    failure.dump(output/"extraction-summary.json", result)
    bind_extraction(output)
    return {"results_root":str(output), "cases":len(cases), "usable_prefixes":sum(r["initiation_prefix_qualified"] for r in statuses)}


def bind_extraction(output):
    """Add small-file integrity bindings without repeating any ODB extraction."""
    import validation_history_audit as history
    from validation_failure_audit import dump

    output = Path(output)
    summary_path = output/"extraction-summary.json"
    previous_bytes = summary_path.read_bytes()
    summary = json.loads(previous_bytes)
    if summary["state"] != "complete":
        raise ValueError("Only a complete extraction may be bound for calculations")
    source_history_path = output/"extraction-code-history.json"
    source_history = json.loads(source_history_path.read_text()) if source_history_path.exists() else {"versions":{}}
    extraction_digest = summary["extractor_sha256"]
    # Previous extraction source is resolvable even after the maintained helper
    # evolves. Store source text as JSON evidence, never executable data copies.
    current_bytes = Path(__file__).read_bytes()
    current_digest = hashlib.sha256(current_bytes).hexdigest()
    source_history["versions"].setdefault(current_digest, {
        "sha256":current_digest, "source_path":str(Path(__file__).resolve()),
        "source_text":current_bytes.decode("utf-8"), "role":"Current extraction/binding implementation"})
    if extraction_digest not in source_history["versions"]:
        raise ValueError("Actual extraction code identity is not retained")
    dump(output/"extraction-code-history.json", source_history)
    rows = []
    for receipt in summary["cases"]:
        directory = Path(receipt["history_export_directory"])
        receipt_path = directory/"extraction.json"
        previous_receipt = receipt_path.read_bytes()
        previous_digest = hashlib.sha256(previous_receipt).hexdigest()
        history_dir = directory/"extraction-history"
        history_dir.mkdir(exist_ok=True)
        if receipt.get("schema_version") != 2:
            (history_dir/(previous_digest+".json")).write_bytes(previous_receipt)
            receipt["binding_upgrade"] = {"previous_receipt_sha256":previous_digest,
                "previous_receipt":str(history_dir/(previous_digest+".json")),
                "note":"Added integrity bindings and CSV-derived diagnostics; source ODB and scientific export bytes unchanged"}
        event_path = Path(receipt["event_json"])
        if receipt.get("event_json_sha256") and sha(event_path) != receipt["event_json_sha256"]:
            raise ValueError("Pinned initiation event changed before binding")
        event = json.loads(event_path.read_text())
        energy_path = directory/"native_energy.csv"
        with energy_path.open(newline="") as file:
            energy = [[float(row[k]) for k in ("time_s", "ALLKE_Nmm", "ALLIE_Nmm", "KE_over_IE")]
                      for row in csv.DictReader(file)]
        cutoff = event.get("prev_native_history", {}).get("time_s")
        receipt.update(schema_version=2, extractor_sha256=receipt.get("extractor_sha256", extraction_digest),
                       event_json_sha256=sha(event_path),
                       binding_code_sha256=current_digest,
                       energy_prefix_check=history.energy_window(energy, cutoff, True, "through_last_saved_precriterion_sample"))
        for item in receipt["exports"]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Scientific export changed before binding")
        inp = Path(receipt["case"]["inp"])
        source_inp = Path(receipt["case"]["source_odb"]).with_suffix(".inp")
        if sha(inp) != receipt["case"]["inp_sha256"] or sha(source_inp) != receipt["case"]["inp_sha256"]:
            raise ValueError("Prepared/retained input changed before binding")
        receipt.update(source_input_hash_verified=True, retained_solver_input_hash_verified=True)
        dump(receipt_path, receipt)
        rows.append(receipt)
    history_dir = output/"extraction-history"
    history_dir.mkdir(exist_ok=True)
    if summary.get("schema_version") != 2:
        (history_dir/(hashlib.sha256(previous_bytes).hexdigest()+".json")).write_bytes(previous_bytes)
    summary.update(cases=rows, schema_version=2, binding_code_sha256=current_digest,
                   event_csv_sha256=sha(output/"audit/failure/failure-initiation.csv"),
                   stored_energy_summary_sha256=sha(output/"audit/history/stored-energy-summary.json"),
                   case_index_sha256=sha(output/"case-index.csv"))
    dump(summary_path, summary)
    return {"bound_cases":len(rows), "source_ODB_reads":0, "summary":str(summary_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refresh-unsubmitted", action="store_true")
    parser.add_argument("--datacheck", action="store_true")
    parser.add_argument("--extract-finished", action="store_true")
    parser.add_argument("--include-interrupted", action="store_true")
    parser.add_argument("--bind-extraction", action="store_true")
    parser.add_argument("--queue-root", type=Path, help="Explicit reviewed queue; auxiliary jobs are skipped and fracture parent_job maps to prepared identity")
    args = parser.parse_args()
    print(json.dumps(bind_extraction(args.root/"audit/new-validation/results") if args.bind_extraction else
                     extract_finished(args.root, args.include_interrupted, args.queue_root) if args.extract_finished else
                     datacheck(args.root) if args.datacheck else
                     prepare(args.root, args.refresh_unsubmitted), indent=2))
