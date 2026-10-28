"""Extend exact-source B21 conversion and initial compliance controls to all cases.

Preparation only writes derived inputs. Submission is an explicit serial-queue
action. Existing completed controls are referenced, never copied or overwritten.
Compliance preparation reuses the modal builder with unit_load only; gathering
uses completed serial-queue deliveries and reads their ODBs without upgrading.
"""
import argparse
import math
import json
from pathlib import Path
from types import SimpleNamespace

from validation_diagnostics import periodic_patch, prepare_rve, sha, source_mesh, table, extract
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
    parser.add_argument("action", choices=("prepare", "gather", "prepare-compliance", "gather-compliance"))
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--out", type=Path, default=Path("Z:/p1/sims/Ti/Validation/audit/controls"))
    args = parser.parse_args()
    globals()[args.action.replace("-", "_")](args.root, args.out)
