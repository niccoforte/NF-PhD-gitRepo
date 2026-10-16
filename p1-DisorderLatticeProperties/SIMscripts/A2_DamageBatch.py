"""Manifest-only FCC STATUS processing; prepare is ordinary Python, run is Abaqus Python.

Only the existing Ti FCC periodic and disNodes/0.2 archive roots are accepted.
No recursive discovery, ODB upgrade, FEA submission, or existing field rewrite.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile


ARCHIVE = Path("/data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data")
SCHEMA = "fcc_status_batch_v1"
PATTERN = re.compile(r"^(Ductile|Fracture)-FCC-20-(?:per-1|20disNodes-lhs-all-([1-9][0-9]*))$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprints():
    here = Path(__file__).resolve()
    return {"batch_driver": fingerprint(here),
            "exporter": fingerprint(here.with_name("A2_DamageOUTpostProcess.py")),
            "damage_core": fingerprint(here.parents[2] / "resources" / "damage.py")}


def file_stat(path):
    info = Path(path).stat()
    require(Path(path).is_file() and info.st_size > 0, "Missing/empty source: " + str(path))
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), prefix="." + path.name,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def validate_record(record, archive_root=ARCHIVE):
    """Constrain manifest entries as well as prepare-time discovery, including symlinks."""
    odb = Path(record["odb"])
    stem = odb.stem
    match = PATTERN.fullmatch(stem)
    require(match is not None, "Unsupported FCC source stem: " + stem)
    task = "UT" if match[1] == "Ductile" else "FT"
    simulation_id = int(match[2]) if match[2] else 0
    require(record["task"] == task and record["simulation_id"] == simulation_id,
            "Manifest task/sample mismatch")
    job_id = str(record["archive_job_id"])
    require(job_id.isdigit(), "Archive job ID must be numeric")
    case_root = (archive_root / "disNodes" / "0.2" / "FCC" if simulation_id else
                 archive_root / "per" / "0.0" / "FCC")
    require(simulation_id or job_id == "6538532", "Unreviewed periodic archive job")
    directory = case_root / job_id / "zip"
    expected = dict(odb=directory / (stem + ".odb"), inp=directory / (stem + ".inp"),
                    field=directory / "transfer" / ("FIELDu-" + stem + ".npz"),
                    output=directory / "transfer" / ("FIELDd-" + stem + ".npz"))
    require(case_root.resolve().is_relative_to(archive_root.resolve()), "Archive case redirects outside reviewed root")
    require(not any(path.is_symlink() for path in (case_root / job_id, directory, directory / "transfer")),
            "Redirected archive job/zip/transfer directory")
    for key, path in expected.items():
        require(Path(record[key]).absolute() == path.absolute() and not Path(record[key]).is_symlink(),
                "Unreviewed/redirected " + key + " path")
    require(record["source_stem"] == stem, "Manifest stem mismatch")
    return expected


def prepare_manifest(archive_root=ARCHIVE, preflight=False):
    """Inventory exact reviewed roots using directory names and source stat only; no ODB reads."""
    archive_root = Path(archive_root).absolute()
    disordered = archive_root / "disNodes" / "0.2" / "FCC"
    require(disordered.is_dir(), "Missing FCC archive root: " + str(disordered))
    jobs = [("disordered", p / "zip") for p in sorted(disordered.iterdir()) if p.is_dir() and p.name.isdigit()]
    jobs += [("periodic", archive_root / "per" / "0.0" / "FCC" / "6538532" / "zip")]
    batches, identities = [], set()
    for case, directory in jobs:
        require(directory.is_dir(), "Missing archive zip directory: " + str(directory))
        records = []
        for odb in sorted(directory.glob("*.odb")):
            match = PATTERN.fullmatch(odb.stem)
            require(match is not None, "Unexpected ODB in reviewed FCC job: " + str(odb))
            simulation_id = int(match[2]) if match[2] else 0
            require((case == "periodic") == (simulation_id == 0), "Periodic/disordered directory mismatch")
            if preflight and simulation_id not in {0, 1, 4501, 8901}:
                continue
            task = "UT" if match[1] == "Ductile" else "FT"
            identity = (task, simulation_id)
            require(identity not in identities, "Duplicate task/sample: " + str(identity))
            identities.add(identity)
            record = dict(task=task, simulation_id=simulation_id, source_stem=odb.stem,
                          archive_job_id=directory.parent.name, odb=str(odb),
                          inp=str(odb.with_suffix(".inp")),
                          field=str(directory / "transfer" / ("FIELDu-" + odb.stem + ".npz")),
                          output=str(directory / "transfer" / ("FIELDd-" + odb.stem + ".npz")))
            validate_record(record, archive_root)
            record["source_stat"] = {key: file_stat(record[key]) for key in ("odb", "inp", "field")}
            records.append(record)
        if records:
            batches.append(dict(archive_job_id=directory.parent.name, case=case, records=records))
    expected_ids = {0, 1, 4501, 8901} if preflight else set(range(9001))
    for task in ("UT", "FT"):
        require({i for t, i in identities if t == task} == expected_ids,
                "Incomplete/unexpected " + task + " FCC sample inventory")
    if preflight:
        batches = [dict(archive_job_id="preflight", case="preflight",
                        records=[record for batch in batches for record in batch["records"]])]
    return dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                preflight=preflight, code_sha256=code_fingerprints(), batches=batches,
                records=sum(len(batch["records"]) for batch in batches),
                frame_alignment="saved_frame_index_order; retain actual per-specimen times; no interpolation")


def load_manifest(path, archive_root=ARCHIVE):
    manifest = json.loads(Path(path).read_text())
    require(manifest["schema_version"] == SCHEMA, "Unsupported batch manifest")
    require(manifest["code_sha256"] == code_fingerprints(), "Manifest/code fingerprint mismatch")
    seen = set()
    for batch in manifest["batches"]:
        require(batch["records"], "Empty archive-job batch")
        for record in batch["records"]:
            validate_record(record, archive_root)
            require(record["archive_job_id"] == batch["archive_job_id"] or
                    (manifest["preflight"] and batch["case"] == "preflight"), "Mixed archive jobs in one batch")
            identity = (record["task"], record["simulation_id"])
            require(identity not in seen, "Duplicate manifest sample")
            seen.add(identity)
    require(len(seen) == manifest["records"], "Manifest record count mismatch")
    return manifest


def check_source_stats(record):
    for key in ("odb", "inp", "field"):
        require(file_stat(record[key]) == record["source_stat"][key], "Source changed since inventory: " + record[key])


def array_header(npz_path, name):
    """Read array shape without loading the (potentially large) displacement array."""
    import numpy as np
    with zipfile.ZipFile(str(npz_path)) as archive:
        with archive.open(name + ".npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
    require(not dtype.hasobject, "Object/pickle array in FIELDu")
    return shape


def validate_field_alignment(record, arrays):
    import numpy as np
    from resources.damage import read_beam_inp
    with np.load(record["field"], allow_pickle=False) as field:
        for key, expected in (("sample_id", record["simulation_id"]), ("mode", record["task"])):
            require(str(np.asarray(field[key]).reshape(-1)[0]) == str(expected), "FIELDu " + key + " mismatch")
        require(str(np.asarray(field["sample_stem"]).reshape(-1)[0]) == record["source_stem"], "FIELDu stem mismatch")
        labels = np.asarray(field["node_labels"], dtype=np.int64)
        coords = np.asarray(field["node_coords"])
        times = np.asarray(field["frame_values"], dtype=float)
    require(len(labels) == len(set(labels.tolist())), "Duplicate FIELDu node labels")
    # Raw UT FIELDu also contains original grip nodes. Validate against the full
    # INP instance, while fracture targets remain restricted to the selected body.
    instances = np.unique(arrays["element_instance"])
    require(len(instances) == 1, "FCC field validation requires one original instance")
    original_mesh = read_beam_inp(record["inp"])
    mesh = {name.upper(): values for name, values in original_mesh.items()}
    require(len(mesh) == len(original_mesh), "Ambiguous INP instance names after Abaqus case normalization")
    require(str(instances[0]).upper() in mesh, "FIELDu/INP instance mismatch")
    lookup = mesh[str(instances[0]).upper()]["nodes"]
    require(set(labels).issubset(lookup), "FIELDu node labels missing from full INP instance")
    require(set(np.asarray(arrays["endpoint_node_label"]).reshape(-1)).issubset(set(labels)),
            "Native parent endpoints missing from raw FIELDu original nodes")
    expected = np.array([lookup[int(label)] for label in labels])
    tolerance = max(1e-6, float(np.finfo(coords.dtype).eps) * max(1., float(np.abs(expected).max())) * 4)
    require(coords.shape == expected.shape and np.allclose(coords, expected, atol=tolerance, rtol=0),
            "FIELDu/INP coordinates mismatch by exact node label")
    shape = array_header(record["field"], "Y")
    require(shape == (len(times), len(labels), 2), "Unexpected FIELDu fnc shape")
    require(times.shape == np.asarray(arrays["frame_time"]).shape and times[0] == 0 and
            np.isfinite(times).all() and (np.diff(times) > 0).all(), "FIELDu/STATUS saved frame layout mismatch")
    return dict(rule="saved_frame_index_order", frames=len(times), field_nodes=len(labels),
                field_nodes_outside_selected_body=int((~np.isin(labels, arrays["node_label"])).sum()),
                node_matching="exact original INP labels; coordinate tolerance for FIELDu floating dtype",
                maximum_coordinate_error_mm=float(np.abs(coords - expected).max()),
                coordinate_tolerance_mm=tolerance,
                maximum_time_difference_s=float(np.abs(times - arrays["frame_time"]).max()),
                interpolation=False)


def load_valid_product(path):
    import numpy as np
    from resources.damage import validate_status_arrays
    with np.load(str(path), allow_pickle=False) as product:
        arrays = {key: product[key] for key in product.files}
    return arrays, validate_status_arrays(arrays)


def reusable_product(record, manifest):
    output = Path(record["output"])
    if not output.exists() and not output.with_suffix(".json").exists():
        return False
    arrays, summary = load_valid_product(output)
    metadata = json.loads(str(arrays["metadata_json"]))
    require(str(arrays["task"]) == record["task"] and int(arrays["simulation_id"]) == record["simulation_id"],
            "Existing FIELDd task/sample mismatch")
    batch = metadata["batch"]
    require(batch["code_sha256"] == manifest["code_sha256"], "Existing FIELDd code mismatch")
    require(metadata["code_sha256"] == {key: manifest["code_sha256"][key] for key in ("exporter", "damage_core")},
            "Existing FIELDd exporter fingerprint mismatch")
    require(metadata["odb_path"] == record["odb"] and metadata["inp_path"] == record["inp"],
            "Existing FIELDd source path mismatch")
    require(batch["archive_job_id"] == record["archive_job_id"], "Existing FIELDd archive job mismatch")
    require(metadata["source_sha256"] == {key: fingerprint(record[key]) for key in ("odb", "inp")},
            "Existing FIELDd source fingerprint mismatch")
    require(batch["source_field_sha256"] == fingerprint(record["field"]), "Existing FIELDd FIELDu mismatch")
    validate_field_alignment(record, arrays)
    sidecar = json.loads(output.with_suffix(".json").read_text())
    require(sidecar["npz_sha256"] == fingerprint(output), "Existing FIELDd JSON/NPZ checksum mismatch")
    require(sidecar["metadata"] == metadata and all(sidecar[key] == value for key, value in summary.items()),
            "Existing FIELDd sidecar validation mismatch")
    return True


def publish_product(record, staged, arrays, summary):
    """Publish only this new family; NPZ rename is the completion marker for the pair."""
    import numpy as np
    output = Path(record["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(str(arrays["metadata_json"]))
    np.savez_compressed(str(staged), **arrays)
    _, verified = load_valid_product(staged)
    require(verified == summary, "Batch NPZ roundtrip changed validation")
    temporary = output.with_name("." + output.name + "." + str(os.getpid()) + ".tmp")
    shutil.copyfile(str(staged), str(temporary))
    checksum = fingerprint(staged)
    require(fingerprint(temporary) == checksum, "Archive copy checksum mismatch")
    sidecar = dict(summary, metadata=metadata, output=str(output), npz_sha256=checksum)
    atomic_json(output.with_suffix(".json"), sidecar)
    os.replace(str(temporary), str(output))
    return checksum


def export_one(record, manifest, scratch, export_status, replace_invalid=False):
    import numpy as np
    check_source_stats(record)
    try:
        if reusable_product(record, manifest):
            return dict(state="skipped_valid", output=record["output"])
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        require(replace_invalid, "Existing FIELDd is not reusable; inspect or pass --replace-invalid: " + str(error))
    require(not scratch.exists(), "Record scratch path already exists")
    scratch.mkdir()
    copied = {}
    source_sha = {}
    for key in ("odb", "inp"):
        source = Path(record[key])
        source_sha[key] = fingerprint(source)
        copied[key] = scratch / source.name
        shutil.copy2(str(source), str(copied[key]))
        require(fingerprint(copied[key]) == source_sha[key], "Scratch source copy checksum mismatch")
    staged = scratch / Path(record["output"]).name
    export_status(copied["odb"], copied["inp"], record["task"], staged,
                  source_paths={key: record[key] for key in ("odb", "inp")})
    arrays, summary = load_valid_product(staged)
    metadata = json.loads(str(arrays["metadata_json"]))
    require(metadata["source_sha256"] == source_sha, "Exporter/source copy fingerprint mismatch")
    require(metadata["code_sha256"] == {key: manifest["code_sha256"][key] for key in ("exporter", "damage_core")},
            "Exporter/code manifest fingerprint mismatch")
    metadata["batch"] = dict(code_sha256=manifest["code_sha256"], archive_job_id=record["archive_job_id"],
                             source_field=record["field"], source_field_sha256=fingerprint(record["field"]),
                             field_alignment=validate_field_alignment(record, arrays))
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    check_source_stats(record)
    checksum = publish_product(record, staged, arrays, summary)
    shutil.rmtree(str(scratch))  # Only the exact fresh per-record directory created above.
    return dict(state="produced", output=record["output"], npz_sha256=checksum, validation=summary)


def run_batch(manifest_path, batch_index, scratch_root=None, results_dir=None, dry_run=False,
              replace_invalid=False, archive_root=ARCHIVE, manifest_sha256=None):
    manifest_path = Path(manifest_path).resolve()
    if manifest_sha256 is not None:
        require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) and
                fingerprint(manifest_path) == manifest_sha256, "Reviewed manifest SHA-256 mismatch")
    manifest = load_manifest(manifest_path, archive_root)
    require(0 <= batch_index < len(manifest["batches"]), "Batch index out of range")
    batch = manifest["batches"][batch_index]
    if dry_run:
        for record in batch["records"]:
            check_source_stats(record)
        return dict(state="dry_run", batch_index=batch_index, archive_job_id=batch["archive_job_id"],
                    records=len(batch["records"]), outputs=[record["output"] for record in batch["records"]])
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit(), "Damage processing requires a Slurm compute job")
    user = os.environ.get("HPC_USER", os.environ.get("USER", "exy053"))
    root = Path(scratch_root or "/gpfs/scratch/{}/{}".format(user, job)).resolve()
    require(root != Path(root.anchor) and root.name == job, "Scratch root must end in exact Slurm job ID")
    scratch = root / ("damage-batch-" + str(batch_index))
    require(not scratch.exists(), "Batch scratch already exists; retain/inspect before explicit rerun")
    scratch.mkdir(parents=True)
    exporter_path = Path(__file__).with_name("A2_DamageOUTpostProcess.py")
    spec = importlib.util.spec_from_file_location("damage_status_exporter", str(exporter_path))
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    result = dict(schema_version=SCHEMA, manifest_sha256=fingerprint(manifest_path),
                  code_sha256=manifest["code_sha256"], batch_index=batch_index,
                  archive_job_id=batch["archive_job_id"], slurm_job_id=job,
                  scratch=str(scratch), records=[])
    results_dir = Path(results_dir) if results_dir else manifest_path.parent / "results"
    result_path = results_dir / ("{}-batch-{}.json".format(job, batch_index))
    for index, record in enumerate(batch["records"]):
        started = time.monotonic()
        entry = dict(task=record["task"], simulation_id=record["simulation_id"], output=record["output"])
        try:
            entry.update(export_one(record, manifest, scratch / ("record-" + str(index)),
                                    exporter.export_status, replace_invalid))
        except Exception as error:
            entry.update(state="failed", error=type(error).__name__ + ": " + str(error))
        entry["elapsed_seconds"] = time.monotonic() - started
        result["records"].append(entry)
        result["failed"] = sum(entry["state"] == "failed" for entry in result["records"])
        atomic_json(result_path, result)
        print(json.dumps(entry, sort_keys=True), flush=True)
    if result["failed"] == 0:
        require(scratch == root / ("damage-batch-" + str(batch_index)), "Unsafe scratch cleanup")
        scratch.rmdir()  # All successful per-record directories were already removed.
    result["state"] = "failed" if result["failed"] else "complete"
    atomic_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Inventory reviewed FCC roots; never opens ODBs")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--preflight", action="store_true")
    run = commands.add_parser("run", help="Run one manifest archive-job batch with Abaqus Python")
    run.add_argument("--manifest", required=True)
    run.add_argument("--manifest-sha256", help="Expected SHA-256 of the reviewed immutable manifest")
    run.add_argument("--batch-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    run.add_argument("--scratch-root")
    run.add_argument("--results-dir")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--replace-invalid", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        require(not Path(args.output).exists(), "Manifest exists; choose a fresh reviewed filename")
        manifest = prepare_manifest(preflight=args.preflight)
        atomic_json(args.output, manifest)
        print(json.dumps(dict(manifest=args.output, batches=len(manifest["batches"]), records=manifest["records"])))
    else:
        result = run_batch(args.manifest, args.batch_index, args.scratch_root, args.results_dir,
                           args.dry_run, args.replace_invalid, manifest_sha256=args.manifest_sha256)
        print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
        return int(result["state"] == "failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
