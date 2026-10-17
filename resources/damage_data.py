"""Build additive native-strut fracture banks aligned to existing MLdata fields.

Only field identities, geometry and array headers are read: displacement targets
and their large validity arrays are never loaded. FIELDd sources remain separate
from FIELDu, and unknown binary targets always have a false validity mask.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from resources.damage import require, validate_status_arrays


BANKS = (("UT", "UT"), ("FT", "FT"), ("MULTI", "UT"), ("MULTI", "FT"))
SAMPLE_KEYS = ("D", "valid_mask", "sample_ids", "sample_stems", "sample_node_coords",
               "frame_time", "frame_present", "sample_frame_count", "first_failure_frame", "source_damage_files",
               "source_damage_sha256", "source_inp_sha256", "source_odb_sha256")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _header(path, key):
    with zipfile.ZipFile(path) as archive, archive.open(key+".npy") as stream:
        version = np.lib.format.read_magic(stream)
        return np.lib.format._read_array_header(stream, version)


def read_field_layout(path):
    """Read saved membership and geometry without materialising Y/valid_mask."""
    path = Path(path)
    shape, _, dtype = _header(path, "Y")
    require(len(shape) == 4 and not dtype.hasobject, "Invalid existing field target header")
    with np.load(path, allow_pickle=False) as source:
        layout = {key: source[key] for key in ("sample_ids", "sample_stems", "node_coords",
                                                "sample_node_coords", "frame_values")}
    ids = np.asarray(layout["sample_ids"]).astype(str)
    require(all(str(int(v)) == v and int(v) >= 0 for v in ids), "Invalid field simulation IDs")
    require(len(ids) == len(set(ids)) == shape[0] and "0" in ids,
            "Field IDs must be unique and retain periodic reference 0")
    layout["sample_ids"] = ids
    require(layout["sample_stems"].shape == (shape[0],), "Invalid field source identities")
    require(layout["node_coords"].shape == (shape[2], 2) and
            layout["sample_node_coords"].shape == (shape[0], shape[2], 2) and
            np.isfinite(layout["node_coords"]).all() and np.isfinite(layout["sample_node_coords"]).all(),
            "Invalid existing field geometry")
    time = layout["frame_values"]
    require(time.shape == (shape[1],) and np.isfinite(time).all() and time[0] == 0 and
            (np.diff(time) > 0).all(), "Invalid field frame order or missing initial frame")
    require(len(set(map(tuple, layout["node_coords"]))) == shape[2], "Duplicate ML reference nodes")
    stat = path.stat()
    layout["provenance"] = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                Y_shape=list(shape), Y_dtype=dtype.str,
                                member_sha256={k: _array_hash(v) for k, v in layout.items()})
    return layout


def index_damage_sources(transfer_roots=(), inventory=None, required_identities=None, audit=None):
    """Index authoritative product IDs, optionally scoped to existing membership.

    An excluded manifest record is recorded without requiring its product to
    exist. Every included identity is still mandatory at the bank-build boundary.
    This does not assess whether the separate raw-corpus processing completed.
    """
    required = None if required_identities is None else {(task, str(int(sample_id)))
                                                        for task, sample_id in required_identities}
    paths, declared, manifest_ids, excluded = [], {}, set(), []
    if inventory is not None:
        manifest = json.loads(Path(inventory).read_text())
        require(manifest["schema_version"] == "fcc_status_batch_v1", "Unsupported batch manifest")
        for batch in manifest["batches"]:
            for record in batch["records"]:
                path = Path(record["output"])
                require(path not in declared, "Duplicate manifest product path")
                identity = record["task"], int(record["simulation_id"]), record["source_stem"]
                key = identity[0], str(identity[1])
                require(identity[0] in {"UT", "FT"} and identity[1] >= 0, "Invalid manifest identity")
                require(key not in manifest_ids, "Duplicate manifest task/simulation ID")
                manifest_ids.add(key)
                declared[path] = identity
                if required is not None and key not in required:
                    excluded.append(dict(task=identity[0], simulation_id=identity[1], source_stem=identity[2],
                                         output=str(path), product_exists=path.is_file()))
                else:
                    paths.append(path)
    for root in transfer_roots:
        root = Path(root)
        require(root.is_dir(), "Missing explicit transfer root: "+str(root))
        paths.extend(sorted(root.rglob("FIELDd-*.npz")))
    require(paths or excluded, "No explicit FIELDd products found")
    index = {}
    for path in paths:
        if path in declared and required is not None and (declared[path][0], str(declared[path][1])) not in required:
            continue
        require(path.is_file(), "Missing FIELDd product: "+str(path))
        with np.load(path, allow_pickle=False) as source:
            require(str(source["schema_version"]) == "strut_status_v1", "Wrong FIELDd schema: "+str(path))
            task, sample_id = str(source["task"]), int(source["simulation_id"])
            meta = json.loads(str(source["metadata_json"]))
        identity = task, sample_id, meta["source_stem"]
        require(task in {"UT", "FT"} and sample_id >= 0 and
                meta["task"] == task and meta["simulation_id"] == sample_id, "FIELDd identity mismatch")
        require(path.name == "FIELDd-"+identity[2]+".npz", "FIELDd filename/source mismatch")
        require(path not in declared or declared[path] == identity, "Manifest/FIELDd identity mismatch")
        key = task, str(sample_id)
        if required is not None and key not in required:
            continue
        require(key not in index, "Duplicate FIELDd task/simulation ID: "+str(key))
        index[key] = dict(path=path, stem=identity[2], stat=(path.stat().st_size, path.stat().st_mtime_ns))
    if audit is not None:
        audit.update(required_identity_count=None if required is None else len(required),
                     indexed_product_count=len(index), manifest_record_count=len(manifest_ids),
                     excluded_manifest_record_count=len(excluded), excluded_manifest_records=excluded,
                     excluded_manifest_ids_by_task={task: sorted(record["simulation_id"] for record in excluded
                                                                if record["task"] == task) for task in ("UT", "FT")},
                     membership_policy="exact existing allFIELD IDs; no new filtering or sample removal",
                     raw_corpus_completion="not assessed by the MLdata bank builder")
    return index


def _edge_mapping(source, layout, tolerance):
    reference = np.asarray(source["canonical_xy"])
    ml_reference = layout["node_coords"]
    delta = np.abs(reference[:, None]-ml_reference[None, :]).max(axis=2)
    matches = delta <= tolerance
    present = np.asarray(source["canonical_node_present"])
    require((matches.sum(axis=1)[present] == 1).all() and
            (matches.sum(axis=1)[~present] == 0).all() and (matches.sum(axis=0) == 1).all(),
            "Native canonical nodes do not uniquely match saved ML field nodes")
    remap = np.full(len(reference), -1, np.int32)
    remap[present] = np.argmax(matches[present], axis=1)
    edge_index = remap[source["canonical_endpoint"]]
    require((edge_index >= 0).all(), "Native edge touches absent ML node")
    return edge_index


def validate_damage_bank(arrays):
    """Check the additive bank schema independently of any model or DATA class."""
    require(str(arrays["schema_version"]) == "strut_status_bank_v1", "Unsupported fracture bank schema")
    require(all(not np.asarray(v).dtype.hasobject for v in arrays.values()), "Object arrays forbidden")
    task, mech = str(arrays["task"]), str(arrays["mech_mode"])
    require(task in {"UT", "FT"} and mech in {task, "MULTI"}, "Invalid fracture bank task")
    metadata = json.loads(str(arrays["metadata_json"]))
    require(metadata["schema_version"] == "strut_status_bank_v1" and metadata["task"] == task and
            metadata["mech_mode"] == mech, "Fracture bank metadata identity mismatch")
    values, known = np.asarray(arrays["D"]), np.asarray(arrays["valid_mask"])
    require(values.ndim == 3 and values.dtype == np.uint8 and known.dtype.kind == "b" and
            known.shape == values.shape and np.isin(values, [0, 1]).all() and
            not values[~known].any(), "Invalid binary targets or unknown mask")
    n, f, s = values.shape
    ids = np.asarray(arrays["sample_ids"])
    require(ids.shape == (n,) and len(set(ids)) == n and "0" in ids,
            "Invalid bank sample identity or missing reference")
    counts, present = np.asarray(arrays["sample_frame_count"]), np.asarray(arrays["frame_present"])
    require(counts.shape == (n,) and counts.dtype.kind in "iu" and
            ((counts > 0) & (counts <= f)).all() and present.shape == (n, f) and
            present.dtype.kind == "b" and
            np.array_equal(present, np.arange(f)[None] < counts[:, None]),
            "Invalid sample frame prefix/count")
    times = np.asarray(arrays["frame_time"])
    require(times.shape == (n, f) and np.isfinite(times[present]).all() and
            np.isnan(times[~present]).all() and (times[:, 0] == 0).all() and
            (np.diff(times, axis=1)[present[:, 1:]] > 0).all(), "Invalid sample frame times")
    require(not known[~present].any(), "Unavailable sample frames must be entirely unknown")
    require(np.asarray(arrays["frame_values"]).shape == (f,) and
            np.array_equal(arrays["frame_index"], np.arange(f)), "Invalid bank frame axes")
    edge = np.asarray(arrays["edge_index"])
    coords = np.asarray(arrays["node_coords"])
    sample_coords = np.asarray(arrays["sample_node_coords"])
    require(edge.shape == (s, 2) and edge.dtype.kind in "iu" and (edge >= 0).all() and
            (edge < len(coords)).all() and sample_coords.shape == (n, len(coords), 2) and
            np.isfinite(sample_coords).all() and np.isfinite(coords).all(), "Invalid bank geometry")
    require(len(set(map(tuple, np.sort(edge, axis=1)))) == s and
            (edge[:, 0] != edge[:, 1]).all(), "Duplicate or closed native struts")
    require(np.array_equal(arrays["strut_id"], np.arange(s)) and
            np.asarray(arrays["strut_present"]).shape == (s,) and arrays["strut_present"].all(),
            "Invalid native strut axis")
    canonical_edge = np.asarray(arrays["canonical_endpoint"])
    canonical = np.asarray(arrays["canonical_xy"])
    require(canonical_edge.shape == (s, 2) and canonical_edge.dtype.kind in "iu" and
            (canonical_edge >= 0).all() and (canonical_edge < len(canonical)).all() and
            np.allclose(canonical[canonical_edge], coords[edge], rtol=0, atol=2e-5),
            "Native-to-ML endpoint mapping mismatch")
    first = np.asarray(arrays["first_failure_frame"])
    require(first.shape == (n, s) and first.dtype.kind in "iu", "Invalid first failure axes")
    expected = np.where(values.any(axis=1), values.argmax(axis=1), -1)
    require(np.array_equal(first, expected), "First failure does not match binary observations")
    for key in SAMPLE_KEYS:
        require(np.asarray(arrays[key]).shape[0] == n, "Invalid sample axis: "+key)
    for key in ("source_damage_sha256", "source_inp_sha256", "source_odb_sha256"):
        require(all(len(value) == 64 and set(str(value)) <= set("0123456789abcdef") for value in arrays[key]),
                "Invalid bank source fingerprints: "+key)
    return dict(samples=n, frames=f, struts=s, unknown_parent_frames=int((~known).sum()),
                unavailable_sample_frames=int((~present).sum()),
                unavailable_parent_frames=int((~present).sum())*s,
                samples_with_unavailable_trailing_frames=int((counts < f).sum()),
                observed_unknown_parent_frames=int(((~known) & present[:, :, None]).sum()),
                failed_parent_frames=int(values.sum()),
                failed_struts_per_frame=values.sum(axis=(0, 2), dtype=np.int64).tolist(),
                unknown_struts_per_frame=(~known).sum(axis=(0, 2), dtype=np.int64).tolist())


def _build_task(layout, task, index, tolerance):
    ids, stems = layout["sample_ids"], layout["sample_stems"]
    reference = None
    firsts, counts, files, hashes, inp_hashes, odb_hashes = [], [], [], [], [], []
    frames = len(layout["frame_values"])
    times = np.full((len(ids), frames), np.nan)
    for row, (sample_id, stem) in enumerate(zip(ids, stems)):
        record = index[(task, sample_id)]
        path = record["path"]
        require(record["stem"] == str(stem), "Field/FIELDd source stem mismatch: "+str(path))
        require(record["stat"] == (path.stat().st_size, path.stat().st_mtime_ns), "FIELDd changed after indexing")
        with np.load(path, allow_pickle=False) as saved:
            source = {key: saved[key] for key in saved.files}
        summary = validate_status_arrays(source)
        keep = summary["frames"]
        require(0 < keep <= frames, "Field/FIELDd frame count mismatch: "+str(path))
        require(str(source["task"]) == task and str(int(source["simulation_id"])) == sample_id,
                "FIELDd identity changed after indexing")
        if reference is None:
            edge = _edge_mapping(source, layout, tolerance)
            reference = {key: source[key] for key in ("canonical_endpoint", "canonical_xy", "strut_present", "strut_id")}
            reference["edge_index"] = edge
            values = np.zeros((len(ids), frames, summary["struts"]), np.uint8)
            known = np.zeros(values.shape, bool)
        else:
            require(all(np.array_equal(source[key], reference[key]) for key in
                        ("canonical_endpoint", "canonical_xy", "strut_present", "strut_id")),
                    "FIELDd native strut axes differ between samples")
        require(np.allclose(source["endpoint_xy0"], layout["sample_node_coords"][row][edge],
                            rtol=0, atol=tolerance), "Field/FIELDd initial sample geometry mismatch: "+str(path))
        values[row, :keep] = source["strut_failed"]
        known[row, :keep] = source["strut_failure_known"]
        times[row, :keep] = source["frame_time"]
        counts.append(keep)
        firsts.append(source["first_failure_frame"])
        meta = json.loads(str(source["metadata_json"]))
        files.append(str(path.resolve()))
        hashes.append(fingerprint(path))
        require(record["stat"] == (path.stat().st_size, path.stat().st_mtime_ns), "FIELDd changed while reading")
        inp_hashes.append(meta["source_sha256"]["inp"])
        odb_hashes.append(meta["source_sha256"]["odb"])
    reference.update(D=values, valid_mask=known, sample_ids=ids, sample_stems=stems,
                     node_coords=layout["node_coords"], sample_node_coords=layout["sample_node_coords"],
                     frame_values=layout["frame_values"], frame_time=times,
                     sample_frame_count=np.asarray(counts, np.int32),
                     frame_present=np.arange(frames)[None] < np.asarray(counts)[:, None],
                     frame_index=np.arange(values.shape[1], dtype=np.int32),
                     first_failure_frame=np.asarray(firsts, np.int32), source_damage_files=np.asarray(files),
                     source_damage_sha256=np.asarray(hashes), source_inp_sha256=np.asarray(inp_hashes),
                     source_odb_sha256=np.asarray(odb_hashes))
    return reference


def _subset_bank(base, layout, tolerance):
    positions = {value: i for i, value in enumerate(base["sample_ids"])}
    require(set(layout["sample_ids"]) <= positions.keys(), "MULTI contains samples missing from task field bank")
    rows = np.asarray([positions[v] for v in layout["sample_ids"]])
    require(np.array_equal(base["sample_stems"][rows], layout["sample_stems"]) and
            np.allclose(base["node_coords"], layout["node_coords"], rtol=0, atol=tolerance) and
            np.allclose(base["sample_node_coords"][rows], layout["sample_node_coords"], rtol=0, atol=tolerance) and
            len(base["frame_values"]) == len(layout["frame_values"]), "MULTI/task field layouts differ")
    arrays = {key: (value[rows] if key in SAMPLE_KEYS else value) for key, value in base.items()}
    arrays["frame_values"] = layout["frame_values"]
    return arrays


def build_damage_banks(mldata_dir, output_dir, transfer_roots=(), inventory=None, overwrite=False, tolerance=2e-5):
    """Stage and validate all four banks before promoting any final output.

    This prepares products only. Publishing/copying to local or HPC archives is
    deliberately left to the caller. Existing allFIELD files are never modified.
    """
    mldata_dir, output_dir = Path(mldata_dir), Path(output_dir)
    source_root, staging_root = mldata_dir.resolve(), output_dir.resolve()
    require(source_root not in (staging_root, *staging_root.parents) and
            all(part.casefold() != "mldata" for part in staging_root.parts),
            "Build bank NPZ/JSON in staging outside MLdata; publish only NPZ training inputs")
    require(tolerance > 0 and np.isfinite(tolerance), "Invalid geometry tolerance")
    targets = [output_dir/f"{mech}-{task}-disNodes-allD.npz" for mech, task in BANKS]
    require(overwrite or not any(p.exists() or p.with_suffix(".json").exists() for p in targets),
            "A fracture bank exists; use a new staging directory or explicit --overwrite")
    layouts = {(mech, task): read_field_layout(mldata_dir/f"{mech}-{task}-disNodes-allFIELD.npz")
               for mech, task in BANKS}
    require(np.array_equal(layouts[("MULTI", "UT")]["sample_ids"], layouts[("MULTI", "FT")]["sample_ids"]),
            "Paired field sample IDs differ")
    required = {(task, sample_id) for (_, task), layout in layouts.items() for sample_id in layout["sample_ids"]}
    source_scope = {}
    index = index_damage_sources(transfer_roots, inventory, required, source_scope)
    missing = sorted(required-index.keys())
    require(not missing, "Missing required FIELDd products: "+str(missing[:20]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pending, task_banks, reports = [], {}, []
    try:
        for (mech, task), target in zip(BANKS, targets):
            layout = layouts[(mech, task)]
            if mech == task:
                arrays = _build_task(layout, task, index, tolerance)
                task_banks[task] = arrays
            else:
                arrays = _subset_bank(task_banks[task], layout, tolerance)
            metadata = dict(schema_version="strut_status_bank_v1", task=task, mech_mode=mech,
                            failure_rule="any member whole-element STATUS=0; active only when all members known active",
                            unknown_rule="D storage is 0 with valid_mask=false; never supervise unknown entries",
                            frame_alignment="saved frame index/order; raw prefix retained; unavailable trailing frames masked; no interpolation or carry",
                            frame_presence="sample_frame_count counts the observed prefix; frame_present marks it; absent frame_time is NaN",
                            edge_axis="native task struts; edge_index remaps audit canonical endpoints to saved ML node order",
                            source_field_layout=layout["provenance"],
                            source_scope=source_scope,
                            builder_sha256=fingerprint(__file__), geometry_tolerance_mm=tolerance)
            arrays = dict(arrays, schema_version=np.array("strut_status_bank_v1"), task=np.array(task),
                          mech_mode=np.array(mech), metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
            summary = validate_damage_bank(arrays)
            with tempfile.NamedTemporaryFile(prefix="."+target.stem+"-", suffix=".npz", dir=output_dir,
                                             delete=False) as stream:
                temporary = Path(stream.name)
                pending.append((temporary, target))
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            with np.load(temporary, allow_pickle=False) as saved:
                require(validate_damage_bank(saved) == summary, "Fracture bank round trip failed")
                require(all(np.array_equal(saved[key], value, equal_nan=np.issubdtype(value.dtype, np.inexact))
                            for key, value in arrays.items()),
                        "Fracture bank array round trip differs")
            report = dict(output=str(target.resolve()), npz_sha256=fingerprint(temporary), **summary,
                          metadata=metadata, source_products=len(arrays["source_damage_files"]),
                          source_scope=source_scope,
                          sample_ids=arrays["sample_ids"].tolist(),
                          source_damage_sha256=arrays["source_damage_sha256"].tolist(),
                          sample_frame_count=arrays["sample_frame_count"].tolist(),
                          samples_with_unavailable_trailing_frame_ids=arrays["sample_ids"][
                              arrays["sample_frame_count"] < len(arrays["frame_values"])].tolist(),
                          final_failed_struts_per_sample=arrays["D"][:, -1].sum(axis=1).tolist(),
                          final_unknown_struts_per_sample=(~arrays["valid_mask"][:, -1]).sum(axis=1).tolist(),
                          last_observed_failed_struts_per_sample=arrays["D"][
                              np.arange(len(arrays["sample_ids"])), arrays["sample_frame_count"]-1].sum(axis=1).tolist(),
                          maximum_frame_time_offset_from_field_reference=float(
                              np.nanmax(np.abs(arrays["frame_time"]-arrays["frame_values"][None]))))
            with tempfile.NamedTemporaryFile(prefix="."+target.stem+"-", suffix=".json", dir=output_dir,
                                             delete=False, mode="w") as stream:
                temporary_json = Path(stream.name)
                pending.append((temporary_json, target.with_suffix(".json")))
                json.dump(report, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            reports.append(report)
        for temporary, target in pending:
            require(overwrite or not target.exists(), "Output appeared while building: "+str(target))
        for temporary, target in pending:
            os.replace(temporary, target)
        return reports
    finally:
        for temporary, _ in pending:
            if temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mldata-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Explicit staging destination; no archive publishing")
    parser.add_argument("--transfer-root", action="append", default=[], help="Explicit periodic/disordered transfer root")
    parser.add_argument("--inventory", help="fcc_status_batch_v1 manifest containing exact FIELDd output paths")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reports = build_damage_banks(args.mldata_dir, args.output_dir, args.transfer_root, args.inventory, args.overwrite)
    print(json.dumps([{k: report[k] for k in ("output", "npz_sha256", "samples", "frames", "struts",
                                            "unknown_parent_frames", "failed_parent_frames")} for report in reports], indent=2))


if __name__ == "__main__":
    main()
