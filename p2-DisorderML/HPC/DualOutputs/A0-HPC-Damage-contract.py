#!/usr/bin/env python3
"""Bounded saved-fixture replay, not a damage trainer or ODB extractor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from resources.damage import fcc_map, require, validate_damage_arrays
from resources.MLfield import reference_field_edges


CASES = (
    ("UT-periodic-damage", "UT", "damage-audit", "Ductile-FCC-20-per-1", 83, 83),
    ("FT-periodic-damage", "FT", "damage-audit", "Fracture-FCC-20-per-1", 7, 7),
    ("UT-397-maxD", "UT", "damage-disordered", "Ductile-FCC-20-20disNodes-lhs-all-397-maxD", 95, 97),
    ("UT-747-maxB", "UT", "damage-disordered", "Ductile-FCC-20-20disNodes-lhs-all-747-maxB", 92, 93),
    ("FT-747-maxB", "FT", "damage-disordered", "Fracture-FCC-20-20disNodes-lhs-all-747-maxB", 9, 9),
    ("FT-747-maxK", "FT", "damage-disordered", "Fracture-FCC-20-20disNodes-lhs-all-747-maxK", 9, 9),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def prepare(samples, inps, consumer, output):
    """Copy only the six authorised NPZ/INP pairs and one consumer descriptor."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "cases": [], "files": {}}
    for label, task, folder, stem, failed, elements in CASES:
        product = samples/folder/(label+".npz")
        source = inps/(stem+".inp")
        meta = json.loads(str(arrays(product)["metadata_json"]))
        require(meta["task"] == task and meta["source_stem"] == stem, "Wrong fixture identity")
        require(digest(source) == meta["source_sha256"]["inp"], "Original INP fingerprint differs")
        for path in (product, source):
            shutil.copy2(path, output/path.name)
            require(digest(path) == digest(output/path.name), "Fixture copy differs")
            manifest["files"][path.name] = digest(path)
        manifest["cases"].append(dict(label=label, task=task, npz=product.name,
            inp=source.name, failed_struts=failed, failed_elements=elements))
    shutil.copy2(consumer, output/"consumer-data.json")
    manifest["files"]["consumer-data.json"] = digest(consumer)
    manifest["consumer"] = "consumer-data.json"
    path = output/"manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared six fixtures; manifest SHA256 {digest(path)}", flush=True)


def consumer_layout(a, consumer, task):
    """Exercise layout only: no specimen join, prediction evaluation or time interpolation."""
    coords = np.asarray(consumer["metadata"]["canonical_coords"])
    lookup = {tuple(xy): i for i, xy in enumerate(coords)}
    require(len(lookup) == len(coords) == len(a["canonical_xy"]), "Duplicate/missing reference coordinate")
    require(all(tuple(xy) in lookup for xy in a["canonical_xy"]), "Consumer reference grid differs")
    mapping = np.array([lookup[tuple(xy)] for xy in a["canonical_xy"]])
    present = np.zeros(len(coords), bool)
    present[mapping] = a["canonical_node_present"]
    require(np.array_equal(present, consumer["node_masks"][task]), "Consumer presence mask differs")
    endpoints = mapping[a["canonical_endpoint"]]
    require(present[endpoints].all(), "Edge touches an initially absent node")
    expected = reference_field_edges(coords, task, present)
    require({tuple(sorted(e)) for e in endpoints} == {tuple(sorted(e)) for e in expected},
            "Remapped initial connectivity differs")
    times = a["frame_time"]
    require(times[0] == 0 and (np.diff(times) > 0).all(), "Invalid actual recorded frames")
    shared_times = np.asarray(consumer["metadata"]["field_frame_values"][task])
    require(len(times)-1 == len(shared_times), "Frame counts differ after removing only frame0")
    require(a["strut_failed"].shape == a["strut_failure_known"].shape == (len(times), len(endpoints)),
            "Binary target/mask axes differ")
    return dict(endpoint_index=endpoints, node_present=present,
        failed=a["strut_failed"][1:].astype(bool), known=a["strut_failure_known"][1:].astype(bool),
        frame_time=times[1:], consumer_frame_time=shared_times, audit_to_consumer=mapping)


def replay(manifest_path, manifest_hash, output, require_cuda=False):
    require(digest(manifest_path) == manifest_hash, "Manifest fingerprint differs")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["version"] == 1, "Unsupported fixture manifest")
    require([c["label"] for c in manifest["cases"]] == [c[0] for c in CASES], "Expected exactly six fixtures")
    expected_files = {"consumer-data.json"} | {c[0]+".npz" for c in CASES} | {c[3]+".inp" for c in CASES}
    require(set(manifest["files"]) == expected_files and manifest["consumer"] == "consumer-data.json",
            "Only the six explicit pairs and one descriptor may be staged")
    output.mkdir(parents=True, exist_ok=False)
    staged = output/"inputs"
    staged.mkdir()
    for name, sha in manifest["files"].items():
        require(Path(name).name == name, "Fixture path must be a basename")
        source = manifest_path.parent/name
        require(digest(source) == sha, "Source fixture differs: "+name)
        shutil.copy2(source, staged/name)
        require(digest(staged/name) == sha, "Staged fixture differs: "+name)
    consumer = json.loads((staged/manifest["consumer"]).read_text())
    import torch
    require(not require_cuda or torch.cuda.is_available(), "GPU replay requires CUDA")
    device = "cuda" if require_cuda else "cpu"
    rows, fingerprints = [], []
    for item, spec in zip(manifest["cases"], CASES):
        label, task, _, stem, failed, elements = spec
        require((item["task"], item["failed_struts"], item["failed_elements"]) == (task, failed, elements),
                "Manifest expected values differ")
        a = arrays(staged/item["npz"])
        meta = json.loads(str(a["metadata_json"]))
        require(meta["task"] == task and meta["source_stem"] == stem, "Fixture provenance differs")
        require(str(a["task"]) == task and int(a["simulation_id"]) == meta["simulation_id"],
                "Saved identity arrays differ from provenance")
        require(digest(staged/item["inp"]) == meta["source_sha256"]["inp"], "INP source mismatch")
        summary = validate_damage_arrays(a)
        summary["legacy_severity_assigned"] = summary.pop("valid_labels")
        require((summary["final_failed_struts"], summary["final_failed_elements"]) == (failed, elements),
                "Failure counts differ")
        geometry, _, _ = fcc_map(staged/item["inp"], task)
        for key, value in geometry.items():
            if value.dtype.kind == "f":
                require(np.allclose(value, a[key], rtol=0, atol=1e-10), "INP geometry differs: "+key)
            else:
                require(np.array_equal(value, a[key]), "INP identity differs: "+key)
        parents = a["parent_strut_index"]
        counts = np.bincount(parents)
        for frame, status in enumerate(a["status"]):
            observed_failed = np.bincount(parents, weights=status == 0) > 0
            known = observed_failed | (np.bincount(parents, weights=np.isfinite(status)) == counts)
            require(np.array_equal(observed_failed, a["strut_failed"][frame]), "STATUS parent rule differs")
            require(np.array_equal(known, a["strut_failure_known"][frame]), "STATUS known mask differs")
        layout = consumer_layout(a, consumer, task)
        path = output/(label+"-consumer.npz")
        np.savez(path, **layout)
        loaded = arrays(path)
        require(all(np.array_equal(v, loaded[k]) for k, v in layout.items()), "NPZ round trip differs")
        tensors = {k: torch.as_tensor(v.copy(), device=device) for k, v in loaded.items()}
        torch.save(tensors, output/(label+"-consumer.pt"))
        restored = torch.load(output/(label+"-consumer.pt"), map_location="cpu", weights_only=True)
        require(all(np.array_equal(v, restored[k].numpy()) for k, v in loaded.items()), "Torch round trip differs")
        content = hashlib.sha256()
        for key in sorted(set(a)-{"metadata_json"}):
            content.update(key.encode()+a[key].dtype.str.encode()+str(a[key].shape).encode()+a[key].tobytes())
        fingerprints.append(content.hexdigest())
        row = dict(label=label, **summary, target_known=int(a["strut_failure_known"].sum()),
            exact_consumer_times=bool(np.array_equal(layout["frame_time"], layout["consumer_frame_time"])),
            max_consumer_time_difference_s=float(np.abs(layout["frame_time"]-layout["consumer_frame_time"]).max()),
            remapped_nodes=int((layout["audit_to_consumer"] != np.arange(len(layout["audit_to_consumer"]))).sum()))
        rows.append(row)
        print("PASS "+json.dumps(row), flush=True)
    require(len(set(fingerprints)) == 5 and fingerprints[4] == fingerprints[5], "Expected FT duplicate changed")
    report = dict(kind="damage-contract-replay", source_revision=os.environ.get("ML_SOURCE_REVISION"),
        manifest_sha256=manifest_hash, device=device, cases=rows, unique_task_responses=5,
        specimen_join=False, prediction_evaluation=False, training=False, live_odb_replay=False,
        frame_note="Only frame0 removed for layout replay. Actual times retained; shared-time differences do not authorise binary interpolation or specimen joins.")
    (output/"contract-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DAMAGE_CONTRACT_REPORT="+json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--samples-root", type=Path)
    parser.add_argument("--inp-root", type=Path)
    parser.add_argument("--consumer-data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-label", default="damage-contract-preflight")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if any(x is None for x in (args.samples_root, args.inp_root, args.consumer_data, args.output_dir)):
            parser.error("Preparation requires explicit samples, INP, consumer and NEW output paths")
        prepare(args.samples_root, args.inp_root, args.consumer_data, args.output_dir)
    else:
        if args.manifest is None or not args.manifest_sha256:
            parser.error("Replay requires the explicit manifest and its pinned SHA256")
        output = args.output_dir or Path(os.environ["ML_RUN_ROOT"])/args.run_label
        replay(args.manifest, args.manifest_sha256, output, args.require_cuda)


if __name__ == "__main__":
    main()
