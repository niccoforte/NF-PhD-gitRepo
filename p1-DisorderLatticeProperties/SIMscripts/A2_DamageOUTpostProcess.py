"""Opt-in, explicit one-ODB strut export; never discovers files or upgrades ODBs.

Run from the repository root with Abaqus Python; see README for the runpy command.
The legacy A1/A2 positional contract and FIELDu/curve exporters are unchanged.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from resources.damage import periodic_fcc_map, reduce_damage, require, validate_damage_arrays


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_key(value):
    section = value.sectionPoint
    return (value.instance.name, value.elementLabel, str(value.position),
            value.integrationPoint if value.integrationPoint is not None else -1,
            section.number if section else -1, section.description if section else "")


def nodal_component(frame, assembly, set_name, variable, component, reduction):
    field = frame.fieldOutputs[variable]
    values = field.getSubset(region=assembly.nodeSets[set_name]).values
    require(len(values) > 0, "Empty loading node set: " + set_name)
    index = list(field.componentLabels).index(component)
    numbers = np.array([v.data[index] for v in values], float)
    require(np.isfinite(numbers).all(), "Missing loading value")
    return float(numbers.mean() if reduction == "mean" else numbers.sum())


def export_damage(odb_path, inp_path, task, output, partial_rule, step_name="Step-1", carry_failure=False):
    from odbAccess import openOdb
    odb_path, inp_path, output = map(lambda p: Path(p).resolve(), (odb_path, inp_path, output))
    require(odb_path.stem == inp_path.stem and "-per-" in inp_path.stem,
            "This validated profile accepts only matching periodic ODB/INP stems")
    require(output.suffix == ".npz" and output not in {odb_path, inp_path}, "Invalid output path")
    require(task == ("UT" if inp_path.stem.startswith("Ductile-") else "FT"), "Task/stem mismatch")
    before = dict(inp=fingerprint(inp_path), odb=fingerprint(odb_path))
    arrays, mesh, mapping = periodic_fcc_map(inp_path, task)
    metadata = dict(schema_version="strut_damage_v1", task=task, simulation_id=0,
                    source_stem=inp_path.stem, inp_path=str(inp_path), odb_path=str(odb_path),
                    source_sha256=before, mapping=mapping, profile="periodic_fcc20_body_v1",
                    canonical_order="audit-only periodic 800 coordinates sorted (y,x), not saved ML order",
                    coordinate_units="mm", time_units="s", force_units="N", unit_basis="Ti producer convention",
                    partial_rule=partial_rule, point_rule="max_observed_section_SDEG_with_complete_coverage",
                    failure_rule="whole_element_STATUS_zero_v1", carry_failure=carry_failure,
                    requested_initiation_variable="DMICRT", actual_initiation_variable="DUCTCRT",
                    scope="one explicitly selected periodic ODB; UT main body excludes grips",
                    coverage_definition="strut_coverage: fraction assigned member D; strut_observation_coverage: fraction members with raw STATUS and all expected reported SDEG points",
                    code_sha256={"exporter":fingerprint(__file__),
                                 "damage_core":fingerprint(Path(__file__).resolve().parents[2]/"resources"/"damage.py")},
                    extraction_utc=datetime.now(timezone.utc).isoformat())
    element_lookup = {int(e): i for i, e in enumerate(arrays["element_label"])}
    node_lookup = {int(n): i for i, n in enumerate(arrays["node_label"])}
    odb = openOdb(str(odb_path), readOnly=True)
    try:
        require(mapping["instance"] in odb.rootAssembly.instances, "INP/ODB instance mismatch")
        inst = odb.rootAssembly.instances[mapping["instance"]]
        odb_elements = {e.label: tuple(e.connectivity) for e in inst.elements}
        odb_nodes = {n.label: np.array(n.coordinates[:2]) for n in inst.nodes}
        require(set(odb_elements) == set(mesh["elements"]), "INP/ODB full element ID mismatch")
        require(all(odb_elements[e] == conn for e, conn in mesh["elements"].items()), "INP/ODB connectivity mismatch")
        require(set(odb_nodes) == set(mesh["nodes"]), "INP/ODB node ID mismatch")
        coordinate_error = max(np.linalg.norm(odb_nodes[n] - xy) for n, xy in mesh["nodes"].items())
        require(coordinate_error <= mapping["tolerance"], "INP/ODB initial coordinate mismatch")
        require({e.label for e in inst.elementSets[mapping["selected_set"]].elements} == set(element_lookup),
                "INP/ODB body selection mismatch")
        metadata.update(odb_version=str(odb.jobData.version), odb_initial_coordinate_error=float(coordinate_error))
        require(step_name in odb.steps, "Explicit step not found")
        step = odb.steps[step_name]
        frames = step.frames
        require(len(frames) >= 2 and frames[0].frameValue == 0, "Initial frame missing")
        first = frames[0].fieldOutputs["SDEG"]
        point_keys = sorted(point_key(v) for v in first.values
                            if v.instance.name == mapping["instance"] and v.elementLabel in element_lookup)
        require(len(set(point_keys)) == len(point_keys), "Duplicate SDEG point identity")
        require(point_keys and {k[1] for k in point_keys} == set(element_lookup), "Incomplete initial point inventory")
        point_lookup = {key: i for i, key in enumerate(point_keys)}
        f, e, p = len(frames), len(element_lookup), len(point_keys)
        arrays.update(sdeg=np.full((f, p), np.nan), ductcrt=np.full((f, p), np.nan),
                      status=np.full((f, e), np.nan), node_displacement=np.full((f, len(node_lookup), 2), np.nan),
                      global_loading=np.full(f, np.nan), load_displacement=np.full(f, np.nan),
                      reaction_force=np.full(f, np.nan), body_strain=np.full(f, np.nan),
                      frame_time=np.array([v.frameValue for v in frames]),
                      frame_increment=np.array([v.incrementNumber for v in frames]),
                      point_element_index=np.array([element_lookup[k[1]] for k in point_keys]),
                      point_instance=np.array([k[0] for k in point_keys]), point_position=np.array([k[2] for k in point_keys]),
                      point_integration=np.array([k[3] for k in point_keys]), point_section=np.array([k[4] for k in point_keys]),
                      point_section_description=np.array([k[5] for k in point_keys]))
        height = float(np.ptp(arrays["endpoint_xy0"][:, :, 1]))
        require(height > 0, "Invalid initial body height")
        metadata.update(step=step_name, body_height=height,
            loading_definition="mean SET-TOP U2 / initial body H" if task == "UT" else "LOAD reference-point U2",
            loading_units="dimensionless" if task == "UT" else "mm",
            body_strain_definition="(mean SET-TOPBODY U2 - mean SET-BOTTOMBODY U2)/initial body H" if task == "UT" else "not defined",
            field_definitions={k:dict(description=v.description, components=list(v.componentLabels))
                               for k, v in frames[0].fieldOutputs.items()},
            section_controls=[line for line in inp_path.read_text().splitlines() if line.lower().startswith("*section controls")])
        for j, frame in enumerate(frames):
            for field_name, array_name in (("SDEG", "sdeg"), ("DUCTCRT", "ductcrt")):
                seen = set()
                if field_name not in frame.fieldOutputs:
                    continue  # missing remains NaN
                for value in frame.fieldOutputs[field_name].values:
                    if value.instance.name != mapping["instance"] or value.elementLabel not in element_lookup:
                        continue
                    key = point_key(value)
                    require(key in point_lookup and key not in seen, "Changed/duplicate point layout")
                    require(str(value.position) == "INTEGRATION_POINT", "Unexpected SDEG/initiation output position")
                    arrays[array_name][j, point_lookup[key]] = float(value.data)
                    seen.add(key)
            seen = set()
            if "STATUS" in frame.fieldOutputs:
                for value in frame.fieldOutputs["STATUS"].values:
                    if value.instance.name != mapping["instance"] or value.elementLabel not in element_lookup:
                        continue
                    i = element_lookup[value.elementLabel]
                    require(str(value.position) == "WHOLE_ELEMENT" and i not in seen, "Invalid STATUS layout")
                    arrays["status"][j, i] = float(value.data)
                    seen.add(i)
            displacement_key = "UT" if "UT" in frame.fieldOutputs else "U"
            field = frame.fieldOutputs[displacement_key]
            component_indices = [list(field.componentLabels).index(k) for k in ("U1", "U2")]
            for value in field.values:
                if value.instance is not None and value.instance.name == mapping["instance"] and value.nodeLabel in node_lookup:
                    arrays["node_displacement"][j, node_lookup[value.nodeLabel]] = np.array(value.data)[component_indices]
            assembly = odb.rootAssembly
            loading_set = "SET-TOP" if task == "UT" else "LOAD"
            displacement = nodal_component(frame, assembly, loading_set, displacement_key, "U2", "mean")
            arrays["load_displacement"][j] = displacement
            arrays["global_loading"][j] = displacement / height if task == "UT" else displacement
            arrays["reaction_force"][j] = nodal_component(frame, assembly, loading_set, "RF", "RF2", "sum")
            if task == "UT":
                top = nodal_component(frame, assembly, "SET-TOPBODY", displacement_key, "U2", "mean")
                bottom = nodal_component(frame, assembly, "SET-BOTTOMBODY", displacement_key, "U2", "mean")
                arrays["body_strain"][j] = (top-bottom)/height
        # Preserve raw load history grids; no zero-fill, resampling, or ordinal joins.
        if task == "UT":
            regions = ["Node {}.{}".format(mapping["instance"], n) for n in sorted(mesh["nsets"]["SET-TOP"])]
        else:
            regions = ["Node ASSEMBLY.1"]
        history = []
        for i, region in enumerate(regions):
            require(region in step.historyRegions, "Load history region missing: " + region)
            for variable_index, variable in enumerate(("U2", "RF2")):
                require(variable in step.historyRegions[region].historyOutputs, "Load history variable missing")
                history.extend((i, variable_index, time, value)
                               for time, value in step.historyRegions[region].historyOutputs[variable].data)
        h = np.array(history)
        arrays.update(history_region=np.array(regions), history_variable=np.array(["U2", "RF2"]),
                      history_region_index=h[:, 0].astype(int), history_variable_index=h[:, 1].astype(int),
                      history_time=h[:, 2], history_value=h[:, 3])
    finally:
        odb.close()
    require(before == dict(inp=fingerprint(inp_path), odb=fingerprint(odb_path)), "Input changed during read-only export")
    arrays.update(sdeg_valid=np.isfinite(arrays["sdeg"]), status_valid=np.isfinite(arrays["status"]),
                  node_displacement_valid=np.isfinite(arrays["node_displacement"]).all(axis=2))
    arrays.update(reduce_damage(arrays["sdeg"], arrays["point_element_index"], arrays["status"],
                                 arrays["parent_strut_index"], partial_rule, carry_failure))
    metadata["source_unchanged_sha256"] = True
    metadata["status_reactivations"] = int(((arrays["status"][1:] == 1) &
                                            np.maximum.accumulate(arrays["status"] == 0, axis=0)[:-1]).sum())
    arrays.update(schema_version=np.array("strut_damage_v1"), simulation_id=np.array(0), task=np.array(task),
                  partial_rule=np.array(partial_rule), carry_failure=np.array(carry_failure),
                  metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    summary = validate_damage_arrays(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **arrays)
    with np.load(str(output), allow_pickle=False) as saved:
        require(validate_damage_arrays(saved) == summary, "Saved NPZ round trip failed")
    summary.update(output=str(output), missing_status=int(np.isnan(arrays["status"]).sum()),
                   missing_sdeg=int(np.isnan(arrays["sdeg"]).sum()), metadata=metadata)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odb", required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--task", choices=("UT", "FT"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial-rule", choices=("max", "failure-only"), required=True)
    parser.add_argument("--step", default="Step-1")
    parser.add_argument("--carry-failure", action="store_true", help="Explicit irreversible STATUS carry; observed/carried masks remain separate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require(args.overwrite or not Path(args.output).exists(), "Output exists; pass --overwrite explicitly")
    result = export_damage(args.odb, args.inp, args.task, args.output, args.partial_rule, args.step, args.carry_failure)
    print(json.dumps({k:v for k,v in result.items() if k != "metadata"}, indent=2))


if __name__ == "__main__":
    main()
