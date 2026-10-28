"""Read-only beam section failure audit; run with Abaqus Python, never submits FEA.

Record section-level initiation separately from whole-element deletion. Native
field times remain brackets; absent section points are never inferred to fail.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path("Z:/p1/sims/Ti/Validation")
DEFAULT_CASES = ("10/0.2/Fracture-hex-60-per-1.odb",
                 "3/0.16/Fracture-hex-80-per-1.odb")
SOURCES = {
    "element_deletion": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-materialdata.htm",
    "section_controls": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-sectioncontrols.htm",
    "rect_section": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-beamcrosssectlib.htm",
    "B21_single_axial_point": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-nforcso.htm",
    "ductile_damage": "https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-damageevolductile.htm",
}
MANUAL_PROOFS = [
    {"url": SOURCES["element_deletion"], "edition": "2025", "section": "Element Deletion Driven by Material Failure in Abaqus/Explicit",
     "quote": "material fails at all of the section points at any one integration location of an element",
     "interpretation": "A failed outer material point is not sufficient for whole-beam deletion; all section points at an axial integration location must fail."},
    {"url": SOURCES["rect_section"], "edition": "2025", "section": "Rectangular Section",
     "quotes": ["Beam in a plane: 5 points", "Bottom and top (points 1 and 5 above for default integration)."],
     "interpretation": "Default planar RECT integration uses five points, while default field output exposes only outer points1/5."},
    {"url": SOURCES["section_controls"], "edition": "2025", "section": "MAX DEGRADATION",
     "quote": "the default value is 1.0 if the element is deleted from the mesh",
     "interpretation": "No explicit source override was found; do not call observed0.99999 the documented nominal default."},
    {"url": SOURCES["B21_single_axial_point"], "edition": "2025", "section": "Results and discussion",
     "quote": "these elements possess a single integration point along the length of the beam",
     "interpretation": "B21 has one axial integration location; its cross-section still has multiple material points."},
]


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    stat = os.stat(path)
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()}


def dump(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if math.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        return item
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def cards(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    wanted = ("*beam section", "*section controls", "*damage initiation", "*damage evolution", "*user defined field")
    found = []
    for index, line in enumerate(lines):
        if line.lower().startswith(wanted):
            block = [line]
            for later in lines[index+1:]:
                if later.startswith("*"):
                    break
                block.append(later)
            found.append({"line": index+1, "text": block})
    return found


def maximum(field, component=0):
    best = None
    for block in field.bulkDataBlocks:
        data = np.asarray(block.data, dtype=float)
        if data.ndim == 2:
            data = data[:, component]
        finite = np.isfinite(data)
        if not finite.any():
            continue
        index = int(np.argmax(np.where(finite, data, -np.inf)))
        if best is None or data[index] > best["value"]:
            best = {"value": float(data[index]), "element": int(block.elementLabels[index]),
                    "instance": block.instance.name,
                    "section_point": getattr(block.sectionPoint, "number", None),
                    "section_description": getattr(block.sectionPoint, "description", None),
                    "integration_point": int(block.integrationPoints[index]) if len(block.integrationPoints) else None}
    return best


def snapshot(field, selected):
    result = []
    for block in field.bulkDataBlocks:
        labels = np.asarray(block.elementLabels)
        indices = np.flatnonzero(np.isin(labels, list(selected)))
        raw = np.asarray(block.data, dtype=float)
        for index in indices:
            result.append({"element": int(labels[index]), "instance": block.instance.name,
                           "section_point": getattr(block.sectionPoint, "number", None),
                           "section_description": getattr(block.sectionPoint, "description", None),
                           "integration_point": int(block.integrationPoints[index]) if block.integrationPoints is not None else None,
                           "values": np.asarray(raw[index]).reshape(-1).tolist()})
    return {"components": list(field.componentLabels), "records": result}


def initiation_only(path, case, out):
    """First positive axial LE11 crossing; sequential scan avoids monotonicity assumptions."""
    from odbAccess import openOdb

    before = fingerprint(path)
    history_dir = out.parent / "history" / path.parent.parent.name / path.parent.name / path.stem
    with (history_dir / "native_curve.csv").open(newline="", encoding="utf-8") as handle:
        load = [{key: float(val) for key, val in row.items()} for row in csv.DictReader(handle)]
    with (history_dir / "native_energy.csv").open(newline="", encoding="utf-8") as handle:
        energy = [{key: float(val) for key, val in row.items()} for row in csv.DictReader(handle)]
    rec = {"case_id": case["case_id"], "odb": case["odb"], "topology": case["topology"],
           "l_mm": case["l_mm"], "nominal_relative_density": case["nominal_relative_density"],
           "nnx": case["nnx"], "source": str(path), "before": before,
           "criterion": "maximum recorded surface tensile LE11 >=0.1105 (logarithmic axial strain; engineering-strain interpretation separately bracketed)",
           "nonfinite_before_threshold": [], "first_status_time": next((r["time_s"] for r in load if r["observed_STATUS_zero"] > 0), None)}
    odb = openOdb(path=str(path), readOnly=True)
    previous, previous_time = None, None
    found = False
    try:
        for index, frame in enumerate(odb.steps["Step-1"].frames):
            if "LE" not in frame.fieldOutputs:
                rec["nonfinite_before_threshold"].append({"frame_index": index, "time": frame.frameValue, "reason": "LE not recorded"})
                continue
            field = frame.fieldOutputs["LE"]
            if "LE11" not in field.componentLabels:
                raise ValueError("Cannot identify axial LE11 component")
            component = list(field.componentLabels).index("LE11")
            for block in field.bulkDataBlocks:
                data = np.asarray(block.data, dtype=float)
                bad = int(np.count_nonzero(~np.isfinite(data[:, component])))
                if bad:
                    rec["nonfinite_before_threshold"].append({"frame_index": index, "time": frame.frameValue,
                                                            "nonfinite_axial_entries": bad})
            top = maximum(field, component=component)
            if top is None:
                continue
            if top["value"] >= math.log1p(.1105) and "engineering_interpretation_crossing" not in rec:
                rec["engineering_interpretation_crossing"] = {"prev_time": previous_time, "cross_time": frame.frameValue,
                                                                "prev_max_LE11": previous["value"] if previous else None,
                                                                "cross_max_LE11": top["value"], "maximum": top}
            if top["value"] >= .1105:
                found = True
                rec.update(prev_time=previous_time, cross_time=frame.frameValue,
                           prev_max_LE11=previous["value"] if previous else None,
                           cross_max_LE11=top["value"], frame_index=index, first_point=top)
                for label, target in (("prev", previous_time), ("cross", frame.frameValue)):
                    nearest = min(range(len(load)), key=lambda i: abs(load[i]["time_s"]-target))
                    row = load[nearest]
                    ei = min(range(len(energy)), key=lambda i: abs(energy[i]["time_s"]-row["time_s"]))
                    rec[label + "_native_history"] = {"index": nearest, "time_error_to_field_s": row["time_s"]-target,
                                                       **row, "KE_over_IE": energy[ei]["KE_over_IE"]}
                rec["point_snapshots"] = {variable: snapshot(frame.fieldOutputs[variable], {top["element"]})
                                          for variable in ("LE", "S", "SDEG", "PEEQ", "DUCTCRT", "STATUS")
                                          if variable in frame.fieldOutputs}
                break
            previous, previous_time = top, frame.frameValue
        rec["threshold_found"] = found
    finally:
        odb.close()
    after = fingerprint(path)
    rec.update(after=after, source_unchanged=before == after,
               recommended_prefix_available=found and not rec["nonfinite_before_threshold"])
    destination = out / path.parent.parent.name / path.parent.name / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    dump(destination / "initiation.json", rec)
    return rec


def save_initiations(out, records):
    prefixes = [rec["damage_prefix"] for rec in records if "damage_prefix" in rec]
    event_names = ("first_SDEG_gt_1e_minus8", "first_DUCTCRT_ge_one", "first_PEEQ_ge_source_threshold")
    summary = {
        "cases": len(records), "thresholds_found": sum(rec.get("threshold_found", False) for rec in records),
        "damage_prefix_scans": len(prefixes),
        "source_fingerprints_unchanged": sum(rec.get("source_unchanged", False) and
            rec.get("damage_prefix", {}).get("source_unchanged", False) for rec in records),
        "nonfinite_LE_prefix_cases": sum(bool(rec.get("nonfinite_before_threshold")) for rec in records),
        "nonfinite_damage_prefix_cases": sum(bool(prefix.get("nonfinite_fields")) for prefix in prefixes),
        "undamaged_through_primary_lower_bound": sum(rec.get("undamaged_prefix_through_primary_lower_bound_verified", False) for rec in records),
        "undamaged_verification_scope": "All recorded surface-section fields through the native lower-bound frame; unsaved interior section-point states are not reconstructed.",
        "first_events_same_recorded_frame_as_tensile_crossing": {event: sum(
            rec.get("damage_prefix", {}).get("events", {}).get(event, {}).get("frame_index") == rec.get("frame_index")
            for rec in records) for event in event_names},
        "max_last_pre_crossing_KE_over_IE": max((rec.get("prev_native_history", {}).get("KE_over_IE", -math.inf)
                                                 for rec in records), default=None),
        "all_history_output_key_union": sorted(set(key for prefix in prefixes for key in
            prefix.get("inventory", {}).get("all_history_output_keys", []))),
        "field_output_key_union_through_crossing": sorted(set(key for prefix in prefixes for key in
            prefix.get("inventory", {}).get("field_output_keys_through_crossing", []))),
    }
    dump(out / "failure-initiation.json", {"cases": records, "sources": SOURCES,
         "manual_proofs": MANUAL_PROOFS,
         "summary": summary,
         "limits": ["Native field brackets only; no interpolated failure status.",
                    "LE11 is logarithmic strain; exact Hsieh strain measure is not explicit in the paper.",
                    "Positive maximum only: compression magnitude is not a tensile failure criterion.",
                    "Endpoint alone cannot undo damage elsewhere before that crossing."]})
    names = ["case_id", "odb", "topology", "l_mm", "nominal_relative_density", "nnx", "threshold_found",
             "prevtime_s", "crosstime_s", "prevmaxLE11", "crossmaxLE11", "element", "section_point",
             "firstSTATUS_time_s", "prev_native_index", "cross_native_index", "prev_KE_IE", "source_sha256", "hash_unchanged", "nonfinite_before_threshold",
             "first_SDEG_time_s", "first_DUCTCRT_time_s", "damage_before_primary_frame", "undamaged_prefix_verified"]
    with (out / "failure-initiation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for rec in records:
            writer.writerow({"case_id": rec["case_id"], "odb": rec["odb"], "topology": rec["topology"],
                             "l_mm": rec["l_mm"], "nominal_relative_density": rec["nominal_relative_density"], "nnx": rec["nnx"],
                             "threshold_found": rec["threshold_found"], "prevtime_s": rec.get("prev_time"),
                             "crosstime_s": rec.get("cross_time"), "prevmaxLE11": rec.get("prev_max_LE11"),
                             "crossmaxLE11": rec.get("cross_max_LE11"), "element": rec.get("first_point", {}).get("element"),
                             "section_point": rec.get("first_point", {}).get("section_point"), "firstSTATUS_time_s": rec["first_status_time"],
                             "prev_native_index": rec.get("prev_native_history", {}).get("index"),
                             "cross_native_index": rec.get("cross_native_history", {}).get("index"),
                             "prev_KE_IE": rec.get("prev_native_history", {}).get("KE_over_IE"),
                             "source_sha256": rec["before"]["sha256"], "hash_unchanged": rec["source_unchanged"],
                             "nonfinite_before_threshold": len(rec["nonfinite_before_threshold"]),
                             "first_SDEG_time_s": rec.get("damage_prefix", {}).get("events", {}).get("first_SDEG_gt_1e_minus8", {}).get("field_time"),
                             "first_DUCTCRT_time_s": rec.get("damage_prefix", {}).get("events", {}).get("first_DUCTCRT_ge_one", {}).get("field_time"),
                             "damage_before_primary_frame": rec.get("observed_damage_strictly_before_primary_crossing_frame"),
                             "undamaged_prefix_verified": rec.get("undamaged_prefix_through_primary_lower_bound_verified")})


def damage_prefix_audit(out):
    """Check earliest material damage before each already saved tensile crossing."""
    from odbAccess import openOdb

    document = json.loads((out / "failure-initiation.json").read_text(encoding="utf-8"))
    records = document["cases"]
    for rec in records:
        print("Checking damage prefix " + rec["odb"], flush=True)
        path = Path(rec["source"])
        before = fingerprint(path)
        events = {}
        previous = None
        complete = True
        bad_fields = []
        inventory = {}
        odb = openOdb(path=str(path), readOnly=True)
        try:
            step = odb.steps["Step-1"]
            inventory["all_history_output_keys"] = sorted(set(key for region in step.historyRegions.values()
                                                              for key in region.historyOutputs.keys()))
            inventory["field_output_keys_through_crossing"] = set()
            for index, frame in enumerate(step.frames):
                if index > rec.get("frame_index", len(step.frames)-1):
                    break
                inventory["field_output_keys_through_crossing"].update(frame.fieldOutputs.keys())
                for variable, event, threshold in (("SDEG", "first_SDEG_gt_1e_minus8", 1e-8),
                                                   ("DUCTCRT", "first_DUCTCRT_ge_one", 1.),
                                                   ("PEEQ", "first_PEEQ_ge_source_threshold", .102268)):
                    if variable not in frame.fieldOutputs:
                        complete = False
                        continue
                    field = frame.fieldOutputs[variable]
                    for block in field.bulkDataBlocks:
                        invalid = int(np.count_nonzero(~np.isfinite(np.asarray(block.data, dtype=float))))
                        if invalid:
                            bad_fields.append({"frame_index": index, "time": frame.frameValue,
                                               "variable": variable, "nonfinite_entries": invalid})
                    best = maximum(field)
                    if best is None:
                        complete = False
                        continue
                    crossed = best["value"] > threshold if variable == "SDEG" else best["value"] >= threshold
                    if crossed and event not in events:
                        element = best["element"]
                        events[event] = {"previous_field_time": previous, "field_time": frame.frameValue,
                                         "frame_index": index, "maximum": best,
                                         "point_snapshots": {key: snapshot(frame.fieldOutputs[key], {element})
                                                             for key in ("LE", "S", "SDEG", "DUCTCRT", "PEEQ", "STATUS")
                                                             if key in frame.fieldOutputs}}
                previous = frame.frameValue
            inventory["field_output_keys_through_crossing"] = sorted(inventory["field_output_keys_through_crossing"])
        finally:
            odb.close()
        after = fingerprint(path)
        rec["damage_prefix"] = {"events": events, "required_fields_available": complete,
                                 "before": before, "after": after, "source_unchanged": before == after,
                                 "inventory": inventory, "nonfinite_fields": bad_fields}
        damage = events.get("first_SDEG_gt_1e_minus8")
        primary_lower = rec.get("prev_time")
        rec["observed_damage_strictly_before_primary_crossing_frame"] = bool(damage and damage["frame_index"] < rec.get("frame_index", -1))
        rec["observed_damage_at_or_before_primary_lower_bound"] = bool(damage and primary_lower is not None and damage["field_time"] <= primary_lower)
        rec["undamaged_prefix_through_primary_lower_bound_verified"] = bool(complete and primary_lower is not None
            and not rec["observed_damage_at_or_before_primary_lower_bound"] and not rec["nonfinite_before_threshold"] and not bad_fields)
        dump(out / path.parent.parent.name / path.parent.name / path.stem / "initiation.json", rec)
        save_initiations(out, records)


def critical_strut_mesh(out, manifest_path):
    """Trace original B21 connectivity only; never opens or submits an ODB/job."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = {rec["case_id"]: rec for rec in json.loads((out / "failure-initiation.json").read_text(encoding="utf-8"))["cases"]}
    records = []
    for case in manifest:
        event = events[case["case_id"]]
        source = Path(case["source_inp"])
        before = fingerprint(source)
        instance = event["first_point"]["instance"].lower()
        nodes, elements, sections, cracktip = {}, {}, [], set()
        current_instance, block, parameters, options = None, None, {}, set()
        for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                parts = [part.strip() for part in line.split(",")]
                block = parts[0].lower()
                options = {part.lower() for part in parts[1:] if "=" not in part}
                parameters = {part.split("=", 1)[0].strip().lower(): part.split("=", 1)[1].strip()
                              for part in parts[1:] if "=" in part}
                if block == "*instance":
                    current_instance = parameters.get("name", "").lower()
                elif block == "*end instance":
                    current_instance = None
                continue
            fields = [part.strip() for part in line.split(",") if part.strip()]
            if current_instance == instance and block == "*node":
                nodes[int(fields[0])] = [float(value) for value in fields[1:]]
            elif current_instance == instance and block == "*element":
                if parameters.get("type", "").upper() != "B21":
                    raise ValueError("Non-B21 source element block: " + str(source))
                if len(fields) != 3:
                    raise ValueError("Unexpected B21 connectivity: " + line)
                elements[int(fields[0])] = tuple(int(value) for value in fields[1:])
            elif current_instance == instance and block == "*beam section":
                if not sections or sections[-1]["parameters"] is not parameters:
                    sections.append({"parameters": parameters, "data": []})
                sections[-1]["data"].append([float(value) for value in fields])
            elif block == "*elset" and parameters.get("elset", "").lower() == "set-cracktip" and parameters.get("instance", "").lower() == instance:
                values = [int(value) for value in fields]
                if "generate" in options:
                    if len(values) != 3:
                        raise ValueError("Malformed generated crack-tip row")
                    values = range(values[0], values[1]+1, values[2])
                cracktip.update(values)
        if len(sections) != 1 or sections[0]["parameters"].get("section", "").upper() != "RECT":
            raise ValueError("Expected one uniform RECT beam section: " + str(source))
        adjacency = defaultdict(list)
        for label, pair in elements.items():
            for node in pair:
                adjacency[node].append(label)
        critical = event["first_point"]["element"]
        start, finish = elements[critical]

        def extend(node, incoming):
            """Follow an unbranched mesh chain; flag impossible closed-loop traversal."""
            edge_path, node_path = [], [node]
            visited = {incoming}
            while len(adjacency[node]) == 2:
                edge = next(label for label in adjacency[node] if label != incoming)
                if edge in visited:
                    raise ValueError("Closed degree2 loop at critical strut: " + str(source))
                visited.add(edge)
                pair = elements[edge]
                node = pair[1] if pair[0] == node else pair[0]
                edge_path.append(edge)
                node_path.append(node)
                incoming = edge
            return edge_path, node_path

        left_edges, left_nodes = extend(start, critical)
        right_edges, right_nodes = extend(finish, critical)
        chain_edges = left_edges[::-1] + [critical] + right_edges
        chain_nodes = left_nodes[::-1] + right_nodes
        if len(chain_nodes) != len(chain_edges)+1 or len(set(chain_edges)) != len(chain_edges):
            raise ValueError("Invalid critical-strut chain")
        if any(set(elements[edge]) != {first, second} for edge, first, second in
               zip(chain_edges, chain_nodes[:-1], chain_nodes[1:])):
            raise ValueError("Critical-strut connectivity does not match traced node order")
        points = np.asarray([nodes[label] for label in chain_nodes], dtype=float)
        deltas = np.diff(points, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        if not np.all(lengths > 0):
            raise ValueError("Nonpositive B21 segment length")
        direction = points[-1] - points[0]
        chord = float(np.linalg.norm(direction))
        if chord == 0.:
            raise ValueError("Closed critical-strut chain")
        unit = direction / chord
        offset = points - points[0]
        transverse = offset - np.outer(offset @ unit, unit)
        residual = float(np.max(np.linalg.norm(transverse, axis=1)))
        after = fingerprint(source)
        topology, nominal_l = event["topology"], float(event["l_mm"])
        expected = {"tri": 4, "kagome": 20, "hex": 30}[topology]
        section = sections[0]
        if len(section["data"][0]) != 2 or section["data"][1] != [0., 0., -1.]:
            raise ValueError("Cannot identify in-plane thickness for this section orientation")
        records.append({"case_id": event["case_id"], "odb": event["odb"], "job": case["job"],
            "topology": topology, "l_mm": nominal_l, "rho": float(event["nominal_relative_density"]),
            "nnx": int(event["nnx"]), "critical_element": critical,
            "critical_section_point": event["first_point"]["section_point"],
            "critical_in_cracktip_elset": critical in cracktip,
            "n_e_degree2_chain": len(chain_edges), "Hsieh_Table2_n_e": expected,
            "n_e_over_Hsieh": len(chain_edges) / expected,
            "chain_length_mm": float(lengths.sum()), "chord_length_mm": chord,
            "length_over_nominal_l": float(lengths.sum()) / nominal_l,
            "max_collinearity_residual_mm": residual,
            "straight_nominal_strut_verified": bool(residual < nominal_l*1e-4 and abs(lengths.sum()/nominal_l-1.) < 1e-4),
            "segment_length_min_mm": float(lengths.min()), "segment_length_max_mm": float(lengths.max()),
            "end_node_labels": [chain_nodes[0], chain_nodes[-1]],
            "end_node_degrees": [len(adjacency[chain_nodes[0]]), len(adjacency[chain_nodes[-1]])],
            "end_coordinates_mm": [points[0].tolist(), points[-1].tolist()],
            "critical_end_coordinates_mm": [nodes[start], nodes[finish]],
            "chain_element_labels": chain_edges, "chain_node_labels": chain_nodes,
            "chain_node_coordinates_mm": points.tolist(),
            "RECT_dimensions_mm": section["data"][0], "section_orientation": section["data"][1],
            "in_plane_t_mm": section["data"][0][1],
            "thickness_basis": "RECT second dimension; source section normal is global(0,0,-1), so first dimension is out of plane.",
            "source_inp": str(source), "source_before": before, "source_after": after,
            "source_unchanged": before == after, "matches_loading_manifest_hash": before["sha256"] == case["source_sha256"]})
    dump(out / "critical-strut-mesh-summary.json", {"cases": records, "manifest": str(manifest_path),
         "manifest_fingerprint": fingerprint(manifest_path),
         "Hsieh_reference": {"doi": "https://doi.org/10.1016/j.jmps.2020.103925", "locations": "Table2, p5; AppendixB, p15",
             "n_e": {"tri": 4, "kagome": 20, "hex": 30},
             "convergence_basis": "Hsieh varies B21 elements/bar at rho0.13,l3mm, refining the crack-tip rectangle; selected counts keep surface-failure load-line displacement within10percent of the final converged value."},
         "method": "Original INP B21 graph: extend critical element at both ends through nodes of degree2; stop at degree!=2. Check straightness and nominal physical-strut length rather than assuming current producer settings.",
         "limits": ["Only11 selected critical struts are audited, not a whole-mesh convergence study.",
                    "Element-count difference alone cannot prove a result error. A refinement control must preserve the physical geometry, section and loading and compare the first surface-strain crossing and energies."]})
    columns = ["case_id", "odb", "topology", "l_mm", "rho", "nnx", "critical_element", "critical_in_cracktip_elset",
               "n_e_degree2_chain", "Hsieh_Table2_n_e", "n_e_over_Hsieh", "chain_length_mm", "length_over_nominal_l",
               "straight_nominal_strut_verified", "segment_length_min_mm", "segment_length_max_mm", "end_node_labels", "end_node_degrees",
               "end_coordinates_mm", "critical_end_coordinates_mm", "RECT_dimensions_mm", "in_plane_t_mm", "max_collinearity_residual_mm",
               "source_inp", "source_sha256", "source_unchanged", "matches_loading_manifest_hash"]
    with (out / "critical-strut-mesh.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            row = {key: rec[key] for key in columns if key in rec}
            row["source_sha256"] = rec["source_before"]["sha256"]
            writer.writerow(row)


def audit(path, out, detailed):
    from odbAccess import openOdb

    before = fingerprint(path)
    rec = {"source": str(path), "before": before, "read_only": True,
           "input_cards": cards(path.with_suffix(".inp")),
           "input_fingerprint": fingerprint(path.with_suffix(".inp")),
           "section_output_limit": "Only requested/saved section points are observable; interior points must not be inferred.",
           "thresholds": {"source_ductile_PEEQ": .102268,
                          "Hsieh_epsilon_f_direct_LE_sensitivity": .1105,
                          "Hsieh_engineering_to_log_sensitivity": math.log1p(.1105)},
           "events": {}, "frames": []}
    prior = None
    selected = set()
    odb = openOdb(path=str(path), readOnly=True)
    try:
        step = odb.steps["Step-1"]
        rec["section_categories"] = {name: {"description": category.description,
                                           "section_points": [{"number": point.number, "description": point.description}
                                                              for point in category.sectionPoints]}
                                     for name, category in odb.sectionCategories.items()}
        rec["element_types"] = {name: sorted(set(element.type for element in instance.elements))
                                for name, instance in odb.rootAssembly.instances.items()}
        status_times = []
        for key, region in step.historyRegions.items():
            if key.startswith("Element ") and "STATUS" in region.historyOutputs:
                status_times.extend(t for t, val in region.historyOutputs["STATUS"].data if val == 0.)
        rec["first_monitored_STATUS_zero"] = min(status_times) if status_times else None
        for index, frame in enumerate(step.frames):
            row = {"index": index, "time": frame.frameValue}
            for variable in ("LE", "PEEQ", "SDEG", "DUCTCRT"):
                if variable in frame.fieldOutputs:
                    row[variable] = maximum(frame.fieldOutputs[variable])
            tests = (("first_SDEG_positive", "SDEG", lambda x: x > 0.),
                     ("first_DUCTCRT_ge_one", "DUCTCRT", lambda x: x >= 1.),
                     ("first_PEEQ_ge_source_threshold", "PEEQ", lambda x: x >= .102268),
                     ("first_tensile_LE_ge_0p1105", "LE", lambda x: x >= .1105),
                     ("first_tensile_LE_ge_log1p_0p1105", "LE", lambda x: x >= math.log1p(.1105)))
            for event, variable, test in tests:
                if event not in rec["events"] and row.get(variable) and test(row[variable]["value"]):
                    rec["events"][event] = {"lower_time_exclusive": prior, "upper_time_inclusive": frame.frameValue,
                                            "frame_index": index, "maximum": row[variable]}
                    selected.add(row[variable]["element"])
            rec["frames"].append(row)
            prior = frame.frameValue
        if detailed:
            near = set()
            for event in rec["events"].values():
                near.update(range(max(0, event["frame_index"]-1), min(len(step.frames), event["frame_index"]+2)))
            deletion = rec["first_monitored_STATUS_zero"]
            if deletion is not None:
                nearest = min(range(len(step.frames)), key=lambda i: abs(step.frames[i].frameValue-deletion))
                near.update(range(max(0, nearest-1), min(len(step.frames), nearest+2)))
            near.add(len(step.frames)-1)
            rec["selected_elements"] = sorted(selected)
            rec["snapshots"] = []
            for index in sorted(near):
                frame = step.frames[index]
                entry = {"index": index, "time": frame.frameValue, "fields": {}}
                for variable in ("STATUS", "STATUSMP", "SDEG", "DUCTCRT", "PEEQ", "S", "LE", "PE"):
                    if variable in frame.fieldOutputs:
                        entry["fields"][variable] = snapshot(frame.fieldOutputs[variable], selected)
                rec["snapshots"].append(entry)
    finally:
        odb.close()
    after = fingerprint(path)
    rec.update(after=after, source_unchanged=before == after)
    dest = out / path.parent.parent.name / path.parent.name / path.stem
    dest.mkdir(parents=True, exist_ok=True)
    dump(dest / "failure.json", rec)
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "audit/failure")
    parser.add_argument("--case", action="append")
    parser.add_argument("--density-campaign", action="store_true")
    parser.add_argument("--initiation-only", action="store_true", help="sequential LE11 scan stops at first0.1105 tensile crossing; no full damage scan")
    parser.add_argument("--damage-prefix-only", action="store_true", help="add earliestSDEG/DUCTCRT/PEEQ and signedpoint snapshots up to saved tensile crossings")
    parser.add_argument("--summarize-existing", action="store_true", help="refresh saved summaries and citations without opening ODBs")
    parser.add_argument("--critical-strut-mesh", action="store_true", help="trace11 selected critical struts in original INPs only")
    parser.add_argument("--loading-controls-manifest", type=Path, default=ROOT / "audit/work-partition/loading-controls-manifest.json")
    args = parser.parse_args()
    if args.critical_strut_mesh:
        critical_strut_mesh(args.out, args.loading_controls_manifest)
        return
    if args.summarize_existing:
        document = json.loads((args.out / "failure-initiation.json").read_text(encoding="utf-8"))
        save_initiations(args.out, document["cases"])
        detailed_path = args.out / "failure-summary.json"
        if detailed_path.exists():
            detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
            detailed["manual_proofs"] = MANUAL_PROOFS
            dump(detailed_path, detailed)
        return
    if args.damage_prefix_only:
        damage_prefix_audit(args.out)
        return
    cases = args.case or DEFAULT_CASES
    with (args.root / "case-index.csv").open(newline="", encoding="utf-8") as handle:
        indexed = {r["odb"]: r for r in csv.DictReader(handle) if r["campaign"] == "Density campaign"
                   and r["mode"] == "Fracture" and r["topology"] in ("tri", "kagome", "hex")
                   and r["storage_status"] == "retained_odb"}
    if args.density_campaign:
        cases = list(indexed)
        cases = [case for case in DEFAULT_CASES if case in cases] + [case for case in cases if case not in DEFAULT_CASES]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for relative in cases:
        print("Reading native failure fields " + relative, flush=True)
        if args.initiation_only:
            records.append(initiation_only(args.root / relative, indexed[relative], args.out))
            save_initiations(args.out, records)
            print("Done " + relative, flush=True)
            continue
        record = audit(args.root / relative, args.out, detailed=relative in DEFAULT_CASES)
        records.append({key: val for key, val in record.items() if key not in ("frames", "snapshots")})
        dump(args.out / "failure-summary.json", {"sources": SOURCES, "cases": records,
             "manual_proofs": MANUAL_PROOFS,
             "manual_access_limit": "2024 direct pages returned403; official2025 manual used for verified definitions.",
             "physical_limit": "Surface criterion, section degradation and whole-element deletion are distinct; no absent interior-section state is invented."})
        print("Done " + relative, flush=True)


if __name__ == "__main__":
    main()
