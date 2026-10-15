"""Explicit INP parent maps and auditable strut damage reduction (no Abaqus import).

Only the periodic 20-by-19 FCC profile is inferred automatically. Disordered
meshes must supply independently validated original joint/canonical identities.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_beam_inp(path):
    """Read inline, untransformed instance meshes; reject unsupported constructs."""
    instances, current, keyword, options = {}, None, "", {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            fields = [v.strip() for v in line.split(",")]
            keyword = fields[0].lower()
            options = {v.split("=", 1)[0].strip().lower(): v.split("=", 1)[1].strip()
                       for v in fields[1:] if "=" in v}
            options["generate"] = any(v.lower() == "generate" for v in fields[1:])
            require(keyword not in {"*include", "*system", "*transform"},
                    "INP includes/transforms require an explicitly resolved mesh")
            if keyword == "*instance":
                current = options["name"]
                require(current not in instances, "Duplicate instance")
                instances[current] = dict(nodes={}, elements={}, nsets=defaultdict(set),
                                          elsets=defaultdict(set))
            elif keyword == "*end instance":
                current = None
            continue
        target = current or options.get("instance")
        if target not in instances:
            require(keyword not in {"*node", "*element"} or current is None,
                    "Unresolved mesh instance")
            continue  # assembly reference points/empty part definitions
        data = instances[target]
        values = [v.strip() for v in line.split(",") if v.strip()]
        if keyword == "*instance":
            raise ValueError("Transformed instance is not supported")
        if keyword == "*node":
            label = int(values[0])
            require(label not in data["nodes"], "Duplicate node label within instance")
            coord = np.array([float(v) for v in values[1:]], float)
            require(len(coord) in (2, 3) and np.isfinite(coord).all(), "Invalid node coordinates")
            require(len(coord) == 2 or coord[2] == 0, "Only planar B21 meshes are supported")
            data["nodes"][label] = coord[:2]
        elif keyword == "*element":
            require(options["type"].upper() == "B21" and len(values) == 3,
                    "Only two-node B21 elements are supported")
            label, a, b = map(int, values)
            require(label not in data["elements"] and a != b, "Invalid/duplicate element")
            data["elements"][label] = (a, b)
        elif keyword in {"*nset", "*elset"}:
            labels = list(map(int, values))
            if options["generate"]:
                require(len(labels) == 3 and labels[2] > 0, "Invalid generated set")
                labels = range(labels[0], labels[1] + 1, labels[2])
            key = "nset" if keyword == "*nset" else "elset"
            data[key + "s"][options[key].upper()].update(labels)
    require(instances and all(v["elements"] for v in instances.values()), "No inline beam mesh")
    return instances


def map_beam_chains(nodes, elements, selected, joint_to_canonical, tolerance=2e-5):
    """Partition actual FE connectivity using supplied original joint identities.

    Canonical endpoint order is invariant to FE labels, connectivity direction,
    and subdivision count. Interior branching, loops and orphan edges fail shut.
    """
    require(selected and set(selected) <= elements.keys(), "Missing selected elements")
    require(len(set(joint_to_canonical.values())) == len(joint_to_canonical),
            "Canonical joints must be one-to-one")
    adjacency = defaultdict(list)
    for label in sorted(selected):
        a, b = elements[label]
        require(a in nodes and b in nodes and a != b, "Invalid beam connectivity")
        adjacency[a].append((b, label))
        adjacency[b].append((a, label))
    require(set(joint_to_canonical) <= adjacency.keys(), "Unconnected original joint")
    visited, chains = set(), []
    for start in sorted(joint_to_canonical, key=joint_to_canonical.get):
        for neighbour, label in sorted(adjacency[start]):
            if label in visited:
                continue
            path, members = [start, neighbour], [label]
            visited.add(label)
            while path[-1] not in joint_to_canonical:
                require(len(adjacency[path[-1]]) == 2, "Branch/end at unidentified original joint")
                choices = [(n, e) for n, e in adjacency[path[-1]] if e != members[-1]]
                require(len(choices) == 1 and choices[0][1] not in visited, "Loop in beam chain")
                neighbour, label = choices[0]
                path.append(neighbour)
                members.append(label)
                visited.add(label)
            require(path[0] != path[-1], "Closed parent strut")
            if joint_to_canonical[path[0]] > joint_to_canonical[path[-1]]:
                path.reverse()
                members.reverse()
            xy = np.array([nodes[n] for n in path])
            vector = xy[-1] - xy[0]
            require(np.dot(vector, vector) > 0, "Zero-length parent")
            t = (xy - xy[0]) @ vector / np.dot(vector, vector)
            residual = float(np.linalg.norm(xy - xy[0] - t[:, None]*vector, axis=1).max())
            require(residual <= tolerance and (np.diff(t) > 0).all(), "Non-straight/non-monotonic parent")
            chains.append(dict(nodes=path, elements=members, residual=residual,
                               canonical=[joint_to_canonical[path[0]], joint_to_canonical[path[-1]]]))
    require(visited == set(selected), "Unassigned elements")
    chains.sort(key=lambda c: c["canonical"])
    require(len({tuple(c["canonical"]) for c in chains}) == len(chains), "Duplicate parent endpoints")
    return chains


def periodic_fcc_map(inp, task, tolerance=2e-5):
    """Checked periodic reference profile only; simulation id 0, body-only UT."""
    require(task in {"UT", "FT"}, "Task must be UT or FT")
    instances = read_beam_inp(inp)
    require(len(instances) == 1, "Periodic profile requires exactly one physical instance")
    instance, mesh = next(iter(instances.items()))
    selected_set = "BODYSET" if task == "UT" else "ALLPARTSET"
    selected = mesh["elsets"][selected_set]
    canonical_xy = np.array(sorted(
        {(10*x, 10*y) for y in range(20) for x in range(21)} |
        {(10*x+5, 10*y+5) for y in range(19) for x in range(20)}, key=lambda p: (p[1], p[0])), float)
    lookup = {tuple(p): i for i, p in enumerate(canonical_xy)}
    expected = set(lookup)
    if task == "FT":
        expected -= {(10*x+5, 95) for x in range(12)}
    labels = {n for e in selected for n in mesh["elements"][e]}
    joints, found = {}, set()
    for n in labels:
        xy = mesh["nodes"][n]
        key = tuple(np.rint(xy))
        if key in expected and np.linalg.norm(xy - key) <= tolerance:
            require(key not in found, "Ambiguous periodic joint")
            joints[n] = lookup[key]
            found.add(key)
    require(found == expected, "Not the validated periodic FCC body; supply original joint identities")
    chains = map_beam_chains(mesh["nodes"], mesh["elements"], selected, joints, tolerance)
    require(len(chains) == (2319 if task == "UT" else 2259), "Unexpected periodic strut count")
    members = [e for c in chains for e in c["elements"]]
    parents = np.repeat(np.arange(len(chains)), [len(c["elements"]) for c in chains])
    node_labels = np.array(sorted(labels), dtype=np.int64)
    arrays = dict(
        strut_id=np.arange(len(chains), dtype=np.int64),
        canonical_endpoint=np.array([c["canonical"] for c in chains], dtype=np.int64),
        endpoint_node_label=np.array([[c["nodes"][0], c["nodes"][-1]] for c in chains], dtype=np.int64),
        endpoint_instance=np.full((len(chains), 2), instance.upper()),
        endpoint_xy0=np.array([[mesh["nodes"][c["nodes"][0]], mesh["nodes"][c["nodes"][-1]]] for c in chains]),
        subdivision_count=np.array([len(c["elements"]) for c in chains], dtype=np.int64),
        mapping_residual=np.array([c["residual"] for c in chains]),
        element_label=np.array(members, dtype=np.int64), element_instance=np.full(len(members), instance.upper()),
        element_connectivity=np.array([mesh["elements"][e] for e in members], dtype=np.int64),
        parent_strut_index=parents, chain_order=np.concatenate([np.arange(len(c["elements"])) for c in chains]),
        node_label=node_labels, node_xy0=np.array([mesh["nodes"][n] for n in node_labels]),
        canonical_xy=canonical_xy,
        canonical_node_present=np.array([i in joints.values() for i in range(len(canonical_xy))]),
        strut_present=np.ones(len(chains), bool),
    )
    return arrays, mesh, dict(instance=instance.upper(), selected_set=selected_set, joints=len(joints),
                              excluded_elements=len(mesh["elements"])-len(members), tolerance=tolerance)


def reduce_damage(sdeg, point_element, status, parents, partial_rule, carry_failure=False):
    """Max section/IP severity; whole-element STATUS=0 establishes failure.

    Missing is unknown. Any verified failed element establishes parent failure
    even if another constituent is unobserved. A point at SDEG=1 is not itself a
    whole-element failure. No near-one threshold or sum defines failure.
    """
    sdeg, status = np.asarray(sdeg, float), np.asarray(status, float)
    point_element, parents = np.asarray(point_element, int), np.asarray(parents, int)
    require(partial_rule in {"max", "failure-only"}, "Explicit partial rule required")
    require(status.ndim == 2 and sdeg.ndim == 2 and len(sdeg) == len(status), "Invalid frame axes")
    require(sdeg.shape[1] == len(point_element) and status.shape[1] == len(parents), "Invalid entity axes")
    require(len(parents) > 0 and (parents >= 0).all() and set(parents) == set(range(parents.max()+1)),
            "Parent indices must be contiguous and nonempty")
    require(len(point_element) > 0 and (point_element >= 0).all() and (point_element < len(parents)).all(),
            "Invalid point-to-element map")
    require(np.isin(status[np.isfinite(status)], [0., 1.]).all(), "STATUS is not binary 0/1")
    require(not np.isinf(status).any() and not np.isinf(sdeg).any(), "Infinite raw values are invalid")
    finite = sdeg[np.isfinite(sdeg)]
    require(((finite >= 0) & (finite <= 1)).all(), "SDEG outside [0,1]")
    observed_failed, known = status == 0, np.isfinite(status)
    failed = observed_failed.copy()
    carried = np.zeros_like(failed)
    if carry_failure:
        ever = np.maximum.accumulate(observed_failed, axis=0)
        require(not (ever & (status == 1)).any(), "Element reactivation contradicts absorbing failure")
        carried = ever & ~known
        failed |= carried
        known |= carried
    damage = np.full(status.shape, np.nan)
    complete = np.zeros(status.shape, bool)
    observed_max = np.full(status.shape, np.nan)
    for element in range(status.shape[1]):
        points = sdeg[:, point_element == element]
        if points.shape[1]:
            complete[:, element] = np.isfinite(points).all(axis=1)
            values = np.max(np.where(np.isfinite(points), points, -np.inf), axis=1)
            values[values == -np.inf] = np.nan
            observed_max[:, element] = values
            valid = complete[:, element] & (status[:, element] == 1)
            if partial_rule == "max":
                damage[valid, element] = values[valid]
            else:
                damage[valid & (values == 0), element] = 0
        damage[failed[:, element], element] = 1
    shape = (status.shape[0], parents.max()+1)
    strut_damage = np.full(shape, np.nan)
    strut_failed, strut_known = np.zeros(shape, bool), np.zeros(shape, bool)
    coverage, observation_coverage, failure_count = np.zeros(shape), np.zeros(shape), np.zeros(shape, int)
    for parent in range(shape[1]):
        mask = parents == parent
        values = damage[:, mask]
        strut_failed[:, parent] = failed[:, mask].any(axis=1)
        strut_known[:, parent] = strut_failed[:, parent] | known[:, mask].all(axis=1)
        coverage[:, parent] = np.isfinite(values).mean(axis=1)
        observation_coverage[:, parent] = (np.isfinite(status[:,mask]) & complete[:,mask]).mean(axis=1)
        valid = np.isfinite(values).all(axis=1)
        strut_damage[valid, parent] = values[valid].max(axis=1)
        strut_damage[strut_failed[:, parent], parent] = 1
        failure_count[:, parent] = failed[:, mask].sum(axis=1)
    first = np.array([np.flatnonzero(v)[0] if v.any() else -1 for v in strut_failed.T])
    previous = np.full(shape[1], -1, int)
    for parent, frame in enumerate(first):
        if frame > 0:
            before = np.flatnonzero(strut_known[:frame, parent] & ~strut_failed[:frame, parent])
            if len(before):
                previous[parent] = before[-1]
    return dict(element_damage=damage, element_sdeg_max=observed_max, element_sdeg_complete=complete,
                element_failure_observed=observed_failed, element_failure_known=known,
                element_failed=failed, element_failure_carried=carried,
                strut_damage=strut_damage, strut_valid=np.isfinite(strut_damage),
                strut_failed=strut_failed, strut_failure_known=strut_known,
                strut_coverage=coverage, strut_observation_coverage=observation_coverage,
                failed_element_count=failure_count,
                first_failure_frame=first, previous_active_frame=previous)


def validate_damage_arrays(arrays):
    """Validate a saved v1 product without importing Abaqus or an ML loader."""
    require(str(arrays["schema_version"]) == "strut_damage_v1", "Unsupported damage schema")
    require(all(not np.asarray(v).dtype.hasobject for v in arrays.values()), "Pickled/object arrays forbidden")
    f, s = arrays["strut_damage"].shape
    e, p = len(arrays["element_label"]), len(arrays["point_element_index"])
    require(arrays["status"].shape == (f, e) and arrays["sdeg"].shape == (f, p), "Raw shape mismatch")
    require(len(arrays["frame_time"]) == f and len(arrays["strut_id"]) == s, "Identity/time shape mismatch")
    require((np.diff(arrays["frame_time"]) > 0).all(), "Non-increasing recorded times")
    require(len(set(zip(arrays["element_instance"], arrays["element_label"]))) == e, "Duplicate element identity")
    require(np.array_equal(np.isfinite(arrays["strut_damage"]), arrays["strut_valid"]), "Invalid label mask")
    require((arrays["strut_damage"][arrays["strut_failed"]] == 1).all(), "Failure override lost")
    require(arrays["frame_time"][0] == 0, "Initial frame must be retained")
    expected = reduce_damage(arrays["sdeg"], arrays["point_element_index"], arrays["status"],
                             arrays["parent_strut_index"], str(arrays["partial_rule"]),
                             bool(arrays["carry_failure"]))
    for key, value in expected.items():
        require(np.array_equal(value, arrays[key], equal_nan=True), "Derived array mismatch: " + key)
    return dict(frames=f, struts=s, elements=e, points=p, valid_labels=int(arrays["strut_valid"].sum()),
                final_failed_struts=int(arrays["strut_failed"][-1].sum()),
                final_failed_elements=int(arrays["element_failed"][-1].sum()))
