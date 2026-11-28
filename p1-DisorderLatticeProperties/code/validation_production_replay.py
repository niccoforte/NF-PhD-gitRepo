"""Replay Paper 1 properties on source-bound native histories.

The CLI runs the frozen 64-history production pilot. It keeps per-metric
availability and writes a single audit receipt. Existing property tables,
machine-learning inputs and figures are outside this entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from resources.calculations import (
    calc_FaW, compact_tension_initiation, filter_force, ortega_crack_axis_factor,
)
from resources.lattices import prismatic_plane_strain
from validation_fixed_filter_pilot import (
    REGISTRY, ROOT, frozen_records, identity, material_event_index, planned_time, read_json,
    curve_report,
)
from validation_controls import body_tensor_source
from validation_fcc import source_geometry
from validation_stiffness import _write_immutable_targets, sha


PILOT = Path("audit/curve-processing/accepted-filter-study/fixed-cutoff-pilot")
GEOMETRY_KEYS = ("a_mm", "B_mm", "W_mm")


def minimum_body_width(body):
    """Narrowest horizontal distance between the outer strut centrelines.

    Between node elevations the intersected centrelines are straight and
    their outer span is convex. Evaluate the one-sided limits and any
    minimum between them.
    """
    nodes = body["nodes"]
    sides = body["boundary_sides"]
    bottom, top = (float(np.mean([nodes[i][1] for i in sides[name]]))
                   for name in ("set-bottombody", "set-topbody"))
    if top <= bottom:
        raise ValueError("Expected positive main-body gauge height")
    if not body["elements"]:
        raise ValueError("Expected main-body struts")
    segments = np.array([[nodes[i], nodes[j]] for i, j in body["elements"].values()])
    segments = np.where((segments[:, 0, 1] > segments[:, 1, 1])[:, None, None],
                        segments[:, ::-1], segments)
    starts, ends = segments[:, 0], segments[:, 1]
    lower = np.maximum(starts[:, 1], bottom)
    upper = np.minimum(ends[:, 1], top)
    intersects = upper > lower
    lower, upper = lower[intersects], upper[intersects]
    starts, ends = starts[intersects], ends[intersects]
    slopes = (ends[:, 0] - starts[:, 0]) / (ends[:, 1] - starts[:, 1])
    adds, removes = {}, {}
    for i, (a, b) in enumerate(zip(lower, upper)):
        adds.setdefault(a, []).append(i)
        removes.setdefault(b, []).append(i)

    levels = sorted({bottom, top, *adds, *removes})
    active = set()
    width = np.inf
    for a, b in zip(levels, levels[1:]):
        active.difference_update(removes.get(a, ()))
        active.update(adds.get(a, ()))
        if not active:
            raise ValueError("No body struts intersect part of the main-body gauge")
        ids = np.fromiter(active, dtype=int, count=len(active))
        m = slopes[ids]
        xa = starts[ids, 0] + m * (a - starts[ids, 1])
        xb = starts[ids, 0] + m * (b - starts[ids, 1])
        width = min(width, np.ptp(xa), np.ptp(xb))
        # With different active outer inclinations at either end, the span
        # can reach a smaller value inside this interval.
        ml = m[np.argmax(xa)] - m[np.argmin(xa)]
        mr = m[np.argmax(xb)] - m[np.argmin(xb)]
        if ml < 0 < mr:
            lo, hi = 0., b - a
            while hi - lo > 1e-8:
                mid = (lo + hi) / 2
                x = xa + m * mid
                left, right = np.argmin(x), np.argmax(x)
                width = min(width, x[right] - x[left])
                if m[right] > m[left]:
                    hi = mid
                else:
                    lo = mid
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Expected a positive finite minimum body width")
    return float(width)


def kagome_stress_area(source, archived_stress_scale):
    """Return the source-specific area and correction to archived nominal stress."""
    body = body_tensor_source(source)
    thickness = body["section_B_t_mm"][0]
    if not np.isfinite(archived_stress_scale) or archived_stress_scale <= 0:
        raise ValueError("Expected a positive archived force-to-stress scale")
    width = minimum_body_width(body)
    archived_area = 1 / archived_stress_scale
    area = thickness * width
    return {"method": "minimum_main_body_centreline_span",
            "width_mm": width, "thickness_mm": thickness, "area_mm2": area,
            "archived_gross_area_mm2": archived_area,
            "stress_scale_from_archive": archived_area / area}


def compact_tension_properties(row, time, x, y, cutoff, geometry, modulus, elastic):
    """Use the full-curve 40% fit only if its fitted prefix precedes initiation."""
    endpoint = row["material_endpoint"]
    result = {"status": endpoint["status"]}
    if "native_index" not in endpoint:
        return result
    i = endpoint["native_index"]
    fixed = row["methods"]["fixed"]
    fit = fixed["initial_work_fits"]["0.4"]
    if fit["status"] != "resolved_estimate" or fit["end_index"] > i:
        return dict(result, status="unavailable_initial_compliance")
    if modulus is None:
        observed = endpoint["fixed"]
        return dict(result, status="unavailable_paired_uat_modulus",
                    force_N=observed["force_N"], displacement_mm=observed["opening_mm"],
                    work_N_mm=observed["work_N_mm"],
                    Apl_N_mm=observed["signed_initial_triangle_area_N_mm"],
                    geometry=geometry, event_time_s=endpoint["time_s"])
    a, B, W = (geometry[k] for k in GEOMETRY_KEYS)
    factor = (ortega_crack_axis_factor(a, W, elastic) if elastic is not None else calc_FaW(a, W))
    opening, _ = filter_force(time[:i+1], x[:i+1], cutoff)
    force, _ = filter_force(time[:i+1], y[:i+1], cutoff)
    try:
        result.update(compact_tension_initiation(
            opening, force, i, a, B, W, modulus, fit["estimate"], factor))
    except ValueError as error:
        return dict(result, status="unavailable_endpoint_mechanics", reason=str(error))
    result.update(
        status="calculated" if result["valid_plastic_area"] else "unavailable_negative_plastic_work",
        geometry_factor=float(factor),
        geometry=geometry,
        event_time_s=endpoint["time_s"],
        stiffness_fit_end_index=fit["end_index"],
        minimum_k0_for_nonnegative_area_N_per_mm=(
            result["force_N"]**2 / (2 * result["work_N_mm"]) if result["work_N_mm"] > 0 else None
        ),
    )
    return result


def fcc_matched_field_check(root, elastic, bind):
    """Swap the conversion tensor at the same saved force, work and crack."""
    path = bind(root/"audit/fcc-calibration/eta-verification/approved-cutoff-check.json")
    data = read_json(path)
    if data["registry_sha256"] != sha(REGISTRY):
        raise ValueError("Matched field reference uses a different cutoff registry")
    rows = []
    M = elastic["M_I_pe"]
    for reference in data["cases"]:
        case = read_json(bind(reference["source"], reference["sha256"]))["case"]
        a, B, W = (case["geometry"][k] for k in GEOMETRY_KEYS)
        factor = ortega_crack_axis_factor(a, W, elastic)
        ki = reference["filtered_force_N"] * factor / (B * np.sqrt(W))
        jel = ki**2 / M
        eta = reference["eta_ASTM"]
        jpl = eta * reference["signed_plastic_area_Nmm"] / (B * (W - a))
        beam = reference["beam_J_reference_N_mm"]
        rows.append(
            dict(nx=reference["nx"], a_over_W=a/W, eta=eta, M_MPa=float(M),
                 Jel_N_mm=float(jel), Jpl_N_mm=jpl, J_N_mm=float(jel+jpl),
                 independent_beam_J_N_mm=beam,
                 KJ_MPa_sqrt_m=float(np.sqrt(M * (jel+jpl) / 1000)),
                 equivalent_beam_K_MPa_sqrt_m=float(np.sqrt(M * beam / 1000)),
                 difference_J_percent=float(100 * ((jel+jpl) / beam - 1)),
                 difference_equivalent_K_percent=float(100 * (np.sqrt((jel+jpl) / beam) - 1)),
                 earlier_mapped_M_difference_equivalent_K_percent=reference[
                     "Ortega_ASTM_K_difference_percent"]))
    return rows


def replay(root, material_path):
    root = Path(root).resolve()
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if (expected is not None and digest != expected) or bindings.get(str(path), digest) != digest:
            raise ValueError("Replay input fingerprint differs: " + str(path))
        bindings[str(path)] = digest
        return path

    prior = read_json(bind(root/PILOT/"pilot.json"))
    registry = read_json(bind(REGISTRY))
    if prior["cutoff_registry_sha256"] != bindings[str(REGISTRY.resolve())]:
        raise ValueError("Frozen pilot cutoff registry differs from the approved registry")
    materials = read_json(bind(material_path))["materials"]
    expected_topologies = set(registry["cutoffs_hz"])
    if set(materials) != expected_topologies:
        raise ValueError("One material record is required per topology")
    if materials["FCC"]["C_state"] != "plane_stress":
        raise ValueError("FCC production requires physical C and one prismatic reduction")
    fcc = prismatic_plane_strain(materials["FCC"]["C_MPa"], .2, 123000., .3)
    if not np.isclose(fcc["M_I_pe"], materials["FCC"]["M_MPa"], rtol=1e-8):
        raise ValueError("FCC M disagrees with the physical-tensor calculation")
    configs = {identity(row): row for row in prior["records"]}
    records = frozen_records(root, bind)
    geometry_by_case = {}
    for record in records:
        if record["task"] == "FT":
            path = bind(record["source_inp"], record["source_inp_sha256"])
            _, _, _, geometry_by_case[identity(record)], _, _ = source_geometry(path)
    output = {}
    # Resolve all paired UAT stiffnesses before any C(T) conversion.
    for record in sorted(records, key=lambda record: record["task"] == "FT"):
        key = identity(record)
        lat, task, design, sim = key
        old = configs[key]
        inp = bind(record["source_inp"], record["source_inp_sha256"])
        path = bind(record["native"], record["native_sha256"])
        prior_endpoint = old["material_endpoint"]
        event_index = None
        binding = old["event_binding"]
        if binding is not None:
            event = read_json(bind(binding["path"], binding["sha256"]))
        with np.load(path, allow_pickle=False) as values:
            time, x, y = (values[k] for k in ("time", "x", "y"))
        stress_area = None
        if task == "UT" and lat == "kagome":
            stress_area = kagome_stress_area(inp, record["normalisation_factors"][1])
            y = y * stress_area["stress_scale_from_archive"]
        if "native_index" in prior_endpoint:
            source_key = str(Path(record["source_odb"]).resolve())
            verified_odb_sha = prior["source_bindings"].get(source_key)
            if verified_odb_sha is None:
                raise ValueError("The frozen pilot did not verify the event's original ODB")
            event_index = material_event_index(record, time, x, y, event, verified_odb_sha)
            if (event_index != prior_endpoint["native_index"]
                    or not np.isclose(time[event_index], prior_endpoint["time_s"], atol=1e-12, rtol=0)
                    or Path(event["source"]).resolve() != Path(record["source_odb"]).resolve()):
                raise ValueError("Native event evidence disagrees with the frozen pilot")
        cutoff = registry["cutoffs_hz"][lat][task]
        row = curve_report(record, time, x, y, cutoff, planned_time(inp), event_index)
        if row["observed_query_mask"] != old["observed_query_mask"]:
            raise ValueError("Native sample support differs from the frozen pilot")
        fixed = row["methods"]["fixed"]
        fit = fixed["initial_work_fits"]["0.4"]
        reduced = dict(topology=lat, task=task, design=design, id=sim, cutoff_Hz=cutoff,
                       source_inp=str(inp), source_inp_sha256=record["source_inp_sha256"],
                       source_odb=record["source_odb"], source_odb_identity=old["source_odb_identity"],
                       native=str(path), native_sha256=record["native_sha256"],
                       event_binding=binding, observed_query_mask=row["observed_query_mask"],
                       recorded_samples=len(time), peak=fixed["peak_response"],
                       stiffness_fit_status=fit["status"],
                       stiffness_window_relative_spread=fixed["stiffness_window_relative_spread"])
        if stress_area is not None:
            reduced["UAT_stress_area"] = stress_area
        if task == "FT":
            realised = geometry_by_case[key]
            parent = geometry_by_case[lat, "FT", "periodic", 0]
            geometry = {k: realised[k] for k in GEOMETRY_KEYS}
            if not np.isclose(geometry["B_mm"], parent["B_mm"], atol=.005, rtol=0):
                raise ValueError("Disordered B differs from its ordered A1 geometry")
            if not np.isclose(geometry["W_mm"], parent["W_mm"], atol=.005, rtol=0):
                raise ValueError("Disordered load-line width differs from its ordered A1 geometry")
            geometry["a_mm"] = parent["a_mm"]
            end = fit.get("end_index")
            reduced.update(
                initial_CT_stiffness_N_per_mm=fit["estimate"],
                initial_CT_stiffness_fit_end_index=end,
                initial_CT_stiffness_fit_end_time_s=float(time[end]) if end is not None else None,
                initial_CT_stiffness_eligible_for_J=(
                    fit["status"] == "resolved_estimate" and event_index is not None
                    and end is not None and end <= event_index
                ),
                a0_nominal_mm=parent["a_mm"], a0_realised_mm=realised["a_mm"],
            )
            modulus = materials[lat]["M_MPa"]
            conversion = {"method": "ordered_reference", "M_MPa": modulus}
            if lat == "kagome" and design != "periodic":
                uat = output.get((lat, "UT", design, sim))
                ordered_uat = output[lat, "UT", "periodic", 0]
                current = uat["UAT_apparent_modulus_MPa"] if uat else None
                reference = ordered_uat["UAT_apparent_modulus_MPa"]
                ratio = None
                if current is not None and reference is not None:
                    if np.isfinite([current, reference]).all() and min(current, reference) > 0:
                        ratio = current / reference
                modulus = modulus * ratio if ratio is not None else None
                conversion = {
                    "method": "ordered_reference_scaled_by_paired_uat", "M_MPa": modulus,
                    "UAT_apparent_modulus_ratio": ratio,
                    "paired_UAT_input_sha256": uat["source_inp_sha256"] if uat else None,
                    "ordered_UAT_input_sha256": ordered_uat["source_inp_sha256"],
                }
            reduced["fracture_conversion"] = conversion
            reduced["compact_tension"] = (
                compact_tension_properties(
                    row, time, x, y, cutoff, geometry, modulus, fcc if lat == "FCC" else None
                ) if event_index is not None else prior_endpoint
            )
        else:
            reduced.update(
                UAT_apparent_modulus_MPa=fit["estimate"] if fit["status"] == "resolved_estimate" else None,
                ductility=fixed["ductility25_coordinate"],
                ductility_observed=fixed["ductility25_observed"],
                work_to_one_percent=fixed["work1"],
                work_to_one_percent_observed=fixed["work1_observed"],
                partial_observed_work=fixed["observed_work"],
            )
        output[key] = reduced
    output = [output[identity(record)] for record in records]
    independent = fcc_matched_field_check(root, fcc, bind)
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("Replay input changed during calculation: " + path)
    return dict(
        schema="p1_population_properties_pilot_v2",
        qualification="10 mm, nominal 20%, frozen ordered and paired disorder; nominal A1 notch used",
        records=output, inputs_sha256=bindings, materials=materials,
        FCC_matched_field_comparison=independent,
        counts=dict(Counter(row["compact_tension"]["status"] for row in output if row["task"] == "FT")),
        provenance_limits=[
            "Saved native arrays and event receipts reverified; solver databases previously verified by the fixed-cutoff pilot.",
            "Kagome UAT uses the minimum gross span between each archived main body's outer strut centrelines.",
            "Disordered kagome uses the paired UAT apparent modulus ratio. Other topologies retain their ordered conversion.",
            "Realised notch is a centreline intersection for inspection. Fracture uses the nominal A1 notch.",
            "FCC field comparison changes the conversion chain at fixed recorded load, signed work and notch.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--materials", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    report = replay(root, args.materials)
    path = root/"audit/production/population-properties-pilot-kagome-area.json"
    _write_immutable_targets({path: json.dumps(report, indent=2, allow_nan=False)+"\n"})
    print(json.dumps({"report": str(path), "histories": len(report["records"]), **report["counts"]}))


if __name__ == "__main__":
    main()
