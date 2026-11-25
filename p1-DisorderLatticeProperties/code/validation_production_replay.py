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
from validation_fcc import source_geometry
from validation_diagnostics import source_mesh
from validation_stiffness import _write_immutable_targets, sha


PILOT = Path("audit/curve-processing/accepted-filter-study/fixed-cutoff-pilot")
GEOMETRY_KEYS = ("a_mm", "B_mm", "W_mm")


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
    ordered = {}
    for record in records:
        if record["task"] == "FT" and record["design"] == "periodic":
            path = bind(record["source_inp"], record["source_inp_sha256"])
            _, _, _, ordered[record["topology"]], _, _ = source_geometry(path)
    output = []
    for record in records:
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
                       stiffness=fit["estimate"], stiffness_fit_status=fit["status"],
                       stiffness_window_relative_spread=fixed["stiffness_window_relative_spread"])
        if task == "FT" and event_index is not None:
            geometry = {k: ordered[lat][k] for k in GEOMETRY_KEYS}
            if design != "periodic":
                _, _, _, section, _ = source_mesh(inp)
                if not np.isclose(section[0], geometry["B_mm"], atol=.005, rtol=0):
                    raise ValueError("Disordered B differs from its ordered A1 geometry")
                geometry["B_mm"] = section[0]
            reduced.update(compact_tension=compact_tension_properties(
                row, time, x, y, cutoff, geometry, materials[lat]["M_MPa"],
                fcc if lat == "FCC" else None),
                material_scope="source_family_reference" if design == "periodic" else
                "representative_ordered_tensor_sensitivity")
        elif task == "FT":
            reduced["compact_tension"] = prior_endpoint
        else:
            reduced.update(
                ductility=fixed["ductility25_coordinate"],
                ductility_observed=fixed["ductility25_observed"],
                work_to_one_percent=fixed["work1"],
                work_to_one_percent_observed=fixed["work1_observed"],
                partial_observed_work=fixed["observed_work"],
            )
        output.append(reduced)
    independent = fcc_matched_field_check(root, fcc, bind)
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("Replay input changed during calculation: " + path)
    return dict(
        schema="p1_shared_mechanics_pilot_v1",
        qualification="10 mm, nominal 20%, frozen ordered and paired disorder; nominal A1 notch used",
        records=output, inputs_sha256=bindings, materials=materials,
        FCC_matched_field_comparison=independent,
        counts=dict(Counter(row["compact_tension"]["status"] for row in output if row["task"] == "FT")),
        provenance_limits=[
            "Saved native arrays and event receipts reverified; solver databases previously verified by the fixed-cutoff pilot.",
            "Disorder values use a representative ordered conversion modulus provisionally.",
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
    path = root/"audit/production/shared-mechanics-pilot.json"
    _write_immutable_targets({path: json.dumps(report, indent=2, allow_nan=False)+"\n"})
    print(json.dumps({"report": str(path), "histories": len(report["records"]), **report["counts"]}))


if __name__ == "__main__":
    main()
