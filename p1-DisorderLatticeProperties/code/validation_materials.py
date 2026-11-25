"""Source-bound ordered conversion controls for the retained Ti population.

This module prepares only missing linear periodic B21 controls. Submission stays
with validation_queue. Isotropic controls use their mapped constituent once;
FCC retains the accepted physical-C prismatic reduction, with mapped C separate.
"""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from validation_diagnostics import (extract, periodic_patch, prepare_rve, sha,
                                    source_mesh)


def _write(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")


def canonical_sources(root):
    """Read frozen canonical identities; hash inputs without opening their ODBs."""
    index = Path(root) / "audit/curve-processing/periodic-cutoff-native/index.json"
    selected = [r for r in json.loads(index.read_text())
                if r["task"] == "FT" and r["design"] == "periodic"]
    if len(selected) != 4 or len({r["topology"] for r in selected}) != 4:
        raise ValueError("Expected exactly four canonical periodic fracture inputs")
    for row in selected:
        if sha(row["source_inp"]) != row["source_inp_sha256"]:
            raise ValueError("Canonical input differs from its frozen source receipt")
    return {r["topology"]: r for r in selected}


def qualify_aligned_tensor(matrix, tolerance=2e-6):
    """Remove only bounded numerical residuals from declared aligned controls.

    Full raw C and every residual are retained. This is not a projection of a
    disordered tensor onto an assumed symmetry class. The bound accommodates
    single-precision macro-reaction output from these periodic B21 controls.
    """
    raw = np.asarray(matrix, dtype=float)
    if raw.shape != (3, 3) or not np.isfinite(raw).all():
        raise ValueError("Expected finite engineering-shear C of shape3x3")
    scale = float(np.max(np.abs(raw)))
    if scale <= 0:
        raise ValueError("Tensor scale must be positive")
    symmetry = float(np.max(np.abs(raw-raw.T))/scale)
    coupling = float(np.max(np.abs(np.r_[raw[:2, 2], raw[2, :2]]))/scale)
    if max(symmetry, coupling) > tolerance:
        raise ValueError("Tensor has resolved asymmetry or normal/shear coupling")
    result = (raw+raw.T)/2
    result[:2, 2] = 0.
    result[2, :2] = 0.
    if np.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("Tensor is not positive definite")
    return result, {"tolerance": tolerance, "symmetry_relative_residual": symmetry,
                    "normal_shear_relative_residual": coupling,
                    "C_MPa_raw": raw.tolist(),
                    "action": "Average numerical antisymmetry and zero bounded normal/shear residuals only"}


def prepare(root, out):
    """Prepare the two source-exact canonical controls absent from the campaign."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = canonical_sources(root)
    target = out/"serial-material-proposal.json"
    if target.exists() and (out/"queue/state.json").exists():
        manifest = json.loads(target.read_text())
        for job in manifest["jobs"]:
            metadata = json.loads((Path(job["input"]).parent/"source.json").read_text())
            if sha(job["input"]) != job["input_sha256"] or sha(metadata["source_inp"]) != metadata["source_sha256"]:
                raise ValueError("Submitted material source/input changed")
        return target
    jobs = []
    for topology in ("kagome", "hex"):
        row = sources[topology]
        source = Path(row["source_inp"])
        _, nodes, elements, section, elastic = source_mesh(source)
        patch = periodic_patch(nodes, elements, topology, 10.)
        coords, edges, classes, periods, origin, mismatch = patch
        case = {"topology": topology, "l_mm": "10", "nominal_relative_density": "0.2",
                "nnx": "20", "case_id": topology+"-canonical-population-CT-mapped"}
        metadata = prepare_rve(out, case, source, coords, edges, classes, periods, section, elastic)
        metadata.update(material_state="already mapped source beam constituent; no second reduction",
                        source_receipt_sha256=row["source_inp_sha256"],
                        origin_mm=origin.tolist(), opposite_boundary_mismatch_mm=mismatch)
        _write(Path(metadata["input"]).parent/"source.json", metadata)
        jobs.append({"job": metadata["job"], "input": metadata["input"],
                     "input_sha256": metadata["input_sha256"], "cpus": 8,
                     "kind": "canonical_population_periodic_B21_material",
                     "purpose": "Exact retained canonical source section and subdivision, three elastic strains",
                     "allow_solver_failure": False, "required_outputs": [".odb", ".dat", ".sta"]})
    manifest = {"abaqus": "C:/SIMULIA/Commands/abaqus.bat", "queue_root": str(out/"queue"), "jobs": jobs}
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Refuse to replace a different material-control manifest")
    _write(target, manifest)
    return target


def _mapped_control(record, topology):
    from resources.lattices import prismatic_plane_strain
    C = np.asarray(record["C_MPa"], dtype=float)
    scale = float(np.max(np.abs(C)))
    if (C.shape != (3, 3) or not np.isfinite(C).all() or scale <= 0
            or np.max(np.abs(C-C.T))/scale > 1e-4
            or np.linalg.eigvalsh((C+C.T)/2).min() <= 0):
        raise ValueError("Mapped control must be finite, symmetric and positive definite")
    S = np.linalg.inv(C)
    modulus = float(1/S[0, 0])
    # This sensitivity explicitly drops measured coupling; its M is never the
    # adopted quasi-isotropic conversion or evidence of an aligned source model.
    aligned = (C+C.T)/2
    aligned[:2, 2] = 0.
    aligned[2, :2] = 0.
    orthotropic = prismatic_plane_strain(aligned, C_state="plane_strain")
    shear_residual = float(C[2, 2]/(.5*(C[0, 0]-C[0, 1]))-1)
    isotropic = np.array([[C[0, 0], C[0, 1], 0.], [C[0, 1], C[0, 0], 0.],
                          [0., 0., .5*(C[0, 0]-C[0, 1])]])
    residual = float(np.max(np.abs(C-isotropic))/scale)
    return {"topology": topology, "stress_unit": "MPa", "C_state": "plane_strain",
            "C_MPa": C.tolist(), "M_MPa": modulus,
            "material_model": "source-matched periodic B21, mapped constituent used once",
            "source_inp": record["source_inp"], "source_inp_sha256": record["source_sha256"],
            "source_odb": record["source_odb"], "source_odb_sha256": record["source_odb_sha256"],
            "control_input": record["input"], "control_input_sha256": record["input_sha256"],
            "section_B_t_mm": record["section_B_t_mm"],
            "elastic_MPa_nu": record["elastic_MPa_nu"],
            "energy_checks": record["energy_checks"],
            "diagnostics": {"normalized_isotropic_tensor_error": residual,
                "prior_strict_isotropic_threshold": .001,
                "prior_strict_isotropic_threshold_passed": residual <= .001,
                "shear_isotropy_relative_residual": shear_residual,
                "normal_shear_relative_residual": float(np.max(np.abs(C[:2, 2]))/scale),
                "alternative_1_over_S22_MPa": float(1/S[1, 1]),
                "alternative_1_over_S22_relative_difference": float((1/S[1, 1])/modulus-1),
                "aligned_orthotropic_sensitivity_M_MPa": orthotropic["M_I_pe"],
                "aligned_orthotropic_sensitivity_relative_difference": orthotropic["M_I_pe"]/modulus-1,
                "aligned_orthotropic_sensitivity_limit": "Measured normal/shear couplings explicitly dropped for this sensitivity only; raw C and accepted scalar1/S11 unchanged"},
            "conversion_rule": "1/S11 of raw source-matched mapped C under the accepted quasi-isotropic approximation; no second reduction",
            "approximation": "Ordered constitutive reference; not each disordered specimen's UAT slope or CT k0"}


def _verify_control(record):
    if (record.get("state") != "completed_and_extracted"
            or record.get("read_only") is not True
            or record.get("source_input_hash_verified") is not True):
        raise ValueError("Ordered control lacks completed read-only extraction evidence")
    energies = record.get("energy_checks", [])
    if (len(energies) != 3 or any(r["ALLSE_N_mm"] <= 0 or not np.isfinite(r["relative_error"])
                                or r["relative_error"] > 1e-4 for r in energies)):
        raise ValueError("Ordered control lacks three qualified work-energy checks")
    odb = Path(record["source_odb"])
    if sha(odb) != record["source_odb_sha256"]:
        raise ValueError("Extracted control ODB changed")
    receipt = odb.parent/"delivery.json"
    delivery = json.loads(receipt.read_text())
    if delivery.get("required_outputs_missing"):
        raise ValueError("Incomplete ordered control delivery")
    entries = {r["file"]: r["sha256"] for r in delivery["files"]}
    if (entries.get(odb.name) != record["source_odb_sha256"]
            or entries.get(record["job"]+".inp") != record["input_sha256"]):
        raise ValueError("Ordered control delivery differs from extracted input or ODB")
    return {"delivery_receipt": str(receipt), "delivery_sha256": sha(receipt),
            "extraction_state": record["state"], "control_odb_hash_reverified": True}


def build_catalogue(root, out):
    """Bind the three exact ordered beam controls and accepted physical FCC C."""
    from resources.lattices import Geometry, calcC_mohr, calcC_sims, prismatic_plane_strain
    root, out = Path(root), Path(out)
    sources = canonical_sources(root)
    records = {}
    tri_path = root/"audit/controls/rve_tri_10_rho0p2_nx30/result.json"
    for topology in ("tri", "kagome", "hex"):
        path = tri_path if topology == "tri" else out/f"rve_{topology}_10_rho0p2_nx20/result.json"
        record = json.loads(path.read_text())
        canonical = Path(sources[topology]["source_inp"])
        _, nodes, elements, section, elastic = source_mesh(canonical)
        if not np.array_equal(section, record["section_B_t_mm"]) or not np.array_equal(elastic, record["elastic_MPa_nu"]):
            raise ValueError("Ordered control material/section differs from retained source")
        if sha(record["source_inp"]) != record["source_sha256"] or sha(record["input"]) != record["input_sha256"]:
            raise ValueError("Material-control source/input changed")
        delivery = _verify_control(record)
        # Compare periodic patches explicitly when reusing the independent tri
        # archive source. Different full specimen hashes are retained as such.
        current = periodic_patch(nodes, elements, topology, 10.)
        _, other_nodes, other_elements, _, _ = source_mesh(record["source_inp"])
        original = periodic_patch(other_nodes, other_elements, topology, 10.)
        if any(a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-8)
               for a, b in zip(current[:4], original[:4])):
            raise ValueError("Reused ordered periodic mesh differs from retained source")
        item = _mapped_control(record, topology)
        item.update(canonical_source_inp=str(canonical), canonical_source_sha256=sha(canonical),
                    result_file=str(path), result_sha256=sha(path),
                    source_reuse_proof="Identical periodic coordinates, connectivity, classes, periods, section and assigned elasticity")
        item.update(delivery)
        records[topology] = item
    geom = Geometry("FCC", 10., 20)
    C = calcC_mohr(copy.deepcopy(geom), "unit", E_s=123000.)[0]
    elastic = prismatic_plane_strain(C, geom.rD, E_s=123000., v_s=.3, C_state="plane_stress")
    history_root = Path("Z:/p1/sims/Ti/StiffMatrix/Cmatrix-periodic")
    historical = calcC_sims("FCC", 20, pDir=history_root.as_posix()+"/")[0]
    history_elastic = prismatic_plane_strain(historical, .2, E_s=123000., v_s=.3)
    history_files = [history_root/f"Ductile-FCC-20-Cmatrix{s}-per-1.inp" for s in "ABC"]
    history_files += sorted((history_root/"transfer").glob("OUT-Ductile-FCC-20-Cmatrix?-per-1-?.csv"))
    history_materials = []
    for path in history_files[:3]:
        _, _, _, section, assigned = source_mesh(path)
        history_materials.append({"path": str(path), "section_B_t_mm": section, "elastic_MPa_nu": assigned})
    mapped_path = root/"audit/fcc-calibration/periodic/tensor-results.json"
    mapped = json.loads(mapped_path.read_text())[0]
    mapped_C, roundoff = qualify_aligned_tensor(mapped["C_MPa"])
    mapped_elastic = prismatic_plane_strain(mapped_C, C_state="plane_strain")
    records["FCC"] = {"topology": "FCC", "stress_unit": "MPa", "C_state": "plane_stress",
                      "C_MPa": C.tolist(), "M_MPa": elastic["M_I_pe"],
                      "S_xy_pe_per_MPa": elastic["S_xy_pe"].tolist(),
                      "lambda": elastic["lambda_pe"], "chi": elastic["chi_pe"],
                      "E_s_MPa": 123000., "nu_s": .3, "nominal_relative_density": .2,
                      "crack_axis": 0, "ortega_axis1": 1,
                      "material_model": "Physical canonical periodic axial-affine C plus one accepted prismatic ROM reduction",
                      "source_inp": sources["FCC"]["source_inp"],
                      "source_inp_sha256": sources["FCC"]["source_inp_sha256"],
                      "canonical_t_mm": geom.t,
                      "historical_physical_control": {"C_MPa_projected_by_existing_reader": historical.tolist(),
                          "M_MPa": history_elastic["M_I_pe"],
                          "M_relative_difference": history_elastic["M_I_pe"]/elastic["M_I_pe"]-1,
                          "materials": history_materials,
                          "inputs": {str(p): sha(p) for p in history_files},
                          "limit": "Historical t0.405746 differs from canonical t0.41421356; existing reader averages normal terms and removes couplings. Supporting sensitivity, not exact source-match or independent proof of all tensor entries."},
                      "mapped_source_sensitivity": {"M_MPa": mapped_elastic["M_I_pe"],
                          "result_file": str(mapped_path), "result_sha256": sha(mapped_path),
                          "roundoff_qualification": roundoff,
                          "limit": "Direct mapped B21 tensor; sensitivity only, not the accepted physical-C ROM production route"},
                      "approximation": "Representative ordered FCC tensor; no per-ID tensor inference from UAT stiffness"}
    document = {"schema": "p1_ordered_material_controls_v1", "materials": records,
                "source_index": str(root/"audit/curve-processing/periodic-cutoff-native/index.json"),
                "source_index_sha256": sha(root/"audit/curve-processing/periodic-cutoff-native/index.json"),
                "source_module_sha256": sha(__file__),
                "scope": "Canonical10mm/20% ordered references; applicability to disorder is an explicit constitutive approximation"}
    _write(out/"material-catalogue.json", document)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract", "catalogue"))
    parser.add_argument("--root", default="Z:/p1/sims/Ti/Validation")
    parser.add_argument("--out", default="Z:/p1/sims/Ti/Validation/audit/controls/population-materials")
    args = parser.parse_args()
    if args.action == "extract":
        # Run with Abaqus Python; catalogue uses the ordinary research runtime.
        extract(SimpleNamespace(root=Path(args.root), out=Path(args.out), queue_root=Path(args.out)/"queue"))
    elif args.action == "prepare":
        print(prepare(args.root, args.out))
    else:
        result = build_catalogue(args.root, args.out)
        print(json.dumps({k: v["M_MPa"] for k, v in result["materials"].items()}))
