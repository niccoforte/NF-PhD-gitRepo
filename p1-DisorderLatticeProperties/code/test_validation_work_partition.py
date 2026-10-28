"""Mechanics/information-loss checks for the validation-only work partition."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation_work_partition import (audit_controls, ct_rotation_factor, delivered_odb_hash, distinct_history_times,
    early_elastic_stiffness_windows, eta_energy_metrics, initial_triangle_metrics,
    internal_energy_partition, interpolated_peak,
    measured_recovery_partition, retain_prepared_input, save_result_versioned, work)


class WorkPartitionTests(unittest.TestCase):
    def test_linear_reversible_cycle(self):
        u = np.linspace(0, 3, 100)
        result = measured_recovery_partition(u, 7*u, u[::-1], 7*u[::-1], initial_stiffness=7)
        self.assertAlmostEqual(result["nonrecovered_work_Nmm"], 0, places=12)
        self.assertAlmostEqual(result["initial_triangle_apparent_Apl_Nmm"], 0, places=12)

    def test_geometric_stiffening_is_not_negative_plasticity(self):
        u = np.linspace(0, 3, 1001)
        p = 7*u+.4*u**3
        result = measured_recovery_partition(u, p, u[::-1], p[::-1], initial_stiffness=7)
        self.assertAlmostEqual(result["nonrecovered_work_Nmm"], 0, places=11)
        self.assertLess(result["initial_triangle_apparent_Apl_Nmm"], -10)
        self.assertFalse(result["is_certified_ASTM_Apl"])

    def test_plastic_residual_cycle(self):
        # Piecewise bilinear plastic loading followed by elastic unloading to a
        # residual displacement; independent polygon work gives area1.25.
        result = measured_recovery_partition([0, 1, 2], [0, 2, 3], [2, .5], [3, 0])
        self.assertAlmostEqual(result["nonrecovered_work_Nmm"], 1.25)
        self.assertAlmostEqual(result["residual_displacement_mm"], .5)

    def test_signed_work_preserves_unloading(self):
        self.assertEqual(work([0, 1], [0, 2]), 1)
        self.assertEqual(work([1, 0], [2, 0]), -1)

    def test_reverse_loading_tail_rejected(self):
        with self.assertRaisesRegex(ValueError, "reverse-loading"):
            measured_recovery_partition([0, 1], [0, 2], [1, 0, -.1], [2, 0, -.1])

    def test_unmatched_peak_rejected(self):
        with self.assertRaisesRegex(ValueError, "same measured"):
            measured_recovery_partition([0, 1], [0, 2], [.9, .1], [2, 0])

    def test_missing_loading_origin_is_not_full_recovery_area(self):
        with self.assertRaisesRegex(ValueError, "initial zero"):
            measured_recovery_partition([.5, 1], [1, 2], [1, 0], [2, 0])

    def test_missing_energy_is_not_zero(self):
        result = internal_energy_partition(100, 80, excluded_terms=("ALLCD", "ALLDMD", "ALLDC", "ALLFC"))
        self.assertEqual(result["unknown_components"], ["ALLPD", "ALLAE"])
        self.assertEqual(result["plastic_dissipation_lower_Nmm"], 0)
        self.assertEqual(result["plastic_dissipation_upper_Nmm"], 20)
        self.assertFalse(result["is_ASTM_Apl"])

    def test_recorded_plastic_component_is_identified(self):
        result = internal_energy_partition(100, 80, channels={"ALLPD": 18, "ALLAE": 2},
                    excluded_terms=("ALLCD", "ALLDMD", "ALLDC", "ALLFC"))
        self.assertEqual(result["plastic_dissipation_lower_Nmm"], 18)
        self.assertEqual(result["plastic_dissipation_upper_Nmm"], 18)

    def test_unknown_signed_fluid_energy_prevents_bound(self):
        result = internal_energy_partition(100, 80)
        self.assertIsNone(result["plastic_dissipation_upper_Nmm"])

    def test_source_exclusion_must_not_contradict_observation(self):
        with self.assertRaisesRegex(ValueError, "contradicts"):
            internal_energy_partition(100, 80, channels={"ALLDMD": 1}, excluded_terms=("ALLDMD",))

    def test_rotation_zero_displacement_and_geometry_domain(self):
        self.assertAlmostEqual(float(ct_rotation_factor(0, 100, 50, 37.5, 37.5)), 1)
        self.assertLess(float(ct_rotation_factor(3, 100, 50, 37.5, 37.5)), 1)
        with self.assertRaises(ValueError):
            ct_rotation_factor(1000, 100, 50, 37.5, 37.5)

    def test_eta_zero_area_preserves_KI_and_units(self):
        result = eta_energy_metrics(3, 1000, 2, 10, 20, 0)
        self.assertEqual(result["K_eta_MPa_sqrt_m"], 3)
        self.assertEqual(result["Jel_N_per_mm"], 9)
        self.assertFalse(eta_energy_metrics(3, 1000, 2, 10, 20, -1)["available"])

    def test_hsieh_triangle_keeps_signed_negative_area_unavailable(self):
        zero = initial_triangle_metrics(9, 6, 2, 3, 1000, 2, 10, 20)
        self.assertEqual(zero["signed_Apl_Nmm"], 0)
        self.assertEqual(zero["KJ_MPa_sqrt_m"], 3)
        negative = initial_triangle_metrics(8, 6, 2, 3, 1000, 2, 10, 20)
        self.assertEqual(negative["signed_Apl_Nmm"], -1)
        self.assertIsNone(negative["KJ_MPa_sqrt_m"])
        self.assertFalse(negative["available"])

    def test_elastic_window_excludes_all_post_yield_frames(self):
        curve = [{"U2_mm": 0, "force_N": 0}, {"U2_mm": 1, "force_N": 7},
                 {"U2_mm": 2, "force_N": 14}, {"U2_mm": 3, "force_N": 20}]
        frames = [{"history_index": i, "maximum_PEEQ": 0 if i<3 else .001,
                   "U2_mm": row["U2_mm"], "maximum_tensile_LE11": .001*i} for i,row in enumerate(curve)]
        windows = early_elastic_stiffness_windows(curve, frames)
        self.assertTrue(all(row["samples"]<=2 and row["stiffness_through_origin_N_mm"]==7 for row in windows))

    def test_invalid_nonfinite_observation_is_rejected(self):
        with self.assertRaises(ValueError): work([0, 1], [0, np.nan])
        with self.assertRaises(ValueError): internal_energy_partition(100, np.nan)

    def test_peak_interpolation_is_observed_and_not_extrapolated(self):
        frames = [{"maximum_tensile_LE11": 0, "U2_mm": 0},
                  {"maximum_tensile_LE11": .10, "U2_mm": 8},
                  {"maximum_tensile_LE11": .12, "U2_mm": 10}]
        target, record = interpolated_peak(frames, .11)
        self.assertAlmostEqual(target, 9)
        self.assertAlmostEqual(record["interpolation_fraction"], .5)
        with self.assertRaises(ValueError): interpolated_peak(frames[:2], .11)
        with self.assertRaises(ValueError): interpolated_peak(frames, .111)

    def test_pinned_input_guard_and_idempotence(self):
        with tempfile.TemporaryDirectory(prefix="work-partition-guard-") as temporary:
            directory = Path(temporary)/"job"
            meta = {"source_sha256": "original", "target_U2_mm": 8.1}
            first = retain_prepared_input(directory, "job", "*Heading\noriginal\n", meta)
            timestamp = (directory/"job.inp").stat().st_mtime_ns
            again = retain_prepared_input(directory, "job", "*Heading\noriginal\n", meta)
            self.assertEqual(first, again)
            self.assertEqual(timestamp, (directory/"job.inp").stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, "changed existing"):
                retain_prepared_input(directory, "job", "*Heading\nchanged\n", meta)
            occupied = Path(temporary)/"occupied"; occupied.mkdir()
            (occupied/"state.json").write_text("started")
            with self.assertRaisesRegex(ValueError, "occupied"):
                retain_prepared_input(occupied, "job", "input", meta)

    def test_planar_two_node_beam_has_only_three_rigid_null_modes(self):
        # Independent kinematic rank argument, not a claim to duplicate Abaqus's
        # shear-correction implementation. q=[u1,v1,theta1,u2,v2,theta2].
        L = 3.
        modes = np.array([[-1/L, 0, 0, 1/L, 0, 0],
                          [0, -1/L, -.5, 0, 1/L, -.5],
                          [0, 0, -1/L, 0, 0, 1/L]])
        self.assertEqual(np.linalg.matrix_rank(modes), 3)
        rigid = np.array([[1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0],
                          [0, 0, 1, 0, L, 1]]).T
        np.testing.assert_allclose(modes@rigid, 0, atol=1e-14)
        self.assertEqual(np.linalg.matrix_rank(rigid), 3)

    def test_duplicate_static_history_requires_identical_observations(self):
        arrays = {"u": np.array([[0., 0], [1, 2], [1, 2]])}
        retained, indices = distinct_history_times(arrays)
        self.assertEqual(indices, [2])
        self.assertEqual(len(retained["u"]), 2)
        arrays["energy"] = np.array([[0., 0], [1, 3], [1, 3.1]])
        with self.assertRaisesRegex(ValueError, "differing repeated-time"):
            distinct_history_times(arrays)
        retained, indices = distinct_history_times(arrays, allow_terminal_ambiguity=True)
        self.assertEqual(indices, [1, 2])
        self.assertEqual(len(retained["u"]), 1)
        longer = {"u": np.array([[0., 0], [1, 2], [1, 2], [1, 2.1]])}
        retained, indices = distinct_history_times(longer, allow_terminal_ambiguity=True)
        self.assertEqual(indices, [1, 2, 3])
        self.assertEqual(len(retained["u"]), 1)

    def test_changed_extraction_keeps_hash_addressed_previous_version(self):
        import hashlib
        with tempfile.TemporaryDirectory(prefix="work-partition-version-") as temporary:
            path = Path(temporary)/"result.json"
            save_result_versioned(path, {"point": 1})
            previous = path.read_bytes(); digest = hashlib.sha256(previous).hexdigest()
            save_result_versioned(path, {"point": 2})
            self.assertEqual((path.parent/"extraction-history"/(digest+".json")).read_bytes(), previous)
            self.assertEqual(json.loads(path.read_text())["point"], 2)

    def test_only_verified_completed_odb_delivery_can_be_opened(self):
        import hashlib
        with tempfile.TemporaryDirectory(prefix="work-partition-delivery-") as temporary:
            path = Path(temporary)/"job.odb"; path.write_bytes(b"fictional ODB")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (path.parent/"delivery.json").write_text(json.dumps({"files": [{"file": path.name,
                "sha256": digest, "bytes": path.stat().st_size}]}))
            self.assertEqual(delivered_odb_hash(path), digest)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from completed queue delivery"):
                delivered_odb_hash(path)

    def test_actual_matched_unload_controls(self):
        root = Path("Z:/p1/sims/Ti/Validation")
        if not (root/"audit/diagnostics/epUnloadPre_hex_3_rho0p16_nx80/result.json").exists():
            self.skipTest("Exact recorded controls unavailable")
        with tempfile.TemporaryDirectory(prefix="work-partition-check-") as temporary:
            results = audit_controls(root, Path(temporary))
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]["passed_original_source_criterion"])
        self.assertTrue(results[1]["passed_original_source_criterion"])
        for row in results:
            self.assertGreater(row["nonrecovered_work_Nmm"], 0)
            self.assertLess(row["initial_triangle_apparent_Apl_Nmm"], 0)
            self.assertLess(row["rotation_only_apparent_Apl_Nmm"], 0)
            self.assertLess(row["relative_cycle_closure_error"], 1e-4)
            self.assertEqual(row["ALLAE_loading_Nmm"], 0)


if __name__ == "__main__":
    unittest.main()
