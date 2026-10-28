"""Information/provenance checks for the native validation recalculation."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from validation_results import matched_loading_snapshot, measured_record, native_record, sha, triangle_record, verify_native_exports


class NativeResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p1_validation_result_")
        self.root = Path(self.temp.name)
        self.case = {"case_id": "fixture_tri", "topology": "tri", "l_mm": "10", "nnx": "30",
                     "nominal_relative_density": ".2", "odb": "10/0.2/Fracture-tri-30-per-1.odb",
                     "inp_sha256": "a"*64, "odb_sha256": "b"*64}
        self.event = {"prev_native_index": "1", "cross_native_index": "2", "prevtime_s": ".05",
                      "crosstime_s": ".06", "prevmaxLE11": ".109", "crossmaxLE11": ".115",
                      "undamaged_prefix_verified": "True", "hash_unchanged": "True", "source_sha256": "b"*64}
        self.rve = {"source_sha256": "a"*64, "source_input_hash_verified": True,
                    "state": "completed_and_extracted", "read_only": True,
                    "C_MPa": [[4,1,0],[1,4,0],[0,0,1.5]], "source_odb_sha256": "c"*64,
                    "job": "synthetic_RVE"}
        self.directory = self.root / "audit/history/10/0.2/Fracture-tri-30-per-1"
        self.directory.mkdir(parents=True)
        self.curve = [{"time_s": 0., "RF2_N": 0., "U2_mm": 0.},
                      {"time_s": .05, "RF2_N": 10000., "U2_mm": 1.},
                      {"time_s": .06, "RF2_N": 9500., "U2_mm": 1.1}]
        self.energy = [{"time_s": 0., "ALLIE_Nmm": 0., "ALLSE_Nmm": 0., "ALLKE_Nmm": 0.},
                       {"time_s": .05, "ALLIE_Nmm": 100., "ALLSE_Nmm": 80., "ALLKE_Nmm": .1},
                       {"time_s": .06, "ALLIE_Nmm": 110., "ALLSE_Nmm": 82., "ALLKE_Nmm": 1.}]
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        for name, rows in (("native_curve.csv", self.curve), ("native_stored_energy.csv", self.energy)):
            with (self.directory/name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        np.savez(self.directory/"native_curve.npz", time=[r["time_s"] for r in self.curve],
                 U2=[r["U2_mm"] for r in self.curve], RF2=[r["RF2_N"] for r in self.curve])

    def calculate(self, endpoint="criterion_last_intact"):
        return native_record(self.root, self.case, self.event, self.rve, endpoint)

    def test_native_energy_is_derived_and_no_second_plane_strain_conversion(self):
        r = self.calculate()
        self.assertEqual(r["derived_predamage_PD_Nmm"], 20)
        self.assertAlmostEqual(r["modulus_MPa"], 3.75)
        self.assertGreater(r["KJ_MPa_sqrt_m"], r["KI_MPa_sqrt_m"])
        self.assertFalse(r["recorded_ALLPD_available"])
        self.assertFalse(r["qualified_ASTM_J"])

    def test_crossing_damage_is_never_promoted_to_valid_energy_area(self):
        r = self.calculate("criterion_first_crossed")
        self.assertIsNone(r["KJ_MPa_sqrt_m"])
        self.assertTrue(r["value_status"].startswith("unavailable"))

    def test_wrong_source_or_unexecuted_modulus_rejected(self):
        self.event["source_sha256"] = "d"*64
        with self.assertRaisesRegex(ValueError, "Initiation event"):
            self.calculate()
        self.event["source_sha256"] = "b"*64
        self.rve["state"] = "prepared_not_submitted"
        with self.assertRaisesRegex(ValueError, "Conversion control"):
            self.calculate()

    def test_negative_opening_force_is_not_replaced_by_absolute_value(self):
        self.curve[1]["RF2_N"] = -10000.
        self.save(); r = self.calculate()
        self.assertLess(r["KI_MPa_sqrt_m"], 0)
        self.assertIsNone(r["KJ_MPa_sqrt_m"])

    def test_energy_timestamp_mismatch_rejected(self):
        self.energy[1]["time_s"] = .055
        self.save()
        with self.assertRaisesRegex(ValueError, "not time aligned"):
            self.calculate()

    def test_unverified_damage_or_large_kinetic_energy_masks_result(self):
        self.event["undamaged_prefix_verified"] = "False"
        self.assertIsNone(self.calculate()["KJ_MPa_sqrt_m"])
        self.event["undamaged_prefix_verified"] = "True"
        self.energy[1]["ALLKE_Nmm"] = 10.
        self.save(); self.assertIsNone(self.calculate()["KJ_MPa_sqrt_m"])

    def test_changed_event_index_cannot_promote_crossing_sample(self):
        self.event["prev_native_index"] = "2"
        with self.assertRaisesRegex(ValueError, "bracket"):
            self.calculate()

    def test_event_time_and_strain_bracket_are_bound_to_the_sample(self):
        self.event["prevtime_s"] = ".049"
        with self.assertRaisesRegex(ValueError, "index/time"):
            self.calculate()
        self.event["prevtime_s"] = ".05"
        self.event["prevmaxLE11"] = ".111"
        with self.assertRaisesRegex(ValueError, "bracket"):
            self.calculate()

    def test_negative_stored_or_kinetic_energy_is_invalid(self):
        for channel in ("ALLSE_Nmm", "ALLKE_Nmm"):
            original = self.energy[1][channel]
            self.energy[1][channel] = -1.
            self.save(); self.assertIsNone(self.calculate()["KJ_MPa_sqrt_m"])
            self.energy[1][channel] = original

    def test_negative_triangle_is_preserved_and_not_promoted_to_toughness(self):
        native = self.calculate()
        control = {"source_sha256": self.case["inp_sha256"], "state": "completed_and_extracted",
                   "read_only": True, "source_input_hash_verified": True, "static_stiffness_N_mm": 1.,
                   "source_odb_sha256": "e"*64, "job": "unit_load_fixture"}
        rec = triangle_record(native, control)
        self.assertLess(rec["Apl_Nmm"], 0)
        self.assertIsNone(rec["KJ_MPa_sqrt_m"])
        self.assertEqual(rec["method"], "Hsieh_initial_triangle_native")
        control["source_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "compliance"):
            triangle_record(native, control)

    def test_energy_csv_tampering_is_detected_by_extraction_receipt(self):
        receipt = {"read_only": True, "source_unchanged": True,
                   "before": {"sha256": self.case["odb_sha256"]},
                   "after": {"sha256": self.case["odb_sha256"]},
                   "csv_sha256": sha(self.directory/"native_stored_energy.csv")}
        (self.directory/"summary.json").write_text(json.dumps(receipt))
        proofs = {str((self.root/self.case["odb"]).resolve()).lower(): receipt}
        verify_native_exports(self.root, self.case, proofs)
        self.energy[1]["ALLIE_Nmm"] = 999.
        self.save()
        with self.assertRaisesRegex(ValueError, "differs from extraction"):
            verify_native_exports(self.root, self.case, proofs)

    def test_curve_csv_must_match_the_separate_native_array_export(self):
        receipt = {"read_only": True, "source_unchanged": True,
                   "before": {"sha256": self.case["odb_sha256"]}, "after": {"sha256": self.case["odb_sha256"]},
                   "csv_sha256": sha(self.directory/"native_stored_energy.csv")}
        (self.directory/"summary.json").write_text(json.dumps(receipt))
        proofs = {str((self.root/self.case["odb"]).resolve()).lower(): receipt}
        path = self.directory/"native_curve.csv"
        path.write_text(path.read_text().replace("10000.0", "20000.0"))
        with self.assertRaisesRegex(ValueError, "NPZ export"):
            verify_native_exports(self.root, self.case, proofs)

    def test_measured_recovery_rejects_nonfinite_checks_and_bad_energy_closure(self):
        result = {"case_id": self.case["case_id"], "source_sha256": self.case["inp_sha256"],
                  "state": "completed_and_extracted", "read_only": True,
                  "source_input_hash_verified": True, "retained_solver_input_hash_verified": True,
                  "loading_section_check": {"maximum_tensile_LE11": .11045, "passed_tensile_criterion": True},
                  "precriterion_peak_verified": True,
                  "loading_peak_state": {"force_N": 10000., "U2_mm": 1.},
                  "work_partition": {"nonrecovered_work_Nmm": 100., "loading_work_Nmm": 5000., "measured_recovery_Nmm": 4900.},
                  "peak_metrics": {"conversion_modulus_MPa": 3.75},
                  "maximum_abs_other_energy_channels_Nmm": {k: 0. for k in ("ALLAE", "ALLCD", "ALLDMD", "ALLSD", "ALLVD")},
                  "force_balance_relative_error": 0., "release_force_ramp_relative_error": 0.,
                  "free_loading_pin_RF2_relative": 0., "relative_cycle_closure_error": 1e-5,
                  "residual_elastic_energy_Nmm": 1., "unload_added_plastic_dissipation_Nmm": 1.,
                  "source_odb_sha256": "e"*64, "job": "synthetic_unload", "matched_load_result_sha256": "f"*64}
        self.assertGreater(measured_record(result, self.case, self.rve)["KJ_MPa_sqrt_m"], 0)
        for name, bad in (("force_balance_relative_error", float("nan")),
                          ("relative_cycle_closure_error", .5)):
            original = result[name]; result[name] = bad
            with self.assertRaisesRegex(ValueError, "eligibility"):
                measured_record(result, self.case, self.rve)
            result[name] = original
        result["peak_metrics"]["conversion_modulus_MPa"] = float("nan")
        with self.assertRaisesRegex(ValueError, "eligibility"):
            measured_record(result, self.case, self.rve)

    def test_pinned_loading_snapshot_survives_later_additive_metadata(self):
        original = self.root/"result.json"
        original.write_text(json.dumps({"source_sha256": "a", "source_odb_sha256": "b"}))
        digest = sha(original)
        archive = self.root/"extraction-history"/(digest+".json")
        archive.parent.mkdir(); archive.write_bytes(original.read_bytes())
        original.write_text("{}")
        result = {"matched_load_result": str(original), "matched_load_result_sha256": digest,
                  "source_sha256": "a", "matched_loading_odb_sha256": "b"}
        self.assertEqual(matched_loading_snapshot(result), str(archive))
        archive.unlink()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            matched_loading_snapshot(result)


if __name__ == "__main__":
    unittest.main()
