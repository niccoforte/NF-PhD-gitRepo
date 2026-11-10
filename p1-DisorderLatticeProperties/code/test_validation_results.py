"""Information/provenance checks for the native validation recalculation."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from validation_results import (completion_records, filtered_native_record, geometry_and_ki,
                                matched_loading_snapshot, measured_record, native_record,
                                native_triangle_record, sha, triangle_record, verify_native_exports)


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

    def test_additive_catalogue_uses_the_actual_source_database_identity(self):
        receipt = {"read_only": True, "source_unchanged": True,
                   "before": {"sha256": self.case["odb_sha256"]},
                   "after": {"sha256": self.case["odb_sha256"]},
                   "csv_sha256": sha(self.directory/"native_stored_energy.csv")}
        (self.directory/"summary.json").write_text(json.dumps(receipt))
        self.case["source_odb"] = str(self.root/"retained-queue/result.odb")
        actual = str(Path(self.case["source_odb"]).resolve()).lower()
        verify_native_exports(self.root, self.case, {actual:receipt})
        with self.assertRaises(KeyError):
            verify_native_exports(self.root, self.case,
                                  {str((self.root/self.case["odb"]).resolve()).lower():receipt})

    def test_unsettled_new_extraction_cannot_enter_the_final_results(self):
        out = self.root/"audit/new-validation/results"
        out.mkdir(parents=True)
        (out/"extraction-summary.json").write_text('{"state":"extracting"}')
        with self.assertRaisesRegex(ValueError, "still running"):
            completion_records(self.root)

    def test_restart_uses_frozen_parent_generation_without_changing_attempt_identity(self):
        base = self.root / "audit/new-validation"
        out = base / "results"
        event = out / "audit/failure/failure-initiation.csv"
        event.parent.mkdir(parents=True)
        event.write_text("case_id\nfixture\n")
        energy = out / "audit/history/stored-energy-summary.json"
        energy.parent.mkdir(parents=True)
        source = self.root / "source.inp"
        source.write_text("Frozen source fixture")
        case = {"case_id":"fixture", "job":"original_r1", "parent_job":"original",
                "inp":str(source), "inp_sha256":sha(source), "source_odb":str(self.root/"retry.odb")}
        energy.write_text(json.dumps([{"source":case["source_odb"]}]))
        index = out / "case-index.csv"
        index.write_text("fixture")
        detail = out / "event.json"
        detail.write_text("{}")
        history = out / "history"
        history.mkdir()
        (history / "native_stored_energy.csv").write_text("time_s\n0\n")
        proof = {"case":case, "source_unchanged":True, "read_only":True, "exports":[],
                 "event_json":str(detail), "event_json_sha256":sha(detail),
                 "initiation_prefix_qualified":True, "state":"completed_and_extracted",
                 "energy_prefix_check":{"positive_IE":{"ke_ie_max":2.0},
                     "startup_sensitivity_t_ge_0p1_event_IE_gt_1pct_max":{"ke_ie_max":.2}},
                 "history_export_directory":str(history)}
        (history / "extraction.json").write_text("{}")
        (out / "extraction-summary.json").write_text(json.dumps({"state":"complete", "cases":[proof],
            "event_csv_sha256":sha(event), "stored_energy_summary_sha256":sha(energy),
            "case_index_sha256":sha(index)}))
        generation = {"job":"original", "input_sha256":case["inp_sha256"]}
        (base / "generation-receipt.json").write_text(json.dumps({"cases":[generation]}))
        with patch("validation_controls.completion_controls", return_value=({"source_kind":"exact"}, {})) as controls, \
             patch("validation_results.verify_native_exports"), \
             patch("validation_results.native_record", side_effect=lambda *args: {"native_index":0,"quality_note":"Fixture"}), \
             patch("validation_results.triangle_record", side_effect=lambda *args: {"native_index":0}):
            all_rows, selected, unavailable = completion_records(self.root)
        controls.assert_called_once_with(self.root, case, generation)
        self.assertEqual((len(all_rows), len(selected), unavailable), (3, 2, []))
        self.assertTrue(all(row["source_odb"] == case["source_odb"] for row in all_rows[:2]))
        self.assertTrue(all(row["energy_window_review_required"] for row in all_rows))
        self.assertTrue(all("endpoint KE/IE alone" in row["quality_note"] for row in all_rows))

    def test_changed_settled_initiation_table_is_rejected(self):
        out = self.root/"audit/new-validation/results"
        event = out/"audit/failure/failure-initiation.csv"
        event.parent.mkdir(parents=True)
        event.write_text("case_id,prevtime_s\nfixture,.05\n")
        digest = sha(event)
        (out/"extraction-summary.json").write_text(json.dumps({"state":"complete","event_csv_sha256":digest}))
        event.write_text("case_id,prevtime_s\nfixture,.06\n")
        with self.assertRaisesRegex(ValueError, "differs from its settled extraction"):
            completion_records(self.root)

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


class FilteredNativeTriangleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p1_filtered_result_")
        self.root = Path(self.temp.name)
        source = self.root / "specimen.inp"
        source.write_text("Exact synthetic source fixture")
        receipt = self.root / "modes.json"
        receipt.write_text('{"frequency_Hz":40.0}')
        self.curve_path = self.root / "native_curve.csv"
        t = np.arange(101) / 1000.
        self.curve = [{"time_s": float(tt), "U2_mm": float(3*tt + .002*np.sin(2*np.pi*300*tt)),
                       "RF2_N": float(1000*tt + 10*np.sin(2*np.pi*300*tt))} for tt in t]
        self.save()
        self.record = {"case_id": "synthetic_tri", "topology": "tri", "l_mm": 10.,
                       "relative_density": .2, "nnx": 30, "native_index": 80,
                       "endpoint_time_s": .08, "endpoint": "criterion_last_intact",
                       "loading_state": "native_dynamic", "source_inp": str(source),
                       "source_inp_sha256": sha(source), "source_odb_sha256": "b"*64,
                       "curve_csv_sha256": sha(self.curve_path),
                       "force_N": self.curve[80]["RF2_N"],
                       "displacement_mm": self.curve[80]["U2_mm"], "loading_work_Nmm": 9.6,
                       "KI_MPa_sqrt_m": .001, "KJ_MPa_sqrt_m": .002, "modulus_MPa": 1000.,
                       "value_status": "available_derived_assumptions", "qualified_ASTM_J": False,
                       "ALLIE_Nmm": 100., "ALLSE_Nmm": 80., "derived_predamage_PD_Nmm": 20.,
                       "surface_threshold_LE11": .1105, "previous_max_LE11": .109,
                       "crossing_max_LE11": .115, "criterion_lower_time_s": .08,
                       "criterion_upper_time_s": .081}
        self.modal = {"source_sha256": sha(source), "source_odb_sha256": "b"*64,
                      "state": "completed_source_qualified_released_physical_mode",
                      "read_only": True, "source_input_hash_verified": True,
                      "natural_frequency_Hz": 40., "job": "synthetic_modal",
                      "modal_receipt": str(receipt), "modal_receipt_sha256": sha(receipt),
                      "modal_odb_sha256": "e"*64}
        self.compliance = {"source_sha256": sha(source), "state": "completed_and_extracted",
                           "read_only": True, "source_input_hash_verified": True,
                           "static_stiffness_N_mm": 10000., "source_odb_sha256": "f"*64,
                           "job": "synthetic_compliance"}

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        with self.curve_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.curve[0]))
            writer.writeheader(); writer.writerows(self.curve)

    def test_paired_causal_work_endpoint_and_material_event(self):
        from scipy.signal import butter, sosfilt, sosfilt_zi
        sos = butter(2, 80., fs=1000., output="sos")
        x = np.array([row["U2_mm"] for row in self.curve[:81]])
        p = np.array([row["RF2_N"] for row in self.curve[:81]])
        xf, _ = sosfilt(sos, x, zi=sosfilt_zi(sos)*x[0])
        pf, _ = sosfilt(sos, p, zi=sosfilt_zi(sos)*p[0])
        filtered = filtered_native_record(self.record, self.curve_path, self.modal)
        self.assertAlmostEqual(filtered["force_N"], pf[-1])
        self.assertAlmostEqual(filtered["displacement_mm"], xf[-1])
        self.assertAlmostEqual(filtered["loading_work_Nmm"], np.trapz(pf, xf))
        _, expected_ki = geometry_and_ki(
            {"topology": "tri", "l_mm": 10., "nnx": 30, "nominal_relative_density": .2}, pf[-1])
        self.assertAlmostEqual(filtered["KI_MPa_sqrt_m"], expected_ki)
        self.assertEqual(filtered["native_index"], 80)
        self.assertEqual(filtered["endpoint_time_s"], .08)
        self.assertFalse(filtered["filter_zero_phase"])
        self.assertTrue(filtered["filter_applied"])
        self.assertEqual(filtered["filter_native_prefix_samples"], 81)
        for key in ("ALLIE_Nmm", "ALLSE_Nmm", "derived_predamage_PD_Nmm",
                    "previous_max_LE11", "crossing_max_LE11"):
            self.assertEqual(filtered[key], self.record[key])
        self.assertEqual(self.record["force_N"], self.curve[80]["RF2_N"])
        triangle = triangle_record(filtered, self.compliance)
        self.assertAlmostEqual(triangle["Apl_Nmm"],
                               np.trapz(pf, xf) - pf[-1]**2/(2*10000.))

    def test_later_fracture_cannot_change_the_filtered_event(self):
        before = filtered_native_record(self.record, self.curve_path, self.modal)
        for row in self.curve[81:]:
            row["RF2_N"] = 1e12
            row["U2_mm"] = 1e9
        self.save()
        changed = dict(self.record, curve_csv_sha256=sha(self.curve_path))
        after = filtered_native_record(changed, self.curve_path, self.modal)
        for key in ("force_N", "displacement_mm", "loading_work_Nmm", "KI_MPa_sqrt_m"):
            self.assertEqual(before[key], after[key])

    def test_wrong_modal_source_or_unverified_state_is_rejected(self):
        for key, bad in (("source_sha256", "a"*64), ("source_odb_sha256", "c"*64),
                         ("state", "prepared_not_submitted"), ("read_only", False),
                         ("modal_odb_sha256", None)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frequency"):
                filtered_native_record(self.record, self.curve_path, dict(self.modal, **{key: bad}))
        Path(self.modal["modal_receipt"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "modal receipt"):
            filtered_native_record(self.record, self.curve_path, self.modal)

    def test_changed_input_curve_or_event_time_is_rejected(self):
        Path(self.record["source_inp"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Original input"):
            filtered_native_record(self.record, self.curve_path, self.modal)
        Path(self.record["source_inp"]).write_text("Exact synthetic source fixture")
        with self.assertRaisesRegex(ValueError, "index/time"):
            filtered_native_record(dict(self.record, endpoint_time_s=.081), self.curve_path, self.modal)
        self.curve_path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Native curve"):
            filtered_native_record(self.record, self.curve_path, self.modal)

    def test_selected_available_triangle_requires_map_but_diagnostics_do_not(self):
        with self.assertRaisesRegex(ValueError, "no verified periodic frequency"):
            native_triangle_record(self.record, self.compliance, self.curve_path, {}, selected=True)
        diagnostic = native_triangle_record(self.record, self.compliance, self.curve_path, {})
        self.assertEqual(diagnostic["response_processing"], "native_unfiltered_diagnostic")
        result = native_triangle_record(self.record, self.compliance, self.curve_path,
                                       {self.record["source_inp_sha256"]: self.modal}, selected=True)
        self.assertEqual(result["response_processing"], "causal_Butterworth_paired_2fn")

    def test_unavailable_negative_area_has_no_raw_fallback_claim(self):
        control = dict(self.compliance, static_stiffness_N_mm=.001)
        result = native_triangle_record(self.record, control, self.curve_path, {}, selected=True)
        self.assertLess(result["Apl_Nmm"], 0)
        self.assertIsNone(result["KJ_MPa_sqrt_m"])
        self.assertEqual(result["response_processing"], "native_unfiltered_unavailable")

    def test_cutoff_at_or_above_nyquist_preserves_native_observations_honestly(self):
        for frequency in (250., 300.):
            with self.subTest(frequency=frequency):
                result = filtered_native_record(
                    self.record, self.curve_path, dict(self.modal, natural_frequency_Hz=frequency))
                for key in ("force_N", "displacement_mm", "loading_work_Nmm", "KI_MPa_sqrt_m",
                            "native_index", "endpoint_time_s", "previous_max_LE11", "crossing_max_LE11"):
                    self.assertEqual(result[key], self.record[key])
                self.assertFalse(result["filter_applied"])
                self.assertEqual(result["filter_status"],
                                 "sampling_unresolved_cutoff_at_or_above_saved_Nyquist")
                self.assertEqual(result["physical_cutoff_Hz"], 2*frequency)
                self.assertEqual(result["saved_Nyquist_Hz"], 500.)
                self.assertEqual(result["filter_modal_job"], self.modal["job"])
                self.assertNotIn("filter_order", result)


if __name__ == "__main__":
    unittest.main()
