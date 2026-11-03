"""Display provenance and normalisation checks using explicitly synthetic inputs."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from validation_comparison import (DENSITIES, LENGTHS, TOPOLOGIES, paper_exports,
                                   paper_supplements, read_rows)


class PaperSupplement(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="codex-p1-validation-display-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        selected = self.root / "selected.csv"
        selected.write_text("synthetic fixture,not research data\n")
        self.inputs = {"selected_recalculation": {"path": str(selected),
                        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest()},
                       "reference": {"sha256": "synthetic-reference"},
                       "original_fig8": {"sha256": "synthetic-fig8"}}
        self.records = []
        self.references, self.fig8 = [], []
        for topology in TOPOLOGIES:
            for rho in DENSITIES:
                self.references.append({"stage": "initial", "topology": topology,
                    "nominal_relative_density": rho, "bar_length_mm": 3,
                    "K_MPa_sqrt_m": 5., "K_graphical_lower": 4.9,
                    "K_graphical_upper": 5.1, "point_id": f"synthetic-{topology}-{rho}"})
                for length in LENGTHS:
                    key = (topology, length, rho)
                    self.fig8.append({"figure": "8a", "topology": topology,
                        "l_mm": length, "relative_density": rho, "literal_ordinate": .1,
                        "reference_identity": str(key), "source_cell": "synthetic"})
                    if rho == .2:
                        self.fig8.append({**self.fig8[-1], "figure": "8b"})
                    if topology == "hex" and rho == .13:
                        continue
                    negative = topology == "hex" and length == 3
                    self.records.append({"method": "Hsieh_initial_triangle_native",
                        "topology": topology, "l_mm": length, "relative_density": rho,
                        "case_id": str(key), "value_status": "diagnostic_only" if negative else "available_provisional",
                        "KJ_MPa_sqrt_m": None if negative else 6.,
                        "Apl_Nmm": -10. if negative else 10.})
        values = {(3., .16): 3., (3., .2): 4.5}
        exponent = math.log(4.5/3.) / math.log(.2/.16)
        values[3., .13] = 3.*(.13/.16)**exponent
        values[10., .13] = values[3., .13]*math.sqrt(10./3.)
        points = []
        for (length, rho), value in values.items():
            point = {"topology": "hex", "l_mm": length, "relative_density": rho,
                     "K_equivalent_MPa_sqrt_m": value, "basis": "Synthetic calculation basis",
                     "independent_original_fracture_KJ": False,
                     "density_extrapolation": rho == .13, "length_scaling": length == 10.}
            if rho != .13:
                point["case_id"] = str(("hex", length, rho))
                self.records.append({**point, "method": "qs_measured_recovery_eta",
                    "value_status": "available_provisional", "KJ_MPa_sqrt_m": value,
                    "loading_state": "counterfactual_damage_off_QS",
                    "source_work_convention": "measured_nonlinear_recovery"})
            points.append(point)
        self.document = {"adopted_for_paper_display": True, "points": points,
                         "source_results": str(selected), "density_exponent": exponent,
                         "source_results_sha256": self.inputs["selected_recalculation"]["sha256"]}
        self.path = self.root / "supplement.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.document))

    def load(self):
        return paper_supplements(self.records, self.inputs, self.path)

    def test_recovery_and_own_data_predictions_leave_records_unchanged(self):
        original = copy.deepcopy(self.records)
        values, metadata = self.load()
        self.assertEqual(self.records, original)
        self.assertEqual(len(values), 4)
        self.assertAlmostEqual(values["hex", 10., .13]["K_equivalent_MPa_sqrt_m"],
                               values["hex", 3., .13]["K_equivalent_MPa_sqrt_m"]*math.sqrt(10/3))
        self.assertEqual(metadata["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_rejects_stale_duplicate_unexpected_and_changed_predictions(self):
        changes = [lambda d: d.update(source_results_sha256="stale"),
                   lambda d: d["points"].append(copy.deepcopy(d["points"][0])),
                   lambda d: d["points"][0].update(topology="tri"),
                   lambda d: d["points"][2].update(K_equivalent_MPa_sqrt_m=999.)]
        original = copy.deepcopy(self.document)
        for change in changes:
            with self.subTest(change=change):
                self.document = copy.deepcopy(original)
                change(self.document)
                self.save()
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_recovery_changes_and_available_native_replacement(self):
        self.records[-1]["KJ_MPa_sqrt_m"] *= 2
        with self.assertRaisesRegex(ValueError, "load-unload"):
            self.load()
        self.records[-1]["KJ_MPa_sqrt_m"] /= 2
        native = next(r for r in self.records if r["method"] == "Hsieh_initial_triangle_native"
                      and r["topology"] == "hex" and r["l_mm"] == 3.)
        native.update(value_status="available_provisional", KJ_MPa_sqrt_m=3., Apl_Nmm=1.)
        with self.assertRaisesRegex(ValueError, "replace an available"):
            self.load()

    def test_requires_explicit_adoption(self):
        self.document["adopted_for_paper_display"] = False
        self.save()
        self.assertEqual(self.load(), ({}, None))

    def test_exports_normalisation_evidence_and_hypothesis_factors(self):
        before = copy.deepcopy((self.records, self.references, self.fig8))
        out = self.root / "paper"
        paper_exports(self.records, self.fig8, self.references, {"parent_UTS_workbook_MPa": 1000.},
                      out, self.inputs, "synthetic-pdf", self.path)
        self.assertEqual((self.records, self.references, self.fig8), before)
        points = read_rows(out / "points.csv")
        current = [r for r in points if r["series"] == "Simulation"]
        self.assertEqual(len(points), 108)
        self.assertEqual(sum(r["plotted"] == "True" for r in current), 54)
        self.assertTrue((out / "exclusions.csv").read_text().startswith("figure,"))
        self.assertEqual(read_rows(out / "exclusions.csv"), [])
        for row in current:
            value = float(row["display_K_MPa_sqrt_m"])
            figure = row["figure"]
            denominator = 1000*math.sqrt(float(row["l_mm"])/1000) if figure == "8a" else 1 if figure == "10" else 1000
            factor = math.sqrt(3) if figure == "8b_my_fix" else 1
            self.assertAlmostEqual(float(row["ordinate"]), value/denominator*factor)
            if row["evaluation_basis"] == "prediction":
                self.assertEqual(row["source_odb"], "")
                self.assertEqual(row["endpoint_time_s"], "")
                self.assertEqual(row["KJ_MPa_sqrt_m"], "")
            elif row["evaluation_basis"] == "nonlinear_recovery":
                self.assertEqual(float(row["Apl_Nmm"]), -10.)
                self.assertEqual(row["KJ_MPa_sqrt_m"], "")
        for row in points:
            if row["figure"] == "8b_hsieh_fix" and row["series"] == "Hsieh et al.":
                self.assertAlmostEqual(float(row["ordinate"]), .1/math.sqrt(3))
        meta = json.loads((out / "metadata.json").read_text())
        self.assertEqual(meta["display_cases_by_basis"], {"original_fracture": 14, "nonlinear_recovery": 2, "prediction": 2})
        for figure in ("8a", "8b", "10", "8b_hsieh_fix", "8b_my_fix"):
            for extension in ("png", "pdf", "svg"):
                self.assertGreater((out / f"figure{figure}.{extension}").stat().st_size, 1000)
            svg = ET.parse(out / f"figure{figure}.svg")
            text = " ".join(svg.getroot().itertext())
            self.assertIn("Hsieh et al.", text)
            self.assertIn("Current study", text)
            self.assertNotIn("Current study:", text)
            self.assertNotIn("Simulation", text)
        self.assertEqual(meta["marker_design"]["legend_entries"], ["Hsieh et al.", "Current study"])
        self.assertIn("hex 3mm/16%", meta["captions"]["10"])
        self.assertIn("hex 3mm/13%", meta["captions"]["10"])


if __name__ == "__main__":
    unittest.main()
