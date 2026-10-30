"""Portable preparation/compliance checks; no ODB access or solver launches."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from validation_controls import compliance_values, gather_compliance, gather_completion_controls, similarity_controls
from validation_modal import prepare, sha256


class ComplianceControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p1_compliance_controls_")
        self.root = Path(self.temp.name)
        self.relative = "source.inp"
        self.source = self.root / self.relative
        self.source.write_text("""*Heading
*Part, name=Part-1
*Node
1, 0., 0.
2, 1., 0.
*Element, type=B21
1, 1, 2
*Beam Section, elset=all, material=Ti, section=RECT
1., .1
0., 0., -1.
*End Part
*Material, name=Ti
*Elastic
135165., .428571
*Density
4.43e-9
*Plastic
932., 0.
*Damage Initiation, criterion=DUCTILE
.102268, .333333, 0.
*Damage Evolution, type=DISPLACEMENT
.001
*Amplitude, name=move
0., 0., 1., 1.
*Step, name=Original, nlgeom=YES
*Dynamic, Explicit
, .1
*Boundary
load, 1, 1
load, 2, 2, 1.
fixity, 1, 1
fixity, 2, 2
*End Step
""", encoding="ascii")
        self.original_hash = sha256(self.source)
        row = {"case_id": "fixture", "inp": self.relative, "inp_sha256": self.original_hash,
               "odb": "source.odb", "topology": "tri", "l_mm": "10",
               "nominal_relative_density": "0.2", "nnx": "30", "campaign": "Density campaign",
               "mode": "Fracture", "storage_status": "retained_odb"}
        with (self.root / "case-index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, folder, *, stiffness=False, controls=None):
        out = self.root / folder
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(SimpleNamespace(root=self.root, out=out, inp=[self.relative], modes=10,
                                    stiffness=stiffness, controls=controls))
        return out

    def test_unit_load_only_retains_source_and_omits_unwanted_modal_inputs(self):
        out = self.stage("unit", controls=("unit_load",))
        files = list(out.glob("*/*.inp"))
        self.assertEqual(len(files), 1)
        text = files[0].read_text()
        self.assertIn("*Step, name=Compliance, nlgeom=NO", text)
        self.assertIn("*Cload\nload, 2, 1.", text)
        self.assertIn("*Element, type=B21\n1, 1, 2", text)
        self.assertNotIn("*Frequency", text)
        self.assertNotIn("*Plastic", text)
        self.assertNotIn("*Damage", text)
        self.assertEqual(sha256(self.source), self.original_hash)
        metadata = json.loads((files[0].parent / "source.json").read_text())
        self.assertEqual(metadata["source_inp_sha256"], self.original_hash)
        self.assertEqual(metadata["input_sha256"], sha256(files[0]))

    def test_existing_default_control_selection_is_unchanged(self):
        default = self.stage("default", stiffness=True)
        explicit = self.stage("explicit", controls=("released", "held", "unit_load"))
        a = {path.name: path.read_bytes() for path in default.glob("*/*.inp")}
        b = {path.name: path.read_bytes() for path in explicit.glob("*/*.inp")}
        self.assertEqual(len(a), 3)
        self.assertEqual(a, b)
        self.assertEqual(len(list(self.stage("modal_default").glob("*/*.inp"))), 2)

    def test_invalid_or_duplicate_control_selection_is_rejected(self):
        for controls in ((), ("bad",), ("unit_load", "unit_load")):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                self.stage("invalid", controls=controls)

    def test_compliance_units_and_force_balance(self):
        result = compliance_values(.002, -1.)
        self.assertEqual(result["compliance_mm_N"], .002)
        self.assertEqual(result["static_stiffness_N_mm"], 500.)
        self.assertEqual(result["force_balance_relative"], 0.)
        for displacement, reaction in ((0., -1.), (-.002, -1.), (.002, -.9), (float("nan"), -1.)):
            with self.subTest(displacement=displacement, reaction=reaction), self.assertRaises(ValueError):
                compliance_values(displacement, reaction)

    def queued_fixture(self):
        out = self.stage("controls", controls=("unit_load",))
        directory = next(out.iterdir())
        metadata_path = directory / "source.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(source_sha256=self.original_hash, input=str(directory / (metadata["job"] + ".inp")))
        metadata_path.write_text(json.dumps(metadata))
        queue = out / "queue-compliance-campaign-20261028"
        run = queue / "runs" / metadata["job"]
        run.mkdir(parents=True)
        (queue / "state.json").write_text(json.dumps({"jobs": [{"job": metadata["job"], "status": "completed"}]}))
        files = []
        for suffix, data in ((".inp", Path(metadata["input"]).read_bytes()), (".odb", b"synthetic-odb"), (".dat", b"completed")):
            path = run / (metadata["job"] + suffix)
            path.write_bytes(data)
            files.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
        (run / "delivery.json").write_text(json.dumps({"job": metadata["job"], "files": files, "required_outputs_missing": []}))
        return out, run, metadata

    def test_gather_checks_delivery_and_extracts_read_only(self):
        out, run, metadata = self.queued_fixture()
        frame = SimpleNamespace(fieldOutputs={
            "U": SimpleNamespace(getSubset=lambda region: SimpleNamespace(values=[SimpleNamespace(data=(0., .002))])),
            "RF": SimpleNamespace(getSubset=lambda region: SimpleNamespace(values=[SimpleNamespace(data=(0., -1.))]))})
        closed = []
        odb = SimpleNamespace(steps={"Compliance": SimpleNamespace(frames=[frame])},
                              rootAssembly=SimpleNamespace(nodeSets={"LOAD": object(), "FIXITY": object()}),
                              close=lambda: closed.append(True))
        with patch.dict(sys.modules, {"odbAccess": SimpleNamespace(openOdb=lambda path, readOnly: self.assert_readonly(path, readOnly, odb))}), \
             patch("validation_controls.reused_compliance", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            gather_compliance(self.root, out)
        records = json.loads((out / "compliance-summary.json").read_text())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["static_stiffness_N_mm"], 500.)
        self.assertEqual(records[0]["source_odb_sha256"], sha256(run / (metadata["job"] + ".odb")))
        self.assertTrue(records[0]["read_only"])
        self.assertEqual(closed, [True])

    def assert_readonly(self, path, read_only, odb):
        self.assertTrue(read_only)
        self.assertTrue(Path(path).is_file())
        return odb

    def test_gather_rejects_changed_retained_input_before_odb_access(self):
        out, run, metadata = self.queued_fixture()
        (run / (metadata["job"] + ".inp")).write_text("changed")
        with patch.dict(sys.modules, {"odbAccess": SimpleNamespace(openOdb=lambda *args, **kwargs: self.fail("ODB opened before provenance verification"))}), \
             patch("validation_controls.reused_compliance", return_value=[]), self.assertRaisesRegex(ValueError, "delivery receipt"):
            gather_compliance(self.root, out)


class SimilarityControlTests(unittest.TestCase):
    """Synthetic periodic beam grid exercises source proof, not solver physics."""
    def setUp(self):
        from validation_completion import transform
        self.temp = tempfile.TemporaryDirectory(prefix="p1_similarity_controls_")
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent.inp"
        self.new = self.root / "new.inp"
        points = [(i*math.sqrt(3), j) for j in range(6) for i in range(6)]
        edges = [(j*6+i+1, j*6+i+2) for j in range(6) for i in range(5)]
        edges += [(j*6+i+1, (j+1)*6+i+1) for j in range(5) for i in range(6)]
        text = "*Heading\n*Assembly, name=A\n*Instance, name=I, part=P\n*Node\n"
        text += "".join(f"{i+1}, {x:.15g}, {y:.15g}\n" for i, (x, y) in enumerate(points))
        text += "*Element, type=B21, elset=BODY\n"
        text += "".join(f"{i+1}, {a}, {b}\n" for i, (a, b) in enumerate(edges))
        text += """*Beam Section, elset=BODY, material=Ti, section=RECT
1., .1
0.,0.,-1.
*End Instance
*Node
100, 1., 2.
101, 3., 4.
*Nset, nset=load
100
*Nset, nset=fixity
101
*Coupling, constraint name=pin, ref node=load, surface=SIDE
*Kinematic
*End Assembly
*Material, name=Ti
*Elastic
135165., .428571
*Density
4.43e-9
*Plastic
932., 0.
*Damage Initiation, criterion=DUCTILE
.102268, .333333, 0.
*Damage Evolution, type=DISPLACEMENT, softening=TABULAR
0., 0.
1., .001
*Amplitude, name=move
0., 0., .1, 10.
*Step, name=Original, nlgeom=YES
*Dynamic, Explicit
, .1
*Boundary, amplitude=move
load, 1, 1
load, 2, 2, 1.
fixity, 1, 1
fixity, 2, 2
*Output, field, number interval=100
*Node Output
U, RF
*Output, history
*Energy Output
ALLIE, ALLKE, ALLSE
*End Step
"""
        self.parent.write_text(text, encoding="ascii")
        self.new.write_text(transform(text, .3, 1.), encoding="ascii")
        parent_case = {"case_id": "parent", "inp": "parent.inp", "inp_sha256": sha256(self.parent),
                       "topology": "tri", "l_mm": 1., "nominal_relative_density": .2, "nnx": 6}
        with (self.root / "case-index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(parent_case)); writer.writeheader(); writer.writerow(parent_case)
        self.case = dict(parent_case, case_id="new", inp="new.inp", l_mm=.3, inp_sha256=sha256(self.new))
        self.record = {"input": str(self.new), "input_sha256": sha256(self.new), "parent_input": str(self.parent),
                       "parent_input_sha256": sha256(self.parent), "length_scale": .3, "thickness_density_scale": 1.,
                       "topology": "tri", "nnx": 6, "l_mm": .3, "relative_density": .2}
        out = self.root / "audit/controls"; out.mkdir(parents=True)
        self.controls = []
        for name in ("rve", "compliance"):
            inp, odb = out / (name + ".inp"), out / (name + ".odb")
            inp.write_bytes(b"synthetic-control-input"); odb.write_bytes(b"synthetic-control-odb")
            self.controls.append({"case_id": "parent", "job": name, "source_inp": str(self.parent),
                "source_sha256": sha256(self.parent), "input": str(inp), "input_sha256": sha256(inp),
                "source_odb": str(odb), "source_odb_sha256": sha256(odb), "state": "completed_and_extracted",
                "source_input_hash_verified": True, "read_only": True})
        self.controls[0].update(analysis="linear_static_periodic_rve", elastic_MPa_nu=[135165., .428571],
            section_B_t_mm=[1., .1], C_MPa=[[100., 25., 0.], [25., 100., 0.], [0., 0., 37.5]],
            conversion_modulus_MPa=93.75, volume_mm3=math.sqrt(3), periods_mm=[math.sqrt(3), 1.])
        self.controls[1].update(analysis="linear_static", control="unit_load", compliance_mm_N=.002,
            static_stiffness_N_mm=500., load_N=1., fixed_pin_reaction_y_N=-1.)
        self.write_controls()

    def tearDown(self):
        self.temp.cleanup()

    def write_controls(self):
        folder = self.root / "audit/controls"
        for name, control in zip(("campaign-rve-results.json", "compliance-summary.json"), self.controls):
            (folder / name).write_text(json.dumps([control]))

    def rehash_new(self):
        self.case["inp_sha256"] = self.record["input_sha256"] = sha256(self.new)

    def test_units_scaling_and_parent_lineage_without_source_mutation(self):
        before = {p: sha256(p) for p in self.root.rglob("*") if p.is_file()}
        rve, compliance = similarity_controls(self.root, self.case, self.record)
        self.assertEqual(rve["C_MPa"], self.controls[0]["C_MPa"])
        self.assertEqual(rve["conversion_modulus_MPa"], 93.75)
        self.assertEqual(compliance["static_stiffness_N_mm"], 150.)
        self.assertAlmostEqual(compliance["compliance_mm_N"], .002/.3)
        self.assertEqual(rve["source_sha256"], sha256(self.new))
        self.assertEqual(rve["source_odb"], self.controls[0]["source_odb"])
        self.assertEqual(rve["parent_control_source_inp_sha256"], sha256(self.parent))
        self.assertEqual(rve["source_kind"], "verified_geometric_similarity")
        self.assertFalse(rve["new_control_solve"])
        self.assertEqual(before, {p: sha256(p) for p in self.root.rglob("*") if p.is_file()})

    def test_changed_relative_density_requires_new_controls(self):
        self.case["nominal_relative_density"] = self.record["relative_density"] = .13
        self.record["thickness_density_scale"] = .65
        with self.assertRaisesRegex(ValueError, "identical topology, mesh and relative density"):
            similarity_controls(self.root, self.case, self.record)

    def test_changed_material_boundary_connectivity_and_pin_are_rejected(self):
        original = self.new.read_text()
        changes = (("135165., .428571", "135166., .428571"),
                   ("load, 2, 2, 1.", "load, 2, 2, 2."),
                   ("1, 1, 2\n", "1, 1, 3\n"),
                   ("100, 0.3, 0.6", "100, 0.3, 0.7"))
        for old, new in changes:
            with self.subTest(old=old):
                self.assertIn(old, original)
                self.new.write_text(original.replace(old, new, 1)); self.rehash_new()
                with self.assertRaises(ValueError):
                    similarity_controls(self.root, self.case, self.record)

    def test_changed_parent_control_or_uncompleted_control_is_rejected(self):
        for name, value in (("state", "failed"), ("read_only", False), ("source_sha256", "bad")):
            original = self.controls[0][name]
            with self.subTest(name=name):
                self.controls[0][name] = value; self.write_controls()
                with self.assertRaisesRegex(ValueError, "parent control completion/provenance"):
                    similarity_controls(self.root, self.case, self.record)
            self.controls[0][name] = original
        self.write_controls()
        Path(self.controls[1]["source_odb"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "parent control completion/provenance"):
            similarity_controls(self.root, self.case, self.record)

    def test_explicit_parent_index_and_supplemental_summaries_are_used(self):
        alternate = self.root / "prepared-parent-index.csv"
        alternate.write_bytes((self.root / "case-index.csv").read_bytes())
        paths = [self.root / "supplemental-rve.json", self.root / "supplemental-compliance.json"]
        for path, control in zip(paths, self.controls):
            path.write_text(json.dumps([control]))
        # Make the default summaries unusable: this must use the supplied paths.
        (self.root / "audit/controls/campaign-rve-results.json").write_text("[]")
        rve, compliance = similarity_controls(self.root, self.case, self.record,
                                             parent_cases=alternate, control_paths=paths)
        self.assertEqual(rve["similarity_proof"]["parent_rve_summary_sha256"], sha256(paths[0]))
        self.assertEqual(rve["similarity_proof"]["parent_index_sha256"], sha256(alternate))
        self.assertEqual(compliance["static_stiffness_N_mm"], 150.)

    def completion_fixture(self):
        base = self.root / "audit/new-validation"
        out, queue = base / "controls", base / "queue-resume"
        out.mkdir(parents=True)
        case = dict(self.case, topology="hex", l_mm=3, nominal_relative_density=.13)
        with (base / "prepared-case-index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(case)); writer.writeheader(); writer.writerow(case)
        metadata_rows = []
        for control in self.controls:
            item = dict(control, case_id=case["case_id"], source_inp=str(self.new), source_sha256=sha256(self.new))
            directory = out / ("rve_hex_fixture" if control["job"] == "rve" else "stiffness_hex_fixture")
            directory.mkdir()
            item["job"] = directory.name
            inp = directory / (item["job"] + ".inp")
            inp.write_bytes(b"synthetic-control-input")
            item.update(input=str(inp), input_sha256=sha256(inp))
            (directory / "source.json").write_text(json.dumps(item))
            run = queue / "runs" / item["job"]; run.mkdir(parents=True)
            entries = []
            for suffix, contents in ((".inp", inp.read_bytes()), (".odb", b"synthetic-odb"), (".dat", b"completed")):
                path = run / (item["job"] + suffix); path.write_bytes(contents)
                entries.append({"file": path.name, "sha256": sha256(path)})
            (run / "delivery.json").write_text(json.dumps({"job": item["job"], "files": entries, "required_outputs_missing": []}))
            metadata_rows.append(item)
        (queue / "state.json").write_text(json.dumps({"jobs": [{"job": item["job"], "status": "completed"} for item in metadata_rows]}))
        return out, queue, metadata_rows

    def test_supplemental_gather_checks_delivery_and_reads_compliance_only(self):
        out, queue, metadata = self.completion_fixture()
        original = self.root / "audit/controls/campaign-rve-results.json"
        before = sha256(original)
        def fake_rve_extract(args):
            item = metadata[0]
            odb = queue / "runs" / item["job"] / (item["job"] + ".odb")
            result = dict(item, source_odb=str(odb), source_odb_sha256=sha256(odb))
            (out / "abaqus-rve-results.json").write_text(json.dumps([result]))
        frame = SimpleNamespace(fieldOutputs={
            "U": SimpleNamespace(getSubset=lambda region: SimpleNamespace(values=[SimpleNamespace(data=(0., .004))])),
            "RF": SimpleNamespace(getSubset=lambda region: SimpleNamespace(values=[SimpleNamespace(data=(0., -1.))]))})
        opened = []
        odb = SimpleNamespace(steps={"Compliance": SimpleNamespace(frames=[frame])},
                              rootAssembly=SimpleNamespace(nodeSets={"LOAD": object(), "FIXITY": object()}), close=lambda: None)
        def open_readonly(path, readOnly):
            self.assertTrue(readOnly); opened.append(path); return odb
        with patch("validation_controls.extract", side_effect=fake_rve_extract), \
             patch.dict(sys.modules, {"odbAccess": SimpleNamespace(openOdb=open_readonly)}):
            result = gather_completion_controls(self.root, queue)
        self.assertEqual(result["completed_controls"], 2)
        self.assertEqual(len(opened), 1)
        self.assertEqual(json.loads(Path(result["compliance_results"]).read_text())[0]["static_stiffness_N_mm"], 250.)
        self.assertEqual(sha256(original), before)

    def test_supplemental_gather_rejects_changed_delivery_before_any_extraction(self):
        _, queue, metadata = self.completion_fixture()
        (queue / "runs" / metadata[0]["job"] / (metadata[0]["job"] + ".inp")).write_text("changed")
        with patch("validation_controls.extract", side_effect=AssertionError("RVE opened before delivery checks")), \
             patch.dict(sys.modules, {"odbAccess": SimpleNamespace(openOdb=lambda *args, **kwargs: self.fail("ODB opened early"))}), \
             self.assertRaisesRegex(ValueError, "delivery receipt"):
            gather_completion_controls(self.root, queue)


if __name__ == "__main__":
    unittest.main()
