"""Portable preparation/compliance checks; no ODB access or solver launches."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from validation_controls import compliance_values, gather_compliance
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


if __name__ == "__main__":
    unittest.main()
