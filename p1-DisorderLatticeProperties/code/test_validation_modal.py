"""Exact-source modal controls preserve specimen mechanics and periodic scope."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
from types import SimpleNamespace
import unittest

from validation_modal import periodic_contents, job_directory, iso_map, sha256


MODEL = """*Heading
*Node
1,0,0
2,1,1
*Element,type=B21
1,1,2
*Beam Section,elset=BodySet,material=ti,section=RECT
80.,.405746
0.,0.,-1.
*Beam Section,elset=BracketSet,material=ti,section=RECT
80.,.811491
0.,0.,-1.
*Material,name=ti
*Density
4.43e-9,
*Elastic
135165.,.428571
*Plastic
900.,0.
*Damage Initiation,criterion=DUCTILE
.1,0.,0.
*Amplitude,name=Amp-1
0.,0.,.1,1.
*Step,name=Load
*Dynamic,Explicit
,.1
"""

UT_BC = """*Boundary,amplitude=Amp-1
Set-top,1,1
Set-top,2,2,1.
Set-top,6,6
Set-bottom,1,1
Set-bottom,2,2
Set-bottom,6,6
*End Step
"""
FT_BC = """*Boundary,amplitude=Amp-1
load,1,1
load,2,2,1.
fixity,1,1
fixity,2,2
*End Step
"""


class ModalPreparation(unittest.TestCase):
    def prepare(self, task, contents):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.inp"
            path.write_text(contents, encoding="ascii")
            answer = periodic_contents(path, task)
            self.assertEqual(path.read_text(), contents)
            return answer

    def test_uat_retains_both_sections_material_mass_and_clamps(self):
        result, metadata = self.prepare("UT", MODEL + UT_BC)
        self.assertIn("80.,.405746", result)
        self.assertIn("80.,.811491", result)
        self.assertIn("4.43e-9", result)
        self.assertIn("135165.,.428571", result)
        self.assertIn("Set-top, 1, 2\nSet-top, 6, 6", result)
        self.assertEqual(metadata["control"], "held_grips")
        self.assertNotIn("*Plastic", result)
        self.assertNotIn("*Amplitude", result)
        self.assertNotIn("*Damage", result)

    def test_ct_released_pin_does_not_acquire_a_held_loading_dof(self):
        result, metadata = self.prepare("FT", MODEL + FT_BC)
        self.assertIn("fixity, 1, 2\nload, 1, 1", result)
        self.assertNotIn("load, 2, 2", result)
        self.assertEqual(metadata["control"], "released")

    def test_unrecognised_source_constraints_require_review(self):
        with self.assertRaisesRegex(ValueError, "constraints"):
            self.prepare("UT", MODEL + FT_BC)

    def test_mass_scaling_is_not_silently_transferred(self):
        with self.assertRaisesRegex(ValueError, "Mass-scaled"):
            self.prepare("FT", MODEL + "*Fixed Mass Scaling\n1.\n" + FT_BC)

    def test_output_directory_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            job_directory("root", "../escape")

    def test_frequency_map_preserves_reviewed_failure_and_verifies_modal_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modal = root / "audit/modal"
            out = modal / "iso-periodic"
            folder = out / "reviewed"
            folder.mkdir(parents=True)
            (modal / "summary.json").write_text("[]")
            source = root / "source.inp"
            original = root / "source.odb"
            solved = folder / "reviewed.odb"
            for path in (source, original, solved):
                path.write_bytes(path.name.encode())
            data = {"control": "released", "job": "reviewed", "source_inp": str(source),
                    "source_inp_sha256": sha256(source), "source_odb": str(original),
                    "source_odb_sha256": sha256(original), "solved_odb": str(solved),
                    "solved_odb_sha256": sha256(solved), "boundary": "released pin",
                    "candidate_mode": {"mode": 1, "frequency_Hz": 40.}, "cutoff_2fn_Hz": 80.,
                    "execution_state": "mechanically_completed_with_reviewed_SIM_wrap_failure"}
            (folder / "modes.json").write_text(json.dumps(data))
            (folder / "mode_shapes.png").write_bytes(b"synthetic reviewed fixture")
            (out / "summary.json").write_text(json.dumps([data]))
            args = SimpleNamespace(root=root, out=out)
            with redirect_stdout(io.StringIO()):
                iso_map(args)
            mapped = json.loads((out / "source-frequency-map.json").read_text())[sha256(source)]
            self.assertEqual(mapped["execution_state"], data["execution_state"])
            self.assertEqual(mapped["modal_odb_sha256"], sha256(solved))
            solved.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "modal ODB changed"):
                iso_map(args)


if __name__ == "__main__":
    unittest.main()
