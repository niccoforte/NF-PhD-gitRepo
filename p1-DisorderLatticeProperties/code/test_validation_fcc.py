"""Focused physical/identity checks for the opt-in FCC feasibility preparation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation_fcc import (crack_extension, energy_reference, immutable_json, matched_section_frames,
                            model_prefix, potential_stress, work_derivative)
from validation_diagnostics import blocks, key, rows


class FCCFeasibilityTests(unittest.TestCase):
    def test_complete_strut_and_half_open_tip(self):
        # Two vertical subdivided struts; only x=0 is removed for [0,10).
        nodes = {1: [0., -1.], 2: [0., 0.], 3: [0., 1.],
                 4: [10., -1.], 5: [10., 0.], 6: [10., 1.]}
        edges = [[1, 1, 2], [2, 2, 3], [3, 4, 5], [4, 5, 6]]
        self.assertEqual(crack_extension(nodes, edges, 0., 0., 10.), {1, 2})
        self.assertEqual(crack_extension(nodes, edges, 0., 0., 0.), set())

    def test_strut_traversal_stops_at_turn(self):
        nodes = {1: [0., -1.], 2: [0., 1.], 3: [1., 1.]}
        self.assertEqual(crack_extension(nodes, [[1, 1, 2], [2, 2, 3]], 0., 0., 1.), {1})

    def test_mesh_set_references_survive_notch(self):
        parsed = blocks("*Instance,name=I,part=P\n*Node\n1,0,-1\n2,0,0\n3,0,1\n4,10,0\n"
                        "*Element,type=B21\n1,1,2\n2,2,3\n3,3,4\n"
                        "*Elset,elset=Body,generate\n1,3,1\n*Nset,nset=Body,generate\n1,4,1\n"
                        "*End Instance\n*Node\n1,40,155\n*Elastic\n1000,.3\n*Plastic\n10,0\n")
        text, ne, nn = model_prefix(parsed, {1: [0, -1], 2: [0, 0], 3: [0, 1], 4: [10, 0]},
                                    [[1, 1, 2], [2, 2, 3], [3, 3, 4]], {1, 2}, "elastic", {})
        self.assertEqual((ne, nn), (1, 2))
        elset = next(b for b in blocks(text) if key(b) == "*elset")
        nset = next(b for b in blocks(text) if key(b) == "*nset")
        self.assertEqual(rows(elset), ["3"])
        self.assertEqual(rows(nset), ["3, 4"])
        self.assertNotIn("*Plastic", text)
        self.assertIn("1,40,155", text)  # Assembly pin must not be filtered as mesh node1.

    def test_potential_inversion_and_elastic_limit(self):
        E, sy, alpha, exponent = 1000., 10., .2, 10.
        stress = np.array([0., .1, 5., 10., 12.])
        strain = stress/E+alpha*sy/E*(stress/sy)**exponent
        np.testing.assert_allclose(potential_stress(strain, E, sy, alpha, exponent), stress, atol=1e-9)
        self.assertAlmostEqual(potential_stress([1e-8], E, sy, alpha, exponent)[0]/1e-8, E, places=6)

    def test_potential_crack_energy_sign_and_units(self):
        u = np.linspace(0., 1., 101)
        # Stiffness decreases linearly with a; U=.5*k*u².
        forces = np.outer([100., 90., 80.], u)
        values = energy_reference(u, forces, h=10., B=2., law="potential")
        self.assertAlmostEqual(values["J_h_N_mm"][-1], .25)
        self.assertAlmostEqual(values["J_next_h_N_mm"][-1], .25)
        self.assertAlmostEqual(values["J_2h_N_mm"][-1], .25)
        self.assertAlmostEqual(values["J_tip_second_order_N_mm"][-1], .25)

    def test_incremental_work_difference_cannot_be_labelled_J(self):
        u = np.linspace(0., 1., 11)
        forces = np.outer([100., 90., 80.], u)
        diagnostic = work_derivative(u, forces, h=10., B=2.)
        self.assertAlmostEqual(diagnostic["tip_second_order"][-1], .25)
        self.assertFalse(any(name.startswith("J") for name in diagnostic))
        with self.assertRaisesRegex(ValueError, "potential constitutive"):
            energy_reference(u, forces, h=10., B=2., law="incremental")
        with self.assertRaisesRegex(ValueError, "from zero"):
            work_derivative(u[1:], forces[:, 1:], h=10., B=2.)

    def test_section_frames_join_by_loading_time_not_raw_index(self):
        class Archive(dict):
            @property
            def files(self):
                return list(self)
        original = Archive(time_0=[0.], time_40=[1.])
        potential = Archive(time_0=[0.], time_41=[1.])
        self.assertEqual(matched_section_frames(original, potential), [(0., "0", "0"), (1., "40", "41")])
        potential["time_41"] = [.99]
        with self.assertRaisesRegex(ValueError, "loading time"):
            matched_section_frames(original, potential)

    def test_preparation_identity_cannot_mutate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"source.json"
            immutable_json(path, {"source": "one"})
            immutable_json(path, {"source": "one"})
            with self.assertRaisesRegex(ValueError, "Refusing"):
                immutable_json(path, {"source": "two"})
            self.assertEqual(json.loads(path.read_text()), {"source": "one"})


if __name__ == "__main__":
    unittest.main()
