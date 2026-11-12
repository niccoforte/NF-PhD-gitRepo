"""Focused physical/identity checks for the opt-in FCC feasibility preparation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation_fcc import (crack_extension, energy_reference, immutable_json, matched_section_frames,
                            interval_quantities, model_prefix, potential_stress, prepare_event, response_at_opening,
                            reviewed_failed_delivery_state,
                            work_derivative)
from validation_diagnostics import blocks, key, rows, sha


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

    def test_native_force_work_integration_does_not_depend_on_query_spacing(self):
        u = np.array([0., .2, .7, 1.])
        q = np.array([0., .05, .3, .9, 1.])
        force, work = response_at_opening(u, 2*u, q)
        np.testing.assert_allclose(force, 2*q, atol=1e-14)
        np.testing.assert_allclose(work, q*q, atol=1e-14)
        force, work = response_at_opening([0., 1., 3.], [0., 3., 1.], [0., .5, 1., 2., 3.])
        np.testing.assert_allclose(force, [0., 1.5, 3., 2., 1.])
        np.testing.assert_allclose(work, [0., .375, 1.5, 4., 5.5])

    def test_response_query_cannot_extrapolate_or_reverse_loading(self):
        for query in ([-.1, .5], [0., 1.01], [0., .5, .4]):
            with self.assertRaisesRegex(ValueError, "supported query"):
                response_at_opening([0., .5, 1.], [0., 1., 2.], query)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            response_at_opening([0., .5, .4], [0., 1., 2.], [0., .4])

    def test_repeated_native_opening_adds_no_work_and_retains_last_reaction(self):
        # Vertical segments in P(delta) do no pin work. These occur when tiny
        # converged increments are below the recorded displacement precision.
        force, work = response_at_opening([0., 1., 1., 2., 2.], [0., 2., 3., 5., 6.], [0., 1., 1.5, 2.])
        np.testing.assert_allclose(force, [0., 3., 4., 6.])
        np.testing.assert_allclose(work, [0., 1., 2.75, 5.])

    def test_completed_solver_with_failed_wrapup_keeps_distinct_extraction_state(self):
        launcher = "End Abaqus/Standard Analysis\ndriverExceptions.AbaqusExecutionError: ('SIM Wrap-up', 1, 'job')"
        status = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
        self.assertEqual(reviewed_failed_delivery_state(launcher, status, 1., 1.),
                         "solver_completed_with_failed_wrapup")
        for time, fraction in ((.998, .998), (1., .998), (.998, 1.)):
            self.assertEqual(reviewed_failed_delivery_state(launcher, status, time, fraction),
                             "failed_with_reviewed_converged_prefix")
        self.assertEqual(reviewed_failed_delivery_state("Standard convergence error", status, 1., 1.),
                         "failed_with_reviewed_converged_prefix")

    def test_interval_average_matches_finite_potential_energy_not_tip_value(self):
        cracks = np.array([1., 2., 3.])
        stiffness = 1000.-cracks**4
        opening = np.array([0., .5, 1.])
        force = stiffness[:, None]*opening
        work = .5*stiffness[:, None]*opening**2
        # U=.5*(1000-a^4)*delta²; exact J=a³*delta² for B=2.
        factors = cracks**3/stiffness**2
        result = interval_quantities(cracks, force, work, 1/stiffness, factors, B=2., W=10.)
        full = result["intervals"][-1]
        np.testing.assert_allclose(full["finite_work_release_N_mm"], 10*opening**2)
        np.testing.assert_allclose(full["predicted_total_J_N_mm"], 10*opening**2, atol=1e-12)
        self.assertEqual(full["quadrature"], "three-point Simpson")
        self.assertNotAlmostEqual(full["finite_work_release_N_mm"][-1], cracks[0]**3)

    def test_negative_plastic_areas_remain_ineligible_without_clipping(self):
        cracks = np.array([1., 2., 3.])
        force = np.array([[1., 2.], [1., 2.], [1., 2.]])
        compliance = np.ones(3)
        result = interval_quantities(cracks, force, .25*force**2, compliance,
                                     np.ones(3), B=2., W=10.)
        self.assertTrue(np.all(result["Apl_N_mm"] < 0))
        self.assertTrue(all(not row["eligible_nonnegative_Apl"].any() for row in result["intervals"]))

    def test_nonlinear_manufactured_potential_recovers_known_ASTM_decomposition(self):
        # H(a)=b² exp(-.522a/W) has -H'/H=eta_ASTM/b. For
        # U=H(a)*(.5delta²-.05delta⁴), the exact total J therefore satisfies
        # J=Jel+eta*Apl/(B*b), independently of our finite-interval routine.
        W, B = 10., 2.
        a = np.array([3., 3.01, 3.02])
        b = W-a
        H = b*b*np.exp(-.522*a/W)
        delta = np.array([0., .25, .5, .75, 1.])
        force = H[:, None]*(delta-.2*delta**3)
        work = H[:, None]*(.5*delta**2-.05*delta**4)
        factors = (2/b+.522/W)/(2*B*H)
        result = interval_quantities(a, force, work, 1/H, factors, B, W)
        full = result["intervals"][-1]
        self.assertTrue(full["eligible_nonnegative_Apl"].all())
        self.assertTrue(np.all(result["Apl_N_mm"][:, 1:] > 0))
        np.testing.assert_allclose(full["predicted_total_J_N_mm"],
                                   full["finite_work_release_N_mm"], rtol=1e-9, atol=1e-12)

    def test_interval_quadrature_rejects_irregular_or_invalid_geometry(self):
        values = (np.ones((3, 2)), np.ones((3, 2)), np.ones(3), np.ones(3))
        for cracks in ([1., 2., 4.], [1., 1., 2.], [8., 9., 10.]):
            with self.assertRaisesRegex(ValueError, "equally spaced"):
                interval_quantities(cracks, *values, B=2., W=10.)

    def test_resolved_event_changes_only_opening_and_reuses_elastic_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls, output = root/"controls", root/"event-controls"
            controls.mkdir()
            files = {name: root/name for name in ("original.inp", "original.odb", "dense.inp", "dense.odb")}
            for name, path in files.items():
                path.write_text(name)
            (controls/"material-fit.json").write_text('{}')
            (controls/"periodic").mkdir()
            (controls/"periodic/tensor-results.json").write_text('[]')
            for i in range(3):
                (controls/("fcc_el_a"+str(i))).mkdir()
                (controls/("fcc_el_a"+str(i))/"result.json").write_text('{}')
                for short, law in (("inc", "incremental"), ("pot", "potential")):
                    name = "fcc_{}_a{}".format(short, i)
                    directory = controls/name
                    directory.mkdir()
                    inp = directory/(name+".inp")
                    inp.write_text("*Heading\n** unchanged mesh/material\n*Boundary\nload,2,2,1.\n")
                    (directory/"source.json").write_text(json.dumps({"job": name, "input": str(inp),
                        "input_sha256": sha(inp), "source_inp": str(files["original.inp"]),
                        "source_sha256": sha(files["original.inp"]), "source_odb": str(files["original.odb"]),
                        "law": law, "crack_index": i, "a_mm": 80+10*i, "W_mm": 160., "B_mm": 80.,
                        "material_fit_sha256": sha(controls/"material-fit.json"), "loading_displacement_mm": 1.}))
            receipt = root/"event.json"
            event = {"state": "completed_and_extracted", "read_only": True,
                     "source_equivalence": {"passed": True}, "criterion": "same material LE criterion",
                     "original_source_inp": str(files["original.inp"]), "original_source_inp_sha256": sha(files["original.inp"]),
                     "original_source_odb": str(files["original.odb"]), "original_source_odb_sha256": sha(files["original.odb"]),
                     "event_source_odb": str(files["dense.odb"]), "event_source_odb_sha256": sha(files["dense.odb"]),
                     "input": str(files["dense.inp"]), "event_source_inp_sha256": sha(files["dense.inp"]),
                     "event": {"previous_field": {"max_LE11": .1, "zero_STATUS": 0, "time_s": .04, "opening_mm": 2.},
                               "crossing_field": {"max_LE11": .12, "time_s": .041}}}
            receipt.write_text(json.dumps(event))
            records = prepare_event(controls, output, receipt)
            self.assertEqual(len(records), 6)
            self.assertEqual(len(list(output.glob("fcc_el*"))), 0)
            for row in records:
                self.assertIn("load,2,2,2", Path(row["input"]).read_text())
                self.assertIn("load,2,2,1.", Path(row["parent_input"]).read_text())
            self.assertEqual(prepare_event(controls, output, receipt), records)
            files["dense.odb"].write_text('changed')
            with self.assertRaisesRegex(ValueError, "source changed"):
                prepare_event(controls, output, receipt)

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
