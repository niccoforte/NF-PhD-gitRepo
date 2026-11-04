"""Field-bracket semantics; no Abaqus or research inputs are required."""
import unittest

import numpy as np

from validation_orthotropic import bracket_history, triggered, orthotropic_elastic


class OrthotropicEvents(unittest.TestCase):
    def test_history_inside_field_bracket_is_not_certified_undamaged(self):
        curve = [{"time_s": t, "U2_mm": t, "RF2_N": 2*t} for t in (0., .029, .0295, .030000001)]
        bracket = bracket_history(curve, .029000002, .030000001)
        self.assertEqual(bracket["last_history_before_positive_field"]["index"], 2)
        self.assertEqual(bracket["last_history_at_or_before_last_nontriggering_field"]["index"], 1)

    def test_missing_lower_field_is_not_invented(self):
        bracket = bracket_history([{"time_s": 0.}], None, .01)
        self.assertIsNone(bracket["last_history_at_or_before_last_nontriggering_field"])

    def test_partial_damage_criterion_is_not_initiation(self):
        self.assertFalse(triggered("DUCTCRT_ge_1", 0., .9))
        self.assertTrue(triggered("DUCTCRT_ge_1", 0., 1.))
        self.assertFalse(triggered("global_STATUS_zero", 1., 1.))
        self.assertTrue(triggered("global_STATUS_zero", 0., 1.))

    def test_already_mapped_tensor_uses_consistent_orthotropic_modulus(self):
        C = np.array([[100., 20., 1e-8], [20., 100., -1e-8], [1e-8, -1e-8, 30.]])
        result = orthotropic_elastic(C)
        expected_S = np.linalg.inv(np.array([[100.,20.,0.],[20.,100.,0.],[0.,0.,30.]]))
        chi = (2*expected_S[0,1]+expected_S[2,2])/(2*expected_S[0,0])
        self.assertAlmostEqual(result["M_I_pe_MPa"], np.sqrt(2/(1+chi))/expected_S[0,0])
        self.assertEqual(result["C_state"], "plane_strain")
        self.assertLess(result["alignment_max_relative_correction"], 1e-6)
        self.assertEqual(result["C_raw_MPa"][0][2], 1e-8)

    def test_physical_normal_shear_coupling_is_not_silently_projected(self):
        with self.assertRaisesRegex(ValueError, "not numerically aligned"):
            orthotropic_elastic([[100.,20.,1.],[20.,100.,0.],[1.,0.,30.]])


if __name__ == "__main__":
    unittest.main()
