"""Source-checked elastic limits; no scientific files or solvers are needed."""
import unittest

import numpy as np

from resources.calculations import calc_FaW, calc_FaW_aniso
from resources.lattices import calc_anisoParams, prismatic_plane_strain


class SharedElasticLimits(unittest.TestCase):
    def test_corrected_ortega_fit_has_a_small_nonzero_isotropic_residual(self):
        # Isotropic engineering-shear compliance, independent of a specimen.
        compliance = np.array([[1., -.3, 0.], [-.3, 1., 0.], [0., 0., 2.6]])
        np.testing.assert_allclose(calc_anisoParams(S=compliance), (1., 1.))
        # Independently expanded corrected Ortega Eq.16 / Appendix A.5 and
        # the historical E1820-08a C(T) polynomial, evaluated at a/W=.5.
        self.assertAlmostEqual(calc_FaW(.5, 1.), 9.659078631008239, places=11)
        self.assertAlmostEqual(calc_FaW_aniso(.5, 1., S=compliance), 9.581893027488892, places=11)
        errors = [calc_FaW_aniso(a, 1., S=compliance)/calc_FaW(a, 1.)-1
                  for a in np.linspace(.2, .9, 701)]
        self.assertLess(max(abs(error) for error in errors), .01)
        self.assertGreater(abs(min(errors)), .009)

    def test_physical_and_already_reduced_isotropic_moduli_agree(self):
        young, poisson = 123000., .3
        plane_stress_C = young/(1-poisson**2)*np.array(
            [[1., poisson, 0.], [poisson, 1., 0.], [0., 0., (1-poisson)/2]])
        reduced = prismatic_plane_strain(plane_stress_C, 1., young, poisson)
        expected = young/(1-poisson**2)
        self.assertAlmostEqual(reduced["M_I_pe"], expected, places=8)
        mapped = prismatic_plane_strain(np.linalg.inv(reduced["S_xy_pe"]), C_state="plane_strain")
        self.assertAlmostEqual(mapped["M_I_pe"], expected, places=8)
        self.assertAlmostEqual(mapped["M_I_pe"], 1/mapped["S_pe"][0, 0], places=8)

    def test_tensor_scaling_scales_M_without_validating_a_disorder_model(self):
        tensor = np.array([[10000., 3000., 0.], [3000., 10000., 0.], [0., 0., 2000.]])
        reference = prismatic_plane_strain(tensor, C_state="plane_strain")
        scaled = prismatic_plane_strain(.7*tensor, C_state="plane_strain")
        self.assertAlmostEqual(scaled["M_I_pe"], .7*reference["M_I_pe"], places=8)
        np.testing.assert_allclose((scaled["lambda_pe"], scaled["chi_pe"]),
                                   (reference["lambda_pe"], reference["chi_pe"]))


if __name__ == "__main__":
    unittest.main()
