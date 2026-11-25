"""Checks for numerical tensor qualification, not constitutive validation."""

import unittest

import numpy as np

from resources.lattices import prismatic_plane_strain
from validation_materials import qualify_aligned_tensor


class AlignedTensorQualification(unittest.TestCase):
    def test_retained_fcc_roundoff_does_not_block_its_known_modulus(self):
        # Retained independent source-periodic result: tiny reaction-output
        # couplings and antisymmetry are below its declared numerical bound.
        raw = np.array([[9608.6357421875, 3959.175537109375, 6.226554251043126e-5],
                        [3959.175537109375, 9608.6357421875, -6.226554251043126e-5],
                        [6.226554251043126e-5, -6.226554978638887e-5, 3978.88232421875]])
        before = raw.copy()
        clean, evidence = qualify_aligned_tensor(raw)
        np.testing.assert_array_equal(raw, before)
        self.assertLess(evidence["normal_shear_relative_residual"], 1e-8)
        result = prismatic_plane_strain(clean, C_state="plane_strain")
        self.assertAlmostEqual(result["M_I_pe"], 8945.725327119437, places=7)

    def test_resolved_disorder_coupling_cannot_be_silently_removed(self):
        C = np.array([[7600., 2060., 78.], [2060., 7400., 32.], [78., 32., 2680.]])
        with self.assertRaisesRegex(ValueError, "resolved"):
            qualify_aligned_tensor(C)

    def test_asymmetry_and_nonpositive_matrices_are_rejected(self):
        C = np.array([[10., 3., 0.], [3.01, 10., 0.], [0., 0., 4.]])
        with self.assertRaisesRegex(ValueError, "resolved"):
            qualify_aligned_tensor(C)
        with self.assertRaisesRegex(ValueError, "positive"):
            qualify_aligned_tensor(np.diag([10., -2., 4.]))


if __name__ == "__main__":
    unittest.main()
