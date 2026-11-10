"""Source-parameter and normalization checks for the published lattice CT table."""

import unittest

import numpy as np

from resources.lattices import calc_anisoParams
from validation_literature import A_OVER_W, PUBLISHED, published_ct_comparison


class PublishedLatticeCalibrationTests(unittest.TestCase):
    def test_published_snub_square_geometry_and_range(self):
        self.assertEqual(A_OVER_W, 0.25)
        self.assertEqual(PUBLISHED, ((.2, 7.65, 1.15), (.25, 4.74, 1.10),
                                     (.3, 3.16, 1.06)))
        for _, chi, _ in PUBLISHED:
            self.assertTrue(.1 <= chi <= 10.)

    def test_normalized_compliance_has_published_parameters(self):
        for _, chi, _ in PUBLISHED:
            compliance = np.array([[1., -.433, 0.], [-.433, 1., 0.],
                                   [0., 0., 2 * (chi + .433)]])
            lam, result = calc_anisoParams(S=compliance)
            self.assertAlmostEqual(lam, 1.)
            self.assertAlmostEqual(result, chi)

    def test_published_y_normalization_uses_actual_source_crack_ratio(self):
        rows = published_ct_comparison()
        for row in rows:
            self.assertAlmostEqual(row["difference_percent"],
                                   100 * (row["ortega_Y"] / row["published_Y"] - 1))
        self.assertAlmostEqual(rows[0]["ortega_Y"], 1.2276413627265912)
        self.assertAlmostEqual(rows[-1]["difference_percent"], 3.5103197278219422)


if __name__ == "__main__":
    unittest.main()
