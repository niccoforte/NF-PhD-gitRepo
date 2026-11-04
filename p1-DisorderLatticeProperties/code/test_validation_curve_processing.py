"""Meaningful mechanics/missingness invariants for the opt-in curve pilot."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation_curve_processing import initial_fit, paired_filter, transfer_prefix


class CurveQualification(unittest.TestCase):
    def test_affine_elastic_response_is_recovered_with_nonzero_intercept(self):
        x = np.linspace(0, .004, 20)
        answer = initial_fit(x, 7300*x+.3)
        self.assertEqual(answer["status"], "numerically_stable_candidate")
        self.assertAlmostEqual(answer["selected"]["slope"], 7300.)
        self.assertFalse(answer["material_elasticity_verified"])

    def test_curved_initial_response_is_not_rescued_by_averaging(self):
        x = np.linspace(0, .005, 20)
        answer = initial_fit(x, 7000*x*(1+300*x))
        self.assertEqual(answer["status"], "unresolved_initial_window")

    def test_nan_gap_cannot_be_filtered_as_zero_or_interpolated(self):
        x = np.linspace(0, 1, 20)
        y = x.copy()
        y[10] = np.nan
        with self.assertRaises(ValueError):
            paired_filter(x, y, .2)

    def test_filter_is_affine_in_paired_work_coordinates(self):
        x = np.linspace(0, 1, 201)
        xx, yy = paired_filter(x, 2*x+3, .2)
        np.testing.assert_allclose(yy, 2*xx+3, atol=1e-10)

    def test_all_zero_status_tail_is_not_treated_as_observed_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fixture.csv"
            path.write_text("0,0,1,1\n1,2,1,1\n2,1,0,1\n3,0,0,0\n4,0,0,0\n")
            points, support = transfer_prefix(path)
            self.assertEqual(len(points), 3)
            self.assertEqual(support["first_partial_STATUS_deletion_row"], 2)
            self.assertEqual(support["first_all_zero_STATUS_row"], 3)
            self.assertIn("not_attested", support["native_support"])


if __name__ == "__main__":
    unittest.main()
