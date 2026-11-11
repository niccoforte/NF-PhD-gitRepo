"""Initial-interval sample-count regression; real curves are read-only checks."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from resources.calculations import calcUT


class InitialStiffnessTests(unittest.TestCase):
    @staticmethod
    def curve(peak=19, modulus=1234.):
        strain = np.arange(26, dtype=float) * .001
        stress = modulus * strain
        stress[peak + 1:] = 0.
        return pd.DataFrame({"x": np.r_[peak + 2, strain],
                             "y_sm": np.r_[peak + 2, stress]})

    def test_two_distinct_finite_samples_give_the_linear_modulus(self):
        frame = self.curve()
        self.assertEqual(int(.1 * np.argmax(frame.y_sm.to_numpy()[1:])), 1)
        self.assertAlmostEqual(calcUT(frame)[2], 1234., places=9)

    def test_one_sample_stays_unresolved(self):
        self.assertTrue(np.isnan(calcUT(self.curve(peak=9))[2]))

    def test_zero_strain_span_stays_unresolved(self):
        frame = self.curve()
        frame.loc[1:2, "x"] = 0.
        self.assertTrue(np.isnan(calcUT(frame)[2]))

    def test_nonfinite_initial_coordinates_stay_unresolved(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                frame = self.curve()
                frame.loc[2, "x"] = value
                self.assertTrue(np.isnan(calcUT(frame)[2]))

    def test_nonfinite_initial_response_stays_unresolved(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                frame = self.curve()
                frame.loc[2, "y_sm"] = value
                self.assertTrue(np.isnan(calcUT(frame)[2]))


class SavedHexagonalRegressionTests(unittest.TestCase):
    def check_saved(self, relative_root, simulation_id, expected_modulus):
        base = Path("Z:/p1/data/Ti/disNodes") / relative_root
        curve_path = base / "MLdata/UT-UT-disNodes-allOUT.csv"
        props_path = base / "MLdata/UT-disNodes-allProps.csv"
        if not curve_path.exists() or not props_path.exists():
            self.skipTest("Original research curves are unavailable on this host")
        curves = pd.read_csv(curve_path, index_col=0)
        props = pd.read_csv(props_path, index_col=0)
        frame = pd.DataFrame({"x": curves.iloc[0].to_numpy(float),
                              "y_sm": curves.loc[simulation_id].to_numpy(float)})
        ductility, strength, stiffness, work = calcUT(frame)
        self.assertAlmostEqual(stiffness, expected_modulus, places=7)
        saved = props.loc[simulation_id]
        np.testing.assert_allclose([ductility, strength, work],
                                   saved[["Ductility", "Strength", "WoF"]].to_numpy(float),
                                   rtol=1e-12, atol=1e-12)

    def test_full_id1_recovers_pre_guard_secant(self):
        self.check_saved("0.2/hex", 1, 1644.0563903081725)

    def test_mechanism_id737_recovers_pre_guard_secant(self):
        self.check_saved("Target-xs/0.2/hex", 737, 1491.561701839381)

    def test_three_point_ols_and_other_properties_are_unchanged(self):
        self.check_saved("0.2/hex", 2, 1550.453084112928)


if __name__ == "__main__":
    unittest.main()
