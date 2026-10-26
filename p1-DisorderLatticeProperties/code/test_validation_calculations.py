"""Focused physical/temporal invariants for the validation-only comparison."""

import unittest

import numpy as np

from resources.lattices import Geometry
from validation_calculations import filter_force, fracture_metrics, fit_stiffness


class ValidationCalculationTests(unittest.TestCase):
    def setUp(self):
        self.geom = Geometry("tri", 10, 30, .2)
        self.geom.FTcalc()

    def test_elastic_work_has_zero_plastic_area(self):
        d = np.linspace(0, 2, 101)
        result = fracture_metrics(d, 120 * d, 100, self.geom, 8000, 120)
        self.assertAlmostEqual(result["Apl_N_mm"], 0, places=10)
        self.assertAlmostEqual(result["KJ_MPa_sqrt_m"], result["KI_MPa_sqrt_m"])

    def test_negative_area_remains_visible_and_invalid(self):
        d = np.linspace(0, 2, 101)
        result = fracture_metrics(d, 120 * d, 100, self.geom, 8000, 60)
        self.assertLess(result["Apl_N_mm"], 0)
        self.assertFalse(result["valid_plastic_area"])
        self.assertIsNone(result["KJ_MPa_sqrt_m"])

    def test_loading_point_unit_rescaling_preserves_work_relation(self):
        d = np.linspace(0, 2, 101)
        one = fracture_metrics(d, 120 * d, 100, self.geom, 8000, 120)
        two = fracture_metrics(d, 240 * d, 100, self.geom, 8000, 240)
        self.assertAlmostEqual(two["KI_MPa_sqrt_m"], 2 * one["KI_MPa_sqrt_m"])
        self.assertAlmostEqual(two["work_N_mm"], 2 * one["work_N_mm"])

    def test_causal_filter_does_not_read_future_samples(self):
        t = np.linspace(0, 1, 1001)
        f = np.sin(2 * np.pi * 3 * t)
        changed = f.copy()
        changed[601:] = 1e6
        first, _ = filter_force(t, f, 30)
        second, _ = filter_force(t, changed, 30)
        np.testing.assert_allclose(first[:601], second[:601], atol=1e-10)

    def test_dc_startup_and_noise_attenuation(self):
        t = np.linspace(0, 1, 1001)
        y, _ = filter_force(t, np.full(len(t), 123.), 30)
        np.testing.assert_allclose(y, 123., atol=1e-10)
        noisy = 123 + np.sin(2 * np.pi * 250 * t)
        y, _ = filter_force(t, noisy, 30)
        self.assertLess(np.std(y[200:]), .02)

    def test_missing_interval_and_nyquist_rejected(self):
        t = np.linspace(0, 1, 101)
        with self.assertRaises(ValueError):
            filter_force(np.delete(t, 20), np.ones(100), 10)
        with self.assertRaises(ValueError):
            filter_force(t, np.ones(101), 50)

    def test_initial_fit_recovers_independent_slope(self):
        d = np.linspace(0, 2, 101)
        fit = fit_stiffness(d, 120 * d + 7, 20)
        self.assertAlmostEqual(fit["stiffness_N_mm"], 120)
        self.assertAlmostEqual(fit["intercept_N"], 7)
        self.assertAlmostEqual(fit["r_squared"], 1)

    def test_matched_causal_filter_preserves_elastic_work_decomposition(self):
        t = np.linspace(0, 1, 1001)
        d = t * t
        force = 120 * d
        df, _ = filter_force(t, d, 20)
        ff, _ = filter_force(t, force, 20)
        result = fracture_metrics(df, ff, 990, self.geom, 8000, 120)
        self.assertAlmostEqual(result["Apl_N_mm"], 0, places=8)
        np.testing.assert_allclose(ff, 120 * df, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
