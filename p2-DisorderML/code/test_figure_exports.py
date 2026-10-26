"""Synthetic checks for saved-output figure meaning; no research data required."""
import copy
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resources.MLmetrics import (
    plot_mechanical_tradeoffs, plot_curve_comparison_panels,
    prepare_beam_plot_geometry, plot_lattice_field_comparison,
)


class FigureExportTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_scatter_retains_individual_values_and_reference(self):
        properties = pd.DataFrame(
            {"Ductility": [2., 3., 4.], "K_JIC": [10., 9., 12.],
             "Strength": [100., 95., 90.]}, index=[7, 20, 30])
        original = properties.copy(deep=True)
        _, ax, _ = plot_mechanical_tradeoffs(properties, periodic_id=7, highlight_id=30)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[1.5, .9], [2., 1.2]])
        np.testing.assert_allclose(ax.collections[0].get_array(), [.95, .9])
        np.testing.assert_allclose(ax.collections[1].get_offsets(), [[1., 1.]])
        np.testing.assert_allclose(ax.collections[2].get_offsets(), [[2., 1.2]])
        pd.testing.assert_frame_equal(properties, original)

    def test_scatter_rejects_duplicate_ids_and_nonfinite_observations(self):
        good = pd.DataFrame({"Ductility": [1., 2.], "K_JIC": [1., 3.],
                             "Strength": [1., .9]}, index=[0, 1])
        duplicate = good.copy()
        duplicate.index = [0, 0]
        nonfinite = good.copy()
        nonfinite.loc[1, "Strength"] = np.nan
        for bad in (duplicate, nonfinite):
            with self.subTest(index=list(bad.index)), self.assertRaises(ValueError):
                plot_mechanical_tradeoffs(bad)

    def test_curve_bars_are_absolute_same_axis_residuals(self):
        x = np.array([0., .5, 1.])
        truth = np.array([0., 4., 2.])
        prediction = np.array([1., 3., 2.5])
        originals = [v.copy() for v in (x, truth, prediction)]
        style = {k: copy.deepcopy(plt.rcParams[k]) for k in ("font.size", "font.family", "axes.spines.top")}
        fig, axes = plot_curve_comparison_panels([
            dict(label="Example", xlabel="Displacement (mm)", ylabel="Force (kN)",
                 x=x, truth=truth, prediction=prediction)])
        ax = axes[0]
        self.assertEqual(fig.axes, [ax])
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [1., 1., .5])
        np.testing.assert_allclose([bar.get_width() for bar in ax.patches], 1. / 3)
        np.testing.assert_allclose([bar.get_y() for bar in ax.patches], 0.)
        self.assertTrue(all(bar.get_alpha() == .25 for bar in ax.patches))
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), truth)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), prediction)
        for actual, original in zip((x, truth, prediction), originals):
            np.testing.assert_array_equal(actual, original)
        self.assertEqual({k: plt.rcParams[k] for k in style}, style)

    @staticmethod
    def beam_map():
        actual = np.array([[0., 0.], [1., .1], [2., 0.], [.5, .05], [1.5, .05]])
        return dict(
            canonical_xy=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]),
            canonical_node_present=np.array([True, True, True, False]),
            canonical_endpoint=np.array([[0, 1], [1, 2]]),
            endpoint_node_label=np.array([[10, 11], [11, 12]]),
            endpoint_xy0=actual[np.array([[0, 1], [1, 2]])],
            node_label=np.array([10, 11, 12, 20, 21]), node_xy0=actual,
            element_connectivity=np.array([[20, 10], [11, 20], [21, 11], [12, 21]]),
            parent_strut_index=np.array([0, 0, 1, 1]), chain_order=np.array([0, 1, 0, 1]))

    def test_geometry_joins_saved_order_and_orients_native_segments(self):
        beam = self.beam_map()
        saved = beam["canonical_xy"][[2, 0, 3, 1]]
        mask = np.array([True, True, False, True])
        geometry = prepare_beam_plot_geometry(beam, saved, mask)
        np.testing.assert_array_equal(geometry["edges"], [[1, 3], [3, 0]])
        np.testing.assert_allclose(geometry["node_coords"][[1, 3, 0]], beam["node_xy0"][:3])
        self.assertTrue(np.isnan(geometry["node_coords"][2]).all())
        np.testing.assert_allclose(geometry["beam_paths"][0], [[0, 0], [.5, .05], [1, .1]])
        np.testing.assert_allclose(geometry["beam_paths"][1], [[1, .1], [1.5, .05], [2, 0]])

    def test_geometry_rejects_nonbijective_masked_or_disconnected_maps(self):
        beam = self.beam_map()
        duplicate = beam["canonical_xy"].copy()
        duplicate[3] = duplicate[2]
        mask = beam["canonical_node_present"]
        with self.assertRaises(ValueError):
            prepare_beam_plot_geometry(beam, duplicate, mask)
        with self.assertRaises(ValueError):
            prepare_beam_plot_geometry(beam, beam["canonical_xy"], np.ones(4, bool))
        disconnected = copy.deepcopy(beam)
        disconnected["element_connectivity"][0] = [12, 20]
        with self.assertRaises(ValueError):
            prepare_beam_plot_geometry(disconnected, beam["canonical_xy"], mask)

    def test_field_ribbons_preserve_gaps_mask_endpoints_and_signed_errors(self):
        coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        panel = dict(label="Example", truth=np.arange(4.), prediction=np.arange(4.) + .25,
                     node_coords=coords, edges=np.array([[0, 1], [2, 3]]),
                     beam_paths=[coords[[0, 1]], coords[[2, 3]]], beam_width=.1)
        _, axes = plot_lattice_field_comparison([panel], minimum_stroke_pt=0)
        for ax in axes[0]:
            polygons = ax.collections[0].get_paths()
            self.assertTrue(all(path.vertices[:, 0].max() <= 1 or path.vertices[:, 0].min() >= 2
                                for path in polygons))
        np.testing.assert_allclose(axes[0, 2].collections[0].get_array(), .25)
        masked = dict(panel, valid_mask=np.array([True, True, True, False]))
        _, axes = plot_lattice_field_comparison([masked], minimum_stroke_pt=0)
        self.assertTrue(all(path.vertices[:, 0].max() <= 1
                            for path in axes[0, 0].collections[0].get_paths()))
        invalid_path = dict(panel, beam_paths=[coords[[0, 2]], coords[[2, 3]]])
        with self.assertRaises(ValueError):
            plot_lattice_field_comparison([invalid_path])

    def test_field_limits_cover_visible_material_in_an_explicit_crop(self):
        coords = np.array([[0., 0.], [1., 0.], [10., 0.], [11., 0.]])
        panel = dict(label="Cropped example", truth=np.array([0., 1., 100., 110.]),
                     prediction=np.array([.1, 1.1, 120., 130.]), node_coords=coords,
                     edges=np.array([[0, 1], [2, 3]]),
                     beam_paths=[coords[[0, 1]], coords[[2, 3]]], beam_width=.1,
                     value_limit=2., error_limit=.5)
        cropped = dict(panel, xlim=(-.2, 1.2), ylim=(-.2, .2))
        _, axes = plot_lattice_field_comparison([cropped], minimum_stroke_pt=0)
        self.assertEqual(axes[0, 0].collections[0].norm.vmax, 2.)
        self.assertEqual(axes[0, 2].collections[0].norm.vmax, .5)
        with self.assertRaises(ValueError):
            plot_lattice_field_comparison([panel], minimum_stroke_pt=0)


if __name__ == "__main__":
    unittest.main()
