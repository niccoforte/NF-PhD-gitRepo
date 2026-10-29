"""Focused lattice-helper regressions; no FEA, research files or ML runs."""

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from resources import lattices


def fcc_patch(cell_size, nx=2, ny=2):
    """Independent corner/centre description of a finite FCC patch."""
    corners = [(x, y) for y in range(ny+1) for x in range(nx+1)]
    centres = [(x+.5, y+.5) for y in range(ny) for x in range(nx)]
    xy = np.array(corners + centres)*cell_size
    nodes = np.column_stack([np.arange(1, len(xy)+1), xy])
    ids = {point: i+1 for i, point in enumerate(corners + centres)}
    expected = set()
    for x, y in corners:
        for point in ((x+1, y), (x, y+1)):
            if point in ids:
                expected.add(tuple(sorted((ids[x, y], ids[point]))))
    for x, y in centres:
        for point in ((x-.5, y-.5), (x-.5, y+.5), (x+.5, y-.5), (x+.5, y+.5)):
            expected.add(tuple(sorted((ids[x, y], ids[point]))))
    return nodes, expected


class LatticeValidationTests(unittest.TestCase):
    def test_fcc_connections_match_cell_topology_at_every_scale(self):
        for cell_size in (0.13, 0.2, 3., 10., 20.):
            with self.subTest(cell_size=cell_size):
                nodes, expected = fcc_patch(cell_size)
                geom = SimpleNamespace(l=cell_size)
                actual = lattices.connectivity("FCC", nodes, geom)
                self.assertEqual(len(actual), 28)
                self.assertEqual({tuple(sorted(row[1:])) for row in actual}, expected)
                # Roundoff on either coordinate axis must not change the graph.
                perturbed = nodes.copy()
                perturbed[9:, 1:] += np.array([1e-11, -1e-11])*cell_size
                np.testing.assert_array_equal(lattices.connectivity("FCC", perturbed, geom), actual)
                # Stiffness helpers use [x,y,z] and zero-based endpoints.
                stiff_nodes = np.column_stack([nodes[:, 1:], np.zeros(len(nodes))])
                stiff = lattices.connectivity("FCC", stiff_nodes, geom, stiff=True)
                self.assertEqual({tuple(sorted(row[1:]+1)) for row in stiff}, expected)

    def test_canonical_fcc_body_and_initial_crack_counts(self):
        nodes, expected = fcc_patch(10., nx=20, ny=19)
        geom = lattices.Geometry("FCC", 10., 20)
        elements = lattices.connectivity("FCC", nodes, geom)
        self.assertEqual(len(nodes), 800)
        self.assertEqual(len(elements), 2319)
        self.assertEqual({tuple(sorted(row[1:])) for row in elements}, expected)
        removed = (nodes[:, 2] == 95) & (nodes[:, 1] < 120)
        fractured = nodes[~removed].copy()
        fractured[:, 0] = np.arange(1, len(fractured)+1)
        elements = lattices.connectivity("FCC", fractured, geom)
        midpoints = fractured[elements[:, 1:].astype(int)-1, 1:].mean(axis=1)
        keep = [not lattices.insidePoint((-16, 93), (117.6, 97), point) for point in midpoints]
        self.assertEqual(len(fractured), 788)
        self.assertEqual(int(np.count_nonzero(keep)), 2259)

    def test_boundary_weights_preserve_fractional_and_vector_thickness(self):
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [.5, .5]])
        elements = np.array([[1, 0, 1], [2, 0, 3], [3, 1, 2]])
        for supplied, expected in ((1, [.5, 1., .5]), ([1, 2, 3], [.5, 2., 1.5])):
            geom = lattices.edgeElems(nodes, elements, SimpleNamespace(t=supplied))
            np.testing.assert_array_equal(geom.t, expected)
            self.assertEqual(geom.t.dtype.kind, "f")
        integer = lattices.effProperties("FCC", lattices.Geometry("FCC", 10, 20, t=1))
        floating = lattices.effProperties("FCC", lattices.Geometry("FCC", 10, 20, t=1.))
        np.testing.assert_array_equal(integer, floating)

    def test_density_and_thickness_setters_stay_consistent(self):
        for lat, coefficient in (("FCC", 2*(1+np.sqrt(2))), ("tri", 2*np.sqrt(3)),
                                 ("kagome", np.sqrt(3)), ("hex", 2/np.sqrt(3))):
            with self.subTest(lat=lat):
                geom = lattices.Geometry(lat, 10, 20)
                geom.rDthickness(t=.5)
                self.assertEqual(geom.t, .5)
                self.assertAlmostEqual(geom.rD, coefficient*.5/10)
                geom.rDthickness(rD=.3)
                self.assertEqual(geom.rD, .3)
                self.assertAlmostEqual(geom.t, 10*.3/coefficient)
                geom.rDthickness(t=np.array([.5, 1.]))
                np.testing.assert_array_equal(geom.t, [.5, 1.])
                np.testing.assert_allclose(geom.rD, coefficient*np.array([.5, 1.])/10)
                geom.rDthickness(rD=np.array([.1, .2]))
                np.testing.assert_allclose(geom.t, 10*np.array([.1, .2])/coefficient)
                geom.rDthickness(rD=0.)
                self.assertEqual(geom.rD, 0.)
                self.assertEqual(geom.t, 0.)

    def test_inactive_fracture_mode_deletes_rows_not_candidate_ids(self):
        nodes = np.array([[1., 0., 1.], [2., 1., 1.], [3., 5., 1.], [4., 6., 1.]])
        geom = SimpleNamespace(l=1., W=10., H=2., a0=2.)
        # The left edge crosses the prescribed crack box; the right one survives.
        elements = lattices.connectivity("hex", nodes, geom, mode="fracture")
        np.testing.assert_array_equal(elements, [[1, 3, 4]])
        stiff_nodes = np.column_stack([nodes[:, 1:], np.zeros(len(nodes))])
        elements = lattices.connectivity("hex", stiff_nodes, geom, stiff=True, mode="fracture")
        np.testing.assert_array_equal(elements, [[1, 2, 3]])

    def test_find_nodes_honours_explicit_path_and_preserves_default(self):
        geom = lattices.Geometry("FCC", 10., 20)
        frame = lattices.pd.DataFrame([[0., 0.], [10., 0.]])
        for path, prefix in ((Path("custom-study"), "custom-study"),
                             (None, "Z:/p1/sims/Ti/stiffMatrix")):
            with patch.object(lattices.pd, "read_csv", return_value=frame) as reader:
                lattices.find_nodes("FCC", geom, "per", stiff=True, path=path)
                self.assertEqual(reader.call_args.args[0], prefix + "/transfer/IN-nDuctile-FCC-20-per-1.csv")

    def test_direction_sign_preserves_quadratic_stiffness(self):
        for vector in ((1., 1.), (-1., 1.), (-1., -1.), (1., -1.), (0., -2.)):
            unit = np.array(vector)/np.linalg.norm(vector)
            expected = [[unit[0]**2, unit[1]**2, unit[0]*unit[1]]]
            np.testing.assert_allclose(lattices.get_Nmatrix([lattices.n_values(vector)]), expected, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            lattices.n_values([0., 0.])
        with self.assertRaisesRegex(ValueError, "finite"):
            lattices.n_values([float("nan"), 1.])

    def test_isotropy_uses_general_engineering_shear_relations(self):
        for scale in (1e-12, 1., 1e9):
            for c in (np.array([[4., 1., 0.], [1., 4., 0.], [0., 0., 1.5]]),
                      np.diag([4., 4., 2.])):
                self.assertTrue(lattices.check_isotropy(scale*c))
        self.assertTrue(lattices.check_isotropy([[3, 1, 0], [1, 3, 0], [0, 0, 1]]))
        self.assertFalse(lattices.check_isotropy([[3, 1, .2], [1, 3, .1], [.2, .1, 1]]))
        self.assertFalse(lattices.check_isotropy([[4, 1, 0], [2, 4, 0], [0, 0, 1.5]]))
        self.assertFalse(lattices.check_isotropy(np.eye(2)))

    def test_scalar_plot_draws_every_element_without_mutating_thickness(self):
        geom = SimpleNamespace(t=1)
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.]])
        with patch.object(plt, "show"):
            lattices.plot_lattice(np.array([[1, 0, 1], [2, 1, 2]]), nodes, geom)
            figure = plt.gcf()
            self.assertEqual(len(figure.axes[0].lines), 2)
            self.assertEqual([line.get_linewidth() for line in figure.axes[0].lines], [2., 2.])
            plt.close(figure)
        self.assertEqual(geom.t, 1)

    def test_canonical_unit_stiffness_and_reviewed_effective_properties(self):
        matrices = {
            "FCC": [[8697413406.3, 3602586593.7000003, 0.],
                    [3602586593.7000003, 8697413406.3, 0.], [0., 0., 3602586593.7000003]],
            "tri": [[9225000000., 3075000000., 0.], [3075000000., 9225000000., 0.], [0., 0., 3075000000.]],
        }
        effective = {
            "FCC": [7205173178.221861, .41421356274619014, 8340266482.872675, .4525020836777498],
            "tri": [8200000000., 1/3, 8453608247.422681, .37457044673539514],
            "kagome": [8200000000., 1/3, 8453608247.422681, .37457044673539514],
            "hex": [8200000000., 1/3, 8244520410.215161, .34057242442522956],
        }
        for lat in effective:
            with self.subTest(lat=lat):
                geom = lattices.Geometry(lat, 10., 30 if lat == "tri" else 20)
                before = copy.deepcopy(geom.__dict__)
                matrix = lattices.calcC_mohr(copy.deepcopy(geom), "unit")[0]
                np.testing.assert_array_equal(matrix, matrices["FCC" if lat == "FCC" else "tri"])
                np.testing.assert_array_equal(lattices.effProperties(lat, geom), effective[lat])
                self.assertEqual(geom.__dict__, before)


if __name__ == "__main__":
    unittest.main()
