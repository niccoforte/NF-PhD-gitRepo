"""Elastic constitutive/axis regressions; no research data or FEA is altered."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from resources.lattices import (
    Geometry, calcC_mohr, calc_anisoParams, effProperties,
    prismatic_plane_strain,
)
from resources.calculations import calcFT, calc_FaW_aniso


def inplane_stiffness(E1, E2, nu12, G12):
    return np.linalg.inv([[1/E1, -nu12/E1, 0],
                          [-nu12/E1, 1/E2, 0], [0, 0, 1/G12]])


class PlaneStrainTests(unittest.TestCase):
    def test_transverse_isotropy_recovers_fleck_reduction(self):
        E_s, nu_s, density, E, nu = 123e9, .3, .2, 8.2e9, 1/3
        C = inplane_stiffness(E, E, nu, E/(2*(1+nu)))
        result = prismatic_plane_strain(C, density, E_s, nu_s)
        q = E/(density*E_s)*nu_s**2
        self.assertAlmostEqual(result['E_pe']/E, 1/(1-q))
        self.assertAlmostEqual(result['nu_pe'], (nu+q)/(1-q))
        self.assertAlmostEqual(result['lambda_pe'], 1.)
        self.assertAlmostEqual(result['chi_pe'], 1.)
        self.assertAlmostEqual(result['M_I_pe']/result['E_pe'], 1.)
        self.assertEqual(result['S_xy_pe'][2, 2], 1/(E/(2*(1+nu))))

    def test_full_solid_limit_recovers_isotropic_plane_strain(self):
        E, nu = 123e9, .3
        result = prismatic_plane_strain(inplane_stiffness(E, E, nu, E/(2*(1+nu))), 1., E, nu)
        self.assertAlmostEqual(result['M_I_pe']/E, 1/(1-nu**2))
        self.assertAlmostEqual(result['nu_pe'], nu/(1-nu))

    def test_reciprocity_and_schur_complement(self):
        result = prismatic_plane_strain(inplane_stiffness(10e9, 5e9, .2, 2e9), .2)
        S = result['S_1236']
        np.testing.assert_allclose(S, S.T, atol=1e-25)
        self.assertGreater(np.linalg.eigvalsh(S).min(), 0.)
        self.assertAlmostEqual(-S[0, 2]/S[2, 2], .3)  # nu31
        self.assertAlmostEqual(-S[2, 0]/S[0, 0], .3*10e9/24.6e9)  # nu13
        self.assertAlmostEqual((-S[2, 0]/S[0, 0])/10e9, .3/24.6e9)
        S_ii = S[np.ix_([0, 1, 3], [0, 1, 3])]
        expected = S_ii - np.outer(S[[0, 1, 3], 2], S[2, [0, 1, 3]])/S[2, 2]
        np.testing.assert_allclose(result['S_xy_pe'], expected)

    def test_bao_crack_axes_match_compliance_formula(self):
        C = inplane_stiffness(10e9, 5e9, .2, 2e9)
        for crack_axis in (0, 1):
            p = prismatic_plane_strain(C, .2, crack_axis=crack_axis)
            s = p['S_pe']
            H = np.sqrt(s[0, 0]*s[1, 1]/2)*np.sqrt(
                np.sqrt(s[1, 1]/s[0, 0]) + (2*s[0, 1]+s[2, 2])/(2*s[0, 0]))
            self.assertAlmostEqual(p['M_I_pe']*H, 1.)
        a = prismatic_plane_strain(C, .2)
        b = prismatic_plane_strain(C, .2, crack_axis=1)
        self.assertAlmostEqual(a['lambda_pe']*b['lambda_pe'], 1.)
        self.assertAlmostEqual(a['chi_pe'], b['chi_pe'])

    def test_already_reduced_tensor_is_not_reduced_twice(self):
        C = inplane_stiffness(10e9, 5e9, .2, 2e9)
        a = prismatic_plane_strain(C, .2)
        b = prismatic_plane_strain(np.linalg.inv(a['S_xy_pe']), C_state='plane_strain')
        np.testing.assert_allclose(a['S_pe'], b['S_pe'])
        self.assertAlmostEqual(a['M_I_pe']/b['M_I_pe'], 1.)
        self.assertIsNone(b['E3'])
        self.assertIsNone(b['S_1236'])

    def test_units_scale_moduli_but_not_dimensionless_parameters(self):
        C = inplane_stiffness(10e9, 5e9, .2, 2e9)
        a = prismatic_plane_strain(C, .2)
        b = prismatic_plane_strain(C/1e6, .2, E_s=123e3)
        for key in ('lambda_pe', 'chi_pe', 'nu_pe'):
            self.assertAlmostEqual(a[key], b[key])
        self.assertAlmostEqual(a['M_I_pe']/b['M_I_pe'], 1e6)

    def test_nonphysical_or_unaligned_inputs_are_rejected(self):
        C = inplane_stiffness(10e9, 5e9, .2, 2e9)
        for kwargs in ({'rD':0}, {'rD':2}, {'rD':.2,'E_s':0},
                       {'rD':.2,'C_state':'unknown'}, {'rD':.2,'crack_axis':2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prismatic_plane_strain(C, **kwargs)
        for invalid in (np.eye(2), -C, np.full((3, 3), np.nan), C+np.triu(np.ones((3, 3)), 1)):
            with self.assertRaises(ValueError):
                prismatic_plane_strain(invalid, .2)
        coupled = C.copy();coupled[0, 2] = coupled[2, 0] = 1e8
        with self.assertRaisesRegex(ValueError, 'S16 and S26'):
            prismatic_plane_strain(coupled, .2)
        # An excessively stiff in-plane model cannot be repaired by clipping.
        with self.assertRaisesRegex(ValueError, 'not positive definite'):
            prismatic_plane_strain(inplane_stiffness(123e9,123e9,.3,47e9), .01)

    def test_canonical_fcc_reduced_values_and_independent_moduli(self):
        g = Geometry('FCC', 10., 20)
        C = calcC_mohr(copy.deepcopy(g), 'unit')[0]
        S = np.linalg.inv(C)
        p = prismatic_plane_strain(C, .2)
        old_lambda, old_chi = calc_anisoParams(C=C)
        self.assertEqual(old_lambda, 1.)
        self.assertAlmostEqual(old_chi, .5857864359799835)
        self.assertAlmostEqual(p['chi_pe'], .5745719877991023)
        self.assertAlmostEqual(p['E3']/1e9, 24.6)
        E, nu, M, nu_pe = effProperties('FCC', g)
        self.assertAlmostEqual(E*S[0, 0], 1.)
        self.assertAlmostEqual(M/1e9, 8.340266482872675)
        self.assertNotAlmostEqual(E/M, 1.)
        self.assertAlmostEqual(nu_pe, p['nu_pe'])

    def test_ortega_factor_uses_declared_reduced_compliance(self):
        C = inplane_stiffness(8.2e9,8.2e9,1/3,3.075e9)
        p = prismatic_plane_strain(C, .2)
        self.assertAlmostEqual(calc_FaW_aniso(1., 2., S=p['S_pe']), 9.581893027488883)
        self.assertAlmostEqual(calc_FaW_aniso(1., 2., C=C), calc_FaW_aniso(1., 2., S=np.linalg.inv(C)))
        with self.assertRaisesRegex(ValueError, '0 < a < W'):
            calc_FaW_aniso(0., 2., C=C)

    def test_calcft_shares_reduction_and_requires_matched_M(self):
        C = calcC_mohr(Geometry('FCC',10.,20),'unit')[0]
        p = prismatic_plane_strain(C,.2)
        geom = SimpleNamespace(LAT='FCC',rD=.2,ftcalc_done=True,W=2.,B=1.,ai=[1.],iso=False)
        df = pd.DataFrame({'x':[3.,0.,.1,.2,.3,.4], 'y_sm':[3.,0.,1.,2.,2.5,2.]})
        result = calcFT(df,geom,p['M_I_pe'],C=C)
        expected_K = 2.5/np.sqrt(2)*calc_FaW_aniso(1.,2.,S=p['S_pe'])
        self.assertAlmostEqual(result[2][0],expected_K)
        reduced = calcFT(df,geom,p['M_I_pe'],C=np.linalg.inv(p['S_xy_pe']),C_state='plane_strain')
        np.testing.assert_allclose(result[2:],reduced[2:])
        with self.assertRaisesRegex(ValueError,'same reduced compliance'):
            calcFT(df,geom,123e9,C=C)

    def test_unequal_axial_moduli_require_an_explicit_source_axis_map(self):
        C = inplane_stiffness(10e9,5e9,.2,2e9)
        p = prismatic_plane_strain(C,.2)
        geom = SimpleNamespace(LAT='FCC',rD=.2,ftcalc_done=True,W=2.,B=1.,ai=[1.],iso=False)
        df = pd.DataFrame({'x':[3.,0.,.1,.2,.3,.4], 'y_sm':[3.,0.,1.,2.,2.5,2.]})
        with self.assertRaisesRegex(ValueError,'source-axis mapping'):
            calcFT(df,geom,p['M_I_pe'],C=C)
        # Explicit choices are numerical sensitivity controls, not validation.
        a = calcFT(df,geom,p['M_I_pe'],C=C,ortega_axis1=0)
        b = calcFT(df,geom,p['M_I_pe'],C=C,ortega_axis1=1)
        self.assertNotAlmostEqual(a[2][0],b[2][0])


if __name__ == '__main__':
    unittest.main()
