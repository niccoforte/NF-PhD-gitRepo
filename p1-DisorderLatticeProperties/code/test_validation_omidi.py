"""Independent source/geometry/event conventions for the Omidi lattice reproduction."""
import json
import math
import unittest

import numpy as np

from validation_omidi import (ROOT, PERIOD, author_cell, measured_plastic,
                              positive_endpoint_force, published_points, sha,
                              stl_oracle, tensor_properties)


class SnubSquareTests(unittest.TestCase):
    def test_initiation_force_preserves_recorded_positive_sign(self):
        self.assertEqual(positive_endpoint_force({'force_N':17.25}),17.25)

    def test_negative_zero_missing_and_nonfinite_initiation_force_rejected(self):
        for endpoint in ({'force_N':-17.25},{'force_N':0.},{'force_N':math.nan},
                         {'force_N':math.inf},{'force_N':None},{}):
            with self.subTest(endpoint=endpoint),self.assertRaises(ValueError):
                positive_endpoint_force(endpoint)

    def test_original_twenty_bar_motif_and_orthogonal_translation(self):
        cell=np.asarray(author_cell())
        self.assertEqual(cell.shape,(20,2,2))
        np.testing.assert_allclose(np.linalg.norm(cell[:,1]-cell[:,0],axis=1),6.,atol=1e-12)
        self.assertAlmostEqual(PERIOD,16.392304845413264)

    def test_full_tensor_engineering_shear_isotropic_limit(self):
        E,nu=120.,.3
        C=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])*E/(1-nu*nu)
        properties=tensor_properties(C)
        self.assertAlmostEqual(properties['lambda_crack_axis'],1.)
        self.assertAlmostEqual(properties['chi'],1.)
        self.assertAlmostEqual(properties['M_crack_axis_MPa'],E)

    def test_negative_shear_or_nonsymmetric_tensor_rejected(self):
        for C in (np.diag([1.,1.,-1.]),np.array([[1.,.2,0],[0,1.,0],[0,0,1.]])):
            with self.assertRaises(ValueError):tensor_properties(C)

    @unittest.skipUnless((ROOT/'source/Snub Square_rho0.20.STL').exists(),'Local primary source unavailable')
    def test_cell_phase_matches_published_solid_cad(self):
        inside,_,_=stl_oracle(ROOT/'source/Snub Square_rho0.20.STL')
        samples=np.concatenate([a+np.linspace(.05,.95,10)[:,None]*(b-a)for a,b in author_cell()])
        # Independently checks the 3mm offset discovered from actual printed CAD.
        self.assertTrue(inside(samples+[3+5*PERIOD,3*PERIOD]).all())
        self.assertLess(inside(samples+[5*PERIOD,3*PERIOD]).mean(),.5)

    @unittest.skipUnless((ROOT/'source/Fig 4.xlsx').exists(),'Local primary source unavailable')
    def test_measured_constituent_table_is_monotone_and_source_bound(self):
        table,metadata=measured_plastic(ROOT/'source/Fig 4.xlsx')
        table=np.array(table)
        self.assertTrue(np.isfinite(table).all())
        self.assertTrue((np.diff(table[:,0])>=0).all())
        self.assertTrue((np.diff(table[:,1])>0).all())
        self.assertEqual(table[0,1],0.)
        self.assertEqual(metadata['physical_E_MPa'],2000.)
        self.assertEqual(metadata['sha256'],sha(ROOT/'source/Fig 4.xlsx'))

    @unittest.skipUnless((ROOT/'source/Fig 10.xlsx').exists(),'Local primary source unavailable')
    def test_direct_published_ct_boundary_layer_and_experiments_not_conflated(self):
        data=published_points(ROOT/'source')
        self.assertEqual([r['published_CT_FE']for r in data],[.145,.212,.272])
        self.assertAlmostEqual(data[0]['published_boundary_layer'],.15159551467867666)
        self.assertNotEqual(data[1]['published_CT_FE'],data[1]['published_experiment1'])

    @unittest.skipUnless((ROOT/'prepared-CT.json').exists(),'Prepared inputs unavailable')
    def test_frozen_geometry_thickness_and_input_identity(self):
        records=json.loads((ROOT/'prepared-CT.json').read_text())
        self.assertEqual([r['section_t_mm']for r in records],[.45,.56,.67])
        for row in records:
            self.assertEqual(row['input_sha256'],sha(row['input']))
            self.assertEqual(row['disconnected_nodes_discarded'],0)
            self.assertEqual(row['pin_endpoint_counts'],[10,10])
            self.assertAlmostEqual(row['a_mm']/row['W_mm'],.25)
            self.assertEqual(row['B_mm'],15.)


if __name__=='__main__':unittest.main()
