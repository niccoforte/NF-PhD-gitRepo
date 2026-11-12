"""Independent source/geometry/event conventions for the Omidi lattice reproduction."""
import json
import math
from types import SimpleNamespace
import unittest

import numpy as np

from validation_omidi import (ROOT, PERIOD, author_cell, measured_plastic,
                              elastic_energy_check, history_at_time, maximum_beam_tension,
                              orthotropic_K_displacement,
                              positive_endpoint_force, published_points, sha,
                              stl_oracle, tensor_properties)


class SnubSquareTests(unittest.TestCase):
    def test_corrected_K_field_parity_faces_front_stress_and_isotropic_limit(self):
        E,G,nu=124.,7.,.43
        S=np.array([[1/E,-nu/E,0],[-nu/E,1/E,0],[0,0,1/G]])
        C=np.linalg.inv(S)
        def stress(point):
            point=np.array(point,float);h=1e-5
            dx=(orthotropic_K_displacement(point+[h,0],E,G,nu)-orthotropic_K_displacement(point-[h,0],E,G,nu))/(2*h)
            dy=(orthotropic_K_displacement(point+[0,h],E,G,nu)-orthotropic_K_displacement(point-[0,h],E,G,nu))/(2*h)
            return C@np.array([dx[0],dy[1],dy[0]+dx[1]])
        np.testing.assert_allclose(stress([4.,0.]),[1/math.sqrt(8*math.pi),1/math.sqrt(8*math.pi),0.],atol=1e-8)
        upper=orthotropic_K_displacement([1.,2.],E,G,nu)
        lower=orthotropic_K_displacement([1.,-2.],E,G,nu)
        np.testing.assert_allclose(upper,lower*[1,-1],atol=1e-14)
        self.assertLess(abs(orthotropic_K_displacement([-4.,0.],E,G,nu)[0]),1e-12)
        np.testing.assert_allclose(stress([-4.,1e-3])[1:],0.,atol=3e-5)
        isoE,isoNu=100.,.3;theta=.7;r=4.
        xy=[r*math.cos(theta),r*math.sin(theta)]
        actual=orthotropic_K_displacement(xy,isoE,isoE/(2*(1+isoNu+1e-7)),isoNu)
        kappa=(3-isoNu)/(1+isoNu);scale=(1+isoNu)/isoE*math.sqrt(r/(2*math.pi))
        expected=scale*np.array([math.cos(theta/2)*(kappa-1+2*math.sin(theta/2)**2),
                                 math.sin(theta/2)*(kappa+1-2*math.cos(theta/2)**2)])
        np.testing.assert_allclose(actual,expected,rtol=1e-6)

    def test_nonfinite_elastic_channels_fail_closed(self):
        self.assertEqual(elastic_energy_check(1.,-1.,.02,[.01]),0.)
        for values in ((1.,math.nan,.02,[.01]),(1.,-1.,math.nan,[.01]),
                       (1.,-1.,.02,[math.nan]),(math.inf,-1.,.02,[.01])):
            with self.subTest(values=values),self.assertRaises(ValueError):elastic_energy_check(*values)
        field=SimpleNamespace(componentLabels=['S11'],bulkDataBlocks=[
            SimpleNamespace(data=np.array([[math.nan]]),elementLabels=[101],
                            integrationPoints=[1],sectionPoint=SimpleNamespace(number=1))])
        with self.assertRaises(ValueError):maximum_beam_tension(field,'S11')

    def test_conflicting_final_timestamp_is_unknown_not_a_last_sample_choice(self):
        curve=np.array([[0.,0.,0.],[.5,6.,480.],[.6,7.,490.],[.6,7.,.017]])
        endpoint=history_at_time(curve,.6)
        self.assertFalse(endpoint['history_valid'])
        self.assertIsNone(endpoint['force_N'])
        self.assertEqual(endpoint['native_indices'],[2,3])
        self.assertEqual(endpoint['opening_mm'],7.)
        self.assertEqual(history_at_time(curve,.5)['force_N'],480.)
        self.assertFalse(history_at_time(curve,.55)['history_valid'])

    def test_bulk_surface_maximum_retains_local_identity(self):
        blocks=[SimpleNamespace(data=np.array([[.04,0],[.02,0]]),elementLabels=[101,102],
                                integrationPoints=[1,1],sectionPoint=SimpleNamespace(number=1)),
                SimpleNamespace(data=np.array([[.01,0],[.07,0]]),elementLabels=[101,102],
                                integrationPoints=[1,1],sectionPoint=SimpleNamespace(number=5))]
        maximum,points=maximum_beam_tension(SimpleNamespace(componentLabels=['LE11','LE12'],bulkDataBlocks=blocks))
        self.assertEqual(points,{1,5})
        self.assertEqual(maximum,{'max_LE11':.07,'element':102,'integration_point':1,'section_point':5})
        subset,_=maximum_beam_tension(SimpleNamespace(componentLabels=['LE11','LE12'],bulkDataBlocks=blocks),element_labels={101})
        self.assertEqual(subset['element'],101)
        self.assertEqual(subset['max_LE11'],.04)

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

    @unittest.skipUnless((ROOT/'elastic/prepared.json').exists(),'Prepared elastic inputs unavailable')
    def test_elastic_controls_retain_geometry_and_remove_plasticity_and_opening_ramp(self):
        from validation_diagnostics import blocks,key
        from pathlib import Path
        rows=json.loads((ROOT/'elastic/prepared.json').read_text())
        self.assertEqual(len(rows),3)
        for row in rows:
            text=Path(row['input']).read_text()
            self.assertEqual(row['input_sha256'],sha(row['input']))
            self.assertNotIn('*plastic',text.lower())
            self.assertNotIn('nlgeom',text.lower())
            self.assertIn('*Cload\nLoad,2,1.\n',text)
            for kind in ('*node','*element','*beam section','*elastic','*equation'):
                current=[b for b in blocks(text)if key(b)==kind]
                original=[b for b in blocks(Path(row['source_inp']).read_text())if key(b)==kind]
                self.assertEqual(current,original)


class PublishedNotchPhaseTests(unittest.TestCase):
    def test_journal_phase_changes_only_the_notch_centre_rule(self):
        from validation_omidi import journal_phase_source
        source='W_x = 100.\nCy=W_x/2-(sqrt(3)+1)/4*L\nCrackSketch.rectangle(point1=(-L,Cy-L/4),point2=(ac,Cy+L/4))\n'
        result=journal_phase_source(source)
        self.assertEqual(result,source.replace('Cy=W_x/2-(sqrt(3)+1)/4*L','Cy=W_x/2'))
        with self.assertRaises(ValueError):journal_phase_source(result)


if __name__=='__main__':unittest.main()
