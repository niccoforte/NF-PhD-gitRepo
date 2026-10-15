"""Synthetic scientific-contract regressions; real ODB checks are a separate pass."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from resources.damage import map_beam_chains, read_beam_inp, reduce_damage


class DamageContract(unittest.TestCase):
    def test_nonconsecutive_labels_variable_subdivision_and_direction(self):
        nodes = {8:np.array([0.,0.]), 31:np.array([1.,0.]), 2:np.array([2.,0.]),
                 99:np.array([3.,0.]), 7:np.array([4.,0.])}
        edges = {412:(31,8), 3:(2,31), 900:(99,2), 50:(7,99)}
        result = map_beam_chains(nodes, edges, set(edges), {8:0, 2:1, 7:2})
        self.assertEqual([r['elements'] for r in result], [[412,3],[900,50]])
        reversed_edges = {e:tuple(reversed(v)) for e,v in edges.items()}
        self.assertEqual(result, map_beam_chains(nodes, reversed_edges, set(edges), {7:2,2:1,8:0}))

    def test_unidentified_branch_and_unassigned_cycle_refused(self):
        nodes = {i:np.array(p,float) for i,p in enumerate([(0,0),(1,0),(2,0),(1,1)])}
        with self.assertRaises(ValueError):
            map_beam_chains(nodes, {1:(0,1),2:(1,2),3:(1,3)}, {1,2,3}, {0:0,2:1,3:2})
        nodes.update({4:np.array([4.,1.]),5:np.array([5.,1.]),6:np.array([4.,2.])})
        with self.assertRaisesRegex(ValueError,'Unassigned elements'):
            map_beam_chains(nodes,{1:(0,1),2:(1,2),3:(4,5),4:(5,6),5:(6,4)},
                            {1,2,3,4,5},{0:0,2:1})

    def test_bent_chain_refused(self):
        with self.assertRaises(ValueError):
            map_beam_chains({1:np.array([0.,0.]),2:np.array([1.,.1]),3:np.array([2.,0.])},
                            {9:(1,2),4:(2,3)}, {9,4}, {1:0,3:1})

    def test_duplicate_labels_are_instance_scoped_and_transforms_refused(self):
        text = '*Instance, name={}\n*Node\n1,0,0\n2,1,0\n*Element, type=B21\n1,1,2\n*End Instance\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mesh.inp'
            path.write_text(text.format('A')+text.format('B'))
            self.assertEqual(set(read_beam_inp(path)), {'A','B'})
            path.write_text('*Instance, name=A\n1,0,0\n')
            with self.assertRaises(ValueError):
                read_beam_inp(path)

    def test_status_zero_overrides_near_one_and_unknown_other_element(self):
        raw = np.array([[.99999,.1,np.nan,np.nan]])
        result = reduce_damage(raw, [0,0,1,1], [[0,np.nan]], [0,0], 'max')
        self.assertEqual(result['strut_damage'][0,0], 1)
        self.assertTrue(result['strut_failed'][0,0])
        self.assertEqual(result['strut_coverage'][0,0], .5)
        self.assertEqual(result['failed_element_count'][0,0], 1)

    def test_point_one_does_not_establish_whole_element_failure(self):
        result = reduce_damage([[1.,.2]], [0,0], [[1]], [0], 'max')
        self.assertFalse(result['strut_failed'][0,0])
        self.assertEqual(result['element_sdeg_max'][0,0], 1)

    def test_maximum_not_sum_and_subdivision_invariance(self):
        result = reduce_damage([[.2,.6,.1,.6]], [0,1,2,3], [[1,1,1,1]], [0,0,1,1], 'max')
        np.testing.assert_allclose(result['strut_damage'], [[.6,.6]])
        self.assertFalse(result['strut_failed'].any())

    def test_missing_section_point_and_missing_status_are_unknown(self):
        for raw,status in (([[.3,np.nan]], [[1]]), ([[0.,0.]], [[np.nan]])):
            result = reduce_damage(raw, [0,0], status, [0], 'max')
            self.assertTrue(np.isnan(result['strut_damage'][0,0]))
            self.assertFalse(result['strut_valid'][0,0])

    def test_carry_is_opt_in_and_reactivation_is_rejected(self):
        status = np.array([[1.],[0.],[np.nan]])
        raw = [[0.,0.],[.99999,.99999],[np.nan,np.nan]]
        observed = reduce_damage(raw, [0,0], status, [0], 'max')
        carried = reduce_damage(raw, [0,0], status, [0], 'max', True)
        self.assertTrue(np.isnan(observed['strut_damage'][2,0]))
        self.assertEqual(carried['strut_damage'][2,0],1)
        self.assertTrue(carried['element_failure_carried'][2,0])
        with self.assertRaises(ValueError):
            reduce_damage(raw, [0,0], [[1],[0],[1]], [0], 'max', True)

    def test_failure_bracket_uses_last_observed_active_frame(self):
        result = reduce_damage([[0.],[np.nan],[.99999]], [0], [[1],[np.nan],[0]], [0], 'max')
        self.assertEqual(result['first_failure_frame'][0],2)
        self.assertEqual(result['previous_active_frame'][0],0)

    def test_failure_only_does_not_invent_partial_labels(self):
        result = reduce_damage([[0.],[.2],[.99999]], [0], [[1],[1],[0]], [0], 'failure-only')
        np.testing.assert_allclose(result['strut_damage'][:,0], [0,np.nan,1], equal_nan=True)

    def test_invalid_status_and_damage_fail_shut(self):
        for raw,status in (([[0.]],[[.5]]), ([[1.1]],[[1.]])):
            with self.assertRaises(ValueError):
                reduce_damage(raw,[0],status,[0],'max')


if __name__ == '__main__':
    unittest.main()
