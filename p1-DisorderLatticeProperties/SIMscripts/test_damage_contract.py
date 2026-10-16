"""Synthetic scientific-contract regressions; real ODB checks are a separate pass."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from resources.damage import fcc_map, map_beam_chains, periodic_fcc_map, read_beam_inp, reduce_damage


def write_fcc_fixture(path, task="UT", shifted=True, mutation=None):
    """Independent FCC construction with shuffled labels and 1/3/8 FE per strut."""
    xy = [(10*x, 10*y) for y in range(20) for x in range(21)]
    xy += [(10*x+5, 10*y+5) for y in range(19) for x in range(20)]
    edges = []
    for y in range(20):
        edges.extend([((10*x, 10*y), (10*x+10, 10*y)) for x in range(20)])
    for x in range(21):
        edges.extend([((10*x, 10*y), (10*x, 10*y+10)) for y in range(19)])
    for y in range(19):
        for x in range(20):
            edges.extend([((10*x+5, 10*y+5), (10*x+dx, 10*y+dy))
                          for dx, dy in [(0, 0), (10, 0), (0, 10), (10, 10)]])
    if task == "FT":
        xy = [p for p in xy if not (p[1] == 95 and p[0] < 120)]
        edges = [(a, b) for a, b in edges if a in xy and b in xy and not
                 ((a[0]+b[0])/2 < 118 and (a[1]+b[1])/2 == 95)]
    if mutation == "topology":
        edges[21] = ((5, 5), (15, 5))  # unsupported center-to-center strut
    rng = np.random.RandomState(729)
    labels = dict(zip(xy, 101+17*rng.permutation(len(xy))))
    coordinates = {}
    for point, label in labels.items():
        boundary = point[0] in {0, 200} or point[1] in {0, 190}
        shift = rng.uniform(-1.4, 1.4, 2) if shifted and not boundary else np.zeros(2)
        if point == (10, 10) and mutation == "bound":
            shift = np.array([1.5, 0.])
        if point == (0, 0) and mutation == "boundary":
            shift = np.array([.1, 0.])
        coordinates[int(label)] = np.array(point, float)+shift
    elements, next_node = {}, 20000
    for edge_index, (a, b) in enumerate(edges):
        count = (1, 3, 8)[edge_index % 3]
        path_nodes = [int(labels[a])]
        for segment in range(1, count):
            coordinates[next_node] = (coordinates[int(labels[a])] +
                                      segment/count*(coordinates[int(labels[b])]-coordinates[int(labels[a])]))
            path_nodes.append(next_node)
            next_node += 19
        path_nodes.append(int(labels[b]))
        for segment in range(count):
            label = 17*(len(elements)+1)+7
            pair = (path_nodes[segment], path_nodes[segment+1])
            elements[label] = pair if segment % 2 else pair[::-1]
    body = list(elements)
    if task == "UT":
        coordinates[900000], coordinates[900001] = np.array([-5., -5.]), np.array([-5., -10.])
        elements[800000] = (900000, 900001)
    lines = ["*Instance, name=Part-1-1", "*Node"]
    lines += [f"{n},{p[0]:.12f},{p[1]:.12f}" for n, p in coordinates.items()]
    lines += ["*Element, type=B21"]
    lines += [f"{e},{a},{b}" for e, (a, b) in reversed(list(elements.items()))]
    lines += ["*Elset, elset=AllPartSet", ",".join(map(str, elements))]
    if task == "UT":
        lines += ["*Elset, elset=BodySet", ",".join(map(str, body)),
                  "*Elset, elset=BracketSet", "800000"]
    lines += ["*End Instance"]
    path.write_text("\n".join(lines))


class DamageContract(unittest.TestCase):
    def test_disordered_fcc_uses_topological_joints_despite_nearby_mesh_nodes(self):
        for task, token, count, joints in (("UT", "Ductile", 2319, 800), ("FT", "Fracture", 2259, 788)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/f"{token}-FCC-20-20disNodes-lhs-all-747-maxB.inp"
                write_fcc_fixture(path, task)
                arrays, mesh, metadata = fcc_map(path, task)
                self.assertEqual(len(arrays["strut_id"]), count)
                self.assertEqual(metadata["joints"], joints)
                self.assertEqual(metadata["simulation_id"], 747)
                self.assertEqual(set(arrays["subdivision_count"]), {1, 3, 8})
                self.assertEqual(len(set(arrays["element_label"])), len(arrays["element_label"]))
                self.assertEqual(metadata["excluded_elements"], int(task == "UT"))
                self.assertTrue(metadata["complete_initial_topology_match"])
                self.assertLess(metadata["maximum_joint_component_displacement_mm"], np.sqrt(2))
                # Numerous FE interior points lie close to reference joints. They
                # cannot become extra original joints through coordinate proximity.
                near = sum(((abs(p-arrays["canonical_xy"]) <= np.sqrt(2)).all(axis=1)).any()
                           for p in mesh["nodes"].values())
                self.assertGreater(near, joints)

    def test_periodic_wrapper_remains_strict_and_endpoint_order_is_shared(self):
        with tempfile.TemporaryDirectory() as folder:
            periodic = Path(folder)/"Ductile-FCC-20-per-1.inp"
            disordered = Path(folder)/"Ductile-FCC-20-20disNodes-lhs-all-397-maxD.inp"
            write_fcc_fixture(periodic, shifted=False)
            write_fcc_fixture(disordered)
            a, _, metadata = fcc_map(periodic, "UT")
            b, _, _ = fcc_map(disordered, "UT")
            np.testing.assert_array_equal(a["canonical_endpoint"], b["canonical_endpoint"])
            self.assertEqual(metadata["simulation_id"], 0)
            with self.assertRaisesRegex(ValueError, "unique bounded"):
                periodic_fcc_map(disordered, "UT")

    def test_fcc_rejects_bound_boundary_and_wrong_initial_topology(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"Ductile-FCC-20-20disNodes-lhs-all-747-maxB.inp"
            for mutation, error in (("bound", "unique bounded"), ("boundary", "boundary joint"),
                                    ("topology", "topology mismatch")):
                write_fcc_fixture(path, mutation=mutation)
                with self.assertRaisesRegex(ValueError, error):
                    fcc_map(path, "UT")

    def test_fcc_rejects_unsupported_profile_and_wrong_task(self):
        with self.assertRaisesRegex(ValueError, "Unsupported FCC source stem"):
            fcc_map("Ductile-FCC-20-40disNodes-lhs-all-747.inp", "UT")
        with self.assertRaisesRegex(ValueError, "Task/stem mismatch"):
            fcc_map("Ductile-FCC-20-20disNodes-lhs-all-747-maxB.inp", "FT")

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
