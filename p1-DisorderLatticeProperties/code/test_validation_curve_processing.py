"""Meaningful mechanics/missingness invariants for the opt-in curve pilot."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation_curve_processing import (initial_fit, paired_filter, transfer_prefix,
                                         query_support_mask, event_segmented_filter, threshold_details,
                                         periodic_causal_filter, paired_gallery_selection,
                                         sampled_odb_fingerprint)


class CurveQualification(unittest.TestCase):
    def test_random_gallery_is_reproducible_balanced_and_original_id_scoped(self):
        candidates={topology:{design:list(range(1,1006)) for design in ('full','mechanism')}
                    for topology in ('tri','kagome','hex','FCC')}
        first=paired_gallery_selection(candidates,42)
        self.assertEqual(first,paired_gallery_selection(candidates,42))
        self.assertNotEqual(first,paired_gallery_selection(candidates,43))
        self.assertEqual(len(first),28)
        for topology in candidates:
            rows=[r for r in first if r['topology']==topology]
            self.assertEqual(sum(r['design']=='full' for r in rows),4)
            self.assertEqual(sum(r['design']=='mechanism' for r in rows),3)
        self.assertTrue(all(1<=r['id']<=1000 for r in first))

    def test_sampled_odb_identity_is_not_misrepresented_as_full_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.odb'
            path.write_bytes(b'a'*200000)
            before=sampled_odb_fingerprint(path)
            self.assertFalse(before['full_file_hash'])
            path.write_bytes(b'b'*200000)
            self.assertNotEqual(before,sampled_odb_fingerprint(path))

    def test_physical_filter_is_causal_and_does_not_leak_future_fracture(self):
        time=np.arange(100)*.001
        force=np.r_[np.ones(50)*10,np.zeros(50)]
        (_, filtered),info=periodic_causal_filter(time,time,force,50.)
        np.testing.assert_allclose(filtered[:50],10.,atol=1e-12)
        self.assertFalse(info['zero_phase'])
        self.assertEqual(info['cutoff_Hz'],100.)
        self.assertEqual(info['startup'],'DC')

    def test_physical_cutoff_cannot_be_silently_capped_to_saved_nyquist(self):
        time=np.arange(100)*.001
        with self.assertRaises(ValueError):
            periodic_causal_filter(time,time,time,300.)

    def test_physical_filter_keeps_paired_affine_work_coordinates(self):
        time=np.arange(100)*.001
        (x,y),_=periodic_causal_filter(time,time,3*time+2,50.)
        np.testing.assert_allclose(y,3*x+2,atol=1e-12)

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

    def test_query_mask_uses_physical_support_not_sample_count(self):
        observations = np.linspace(0, .10375, 126)
        query = np.linspace(0, .1, 201)
        self.assertTrue(query_support_mask(observations, query).all())
        self.assertEqual(query_support_mask(np.linspace(0, .08, 126), query).sum(), 161)

    def test_segment_filter_does_not_mix_a_resolved_force_jump(self):
        x = np.linspace(0, 1, 80)
        y = np.r_[np.ones(40)*10, np.ones(40)*2]
        (xx, yy), support = event_segmented_filter(x, y, 40, .4)
        np.testing.assert_allclose(yy, y, atol=1e-12)
        self.assertFalse(support['endpoint_pinning'])

    def test_short_event_segment_is_explicitly_unresolved(self):
        x = np.arange(40, dtype=float)
        _, support = event_segmented_filter(x, x, 8, .4)
        self.assertEqual(support['status'], 'short_side_kept_raw_unresolved')

    def test_threshold_recorded_sample_is_not_replaced_by_interpolation(self):
        result = threshold_details([0.,1.,2.,3.,4.],[0.,8.,6.,3.,1.])
        self.assertEqual(result['threshold_MPa'],2.)
        self.assertEqual(result['recorded_ductility'],4.)
        self.assertEqual(result['interpolated_crossing_for_display'],3.5)
        censored = threshold_details([0.,1.,2.],[0.,8.,6.])
        self.assertIsNone(censored['recorded_ductility'])


if __name__ == "__main__":
    unittest.main()
