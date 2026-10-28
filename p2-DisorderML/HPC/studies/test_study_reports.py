"""Protect mean baselines, percent-error masking and bounded download verification."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from resources.MLstudy_reports import percent_error,verify_download,measure_curve_regions


class StudyReportTests(unittest.TestCase):
    def test_percent_error_masks_zero_tiny_and_invalid_truth_without_changing_input(self):
        truth=np.array([0.,.001,-2.,4.,np.nan]);pred=np.array([9.,1.,-1.,5.,8.])
        result=percent_error(pred,truth,np.ones(5,bool),1.)
        self.assertTrue(np.isnan(result[[0,1,4]]).all())
        np.testing.assert_allclose(result[[2,3]],[50.,25.])
        np.testing.assert_array_equal(pred,[9.,1.,-1.,5.,8.])

    def test_percent_error_keeps_positive_negative_truth_symmetric(self):
        a=percent_error(np.array([2.,-2.]),np.array([1.,-1.]),[True,True],1.)
        np.testing.assert_array_equal(a,[100.,100.])

    def test_curve_regions_use_true_peak_and_fixed_mean_reference(self):
        x=np.arange(5.)
        truth=np.array([[0.,1.,3.,1.,0.],[0.,4.,2.,1.,0.]])
        rows=measure_curve_regions(truth,truth,x,np.ones(5))
        peak=next(r for r in rows if r["region"].startswith("Peak"))
        self.assertEqual(peak["valid_points"],2)
        self.assertEqual(peak["rmse"],0)
        self.assertEqual(peak["skill_percent"],100)
        shifted=measure_curve_regions(np.flip(truth,axis=1),truth,x,np.ones(5))
        self.assertEqual(next(r for r in shifted if r["region"].startswith("Peak"))["valid_points"],2)
        with self.assertRaises(ValueError):measure_curve_regions(np.full_like(truth,np.inf),truth,x,np.ones(5))

    def test_download_receipt_rejects_wrong_hash_count_and_escaping_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"run").mkdir();payload=root/"run/file";payload.write_bytes(b"verified")
            sha=hashlib.sha256(payload.read_bytes()).hexdigest();manifest=root/"manifest"
            manifest.write_text(sha+"  run/file\n")
            self.assertEqual(verify_download(root,manifest,1)["files"],1)
            with self.assertRaises(ValueError):verify_download(root,manifest,2)
            manifest.write_text("0"*64+"  run/file\n")
            with self.assertRaises(ValueError):verify_download(root,manifest,1)
            manifest.write_text(sha+"  ../file\n")
            with self.assertRaises(ValueError):verify_download(root,manifest,1)


if __name__=="__main__":unittest.main()
