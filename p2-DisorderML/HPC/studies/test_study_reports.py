"""Protect mean baselines, percent-error masking and bounded download verification."""
import hashlib
from pathlib import Path
import tempfile
import runpy
import unittest
import numpy as np
from resources.MLstudy_reports import percent_error,verify_download,measure_curve_regions
from resources.MLstudy_reports import write_study_report,replace_study_followup
from resources.MLstudy_reports import verify_study_inventory


class StudyReportTests(unittest.TestCase):
    def test_base_variation_refresh_preserves_completed_nested_diagnostics(self):
        from unittest.mock import patch
        from resources.MLstudy_reports import insert_variation_section
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"README.md"
            write_study_report(path,"# Study\n\n## Global TRAIN mean\n\nReference.\n\n## Local errors\n\nMeasured.\n")
            replace_study_followup(path,"complete","## Completed follow-up\n\n### Prediction variation and collapse\n\nPreserve follow-up diagnostic.\n\n### Cost\n\nPreserve cost.")
            with patch("resources.MLstudy_reports.prediction_variation_section",return_value="## Prediction variation and collapse\n\nRefreshed original diagnostic.\n\n"):
                insert_variation_section(path,[])
                insert_variation_section(path,[])
            result=path.read_text()
            self.assertIn("Preserve follow-up diagnostic.",result)
            self.assertIn("Preserve cost.",result)
            self.assertEqual(result.count("Refreshed original diagnostic."),1)

    def test_curve_changes_match_task_seed_and_region(self):
        import pandas as pd
        fn=runpy.run_path(str(Path(__file__).parent/"Loss/review_curve_terms.py"))["matched_changes"]
        table=pd.DataFrame([dict(mode="UT",seed=42,variant="mse",region="peak",rmse=2.),
            dict(mode="UT",seed=43,variant="mse",region="peak",rmse=10.),
            dict(mode="UT",seed=43,variant="work",region="peak",rmse=5.),
            dict(mode="UT",seed=42,variant="work",region="peak",rmse=3.)])
        np.testing.assert_allclose(fn(table,["mode","seed","region"]),[0,0,-50,50])
        with self.assertRaises(ValueError):fn(pd.concat([table,table.iloc[:1]]),["mode","seed","region"])

    def test_multilayout_inventory_checks_metadata_hashes_and_containment(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/"UT/FieldToCurve/Transformer/run";run.mkdir(parents=True)
            meta=dict(status="complete",checkpoint_reload="passed");payload=run/"run_metadata.json"
            payload.write_text(json.dumps(meta));manifest=root/"inventory.json"
            row=dict(path=payload.relative_to(root).as_posix(),size=payload.stat().st_size,sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
            inventory=dict(runs=[dict(path=run.relative_to(root).as_posix(),metadata=meta,files=[row])])
            manifest.write_text(json.dumps(inventory));self.assertEqual(verify_study_inventory(root,manifest,1)["files"],1)
            payload.write_text("changed")
            with self.assertRaises(ValueError):verify_study_inventory(root,manifest,1)
            payload.write_text(json.dumps(meta));row["path"]="../outside";manifest.write_text(json.dumps(inventory))
            with self.assertRaises(ValueError):verify_study_inventory(root,manifest,1)

    def test_completed_followup_survives_base_rebuild_and_replaces_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"README.md";write_study_report(path,"# Original\n\nInitial evidence.\n")
            replace_study_followup(path,"cnn","## Completed\n\nFirst verdict.")
            replace_study_followup(path,"cnn","## Completed\n\nNew verdict.")
            write_study_report(path,"# Rebuilt\n\nRefreshed initial evidence.\n")
            text=path.read_text();self.assertNotIn("First verdict",text)
            self.assertEqual(text.count("New verdict"),1)
            self.assertLess(text.index("New verdict"),text.index("Refreshed initial evidence"))

    def test_paired_bootstrap_preserves_exact_matched_effect(self):
        fn=runpy.run_path(str(Path(__file__).parent/"Loss/review_spatial_balance.py"))["paired_interval"]
        b=np.array([[1.,2.,3.],[10.,20.,30.]])
        result=fn(b,b*.25,rmse=True,repeats=200)
        for key in ("change","ci_low","ci_high"):self.assertAlmostEqual(result[key],-50.)
        result=fn(b,b+.5,rmse=False,repeats=200)
        for key in ("change","ci_low","ci_high"):self.assertAlmostEqual(result[key],.5)
        with self.assertRaises(ValueError):fn(b,b[:1])

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
