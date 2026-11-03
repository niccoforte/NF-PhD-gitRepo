"""Saved feature masks are checked by task and group without loading Torch."""
import copy
from pathlib import Path
import runpy
import unittest

API=runpy.run_path(str(Path(__file__).with_name("review_feature_ablation.py")))
NAMES=["x0","y0","designable","present","bottom_interface","top_interface",
    "coupled_fixity","coupled_load","tip_dx_ref","tip_dy_ref","tip_distance_ref"]


class FeatureReviewTests(unittest.TestCase):
    def test_expected_ft_pin_mask_does_not_suppress_ut_or_operational_presence(self):
        keep={m:[True]*11 for m in ("UT","FT")};keep["FT"][6:8]=[False,False]
        cfg={"model_config":{"field_model":{"independent_tasks":True,"locality":{"kind":"cnn","blocks":2},"context_keep":keep},"curve_model":{}}}
        self.assertEqual(API["verify_feature_mask"](cfg,NAMES,"FT","no_pins"),["coupled_fixity","coupled_load"])
        altered=copy.deepcopy(cfg);altered["model_config"]["field_model"]["context_keep"]["UT"][6]=False
        with self.assertRaises(ValueError):API["verify_feature_mask"](altered,NAMES,"FT","no_pins")
        with self.assertRaises(ValueError):API["verify_feature_mask"](cfg,NAMES,"FT","control")

    def test_control_and_wrong_architecture(self):
        cfg={"model_config":{"field_model":{"independent_tasks":True,"locality":{"kind":"cnn","blocks":2}},"curve_model":{}}}
        self.assertEqual(API["verify_feature_mask"](cfg,NAMES,"UT","control"),[])
        cfg["model_config"]["field_model"]["locality"]["blocks"]=3
        with self.assertRaises(ValueError):API["verify_feature_mask"](cfg,NAMES,"UT","control")

    def test_cnn_followup_entries_reuse_only_completed_missing_seed_sources(self):
        api=runpy.run_path(str(Path(__file__).parents[1]/"Locality/review_cnn_spatial.py"))
        rows=api["entries"](Path("runs"));self.assertEqual(len(rows),18)
        self.assertEqual(len({(r["mode"],r["variant"],r["seed"]) for r in rows}),18)
        for r in rows:
            name=r["path"].name
            if r["variant"]=="cnn3" and r["seed"]==42:self.assertTrue(name.startswith("cnn-confirmation-261028"))
            if r["variant"]=="cnn2_spatial":self.assertTrue(name.startswith("cnn-spatial-depth-261102"))


if __name__=="__main__":unittest.main()
