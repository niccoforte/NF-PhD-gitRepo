"""Isolated field changes: time-only weights, train-only transforms and reload."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from resources.MLdual import DUAL_MODEL, DualLoss
from resources.MLfield import StructuredFieldLoss
from resources.MLfield_variations import gaussian_frame_weights, residual_field_coordinates, configure_field_variation

BASE=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Locality/test_locality.py"))
RUNNER=runpy.run_path(str(Path(__file__).with_name("A0-FieldVariations.py")))


class FieldVariationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_thirteen_cases_and_ft_only_priors(self):
        cases=RUNNER["CASES"]
        self.assertEqual(len(cases),13);self.assertEqual(len(set(cases)),13)
        self.assertTrue(all(m=="FT" for v,m in cases if v in ("crack_face","box2","box4")))
        self.assertEqual(sum(v.startswith("gaussian") for v,m in cases),8)

    def test_gaussian_is_temporal_symmetric_positive_with_mean_one_valid_weights(self):
        for a in (1,3):
            for width in (.15,.25):
                weights=gaussian_frame_weights(20,a,width)
                self.assertEqual(weights.shape,(1,20,1));self.assertTrue((weights>=1).all())
                np.testing.assert_allclose(weights,weights[:,::-1])
                self.assertIn(int(weights.argmax()),(9,10))
                loss=StructuredFieldLoss([[0,1]],2,fixed_weights=weights,spatial_weight=0,temporal_weight=0)
                target=torch.zeros(2,2,40);mask=torch.ones_like(target,dtype=torch.bool);mask[0,0,:4]=False
                w=loss.localization_weights(target,mask).expand(2,2,20,2)
                valid=mask.reshape_as(w)
                for sample in range(2):self.assertAlmostEqual(float(w[sample][valid[sample]].mean()),1.,places=6)
                # No neighbor/derivative term is active; only weighted nodal error.
                pred=torch.ones_like(target)
                self.assertAlmostEqual(float(loss(pred,target,mask)),1.,places=6)

    def test_residual_fit_ignores_val_test_preserves_physical_targets_and_other_mode(self):
        data=BASE["fixture"]();other=copy.deepcopy(data.normalizers["field"]["UT"])
        physical={s:data.inverse_field("FT",v["field"]["FT"]).copy() for s,v in data.splits.items()}
        second=BASE["fixture"]();second.splits["val"]["field"]["FT"][:]=1e6;second.splits["test"]["field"]["FT"][:]=-1e6
        residual_field_coordinates(data,"FT");residual_field_coordinates(second,"FT")
        for key in ("mean","scale"):
            np.testing.assert_array_equal(data.normalizers["field"]["FT"][key],second.normalizers["field"]["FT"][key])
            np.testing.assert_array_equal(data.normalizers["field"]["UT"][key],other[key])
        for s,v in data.splits.items():
            keep=v["field_mask"]["FT"]
            np.testing.assert_allclose(data.inverse_field("FT",v["field"]["FT"])[keep],physical[s][keep],rtol=1e-5,atol=1e-6)
            np.testing.assert_array_equal(data.datasets[s].fields["FT"].numpy(),v["field"]["FT"])

    def test_crack_face_starts_at_same_function_and_does_not_train_ut(self):
        data=BASE["fixture"]();net=BASE["network"](data);old=copy.deepcopy(net.state_dict())
        feature={m:np.column_stack([data.task_features[m],np.zeros(6) if m=="UT" else [0,0,1,0,1,0]]) for m in ("UT","FT")}
        with patch("resources.MLdual.dual_node_context",return_value=(feature,data.task_feature_names+["initial_crack_face"],{})):
            configure_field_variation(net,data,SimpleNamespace(variant="crack_face",mode="FT"))
        for key,value in old.items():
            current=net.state_dict()[key]
            if "field_model" in key and "context_projection" in key:current=current[:,:-1]
            torch.testing.assert_close(current,value,rtol=0,atol=0)
        self.assertTrue(torch.all(net.field_model.tasks["FT"].context_projection.weight[:,-1]==0))

    def test_real_runner_exports_weighted_and_residual_fits_with_reload(self):
        runner=runpy.run_path(str(Path(__file__).resolve().parents[1]/"Locality/A0-Locality.py"))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=BASE["fixture"]()
            source=DUAL_MODEL(BASE["network"](data),DualLoss(),data=data,device="cpu",batch=2)
            source.save(root/"control",metadata=dict(curve_training_source="mixed",run_config=dict(early_stop_patience=2,early_stop_delta=0.)))
            for variant in ("residual","gaussian_a1_w0.15","box2","crack_face"):
                data=BASE["fixture"]()
                args=RUNNER["parse_args"](["--reference-model-json",str(root/"control/model.json"),"--run-label",variant,
                    "--run-root",str(root/"runs"),"--epochs","1","--nsims","64","--mode","FT","--variant",variant,"--allow-cpu"])
                feature={m:np.column_stack([data.task_features[m],np.zeros(6)]) for m in ("UT","FT")}
                with patch.dict(runner["run"].__globals__,{"load_data":lambda args:data}), \
                     patch("resources.MLfield.reference_field_edges",side_effect=BASE["edges"]), \
                     patch("resources.MLdual.dual_node_context",return_value=(feature,data.task_feature_names+["initial_crack_face"],{})), \
                     patch("resources.MLfield.fcc_initial_crack_region",return_value=(np.array([0,0,0,1,1,1],bool),np.array([[0,0],[2,1]]))):
                    model=runner["run"](args,configure=configure_field_variation,study="field_variations")
                saved=Path(model.save_dir)
                self.assertEqual(json.loads((saved/"run_metadata.json").read_text())["checkpoint_reload"],"passed")
                self.assertTrue((saved/"results/FT_val_field_motion.csv").is_file())
                with np.load(saved/"results/predictions.npz") as arrays:self.assertFalse(any("curve" in k for k in arrays.files))


if __name__=="__main__":unittest.main()
