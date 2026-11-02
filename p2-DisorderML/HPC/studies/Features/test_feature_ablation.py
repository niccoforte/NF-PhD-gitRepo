"""Context attribution preserves refreshed pins, physical masks and reload."""
import copy
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from resources.MLdual import DUAL_MODEL,DualLoss,DualStageTransformer
from resources.MLfeature_ablation import CASES,GROUPS,configure_context_ablation

BASE=runpy.run_path(str(Path(__file__).parents[1]/"Locality/test_locality.py"))
NAMES=["x0","y0","designable","present","bottom_interface","top_interface",
    "coupled_fixity","coupled_load","tip_dx_ref","tip_dy_ref","tip_distance_ref"]


def fixture():
    data=BASE["fixture"]();data.context_feature_size=11;data.task_feature_names=NAMES.copy()
    data.task_features={m:np.ones((6,11),np.float32) for m in ("UT","FT")}
    for dataset in data.datasets.values():
        dataset.task_features={m:torch.from_numpy(v) for m,v in data.task_features.items()}
        dataset.context_spec=dict(pin_feature_indices=[6,7]);dataset.pin_masks=torch.ones(len(dataset),6,2,dtype=torch.bool)
    return data


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_task_cases_and_runtime_masks_preserve_initialization(self):
        self.assertEqual(len(CASES),15);self.assertEqual(len(set(CASES)),15)
        for mode,variant in CASES:
            data=fixture();net=BASE["network"](data);old=copy.deepcopy(net.state_dict())
            masks=copy.deepcopy(data.node_masks);normalizers=copy.deepcopy(data.normalizers)
            with patch("resources.MLfield.reference_field_edges",side_effect=BASE["edges"]):
                configure_context_ablation(net,data,SimpleNamespace(mode=mode,variant=variant))
            for name,value in old.items():torch.testing.assert_close(value,net.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(data.context_feature_size,11)
            for m in ("UT","FT"):np.testing.assert_array_equal(data.node_masks[m],masks[m])
            self.assertEqual(str(data.normalizers),str(normalizers))
            net.eval();stage=net.field_model.tasks[mode];batch=next(iter(data.make_dataloaders(2)["val"]))
            if variant!="control":
                removed=NAMES if variant=="no_context" else GROUPS[variant]
                altered={m:v.clone() for m,v in batch["task_features"].items()}
                for name in removed:altered[mode][:,:,NAMES.index(name)]=1234.
                with torch.no_grad():
                    a=stage(batch["geometry"],batch["task_features"],batch["node_mask"])[mode]
                    b=stage(batch["geometry"],altered,batch["node_mask"])[mode]
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                if mode=="FT":self.assertTrue(torch.all(a[:,1]==0))
            clone=DualStageTransformer.from_config(net.get_config());clone.load_state_dict(net.state_dict(),strict=True);clone.eval()
            with torch.no_grad():
                a=net.field_model(batch["geometry"],batch["task_features"],batch["node_mask"])[mode]
                b=clone.field_model(batch["geometry"],batch["task_features"],batch["node_mask"])[mode]
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_refreshed_pin_channels_have_zero_gradient_and_live_channels_do_not(self):
        data=fixture();net=BASE["network"](data)
        with patch("resources.MLfield.reference_field_edges",side_effect=BASE["edges"]):
            configure_context_ablation(net,data,SimpleNamespace(mode="FT",variant="no_pins"))
        batch=next(iter(data.make_dataloaders(2)["val"]))
        self.assertTrue((batch["task_features"]["FT"][:,:,6:8]==1).all())
        batch["task_features"]["FT"].requires_grad_()
        net.field_model.tasks["FT"](batch["geometry"],batch["task_features"],batch["node_mask"])["FT"].square().mean().backward()
        grad=batch["task_features"]["FT"].grad
        self.assertEqual(float(grad[:,:,6:8].abs().sum()),0.)
        self.assertGreater(float(grad[:,:,:2].abs().sum()),0.)

    def test_historical_config_and_invalid_masks(self):
        data=fixture();net=BASE["network"](data);stage=net.field_model.tasks["UT"]
        self.assertNotIn("context_keep",stage.get_config())
        for keep in ({"UT":[True]*11},{"UT":[1]*11,"FT":[True]*11},{m:[True]*10 for m in ("UT","FT")}):
            with self.assertRaises(ValueError):stage.set_context_keep(keep)

    def test_field_trainer_checkpoint_reload_keeps_task_specific_pin_suppression(self):
        from resources.MLmetrics import postprocess_load_dual_run
        data=fixture();net=BASE["network"](data)
        with patch("resources.MLfield.reference_field_edges",side_effect=BASE["edges"]):
            configure_context_ablation(net,data,SimpleNamespace(mode="FT",variant="no_pins"))
        model=DUAL_MODEL(net,DualLoss(),data=data,device="cpu",batch=2,field_only_modes=("FT",))
        model.train(1,verbose=0)
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder);model.save_results(eval_split="val")
            clone=postprocess_load_dual_run(folder,load_model=True)[-1]
            self.assertEqual(clone.model.field_model.tasks["FT"].context_keep["FT"][6:8],[False,False])
            self.assertTrue(all(clone.model.field_model.tasks["UT"].context_keep["UT"]))


if __name__=="__main__":unittest.main()
