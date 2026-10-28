"""Independent true-field attribution and spatial-region gradient contracts."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest
import numpy as np
import torch
from resources.MLcurve_terms import CurveTermObjective, IndependentCurveTrainer, TERMS, VARIANTS, load_curve_term_model
from resources.MLfield import StructuredFieldLoss


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_requested_terms_and_finite_gradients(self):
        x=np.linspace(0,1,201)
        target=torch.tensor(np.stack([np.sin(np.pi*x),.8*np.sin(np.pi*x)**2]),dtype=torch.float32)
        for variant in VARIANTS:
            pred=(target*.9+.01).requires_grad_()
            loss=CurveTermObjective(variant,0.,1.,x)
            total,raw,weighted=loss.components(pred,target)
            expected=({"range_mse"} if variant=="range_mse" else
                      {"mse",*(TERMS if variant=="combined" else ([variant] if variant!="mse" else []))})
            self.assertEqual(set(weighted),expected)
            if variant!="range_mse":torch.testing.assert_close(raw["mse"],(pred-target).square().mean())
            total.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_spatial_region_does_not_mask_displacement(self):
        loss=StructuredFieldLoss([[0,1],[1,2]],3,n_components=1,
            spatial_node_mask=[True,True,False])
        p=torch.tensor([[[1.],[3.],[20.]]],requires_grad=True);y=torch.zeros_like(p)
        components=loss.component_losses(p,y)
        components["spatial"].backward(retain_graph=True)
        self.assertEqual(p.grad[0,2,0],0.)
        p.grad.zero_();components["displacement"].backward()
        self.assertGreater(p.grad[0,2,0],0.)
        clone=StructuredFieldLoss(**loss.get_config())
        torch.testing.assert_close(clone(p,y),loss(p,y))
        with self.assertRaises(ValueError):
            StructuredFieldLoss([[0,1],[1,2]],3,spatial_node_mask=[True,False,True])

    def test_true_field_only_training_export_and_reload(self):
        helper=runpy.run_path(str(Path(__file__).parents[1]/"Locality/test_locality.py"))
        data=helper["fixture"]()
        for mode in ("UT","FT"):
            net=helper["network"](data)
            stage=net.curve_model.tasks[mode]
            train=dict(batch=2,opt="adamw",lr=.001,scheduler=None,grad_clip=1.)
            trainer=IndependentCurveTrainer(stage,data,mode,"peak_location",train,"cpu")
            self.assertFalse(any("field_model" in k for k in trainer.stage.state_dict()))
            with tempfile.TemporaryDirectory() as td:
                metadata={"run_layout":{"task":mode,"output_kind":"FieldToCurve"}}
                trainer.fit(1,2,0.,td,metadata);trainer.export(td,metadata)
                clone,cfg=load_curve_term_model(td)
                batch=next(iter(trainer.loaders["val"]))
                with torch.no_grad():
                    torch.testing.assert_close(clone(batch["field"],batch["task_features"],batch["node_mask"])[mode],trainer.predict_batch(batch))
                with np.load(Path(td)/"results/predictions.npz") as saved:
                    self.assertEqual(set(saved.files),{"sample_ids",f"{mode}_val_curve_outputs",f"{mode}_val_curve_truth",f"{mode}_curve_x_values"})
                self.assertEqual(cfg["mode"],mode)
                self.assertEqual(json.loads((Path(td)/"results/metrics.json").read_text())["curve_evaluation_source"],"true")


if __name__=="__main__":unittest.main()
