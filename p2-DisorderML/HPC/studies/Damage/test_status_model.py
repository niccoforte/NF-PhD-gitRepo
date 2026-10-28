"""STATUS contract, gradients, endpoint invariance, balance and checkpoint tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from resources.MLdamage import BalancedStatusLoss, StatusFieldNetwork, StatusTrainer, align_status_bank, class_balance, load_status_model, status_metrics
from resources.MLdual import DualStageTransformer

spec=importlib.util.spec_from_file_location("locality_fixture",Path(__file__).parents[1]/"Locality/test_locality.py")
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)


def network(data, variant="weak", mode="UT", edges=None):
    base=DualStageTransformer.from_data(data,field_kwargs=dict(d_model=8,n_heads=2,n_layers=1,dropout=0.,independent_tasks=True),
        curve_kwargs=dict(d_model=8,n_heads=2,n_layers=1,dropout=0.,independent_tasks=True))
    edges=np.array([[0,1],[1,2],[3,4]]) if edges is None else edges
    return StatusFieldNetwork(base.field_model.tasks[mode],mode,edges,data.metadata["canonical_coords"],data.normalizers["geometry"],variant)


class StatusModelTests(unittest.TestCase):
    def test_explicit_auxiliary_weight_keeps_field_term_and_is_saved(self):
        data=fixture_module.fixture();rng=np.random.default_rng(7)
        status={s:dict(target=rng.integers(0,2,(len(data.datasets[s]),3,3),dtype=np.uint8),mask=np.ones((len(data.datasets[s]),3,3),bool)) for s in ("train","val")}
        training=dict(batch=2,lr=.001,opt=["adamw",0.],scheduler=["plateau","min",.5,2,1e-4],grad_clip=None)
        trainer=StatusTrainer(network(data),data,status,training,"UT","weak","cpu",damage_coefficient=.001)
        batch=next(iter(trainer.loaders["train"]))
        total,field,damage=trainer.objective(trainer.network(batch),batch)
        torch.testing.assert_close(total,field+.001*damage)
        total.backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in trainer.network.stage.heads.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            trainer.save(folder,{})
            self.assertEqual(json.loads((Path(folder)/"model.json").read_text())["damage_coefficient"],.001)
        with self.assertRaises(ValueError):
            StatusTrainer(network(data,"damage"),data,status,training,"UT","damage","cpu",damage_coefficient=.001)

    def test_alignment_reorders_ids_and_preserves_unknown_frames(self):
        data=fixture_module.fixture();mode="UT"
        data.metadata["field_to_canonical"]={mode:np.arange(6)}
        data.metadata["field_frame_values"][mode]=[.1,.2,.3]
        ids=data.sample_ids["train"]+data.sample_ids["val"]
        xy=np.asarray(data.metadata["canonical_coords"])
        stats=data.normalizers["geometry"]
        raw=np.concatenate([xy+data.splits[s]["geometry"]*stats["scale"]+stats["mean"] for s in ("train","val")])
        order=np.arange(len(ids))[::-1]
        target=np.zeros((len(ids),4,3),np.uint8);target[0,2:,1]=1
        mask=np.ones_like(target,bool);mask[1,-1]=False
        arrays=dict(schema_version=np.array("strut_status_bank_v1"),task=np.array(mode),mech_mode=np.array("MULTI"),
            sample_ids=np.asarray(ids)[order],node_coords=xy,edge_index=np.array([[0,1],[1,2],[3,4]]),
            frame_values=np.array([0,.1,.2,.3]),frame_index=np.arange(4),D=target[order],valid_mask=mask[order],
            sample_node_coords=raw[order],strut_id=np.arange(3),metadata_json=np.array("{}"))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bank.npz";np.savez(path,**arrays)
            result,edges,_=align_status_bank(path,data,mode)
            np.testing.assert_array_equal(result["train"]["target"][0,1],[0,1,1])
            self.assertFalse(result["train"]["mask"][1,:,-1].any())
            arrays["sample_node_coords"]=raw[order]+1;np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,"initial geometry"):align_status_bank(path,data,mode)

    def test_training_balance_and_unknown_mask(self):
        y=np.array([1]+[0]*99,dtype=np.uint8);mask=np.ones(100,bool)
        balance=class_balance(y,mask);self.assertEqual(balance["pos_weight"],99.)
        logits=torch.zeros(100,requires_grad=True)
        loss=BalancedStatusLoss(balance)(logits,torch.tensor(y),torch.tensor(mask))
        self.assertAlmostEqual(loss.item(),np.log(2),places=5)
        loss.backward();self.assertAlmostEqual(abs(logits.grad[0].item()),float(logits.grad[1:].sum()),places=5)
        mask[0]=False
        with self.assertRaises(ValueError):class_balance(y,mask)

    def test_endpoint_reversal_and_no_true_damage_input(self):
        data=fixture_module.fixture();net=network(data);net.eval()
        batch=next(iter(torch.utils.data.DataLoader(data.val_dataset,batch_size=2)))
        first=net(batch)
        net.edges[:]=net.edges.flip(-1)
        torch.testing.assert_close(first["damage_logits"],net(batch)["damage_logits"])
        batch["damage"]=torch.ones(2,3,3)
        torch.testing.assert_close(first["field"],net(batch)["field"])
        net(batch)["damage_logits"].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.stage.encoder.parameters()))

    def test_old_stage_outputs_and_state_names_remain_identical(self):
        data=fixture_module.fixture();net=network(data).eval();stage=net.stage
        batch=next(iter(torch.utils.data.DataLoader(data.val_dataset,batch_size=2)))
        before=set(stage.state_dict())
        a=stage(batch["geometry"],batch["task_features"],batch["node_mask"])
        b,latent=stage(batch["geometry"],batch["task_features"],batch["node_mask"],return_node_features=True)
        torch.testing.assert_close(a["UT"],b["UT"]);self.assertEqual(before,set(stage.state_dict()))
        self.assertEqual(latent["UT"].shape,(2,6,8))

    def test_all_intact_predictions_are_exposed(self):
        target=np.zeros((2,3,4),np.uint8);target[0,1,2:]=1
        m=status_metrics(np.full(target.shape,.01),target,np.ones_like(target,bool))
        self.assertEqual(m["recall"],0.);self.assertEqual(m["missed_fractures"],2)
        self.assertEqual(m["missed_failed_struts"],1);self.assertEqual(m["predicted_positive_fraction"],0.)

    def test_all_four_arms_train_export_reload(self):
        data=fixture_module.fixture();rng=np.random.default_rng(1)
        status={s:dict(target=rng.integers(0,2,(len(data.datasets[s]),3,3),dtype=np.uint8),mask=np.ones((len(data.datasets[s]),3,3),bool)) for s in ("train","val")}
        training=dict(batch=2,lr=.001,opt=["adamw",0.],scheduler=["plateau","min",.5,2,1e-4],grad_clip=None)
        with tempfile.TemporaryDirectory() as folder:
            for variant in ("control","damage","weak","strong"):
                trainer=StatusTrainer(network(data,variant),data,status,training,"UT",variant,"cpu")
                path=Path(folder)/variant;evaluation=trainer.fit(1,2,1e-5,path,{})
                loaded,cfg=load_status_model(path)
                batch=next(iter(trainer.loaders["val"]));trainer.network.eval()
                torch.testing.assert_close(loaded(batch)["field"],trainer.network(batch)["field"])
                self.assertEqual(cfg["damage_coefficient"],dict(control=0.,damage=1.,weak=.1,strong=1.)[variant])
                runner_spec=importlib.util.spec_from_file_location("status_runner",Path(__file__).with_name("A0-Damage.py"))
                runner=importlib.util.module_from_spec(runner_spec);runner_spec.loader.exec_module(runner)
                with patch("resources.MLmetrics.save_field_motion_diagnostics"):
                    runner.export_results(trainer,evaluation,path,{},dict(schema="synthetic"))
                self.assertTrue((path/"results/metrics.json").is_file())
                metrics=json.loads((path/"results/metrics.json").read_text())
                self.assertEqual(bool(metrics["diagnostics"]["field"]),variant!="damage")


if __name__=="__main__":unittest.main()
