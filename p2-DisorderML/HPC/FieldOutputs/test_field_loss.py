"""CPU contract checks for the active structured displacement loss."""
import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pandas as pd
from resources.MLfield import StructuredFieldLoss, reference_field_edges
from resources.MLfunc import MaskedFieldMSELoss
from resources.MLmodels import _model_loss_to_config, _model_build_loss_from_config


class FieldLossTests(unittest.TestCase):
    def loss(self, **kwargs):
        return StructuredFieldLoss([[0,1],[1,2]], 3, n_components=1, **kwargs)

    def test_identity_and_gradients(self):
        y=torch.tensor([[[0.,1.],[0.,2.],[0.,3.]]])
        for gain in (0.,3.):
            loss=self.loss(localization_gain=gain)
            p=y.clone().requires_grad_()
            self.assertEqual(loss(p,y).item(),0.)
            loss(p,y).backward()
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_masked_baseline(self):
        y=torch.randn(2,3,4); y[0,1,2]=float("nan")
        p=torch.randn_like(y,requires_grad=True)
        result=self.loss(spatial_weight=0,temporal_weight=0)(p,y)
        torch.testing.assert_close(result,MaskedFieldMSELoss()(p,y))
        result.backward(); self.assertTrue(torch.isfinite(p.grad).all())

    def test_signed_jump(self):
        y=torch.tensor([[[0.],[1.],[2.]]])
        loss=self.loss()
        reverse=loss.component_losses(-y,y)["spatial"]
        smooth=loss.component_losses(y*.5,y)["spatial"]
        self.assertGreater(reverse,smooth)
        self.assertEqual(loss.component_losses(y+3,y)["spatial"].item(),0.)

    def test_physical_increment(self):
        mean=[0.,10.];scale=[1.,2.]
        y=torch.zeros(1,3,2)
        p=torch.tensor([[[2.,1.]]*3])  # constant +2 in physical units
        self.assertEqual(self.loss(mean=mean,scale=scale).component_losses(p,y)["temporal"].item(),0.)

    def test_validity_and_nonfinite_prediction(self):
        y=torch.ones(1,3,3);p=y.clone();p[:,2]=1000
        mask=torch.ones_like(y,dtype=torch.bool);mask[:,2]=False
        self.assertEqual(self.loss()(p,y,mask).item(),0.)
        p[:,0]=float("nan")
        with self.assertRaises(FloatingPointError): self.loss()(p,y,mask)

    def test_config(self):
        loss=self.loss(mean=[1,2],scale=[2,3],localization_gain=3)
        restored=_model_build_loss_from_config(_model_loss_to_config(loss))
        self.assertEqual(loss.get_config(),restored.get_config())

    def test_periodic_topology(self):
        xy=np.array([(x*10,y*10) for y in range(20) for x in range(21)] +
                    [(x*10+5,y*10+5) for y in range(19) for x in range(20)])
        ut=reference_field_edges(xy,"UT");ft=reference_field_edges(xy,"FT")
        self.assertEqual(len(ut),2319);self.assertEqual(len(ft),2259)
        np.testing.assert_array_equal(ft,reference_field_edges(xy*.001+[3,4],"FT"))

    def test_degree_averaging_and_temporal_hole(self):
        loss=self.loss()
        values=torch.tensor([[[[1.]],[[9.]]]])
        means,valid=loss._node_average(values,torch.ones_like(values,dtype=torch.bool))
        torch.testing.assert_close(means.flatten(),torch.tensor([1.,5.,9.]))
        y=torch.zeros(1,3,3);p=y.clone();p[:,:,1]=100.
        mask=torch.ones_like(y,dtype=torch.bool);mask[:,:,1]=False
        self.assertEqual(loss.component_losses(p,y,mask)["temporal"].item(),0.)

    def test_epoch_component_collection(self):
        from resources.MLfunc import train_model
        from torch.utils.data import DataLoader,TensorDataset
        x=torch.randn(4,3,2);y=torch.randn(4,3,2)
        net=torch.nn.Linear(2,2);loss=self.loss()
        loader=DataLoader(TensorDataset(x,y),batch_size=2)
        rows=[]
        train_model('tr',net,loss,2,torch.optim.Adam(net.parameters()),loader,loader,
                    device='cpu',verbose=0,selection_metric='mse',epoch_callback=lambda e,r,m:rows.append(r))
        self.assertEqual(len(rows),2)
        self.assertIn('train_spatial',rows[0]);self.assertIn('val_temporal',rows[0])
        self.assertTrue(np.isfinite(list(rows[0].values())).all())

    def test_graph_node_batch_shape(self):
        from torch_geometric.data import Data,Batch
        from resources.MLfunc import _forward_batch
        from resources.MLmodels import GNN
        edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
        batch=Batch.from_data_list([Data(x=torch.randn(3,2),edge_index=edges,y=torch.randn(3,4)) for _ in range(2)])
        net=GNN(in_size=2,h_size=[8],out_size=4,pool='node',block='gcn')
        p,y=_forward_batch('gcn',net,batch,'cpu','test')
        self.assertEqual(tuple(p.shape),(2,3,4))
        self.loss()(p,y).backward()
        self.assertTrue(any(v.grad is not None for v in net.parameters()))

    def test_motion_diagnostics_identity(self):
        from resources.MLmetrics import field_motion_diagnostics
        xy=np.array([(x*10,y*10) for y in range(20) for x in range(21)] +
                    [(x*10+5,y*10+5) for y in range(19) for x in range(20)])
        y=np.random.default_rng(1).normal(size=(2,800,4))
        table=field_motion_diagnostics(y,y,xy,'UT',sample_ids=['a','b'])
        self.assertEqual(table.spatial_rmse.max(),0.)
        self.assertEqual(table.u2_jump_sign_accuracy.min(),1.)
        self.assertEqual(table.activity_node_jaccard.min(),1.)

    def test_frozen_curve_bridge_alignment(self):
        from resources.MLfield import evaluate_independent_curve_bridge
        from sklearn.preprocessing import StandardScaler
        class Curve(torch.nn.Module):
            def forward(self,x):
                return x.mean((1,2))[:,None]*torch.linspace(0,1,201)[None,:]
        coords=pd.DataFrame([[0,0,1,0,2,0]])
        inputs=np.arange(90,dtype=float).reshape(3,3,10)/90
        inscale=StandardScaler().fit(inputs.reshape(-1,10))
        scaled=inscale.transform(inputs.reshape(-1,10)).reshape(inputs.shape)
        outscale=StandardScaler().fit(np.stack([np.zeros(201),np.ones(201)]))
        train=pd.DataFrame(index=['train'])
        data=SimpleNamespace(path='fixture',UT_IN_df=coords,
             UT_val_in_df=pd.DataFrame(index=['b','a']),UT_train_in_df=train,
             UT_field_components=['U1','U2'],UT_field_frame_values=np.array([.5,1.]))
        curves=SimpleNamespace(input_kind='field',output_kind='curve',reduce_dim=False,
             UT_IN_df=coords,UT_val_in_df=pd.DataFrame(index=['a','b','c']),UT_train_in_df=train,
             UT_field_input_components=['U1','U2'],UT_field_input_frame_values=np.array([.5,1.]),
             UT_val_in=scaled,UT_val_out=np.tile(np.linspace(0,1,201),(3,1)),
             UT_INscaler=inscale,UT_OUTscaler=outscale,
             UT_OUT_df=pd.DataFrame([np.r_[0,np.linspace(0,1,201)]],columns=[str(i) for i in range(202)]))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);source=folder/'best_model.json'
            (folder/'best_model_data.json').write_text(json.dumps({'data_config':{}}))
            np.savez(folder/'predictions.npz',UT_val_outputs=inputs[[1,0],:,:4])
            with patch('resources.MLdata.DATA',return_value=curves),patch('resources.MLmodels.MODEL.from_json',return_value=SimpleNamespace(UT_model=Curve())):
                result=evaluate_independent_curve_bridge(data,source,folder,'UT')
            np.testing.assert_allclose(result['true_field'],result['pred_field'],atol=1e-6)
            with np.load(folder/'independent_curve_bridge.npz') as z:
                self.assertEqual(z['sample_ids'].tolist(),['b','a'])

    def test_hpo_preset_exact_training_arguments(self):
        import runpy
        entry=Path(__file__).with_name('A0-HPC_Field-lossTrial.py')
        cfg={'batch':1,'lr':7e-5,'opt':['adamw',2e-9],'scheduler':['plateau','min',.2,12,1e-4],
             'earlyStop':{'params':{'patience':100,'min_delta':1e-5}}}
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'FT/Field/HPO/study/Transformer/best_model.json';source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'reload_config':{'training_config':cfg}}))
            main=runpy.run_path(str(entry))['main'];calls=[]
            with patch('runpy.run_path',return_value={'main':lambda args:calls.append(args)}):
                main(['--task','FT','--hpo-model-json',str(source),'--field-loss-variant','both'])
            args=calls[0]
            self.assertEqual(args[args.index('--batch')+1],'1')
            self.assertEqual(args[args.index('--early-stop-patience')+1],'100')
            self.assertEqual(args[-2:],['--field-loss-variant','both'])

    def test_saved_motion_plots(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from resources.MLmetrics import plot_field_motion_results, plot_true_field_curve_comparison, plot_field_loss_components
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            self.assertIsNone(plot_field_motion_results(folder))
            for mode in ('UT','FT'):
                pd.DataFrame({'sample_id':['a','b'],'field_rmse':[.1,.2],
                              'jump_interval_mae':[1,2],'jump_timing_event_count':[3,4]}).to_csv(folder/f'{mode}_val_field_motion.csv',index=False)
                for source in ('curve','true_field_curve'):
                    pd.DataFrame({'sample_id':['a','b'],'sample_rmse':[.1,.2],
                                  'imputed_field_values':[0,0]}).to_csv(folder/f'{mode}_val_{source}_sample_metrics.csv',index=False)
            self.assertIsNotNone(plot_field_motion_results(folder))
            self.assertIsNotNone(plot_true_field_curve_comparison(folder))
            history=pd.DataFrame({'epoch':[1,2],**{f'{s}_{t}':[.1,.2] for s in ('train','val') for t in ('displacement','spatial','temporal')}})
            figure=plot_field_loss_components(history)
            self.assertTrue(all(ax.get_yscale()=='log' for ax in figure.axes))
            plt.close('all')


if __name__=="__main__": unittest.main()
