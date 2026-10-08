"""Active CPU contracts for specimen-specific weighting and relative reporting."""
import itertools
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from resources.MLfield import StructuredFieldLoss
from resources.MLmodels import _model_build_loss_from_config, _model_loss_to_config


class SuddenWeightingTests(unittest.TestCase):
    def setUp(self):
        self.xy = torch.tensor([[[0.,0.],[1.1,.1],[-.9,.2],[.1,1.2],[-.1,-.8]]])
        self.edges = list(itertools.combinations(range(5),2))
        self.loss = StructuredFieldLoss(self.edges,5,spatial_weight=0.,temporal_weight=0.,
                                        localization_gain=1.,localization_mode="sudden")
        self.y = torch.zeros(1,5,10)

    def test_smooth_affine_motion_is_neutral(self):
        times=torch.arange(5.)
        u=(self.xy@torch.tensor([[2.,1.],[-1.,.5]]))[:,:,None,:]*times[None,None,:,None]
        u += 10*times[None,None,:,None]
        w=self.loss.localization_weights(u.flatten(2),initial_coords=self.xy)
        torch.testing.assert_close(w,torch.ones_like(w),rtol=1e-5,atol=1e-5)
        # Static non-affine offsets are not a sudden loading event either.
        u[:,0] += 100
        w=self.loss.localization_weights(u.flatten(2),initial_coords=self.xy)
        torch.testing.assert_close(w,torch.ones_like(w),rtol=1e-5,atol=1e-5)

    def test_spike_weights_are_detached_capped_and_normalized(self):
        self.y[:,0,4:]=8.
        self.y.requires_grad_()
        w=self.loss.localization_weights(self.y,initial_coords=self.xy)
        self.assertFalse(w.requires_grad)
        self.assertGreater(w[0,0,2,0],w[0,0,0,0])
        self.assertGreaterEqual(w.min().item(),.5)
        self.assertLessEqual(w.max().item(),2.)
        self.assertAlmostEqual(w.mean().item(),1.,places=6)
        p=torch.zeros_like(self.y,requires_grad=True)
        self.loss(p,self.y,initial_coords=self.xy).backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_masks_exclude_absent_nodes_and_missing_intervals(self):
        mask=torch.ones_like(self.y,dtype=torch.bool);mask[:,4]=False
        a=self.loss.localization_weights(self.y,mask,self.xy)
        self.y[:,4]=1e9
        b=self.loss.localization_weights(self.y,mask,self.xy)
        torch.testing.assert_close(a[:,:4],b[:,:4])
        self.y.zero_();self.y[:,:,4:6]=1e9;mask[:,:,4:6]=False
        b=self.loss.localization_weights(self.y,mask,self.xy)
        torch.testing.assert_close(b[mask.reshape_as(b)],torch.ones_like(b[mask.reshape_as(b)]))
        w=self.loss.localization_weights(self.y,torch.zeros_like(mask),self.xy)
        torch.testing.assert_close(w,torch.ones_like(w))

    def test_physical_reconstruction_and_coordinate_units(self):
        self.y[:,0,4:]=8.
        loss=StructuredFieldLoss(self.edges,5,mean=list(range(10)),scale=list(range(1,11)),
            spatial_weight=0.,temporal_weight=0.,localization_gain=1.,localization_mode="sudden")
        scaled=(self.y-torch.arange(10.))/torch.arange(1.,11.)
        a=self.loss.localization_weights(self.y,initial_coords=self.xy)
        b=loss.localization_weights(scaled,initial_coords=self.xy*1000+25)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

    def test_rank_deficient_patch_and_historical_reload(self):
        xy=self.xy.clone();xy[:,:,1]=0
        self.y[:,0,4:]=8.
        w=self.loss.localization_weights(self.y,initial_coords=xy)
        self.assertTrue(torch.isfinite(w).all())
        cfg=_model_loss_to_config(self.loss)
        restored=_model_build_loss_from_config(cfg)
        torch.testing.assert_close(self.loss(self.y,self.y,initial_coords=xy),
                                   restored(self.y,self.y,initial_coords=xy))
        old=StructuredFieldLoss(self.edges,5).get_config();old.pop('localization_mode')
        self.assertEqual(StructuredFieldLoss(**old).localization_mode,'activity')
        with self.assertRaises(ValueError):self.loss(self.y,self.y)

    def test_relative_reporting_rejects_validation_mean(self):
        from resources.MLmetrics import relative_error_summary,field_summary_table,curve_summary_table
        for kind,table in [('field',field_summary_table),('curve',curve_summary_table)]:
            s={'rmse':2.,f'mean_{kind}_baseline_rmse':4.,f'mean_{kind}_baseline_source':f'train_mean_{kind}'}
            self.assertEqual(relative_error_summary(s,kind)['RMSE reduction vs training mean (%)'],50.)
            self.assertIn('(%)',table({'summary':s}).iloc[0]['metric'])
            s[f'mean_{kind}_baseline_source']=f'truth_mean_{kind}'
            self.assertTrue(np.isnan(relative_error_summary(s,kind)['RMSE / training-mean error (%)']))

    def test_opt_in_cli_and_safe_suite_preview(self):
        root=Path(__file__).parent
        parse=runpy.run_path(str(root/'A0-HPC-Dual-test.py'))['parse_args']
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(['--experiment','sudden'])
        self.assertEqual(parse(['--experiment','sudden','--localization-gain','1']).localization_gain,1.)
        preview=subprocess.run(['bash',str(root/'B4_Dual-experiments.sh'),'unit-preview',
            '--variants','baseline,sudden'],capture_output=True,text=True,check=True)
        self.assertIn('2 dependent',preview.stdout)
        self.assertNotIn('Variant: private',preview.stdout)

    def test_dual_training_and_checkpoint(self):
        from resources.MLdual import DUAL_DATA,DUAL_MODEL,DualLoss,DualStageTransformer
        rng=np.random.default_rng(42);modes=('UT','FT');masks={m:np.ones(5,bool) for m in modes}
        splits={s:{'geometry':rng.normal(size=(n,5,2)).astype('float32')*.01,
            'field':{m:rng.normal(size=(n,5,10)).astype('float32') for m in modes},
            'field_mask':{m:np.ones((n,5,10),dtype=bool) for m in modes},
            'curve':{m:rng.normal(size=(n,7)).astype('float32') for m in modes}}
            for s,n in [('train',4),('val',2),('test',2)]}
        data=DUAL_DATA(splits,masks,{m:np.ones((5,1)) for m in modes},
            metadata={'canonical_coords':self.xy[0].numpy()})
        cfg={'d_model':8,'n_heads':2,'n_layers':1,'dropout':0.}
        net=DualStageTransformer.from_data(data,field_kwargs=cfg,curve_kwargs=cfg)
        losses={m:StructuredFieldLoss(self.edges,5,**data.normalizers['field'][m],
            spatial_weight=0.,temporal_weight=0.,localization_gain=1.,localization_mode='sudden') for m in modes}
        trainer=DUAL_MODEL(net,DualLoss(field_loss=losses),data=data,batch=2,device='cpu')
        result=trainer._run_loader(trainer.dataloaders['train'],True)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model'
            path=Path(trainer.save(path))
            desc=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(desc['loss_config']['field_losses']['UT']['params']['localization_mode'],'sudden')
            trainer.load(path,strict=True)
            self.assertTrue(all(np.isfinite(v) for v in trainer._run_loader(trainer.dataloaders['val'],False).values()))

    def test_full_fcc_topology_and_relative_motion(self):
        from resources.MLfield import reference_field_edges
        from resources.MLmetrics import field_motion_diagnostics
        xy=np.array([(10*x,10*y) for y in range(20) for x in range(21)]+
                    [(10*x+5,10*y+5) for y in range(19) for x in range(20)],dtype=np.float32)
        rng=np.random.default_rng(42)
        target=rng.normal(size=(1,800,10)).astype(np.float32)
        for mode,count in [('UT',2319),('FT',2259)]:
            edges=reference_field_edges(xy,mode)
            self.assertEqual(len(edges),count)
            mask=np.ones_like(target,dtype=bool)
            if mode=='FT': mask[:,(xy[:,1]==95)&(xy[:,0]<120)]=False
            loss=StructuredFieldLoss(edges,800,spatial_weight=0.,temporal_weight=0.,
                localization_mode='sudden',localization_gain=1.)
            w=loss.localization_weights(torch.from_numpy(target),torch.from_numpy(mask),
                torch.from_numpy(xy[None]+rng.normal(scale=.1,size=(1,800,2)).astype(np.float32)))
            self.assertTrue(torch.isfinite(w).all())
            self.assertAlmostEqual(w.flatten()[mask.flatten()].mean().item(),1.,places=5)
            table=field_motion_diagnostics(np.zeros_like(target),target,xy,mode,valid_mask=mask)
            self.assertAlmostEqual(table.iloc[0]['local_jump_nrmse_percent'],100.)
            self.assertAlmostEqual(table.iloc[0]['temporal_nrmse_percent'],100.)

    def test_sudden_runner_saves_complete_results(self):
        from resources.MLdual import DUAL_DATA, dual_node_context
        xy=np.array([(10*x,10*y) for y in range(20) for x in range(21)]+
                    [(10*x+5,10*y+5) for y in range(19) for x in range(20)],dtype=np.float32)
        masks={'UT':np.ones(800,bool),'FT':~((xy[:,1]==95)&(xy[:,0]<120))}
        features,names,spec=dual_node_context(xy,(xy[:,0]>0)&(xy[:,0]<200)&(xy[:,1]>0)&(xy[:,1]<190),masks)
        rng=np.random.default_rng(42)
        splits={s:{'geometry':rng.normal(scale=.1,size=(n,800,2)).astype('float32'),
            'field':{m:rng.normal(size=(n,800,10)).astype('float32') for m in masks},
            'field_mask':{m:np.broadcast_to(masks[m][None,:,None],(n,800,10)).copy() for m in masks},
            'curve':{m:rng.normal(size=(n,201)).astype('float32') for m in masks}}
            for s,n in [('train',4),('val',2),('test',2)]}
        data=DUAL_DATA(splits,masks,features,task_feature_names=names,metadata={
            'canonical_coords':xy,'context_spec':spec,'field_components':{m:['U1','U2'] for m in masks},
            'field_frame_values':{m:np.linspace(.2,1,5) for m in masks},
            'curve_x_values':{m:np.linspace(0,1,201) for m in masks}})
        runner=runpy.run_path(str(Path(__file__).with_name('A0-HPC-Dual-test.py')))['main']
        with tempfile.TemporaryDirectory() as tmp, patch.object(DUAL_DATA,'from_files',return_value=data), contextlib.redirect_stdout(io.StringIO()):
            trainer=runner(['--allow-cpu','--epochs','1','--batch','2','--run-root',tmp,
                '--run-label','sudden-contract','--experiment','sudden','--localization-gain','1','--loss','mse',
                '--field-d-model','8','--field-n-heads','2','--field-n-layers','1',
                '--curve-d-model','8','--curve-n-heads','2','--curve-n-layers','1'])
            folder=Path(trainer.results_dir)
            self.assertTrue((folder/'predictions.npz').is_file())
            for mode in masks:
                self.assertTrue((folder/f'{mode}_val_field_motion.csv').is_file())
                self.assertEqual(trainer.lossf.field_losses[mode].localization_mode,'sudden')
            self.assertEqual(json.loads((folder.parent/'run_metadata.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
