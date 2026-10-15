"""Source, nested-correction and opt-in loss contracts; no research accuracy claim."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
import os
import subprocess

import numpy as np
import torch

from test_independence import fixture, model_for, arguments
from resources.MLdual import DualStageTransformer, DualLoss
from resources.MLindependence import prepare_bank, _CurveInputs, _fit_groups
from resources.MLfollowup import prepare_correction_bank, train_followup, new_corrector


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self):
        a=arguments()
        a.field_loss_variant=None
        a.corrector_d_model=8; a.corrector_n_heads=2; a.corrector_n_layers=1
        a.correction_bank=None; a.curve_training_source="predicted";a.reference_model_json=None
        return a

    def test_mixture_is_training_sampling_not_interpolation(self):
        data=fixture(); primary={m:np.zeros_like(data.splits['train']['field'][m]) for m in ('UT','FT')}
        secondary={m:np.ones_like(v) for m,v in primary.items()}
        ds=_CurveInputs(data.datasets['train'],primary,secondary)
        torch.manual_seed(42)
        values=[float(ds[i%len(ds)]['curve_input']['UT'][0,0]) for i in range(100)]
        self.assertEqual(set(values),{0.,1.})
        self.assertTrue(30<sum(values)<70)

    def test_identity_masks_independence_and_reload(self):
        data=fixture();a=self.args();model=model_for(data,'1i')
        net=new_corrector(model.model,data,a)
        b=next(iter(model.dataloaders['val']))
        net.eval();y=net.field_model(b['field'],b['task_features'],b['node_mask'])
        for mode in y:
            torch.testing.assert_close(y[mode],b['field'][mode])
        clone=DualStageTransformer.from_config(net.get_config())
        clone.load_state_dict(net.state_dict(),strict=True)
        y['UT'].sum().backward()
        self.assertTrue(all(p.grad is None for p in net.field_model.tasks['FT'].parameters()))

    def test_nested_bank_and_all_sources_with_frozen_generators(self):
        data=fixture(24);a=self.args()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bank=root/'bank';bank.mkdir()
            model=model_for(data,'1i')
            prepare_bank(model,'1i',a,bank,{})
            model.save(bank)
            a.field_bank=str(bank)
            baseline=model_for(data,'1i')
            meta={'run_config':{'seed':42}}
            (root/'baseline').mkdir()
            train_followup(baseline,a,root/'baseline',meta)
            baseline.save(root/'baseline',metadata=meta)
            a.reference_model_json=str(root/'baseline/model.json')
            correction=model_for(data,'1i');corrector_dir=root/'correction';corrector_dir.mkdir()
            meta={}
            prepare_correction_bank(correction,a,corrector_dir,meta)
            correction.save(corrector_dir,metadata=meta)
            correction.save_results(eval_split='val')
            from resources.MLmetrics import postprocess_load_dual_run
            restored=postprocess_load_dual_run(corrector_dir,load_model=True,data=data,device='cpu')[3]
            self.assertEqual(restored.model.field_model.get_config(),correction.model.field_model.get_config())
            manifest=json.loads((corrector_dir/'correction_bank.json').read_text())
            held=[]
            for fold in manifest['folds']:
                target=set(fold['heldout_train_ids']);held.extend(target)
                self.assertFalse(target & set(fold['fit_ids']))
                self.assertFalse(target & set(fold['selection_ids']))
                inner=json.loads((corrector_dir/'nested'/str(fold['fold'])/'field_bank.json').read_text())
                for f in inner['folds']:
                    self.assertFalse(target & set(f['fit_ids']+f['selection_ids']+f['heldout_train_ids']))
            self.assertEqual(set(held),set(data.sample_ids['train']))
            for source in ('predicted','true','mixed','corrected','corrected_mixed'):
                a.curve_training_source=source;a.correction_bank=str(corrector_dir)
                fit=model_for(data,'1i');dest=root/source;dest.mkdir()
                train_followup(fit,a,dest,{})
                saved=fit.save(dest)
                clone=DualStageTransformer.from_config(fit.model.get_config())
                clone.load_state_dict(fit.model.state_dict(),strict=True)
                self.assertTrue(Path(saved).exists())
                self.assertTrue(np.isfinite(fit.predict('val')['prediction']['curve']['UT']).all())
                field=fit.model.field_model.base if source.startswith('corrected') else fit.model.field_model
                raw=torch.load(bank/'model.mdl',weights_only=False)['model_state_dict']
                for key,value in field.state_dict().items():
                    torch.testing.assert_close(value,raw['field_model.'+key],rtol=0,atol=0)

    def test_original_cli_guards_and_new_study_scope(self):
        parse=runpy.run_path(str(Path(__file__).with_name('A0-HPC-Dual-test.py')))['parse_args']
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse(['--independence-case','1i','--experiment','baseline','--field-bank','x','--field-loss-variant','spatial'])
        args=parse(['--independence-case','1i','--experiment','baseline','--field-bank','x',
                    '--followup-study','losses','--field-loss-variant','spatial'])
        self.assertEqual(args.loss,'mse')
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse(['--followup-study','inputs','--independence-case','3ii','--experiment','baseline'])

    def test_custom_curve_objective_reaches_saved_independent_fit(self):
        from resources.MLfunc import CombinedCurveLoss
        data=fixture();a=self.args();model=model_for(data,'1i')
        objective=DualLoss(curve_loss=CombinedCurveLoss(peak_target_mode='soft'),
                           curve_normalizers=data.normalizers['curve'])
        inputs={s:data.splits[s]['field'] for s in ('train','val')}
        with tempfile.TemporaryDirectory() as folder:
            _fit_groups(model.model,data,[[('curve','UT')]],a,folder,{},cached=inputs,objective=objective)
            d=json.loads((Path(folder)/'groups/curve_UT/model.json').read_text())
            self.assertEqual(d['loss_config']['curve_losses']['UT']['class'],'CombinedCurveLoss')
            self.assertEqual(d['loss_config']['curve_losses']['UT']['params']['peak_target_mode'],'soft')
            self.assertEqual(d['loss_config']['weights']['curve'],{'UT':1.,'FT':0.})

    def test_custom_field_objective_reaches_independent_fit(self):
        from resources.MLfield import StructuredFieldLoss
        data=fixture();a=self.args();model=model_for(data,'1i')
        objective=DualLoss(field_loss={m:StructuredFieldLoss([[0,2],[2,3],[3,4],[4,5]],6,
            spatial_weight=.1,temporal_weight=.1,**data.normalizers['field'][m]) for m in ('UT','FT')})
        with tempfile.TemporaryDirectory() as folder:
            _fit_groups(model.model,data,[[('field','UT')]],a,folder,{},objective=objective)
            d=json.loads((Path(folder)/'groups/field_UT/model.json').read_text())
            self.assertEqual(d['loss_config']['field_losses']['UT']['class'],'StructuredFieldLoss')
            self.assertEqual(d['loss_config']['field_losses']['UT']['params']['spatial_weight'],.1)
            self.assertEqual(d['loss_config']['weights']['field'],{'UT':1.,'FT':0.})

    def test_launcher_dependency_graph_without_slurm(self):
        script=Path(__file__).with_name('B4_Independent-followups.sh').read_text()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';repo.mkdir();archive=root/'archive';home=root/'home'
            hpc=repo/'p2-DisorderML/HPC';hpc.mkdir(parents=True)
            (hpc/'B1_ML-new.sh').write_text('#!/bin/bash\n')
            for command in (['git','init',str(repo)],['git','-C',str(repo),'add','.'],
                            ['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@invalid','commit','-m','fixture']):
                subprocess.run(command,capture_output=True,check=True)
            base=archive/'MULTI/Dual/Transformer'
            anchor=base/'HPO/dual-joint-hpo1/best/model.json';anchor.parent.mkdir(parents=True);anchor.write_text('{}')
            for seed in (42,43,44):
                for role,name in [('bank-1i','field_bank.json'),('1ip','model.json')]:
                    f=base/f'independence-261013-r1-{role}-s{seed}'/name
                    f.parent.mkdir();f.write_text('{}')
            counter=root/'counter';counter.write_text('100')
            stub=root/'sbatch'
            stub.write_text(f'#!/bin/bash\nn=$(cat "{counter.as_posix()}")\nn=$((n+1))\necho "$n" > "{counter.as_posix()}"\necho "$n"\n')
            local=root/'launch.sh'
            local.write_text(script.replace('/opt/slurm/bin/sbatch',f'bash "{stub.as_posix()}"')
                                   .replace('/data/home/exy053/p2',home.as_posix()))
            env=dict(os.environ,REPO_ROOT=repo.as_posix(),ARCHIVE_ROOT=archive.as_posix())
            for study,count in [('inputs',15),('losses',30)]:
                subprocess.run(['bash',str(local),study,study,'--submit'],env=env,capture_output=True,text=True,check=True)
                lines=(home/f'MULTI/Dual/Transformer/{study}/jobs.tsv').read_text().splitlines()[1:]
                self.assertEqual(len(lines),count+1)
                rows=[r.split('\t') for r in lines];gate=rows[0][2]
                for seed in ('42','43','44'):
                    selected={r[0]:r for r in rows if r[1]==seed and r[0]!='preflight'}
                    if study=='inputs':
                        self.assertEqual(selected['correction'][3],gate)
                        self.assertTrue(all(selected[s][3]==selected['correction'][2] for s in ('true','mixed','corrected','corrected_mixed')))
                    else:
                        for variant in ('spatial','temporal','both','sudden'):
                            self.assertEqual(selected[variant][3],selected['bank-'+variant][2])
                        self.assertEqual(selected['combined_soft'][3],gate)


if __name__=='__main__':
    unittest.main()
