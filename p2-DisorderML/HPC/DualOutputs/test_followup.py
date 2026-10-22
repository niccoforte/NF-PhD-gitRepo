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
import ast
import shutil
import copy
import hashlib
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from test_independence import fixture, model_for, arguments
from resources.MLdual import DualStageTransformer, DualLoss, FieldCorrectionStage
from resources.MLindependence import prepare_bank, _CurveInputs, _fit_groups
from resources.MLfollowup import prepare_correction_bank, train_followup, new_corrector, field_objective, reference_model


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self):
        a=arguments()
        a.field_loss_variant=None
        a.corrector_d_model=8; a.corrector_n_heads=2; a.corrector_n_layers=1
        a.correction_bank=None; a.curve_training_source="predicted";a.reference_model_json=None
        a.followup_study="inputs"; a.reference_curve_source="predicted"; a.curve_loss_ablation=None
        a.spatial_weight=.1; a.temporal_weight=.1
        return a

    def test_mixture_is_training_sampling_not_interpolation(self):
        data=fixture(); primary={m:np.zeros_like(data.splits['train']['field'][m]) for m in ('UT','FT')}
        secondary={m:np.ones_like(v) for m,v in primary.items()}
        ds=_CurveInputs(data.datasets['train'],primary,secondary)
        torch.manual_seed(42)
        values=[float(ds[i%len(ds)]['curve_input']['UT'][0,0]) for i in range(100)]
        self.assertEqual(set(values),{0.,1.})
        self.assertTrue(30<sum(values)<70)

    def test_notebook_artifact_roots_are_platform_specific(self):
        notebooks=Path(__file__).resolve().parents[2]/'code'
        for name in ('ML-DualOutputs.ipynb','ML-DualPostProcessing.ipynb','ML-HPOpostProcess.ipynb'):
            nb=json.loads((notebooks/name).read_text(encoding='utf-8'))
            code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
            tree=ast.parse('\n'.join(line for line in code.splitlines() if not line.startswith('%')))
            node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
                isinstance(t,ast.Name) and t.id=='RUN_ROOT' for t in n.targets))
            expression=compile(ast.Expression(node.value),name,'eval')
            for platform,expected in (('nt',Path('Z:/p2')),('posix',None)):
                value=eval(expression,{'os':SimpleNamespace(name=platform),'Path':Path,'REPO':Path('fixture')})
                self.assertEqual(value,expected)
            guard=tree.body[tree.body.index(node)+1]
            with self.assertRaisesRegex(ValueError,'external data directory'):
                exec(compile(ast.Module(body=[guard],type_ignores=[]),name,'exec'),{'RUN_ROOT':None})

    def test_transfers_require_external_destinations(self):
        bash=shutil.which('bash')
        if bash is None:
            self.skipTest('Bash unavailable; transfer guards require a shell')
        scripts=Path(__file__).resolve().parents[1]
        # Stub only transfer commands: exercise the real Git destination guard.
        wrapper='''
rsync() { echo MOCK_TRANSFER; }
scp() { echo MOCK_TRANSFER; }
export -f rsync scp
if command -v cygpath >/dev/null; then export LOCAL_ROOT=$(cygpath -u "$LOCAL_ROOT"); fi
exec bash "$@"
'''
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkout=root/'checkout';checkout.mkdir()
            subprocess.run(['git','init','--quiet',str(checkout)],check=True)
            for name in ('B3_ML-transfer-mac.sh','B3_ML-transfer-windows.sh'):
                script=scripts/name
                for destination,allowed in ((checkout/'data',False),(root/'external',True)):
                    env=dict(os.environ,LOCAL_ROOT=destination.as_posix())
                    result=subprocess.run([bash,'-c',wrapper,'transfer-test',script.as_posix(),
                                           'MULTI/Dual/Transformer/example'],env=env,
                                          capture_output=True,text=True)
                    if allowed:
                        self.assertEqual(result.returncode,0,result.stderr)
                        self.assertIn('MOCK_TRANSFER',result.stdout)
                    else:
                        self.assertEqual(result.returncode,2,result.stderr)
                        self.assertIn('outside all Git checkouts',result.stderr)
                        self.assertFalse(destination.exists())
                if name.endswith('-mac.sh'):
                    env=dict(os.environ);env.pop('LOCAL_ROOT',None)
                    result=subprocess.run([bash,script.as_posix(),'MULTI/Dual/Transformer/example'],
                                          env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,2,result.stderr)
                    self.assertIn('Set LOCAL_ROOT',result.stderr)

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
                if source=='true':
                    # True-trained checkpoint: main results still deploy P;
                    # the additional T evaluation bypasses the field generator.
                    results=Path(fit.save_results(eval_split='val'))
                    fit.save_true_field_curves(results,'val')
                    with np.load(results/'true_field_curves.npz',allow_pickle=False) as exported:
                        self.assertEqual(list(exported['sample_ids']),data.sample_ids['val'])
                        fit.model.eval()
                        batch=next(iter(fit.dataloaders['val']))
                        with torch.no_grad():
                            direct=fit.model.curve_model(batch['field'],batch['task_features'],batch['node_mask'])
                        for mode in ('UT','FT'):
                            expected=data.inverse_curve(mode,direct[mode].numpy())
                            np.testing.assert_allclose(exported[mode+'_outputs'],expected,rtol=1e-6,atol=1e-6)
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
        common=['--followup-study','losses','--independence-case','1i','--experiment','baseline','--field-bank','raw']
        for source in ('predicted','true','mixed','corrected','corrected_mixed'):
            correction=['--correction-bank','new'] if source.startswith('corrected') else []
            for objective in (['--field-loss-variant','spatial'],['--curve-loss-ablation','combined_soft']):
                parsed=parse(common+['--curve-training-source',source]+correction+objective)
                self.assertEqual(parsed.curve_training_source,source)
        for extra in (['--curve-training-source','corrected_mixed'],
                      ['--field-loss-variant','spatial','--curve-loss-ablation','combined_soft'],
                      ['--prepare-correction-bank','--reference-model-json','ref','--curve-loss-ablation','combined_soft'],
                      ['--reference-model-json','ref','--reference-curve-source','mixed'],
                      ['--eval-split','test']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse(common+extra)
        self.assertEqual(parse([]).curve_training_source,'predicted')

    def test_reference_hash_exception_is_bounded_and_checked_before_nested_fits(self):
        data=fixture(); a=self.args(); model=model_for(data,'1i')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            meta={'case':'1ip','run_config':{'seed':42},'field_bank_sha256':'original'}
            model.save(root/'reference',metadata=meta)
            a.reference_model_json=str(root/'reference/model.json')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                reference_model(model,a,'changed')
            a.followup_study='losses'; a.field_loss_variant='spatial'
            provenance={}
            reference_model(model,a,'changed',provenance)
            self.assertEqual(provenance['reference_field_bank_sha256'],'original')
            self.assertTrue(provenance['changed_generator_reference'])
            original=json.loads(Path(a.reference_model_json).read_text())
            for edit,message in ((lambda d:d['metadata']['run_config'].update(seed=43),'seeded'),
                                 (lambda d:d['data']['sample_ids'].update(val=['wrong']),'data/splits/scalers'),
                                 (lambda d:d['data']['normalizers']['field']['UT'].update(scale=[123]),'data/splits/scalers'),
                                 (lambda d:d['model_config']['curve_model'].update(d_model=16),'configuration')):
                descriptor=copy.deepcopy(original);edit(descriptor)
                Path(a.reference_model_json).write_text(json.dumps(descriptor))
                with self.assertRaisesRegex(ValueError,message):
                    reference_model(model,a,'changed')
            Path(a.reference_model_json).write_text(json.dumps(original))
            a.curve_loss_ablation='combined_soft'
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                reference_model(model,a,'changed')
            a.curve_loss_ablation=None; a.followup_study='inputs'
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                reference_model(model,a,'changed')
            a.field_bank=str(root);(root/'field_bank.npz').write_bytes(b'changed')
            destination=root/'correction';destination.mkdir()
            with mock.patch('resources.MLfollowup.load_bank',return_value=({},{})), \
                 mock.patch('resources.MLfollowup.prepare_bank') as nested:
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    prepare_correction_bank(model,a,destination,{})
                nested.assert_not_called()
                self.assertFalse((destination/'nested').exists())
            # Curve losses may not quietly attach a different refiner to the same raw bank.
            corrected=model_for(data,'1i')
            refiner=new_corrector(corrected.model,data,a).field_model.correction
            corrected.model.field_model=FieldCorrectionStage(refiner,corrected.model.field_model)
            corrected.save(root/'cm',metadata=dict(meta,curve_training_source='corrected_mixed'))
            a.reference_model_json=str(root/'cm/model.json');a.reference_curve_source='corrected_mixed'
            a.followup_study='losses';a.field_loss_variant=None;a.curve_loss_ablation='combined_soft'
            reference_model(corrected,a,'original')
            with torch.no_grad():
                next(corrected.model.field_model.correction.parameters()).add_(1.)
            with self.assertRaisesRegex(ValueError,'winning frozen correction pipeline'):
                reference_model(corrected,a,'original')

    def test_changed_spatial_generators_use_fold_local_loss_and_mse_correctors(self):
        data=fixture(24); a=self.args()
        a.followup_study='losses'; a.field_loss_variant='spatial'; a.reference_curve_source='corrected_mixed'
        edges=np.array([[0,2],[2,3],[3,4],[4,5]])
        with tempfile.TemporaryDirectory() as folder, mock.patch('resources.MLfield.reference_field_edges',return_value=edges):
            root=Path(folder); bank=root/'bank';bank.mkdir()
            parent=model_for(data,'1i')
            prepare_bank(parent,'1i',a,bank,{},loss_factory=lambda d:field_objective(d,a))
            parent.save(bank);a.field_bank=str(bank)
            winner=model_for(data,'1i')
            refiner=new_corrector(winner.model,data,a).field_model.correction
            winner.model.field_model=FieldCorrectionStage(refiner,winner.model.field_model)
            winner.save(root/'winner',metadata={'case':'1ip','run_config':{'seed':42,'followup_study':'inputs'},
                'curve_training_source':'corrected_mixed','field_bank_sha256':hashlib.sha256(b'original MSE').hexdigest()})
            a.reference_model_json=str(root/'winner/model.json')
            correction=model_for(data,'1i');correction.lossf=field_objective(data,a)
            dest=root/'correction';dest.mkdir();meta={}
            prepare_correction_bank(correction,a,dest,meta);correction.save(dest,metadata=meta)
            self.assertEqual(correction.lossf.get_config()['field_losses']['UT']['class'],'MaskedFieldMSELoss')
            self.assertNotEqual(meta['field_bank_sha256'],meta['reference_field_bank_sha256'])
            provenance=json.loads((dest/'correction_bank.json').read_text())
            self.assertEqual(provenance['generator_loss_variant'],'spatial')
            self.assertEqual(provenance['corrector_loss'],'mse')
            for outer in provenance['folds']:
                nested=dest/'nested'/str(outer['fold'])
                manifest=json.loads((nested/'field_bank.json').read_text())
                self.assertEqual(manifest['field_loss_variant'],'spatial')
                for descriptor_path in nested.glob('**/groups/field_UT/model.json'):
                    descriptor=json.loads(descriptor_path.read_text());loss=descriptor['loss_config']['field_losses']['UT']
                    if 'corrector' in descriptor_path.parts:
                        self.assertEqual(loss['class'],'MaskedFieldMSELoss')
                    else:
                        self.assertEqual(loss['class'],'StructuredFieldLoss')
                        self.assertEqual(loss['params']['spatial_weight'],.1)
                        normalizers=descriptor['data']['normalizers']['field']['UT']
                        np.testing.assert_allclose(loss['params']['mean'],normalizers['mean'])
                        np.testing.assert_allclose(loss['params']['scale'],normalizers['scale'])
                        self.assertFalse(set(outer['heldout_train_ids']) & set(descriptor['data']['sample_ids']['train']))
            a.curve_training_source='corrected_mixed';a.correction_bank=str(dest)
            fit=model_for(data,'1i');adapted=root/'adapted';adapted.mkdir()
            train_followup(fit,a,adapted,{})
            self.assertTrue((adapted/'fixed_readout_curves.npz').is_file())
            for key,value in correction.model.field_model.state_dict().items():
                torch.testing.assert_close(fit.model.field_model.state_dict()[key],value,rtol=0,atol=0)
            for mode in ('UT','FT'):
                saved=json.loads((adapted/'groups'/('curve_'+mode)/'model.json').read_text())
                self.assertEqual(saved['loss_config']['curve_losses'][mode]['class'],'MSELoss')

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
        bash=shutil.which('bash')
        if bash is None:
            self.skipTest('Bash unavailable; launcher requires a shell')
        script=Path(__file__).with_name('B4_Independent-followups.sh').read_text()
        wrapper=(Path(__file__).resolve().parents[1]/'B1_ML-new.sh').read_text()
        for directive in ('#SBATCH -n 12','#SBATCH --cpus-per-gpu=12','#SBATCH --mem-per-cpu=7500M','#SBATCH --gres=gpu:1'):
            self.assertIn(directive,wrapper)
        parse=runpy.run_path(str(Path(__file__).with_name('A0-HPC-Dual-test.py')))['parse_args']
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
                for role,names in [('correction',('correction_bank.json','model.json')),('corrected_mixed',('model.json',)),('mixed',('model.json',))]:
                    for name in names:
                        f=base/f'winning-inputs-{role}-s{seed}'/name
                        f.parent.mkdir(exist_ok=True);f.write_text('{}')
            reused=home/'MULTI/Dual/Transformer/reused-losses';reused.mkdir(parents=True)
            bank_rows=['role\tseed\tjob_id\tdependency\tsource_revision']
            for si,seed in enumerate((42,43,44)):
                for vi,variant in enumerate(('spatial','temporal','both','sudden')):
                    job=800+si*4+vi
                    bank_rows.append(f'bank-{variant}\t{seed}\t{job}\t-\told-revision')
                    if seed!=44:
                        dest=base/f'reused-losses-bank-{variant}-s{seed}';dest.mkdir()
                        for name in ('field_bank.json','field_bank.npz','model.mdl','model_data.json'):
                            (dest/name).touch()
            (reused/'jobs.tsv').write_text('\n'.join(bank_rows)+'\n')
            sacct=root/'sacct';sacct.write_text('#!/bin/bash\ncase "$4" in 808|809|810|811) echo "PENDING|0:0";; *) echo "COMPLETED|0:0";; esac\n')
            squeue=root/'squeue';squeue.write_text('#!/bin/bash\necho "$3"\n')
            counter=root/'counter';counter.write_text('100')
            calls=root/'calls'
            stub=root/'sbatch'
            stub.write_text(f'#!/bin/bash\nn=$(cat "{counter.as_posix()}")\nn=$((n+1))\necho "$n" > "{counter.as_posix()}"\n'
                f'printf "%s\\t" "$n" "$@" >> "{calls.as_posix()}"\nprintf "\\n" >> "{calls.as_posix()}"\necho "$n"\n')
            local=root/'launch.sh'
            local.write_text(script.replace('/opt/slurm/bin/sbatch',f'bash "{stub.as_posix()}"')
                                   .replace('/opt/slurm/bin/sacct',f'bash "{sacct.as_posix()}"')
                                   .replace('/opt/slurm/bin/squeue',f'bash "{squeue.as_posix()}"')
                                   .replace('/data/home/exy053/p2',home.as_posix()))
            env=dict(os.environ,REPO_ROOT=repo.as_posix(),ARCHIVE_ROOT=archive.as_posix())
            for suite,study,source,count in [('inputs','inputs','predicted',15),('losses-P','losses','predicted',30),
                                             ('losses-CM','losses','corrected_mixed',42),
                                             ('losses-mixed-reuse','losses','mixed',18)]:
                reuse=suite=='losses-mixed-reuse'
                env.update(LOSS_CURVE_SOURCE=source,INPUT_SUITE='winning-inputs',FIELD_BANK_SUITE='reused-losses' if reuse else '')
                before=counter.read_text()
                preview=subprocess.run([bash,str(local),suite,study],env=env,capture_output=True,text=True,check=True)
                self.assertEqual(counter.read_text(),before)
                self.assertFalse((home/f'MULTI/Dual/Transformer/{suite}').exists())
                self.assertEqual(sum(line.startswith('PLAN ') for line in preview.stdout.splitlines()),count)
                subprocess.run([bash,str(local),suite,study,'--submit'],env=env,capture_output=True,text=True,check=True)
                lines=(home/f'MULTI/Dual/Transformer/{suite}/jobs.tsv').read_text().splitlines()[1:]
                self.assertEqual(len(lines),count+1)
                rows=[r.split('\t') for r in lines];gate=rows[0][2]
                commands={r[0]:r[1:] for line in calls.read_text().splitlines() if (r:=line.rstrip('\t').split('\t'))}
                self.assertIn('8:0:0',commands[gate])
                self.assertEqual(commands[gate][commands[gate].index('--loss-curve-source')+1],source)
                for row in rows[1:]:
                    command=commands[row[2]]
                    self.assertIn('240:0:0',command)
                    self.assertIn('--kill-on-invalid-dep=yes',command)
                    self.assertNotIn('--nsims',command)
                    cli=command[command.index('DualOutputs/A0-HPC-Dual-test.py')+1:]
                    anchor_index=cli.index('--base-model-json');del cli[anchor_index:anchor_index+2]
                    parsed=parse(cli)
                    self.assertEqual(parsed.seed,int(row[1]))
                for seed in ('42','43','44'):
                    selected={r[0]:r for r in rows if r[1]==seed and r[0]!='preflight'}
                    if study=='inputs':
                        self.assertEqual(selected['correction'][3],gate)
                        self.assertTrue(all(selected[s][3]==selected['correction'][2] for s in ('true','mixed','corrected','corrected_mixed')))
                    else:
                        for variant in ('spatial','temporal','both','sudden'):
                            upstream='correction-'+variant if source.startswith('corrected') else 'bank-'+variant
                            if reuse:
                                self.assertNotIn(upstream,selected)
                                vi=('spatial','temporal','both','sudden').index(variant)
                                self.assertEqual(selected[variant][3],gate+(f':{808+vi}' if seed=='44' else ''))
                            else:
                                self.assertEqual(selected[variant][3],selected[upstream][2])
                            fit=commands[selected[variant][2]]
                            self.assertEqual(fit[fit.index('--curve-training-source')+1],source)
                            if reuse:
                                self.assertNotIn('--correction-bank',fit)
                                self.assertIn(f'reused-losses-bank-{variant}-s{seed}',' '.join(fit))
                                self.assertIn(f'winning-inputs-mixed-s{seed}/model.json',' '.join(fit))
                            if source.startswith('corrected'):
                                correction=selected['correction-'+variant]
                                self.assertEqual(correction[3],selected['bank-'+variant][2])
                                self.assertIn(f'winning-inputs-corrected_mixed-s{seed}/model.json',' '.join(commands[correction[2]]))
                                self.assertIn(f'losses-CM-correction-{variant}-s{seed}',' '.join(fit))
                        self.assertEqual(selected['combined_soft'][3],gate)
                        for loss in ('combined_soft','combined_no_location'):
                            fit=commands[selected[loss][2]]
                            self.assertNotIn('--field-loss-variant',fit)
                            self.assertIn(f'independence-261013-r1-bank-1i-s{seed}',' '.join(fit))
                            if source.startswith('corrected'):
                                self.assertIn(f'winning-inputs-correction-s{seed}',' '.join(fit))
                if reuse:
                    reused_rows=(home/f'MULTI/Dual/Transformer/{suite}/reused-banks.tsv').read_text().splitlines()
                    self.assertEqual(len(reused_rows),13)
                    self.assertTrue(all('--prepare-field-bank' not in commands[r[2]] for r in rows[1:]))

    def test_cm_preflight_routes_each_loss_through_selected_source(self):
        preflight=runpy.run_path(str(Path(__file__).with_name('A0-HPC-Dual-preflight.py')))['main']
        parse=runpy.run_path(str(Path(__file__).with_name('A0-HPC-Dual-test.py')))['parse_args']
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); calls=[]
            def runner(argv):
                anchor_index=argv.index('--base-model-json');argv=argv[:anchor_index]+argv[anchor_index+2:]
                args=parse(argv);calls.append(args)
                dest=root/args.run_label;dest.mkdir();(dest/'results').mkdir()
                (dest/'results/true_field_curves.npz').touch()
                return SimpleNamespace(model_file=str(dest/'model.mdl'),results_dir=str(dest/'results'),data=None)
            with mock.patch('sys.argv',['preflight','--base-model-json','unused','--run-label','gate',
                                        '--followup-suite','losses','--loss-curve-source','corrected_mixed']), \
                 mock.patch('runpy.run_path',return_value={'main':runner}), \
                 mock.patch('resources.MLmetrics.postprocess_load_dual_run'), contextlib.redirect_stdout(io.StringIO()):
                preflight()
            self.assertEqual(len(calls),18)
            for args in calls:
                self.assertEqual((args.nsims,args.epochs,args.seed,args.split_seed),(64,1,42,42))
                self.assertEqual(args.independence_case,'1i')
            for variant in ('spatial','temporal','both','sudden'):
                selected=[a for a in calls if a.field_loss_variant==variant]
                self.assertEqual(len(selected),3)
                bank,correction,fit=selected
                self.assertTrue(bank.prepare_field_bank)
                self.assertTrue(correction.prepare_correction_bank)
                self.assertEqual(correction.field_bank,str(root/('gate-bank-'+variant)))
                self.assertEqual(fit.correction_bank,str(root/('gate-correction-'+variant)))
                self.assertEqual(fit.curve_training_source,'corrected_mixed')
                self.assertEqual(fit.reference_curve_source,'corrected_mixed')
                self.assertEqual(fit.loss,'mse')
            for args in (a for a in calls if a.curve_loss_ablation):
                self.assertEqual(args.field_bank,str(root/'gate-bank'))
                self.assertEqual(args.correction_bank,str(root/'gate-input-correction'))
                self.assertEqual(args.curve_training_source,'corrected_mixed')
                self.assertEqual(Path(args.reference_model_json),root/'gate-input-corrected_mixed/model.json')


if __name__=='__main__':
    unittest.main()
