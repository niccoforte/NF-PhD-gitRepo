import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from validation_stiffness import archive_filter, initial_work_stiffness, qualify_stiffness
import validation_stiffness as stiffness


class InitialWorkStiffnessTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0., .1, 201)
        self.x = np.linspace(0., .01, 201)

    def test_linear_energy_returns_exact_slope_with_nonuniform_coordinates(self):
        x = self.x**1.2
        result = initial_work_stiffness(x, 4200*x)
        self.assertAlmostEqual(result['estimate'], 4200., places=9)
        self.assertAlmostEqual(result['secant_energy_estimate'], 4200., places=9)

    def test_known_baseline_offsets_do_not_change_slope(self):
        result = initial_work_stiffness(self.x+7, 5700*self.x+43, baseline_response=43.)
        self.assertAlmostEqual(result['estimate'], 5700., places=6)

    def test_paired_filter_preserves_linear_relation_and_native_length(self):
        (x, y), info = archive_filter(self.time, self.x, 3300*self.x, 10000)
        np.testing.assert_allclose(y, 3300*x, rtol=1e-12, atol=1e-10)
        self.assertEqual(len(x), len(self.x))
        self.assertAlmostEqual(info['cutoff_Hz'], 2000/6)
        self.assertTrue(info['capped'])
        json.dumps(info)

    def test_lower_physical_cutoff_is_preserved(self):
        _, info = archive_filter(self.time, self.x, self.x, 70.)
        self.assertAlmostEqual(info['cutoff_Hz'], 140.)
        self.assertFalse(info['capped'])

    def test_sampling_gap_is_not_interpolated_into_observations(self):
        time = self.time.copy()
        time[100:] += .01
        with self.assertRaises(ValueError):
            archive_filter(time, self.x, self.x, 200.)

    def test_high_frequency_oscillation_has_bounded_modulus_error(self):
        y = 5000*self.x+2*np.sin(2*np.pi*700*self.time)
        result = qualify_stiffness(self.time, self.x, y, 10000.)
        self.assertLess(abs(result['estimate']/5000-1), .03)

    def test_low_resolution_is_flagged_instead_of_certified(self):
        x = np.arange(4.)
        result = initial_work_stiffness(x, 20*x)
        self.assertEqual(result['status'], 'low_resolution_estimate')
        self.assertAlmostEqual(result['estimate'], 20.)

    def test_strong_curvature_cannot_pass_the_linearity_screen(self):
        result = qualify_stiffness(self.time, self.x, 100*self.x+100000*self.x**2, 10000.)
        self.assertEqual(result['quality'], 'review_required')
        self.assertTrue(result['flags'])

    def test_pointwise_oscillation_is_not_confused_with_systematic_curvature(self):
        # Work integration is meant to suppress alternating reaction noise.
        y=5000*self.x+3*np.sin(2*np.pi*300*self.time)
        result=qualify_stiffness(self.time,self.x,y,159.)
        self.assertLess(result['energy_fit_residual'],.03)
        self.assertNotIn('initial_linearity_residual_over5percent',result['flags'])

    def test_actual_initial_reaction_noise_is_not_subtracted_as_a_preload(self):
        y=5000*self.x;y[0]=-.15
        result=initial_work_stiffness(self.x,y)
        self.assertAlmostEqual(result['estimate'],5000.,places=7)
        self.assertEqual(result['baseline_y'],0.)


class HoldoutImmutabilityTests(unittest.TestCase):
    def test_reprepare_preserves_evidence_and_late_conflict_prevents_all_writes(self):
        identities=[('full',21)]+[('mechanism',i)for i in(219,274,643,670,698,699,782,901)]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            study=root/'audit/curve-processing/accepted-filter-study'
            study.mkdir(parents=True)
            source=root/'original.inp';source.write_text('*Heading\n')
            rows=[dict(design=d,id=i,stiffness=dict(quality='review_required',estimate=1500.+i,
                                                 flags=['window_sensitive_over5percent']))for d,i in identities]
            replay=study/'hex-population-readonly-qualification.json'
            replay.write_text(json.dumps(dict(records=rows)))
            def render(source,out,name,travel):
                target=Path(out)/name/(name+'.inp');target.parent.mkdir(parents=True)
                target.write_text('*Heading\n'+name+'\n')
                return dict(job=name,input=str(target),input_sha256=stiffness.sha(target),
                            source_inp=str(source),source_sha256=stiffness.sha(source),travel_mm=travel)
            with patch.object(stiffness,'_holdout_sources',return_value=[source]), \
                 patch('validation_controls.prepare_uat_linear',side_effect=render):
                manifest=stiffness.prepare_holdout_controls(root)
                out=study/'holdout-controls'
                snapshot={p:(p.read_bytes(),p.stat().st_mtime_ns)for p in out.rglob('*')if p.is_file()}
                stiffness.prepare_holdout_controls(root)
                self.assertEqual(snapshot,{p:(p.read_bytes(),p.stat().st_mtime_ns)for p in snapshot})
                # A missing early input must not be recreated before a later
                # changed frozen prediction/source replay has been rejected.
                missing=Path(manifest['jobs'][0]['input']);missing.unlink()
                rows[-1]['stiffness']['estimate']+=1
                replay.write_text(json.dumps(dict(records=rows)))
                with self.assertRaisesRegex(ValueError,'different existing holdout evidence'):
                    stiffness.prepare_holdout_controls(root)
                self.assertFalse(missing.exists())
                self.assertEqual({p:v for p,v in snapshot.items()if p!=missing},
                                 {p:(p.read_bytes(),p.stat().st_mtime_ns)for p in snapshot if p!=missing})

    def test_complete_set_preflight_rejects_before_creating_missing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);first=root/'new.txt';later=root/'existing.txt'
            later.write_text('original')
            with self.assertRaisesRegex(ValueError,'different existing holdout evidence'):
                stiffness._write_immutable_targets({first:'new',later:'changed'})
            self.assertFalse(first.exists())
            self.assertEqual(later.read_text(),'original')


class FrozenHoldoutComparisonTests(unittest.TestCase):
    def fixture(self,job,estimate,reference):
        prediction=dict(job=job,design='mechanism',id=int(job),source_inp_sha256='same-source-'+job,
                        frozen_curve_estimate_MPa=estimate,quality_flags=['window_sensitive_over5percent'])
        control=dict(job=job,source_sha256=prediction['source_inp_sha256'],state='completed_and_extracted',
                     read_only=True,E_apparent_MPa=reference,force_balance_relative=0.,work_energy_relative=0.,
                     clamp_work_energy_relative=0.)
        return prediction,control

    def test_reports_every_error_and_does_not_accept_when_one_holdout_fails(self):
        a,ca=self.fixture('1',1500.,1500.)
        b,cb=self.fixture('2',1400.,1500.)
        report=stiffness.evaluate_holdouts([a,b],[ca,cb])
        self.assertEqual(report['holdouts'],2)
        self.assertEqual([r['frozen_curve_estimate_MPa']for r in report['records']],[1500.,1400.])
        self.assertFalse(report['all_holdouts_within_target'])
        self.assertAlmostEqual(report['max_absolute_error_percent'],100/15)
        self.assertEqual(report['verdict'],'retain_sensitivity_flags_independent_holdout_target_not_met')
        self.assertEqual(b['frozen_curve_estimate_MPa'],1400.)

    def test_missing_control_cannot_be_silently_omitted(self):
        a,ca=self.fixture('1',1500.,1500.);b,_=self.fixture('2',1500.,1500.)
        with self.assertRaisesRegex(ValueError,'Complete frozen holdout set'):
            stiffness.evaluate_holdouts([a,b],[ca])

    def test_source_mismatch_is_rejected(self):
        prediction,control=self.fixture('1',1500.,1500.);control['source_sha256']='other-source'
        with self.assertRaisesRegex(ValueError,'source differs'):
            stiffness.evaluate_holdouts([prediction],[control])

    def test_exact_five_percent_is_inside_fixed_acceptance(self):
        prediction,control=self.fixture('1',105.,100.)
        report=stiffness.evaluate_holdouts([prediction],[control])
        self.assertTrue(report['all_holdouts_within_target'])
        self.assertEqual(report['acceptance_tolerance_percent'],5.)


class AcceptedHexStiffnessTests(unittest.TestCase):
    def fixture(self):
        rows=[];predictions=[];controls=[]
        for identity in range(1,10):
            job='holdout'+str(identity);estimate=1500.+identity
            rows.append(dict(design='mechanism',id=identity,path='raw'+str(identity),source_sha256='raw-hash',
                             stiffness=dict(quality='review_required',estimate=estimate,flags=['window_sensitive_over5percent'],
                                            window_relative_spread=.055,cutoff_relative_spread=.01)))
            predictions.append(dict(design='mechanism',id=identity,job=job,source_inp_sha256='source-hash',
                                    frozen_curve_estimate_MPa=estimate))
            # Include agreeing and disagreeing FE values. Selection must depend
            # on the prior curve flag, never on this subsequent error.
            reference=estimate if identity%2 else estimate+200.
            controls.append(dict(job=job,source_sha256='source-hash',E_apparent_MPa=reference,source_inp='source.inp',
                                 input_sha256='input-hash',odb_sha256='odb-hash',delivery_sha256='delivery-hash'))
        rows.append(dict(design='full',id=99,path='raw99',source_sha256='raw-hash99',
                         stiffness=dict(quality='passes_frozen_numerical_screens',estimate=1400.,flags=[],
                                        window_relative_spread=.01,cutoff_relative_spread=.005)))
        return dict(records=rows),dict(predictions=predictions),controls

    def test_all_nine_preflagged_cases_use_controls_regardless_of_holdout_error(self):
        population,freeze,controls=self.fixture();before=json.dumps(population,sort_keys=True)
        rows=stiffness.select_accepted_hex_stiffness(population,freeze,controls,{r['job']for r in controls})
        self.assertEqual(sum(r['estimator']=='source_exact_linear_UAT_preflagged_fallback'for r in rows),9)
        self.assertEqual([r['E_apparent_MPa']for r in rows[:9]],[r['E_apparent_MPa']for r in controls])
        self.assertEqual(rows[-1]['estimator'],'filtered_initial_work_curve')
        self.assertEqual(rows[-1]['E_apparent_MPa'],1400.)
        self.assertEqual(json.dumps(population,sort_keys=True),before)

    def test_missing_or_unverified_fallback_blocks_the_whole_catalogue(self):
        population,freeze,controls=self.fixture();verified={r['job']for r in controls}
        with self.assertRaisesRegex(ValueError,'Missing or unverified'):
            stiffness.select_accepted_hex_stiffness(population,freeze,controls[:-1],verified)
        with self.assertRaisesRegex(ValueError,'Missing or unverified'):
            stiffness.select_accepted_hex_stiffness(population,freeze,controls,verified-{'holdout9'})


class UniformSamplingComparisonTests(unittest.TestCase):
    def test_uniform_sampling_rule_does_not_change_the_frozen_physical_rule(self):
        time=np.linspace(0,.1,201);x=np.linspace(0,.01,201);y=5000*x
        (ux,uy),uniform=stiffness.uniform_fs6_curve(time,x,y)
        _,frozen=stiffness.archive_filter(time,x,y,159.08)
        self.assertAlmostEqual(uniform['cutoff_Hz'],2000/6)
        self.assertAlmostEqual(frozen['cutoff_Hz'],318.16)
        np.testing.assert_allclose(uy,5000*ux,atol=1e-10)
        self.assertEqual(len(ux),201)

    def test_unobserved_thresholds_remain_censored(self):
        time=np.arange(6.);x=np.arange(6.);y=np.arange(6.)
        result=stiffness.response_endpoint_metrics(time,x,y)
        self.assertFalse(result['ductility25_observed'])
        self.assertFalse(result['work1_observed'])
        self.assertIsNone(result['ductility25_coordinate'])
        self.assertIsNone(result['work1_work'])

    def test_selected_common_rule_caps_fast_modes_and_never_raises_slow_modes(self):
        time=np.linspace(0,.1,201);x=np.linspace(0,.01,201);y=5000*x
        for frequency,expected in((52.73,105.46),(159.08,318.16),(323.8,2000/6),(3237.4,2000/6)):
            with self.subTest(frequency=frequency):
                (fx,fy),info=stiffness.final_archive_filter(time,x,y,frequency)
                self.assertAlmostEqual(info['cutoff_Hz'],expected)
                self.assertLessEqual(info['cutoff_Hz'],2*frequency)
                np.testing.assert_allclose(fy,5000*fx,atol=1e-10)


class SelectedWorkflowCLITests(unittest.TestCase):
    def test_default_cli_routes_to_selected_policy_not_old_conditional_run(self):
        import contextlib
        import io
        result=dict(selected_policy='fc=min(2fn,fs/6)',native_histories=64,changed_histories=8,stiffness_windows_passed=64)
        with patch.object(stiffness,'compare_capped_policy',return_value=result) as selected, \
             patch.object(stiffness,'run',side_effect=AssertionError('Historical rule cannot be the default')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stiffness.main(['--root','Z:/custom-validation']),result)
        selected.assert_called_once_with(Path('Z:/custom-validation'))


if __name__ == '__main__':
    unittest.main()
