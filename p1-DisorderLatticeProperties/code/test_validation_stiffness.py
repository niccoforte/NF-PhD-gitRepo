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


if __name__ == '__main__':
    unittest.main()
