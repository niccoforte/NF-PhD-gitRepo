"""Half-model compliance/energy normalisation; no solvers or research files."""
import unittest
import json
import tempfile
from pathlib import Path

from validation_elastic_calibration import digest, elastic_factor, verified_delivery


class HalfCTCalibration(unittest.TestCase):
    def test_smaller_crack_increment_preserves_an_affine_compliance_derivative(self):
        ratios=(.495,.4975,.5,.5025,.505)
        compliance=[.002+.0001*20*r for r in ratios]
        result=elastic_factor(compliance,20.,2.,10000.,crack_ratios=ratios)
        self.assertAlmostEqual(result['dC_half_da_fine_per_N'],.0001)
        self.assertAlmostEqual(result['f_fine'],(10000.*2.*20.*.0001)**.5)

    def test_half_opening_derivative_includes_the_symmetry_factor(self):
        W, B, modulus, slope = 20., 2., 10000., .0001
        compliance = [.002+slope*W*r for r in (.49,.495,.5,.505,.51)]
        value = elastic_factor(compliance,W,B,modulus)
        self.assertAlmostEqual(value["J_fine_N_per_mm"],slope/B)
        self.assertAlmostEqual(value["f_fine"],(modulus*B*W*slope)**.5)

    def test_load_scales_J_but_not_the_geometry_factor(self):
        compliance = [.002+.0001*20*r for r in (.49,.495,.5,.505,.51)]
        one = elastic_factor(compliance,20.,2.,10000.,1.)
        two = elastic_factor(compliance,20.,2.,10000.,2.)
        self.assertAlmostEqual(two["J_fine_N_per_mm"],4*one["J_fine_N_per_mm"])
        self.assertEqual(one["f_fine"],two["f_fine"])

    def test_nonfinite_or_nonincreasing_compliance_is_rejected(self):
        for values in ([.1]*5,[.1,.2,float("nan"),.4,.5],[.5,.4,.3,.2,.1]):
            with self.assertRaises(ValueError):
                elastic_factor(values,20.,1.,10000.)

    def test_nonfinite_dimensions_are_rejected(self):
        compliance=[.1,.2,.3,.4,.5]
        for bad in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                elastic_factor(compliance,bad,1.,10000.)

    def delivery_fixture(self, out):
        prepared=out/'fixture.inp'
        prepared.write_text('Synthetic test input, no research content')
        job={'job':'fixture','input':str(prepared),'input_sha256':digest(prepared)}
        (out/'source.json').write_text(json.dumps(job))
        queue=out/'queue'
        run=queue/'runs/fixture'
        run.mkdir(parents=True)
        files=[]
        for suffix in ('.inp','.odb','.dat'):
            artifact=run/('fixture'+suffix)
            artifact.write_bytes(prepared.read_bytes() if suffix=='.inp' else b'Synthetic test artifact')
            files.append({'file':artifact.name,'sha256':digest(artifact)})
        (run/'delivery.json').write_text(json.dumps({'job':'fixture','files':files,'required_outputs_missing':[]}))
        manifest={'queue_root':str(queue),'jobs':[job]}
        (out/'manifest.json').write_text(json.dumps(manifest))
        (queue/'state.json').write_text(json.dumps({'manifest_sha256':digest(out/'manifest.json'),
                                                  'jobs':[{'job':'fixture','status':'completed'}]}))
        return manifest,job,run

    def test_completed_delivery_rejects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            manifest,job,run=self.delivery_fixture(out)
            source,odb,sha,receipt=verified_delivery(out,manifest,job)
            self.assertEqual(sha,digest(odb))
            self.assertEqual(receipt['queue_recorded_status'],'completed')
            (run/'fixture.odb').write_text('Tampered test artifact')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                verified_delivery(out,manifest,job)

    def test_existing_odb_does_not_qualify_an_incomplete_job(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            manifest,job,run=self.delivery_fixture(out)
            state_path=out/'queue/state.json'
            state=json.loads(state_path.read_text())
            state['jobs'][0]['status']='running'
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'not a completed'):
                verified_delivery(out,manifest,job)

    def test_failed_job_requires_bound_review_and_keeps_failed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            manifest,job,run=self.delivery_fixture(out)
            state_path=out/'queue/state.json'
            state=json.loads(state_path.read_text())
            state['jobs'][0]['status']='failed'
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'explicit hash-bound'):
                verified_delivery(out,manifest,job)
            review={'job':job['job'],'state':'accepted_control_only','queue_recorded_status':'failed',
                    'accepted_for_calculation':True,'input_sha256':job['input_sha256'],
                    'queue_state_sha256':digest(state_path),'delivery_sha256':digest(run/'delivery.json'),
                    'source_odb_sha256':digest(run/'fixture.odb'),
                    'native_validation':{'all_expected_equilibrium_steps_verified':True},
                    'reviewed_file_sha256':{'fixture.odb':digest(run/'fixture.odb')}}
            (out/'queue/reviewed-wrapper-failure.json').write_text(json.dumps(review))
            receipt=verified_delivery(out,manifest,job)[3]
            self.assertEqual(receipt['queue_recorded_status'],'failed')
            self.assertTrue(receipt['calculation_usable'])
            review['queue_state_sha256']='wrong'
            (out/'queue/reviewed-wrapper-failure.json').write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,'identity/status changed'):
                verified_delivery(out,manifest,job)


if __name__ == "__main__":
    unittest.main()
