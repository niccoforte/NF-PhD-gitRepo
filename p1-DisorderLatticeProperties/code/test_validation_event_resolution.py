import json
from pathlib import Path
import tempfile
import unittest

from validation_controls import prepare_event_resolution


class EventResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source.inp'
        self.model = '*Heading\n*Amplitude,name=Amp-1\n0.,0.,.1,15.2\n*Material,name=Ti\n*Elastic\n135165.,.428571\n'
        self.source.write_text(self.model+'*Step,name=Step-1,nlgeom=YES\n*Dynamic, Explicit\n, .1\n'
            '*Boundary,amplitude=Amp-1\nload,2,2,1.\n*Output, field\n*Element Output\nLE,STATUS\n'
            '*Output, history\n*Element Output,elset=tip\nSTATUS\n*Node Output, nset=load\nU2,RF2\n*End Step\n')

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_physics_and_original_amplitude(self):
        before = self.source.read_bytes()
        rec = prepare_event_resolution(self.source,self.root/'control')
        text = Path(rec['input']).read_text()
        self.assertTrue(text.startswith(self.model))
        self.assertIn('*Dynamic, Explicit\n, 0.045',text)
        self.assertIn('*Boundary,amplitude=Amp-1\nload,2,2,1.',text)
        self.assertIn('*Output, history, time interval=1e-05\n*Node Output, nset=load',text)
        self.assertEqual(text.count('time interval=0.0005'),2)
        self.assertEqual(self.source.read_bytes(),before)
        self.assertEqual(json.loads(Path(rec['manifest']).read_text())['jobs'][0]['cpus'],8)

    def test_rejects_new_loading_duration(self):
        with self.assertRaises(ValueError):
            prepare_event_resolution(self.source,self.root/'bad',stop_time=.2)

    def test_refuses_mutating_prepared_control(self):
        prepare_event_resolution(self.source,self.root/'same')
        with self.assertRaises(ValueError):
            prepare_event_resolution(self.source,self.root/'same',stop_time=.04)


if __name__ == '__main__':
    unittest.main()
