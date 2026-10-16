"""Layout-only contracts: no fitted damage model or specimen join."""
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np

from resources.damage import _fcc_reference

layout = runpy.run_path(str(Path(__file__).with_name("A0-HPC-Damage-contract.py")))["consumer_layout"]


class DamageConsumerTests(unittest.TestCase):
    def fixture(self, task="FT"):
        xy, present, edges = _fcc_reference(task)
        time = np.arange(21)*.005
        a = dict(canonical_xy=xy, canonical_node_present=present, canonical_endpoint=edges,
                 frame_time=time, strut_failed=np.zeros((21,len(edges)),bool),
                 strut_failure_known=np.ones((21,len(edges)),bool))
        consumer = dict(metadata=dict(canonical_coords=xy[::-1].tolist(),
            field_frame_values={task:time[1:].tolist()}), node_masks={task:present[::-1].tolist()})
        return a, consumer

    def test_reorders_endpoints_and_presence_not_only_rows(self):
        a, c = self.fixture()
        result = layout(a,c,"FT")
        np.testing.assert_array_equal(result["endpoint_index"],799-a["canonical_endpoint"])
        self.assertEqual(result["failed"].shape,(20,2259))
        self.assertTrue(result["node_present"][result["endpoint_index"]].all())

    def test_absent_node_mask_mismatch_is_rejected(self):
        a,c = self.fixture()
        c["node_masks"]["FT"] = [True]*800
        with self.assertRaises(ValueError): layout(a,c,"FT")

    def test_unknown_is_not_relabelled_and_round_trip_preserves_it(self):
        a,c = self.fixture("UT")
        a["strut_failure_known"][4,8] = False
        result = layout(a,c,"UT")
        self.assertFalse(result["known"][3,8])
        self.assertTrue(np.isnan(np.where(result["known"],result["failed"],np.nan)[3,8]))
        with tempfile.TemporaryDirectory() as folder:
            np.savez(Path(folder)/"consumer.npz",**result)
            with np.load(Path(folder)/"consumer.npz",allow_pickle=False) as saved:
                for key,value in result.items(): np.testing.assert_array_equal(saved[key],value)

    def test_actual_times_retained_not_interpolated_to_consumer(self):
        a,c = self.fixture()
        a["frame_time"][4] += 1e-7
        result = layout(a,c,"FT")
        self.assertNotEqual(result["frame_time"][3],result["consumer_frame_time"][3])
        c["metadata"]["field_frame_values"]["FT"] = [0.]*19
        with self.assertRaises(ValueError): layout(a,c,"FT")


if __name__ == "__main__": unittest.main()
