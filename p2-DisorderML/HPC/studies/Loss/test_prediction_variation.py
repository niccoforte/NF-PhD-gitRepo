"""Distinguish prediction spread from accuracy and keep report placement stable."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from resources.MLstudy_reports import insert_variation_section


class VariationTests(unittest.TestCase):
    def test_collapse_full_spread_and_idempotent_placement(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);y=np.arange(12,dtype=float).reshape(3,4)
            entries=[]
            for variant,p in (("constant",np.broadcast_to(y.mean(0),y.shape)),("wrong_order",y[::-1])):
                path=root/variant;(path/"results").mkdir(parents=True)
                np.savez(path/"results/predictions.npz",sample_ids=np.array(["a","b","c"]),UT_val_curve_truth=y,UT_val_curve_outputs=p)
                entries.append(dict(variant=variant,path=path))
            report=root/"README.md";report.write_text('# Report\n\n## Global performance versus TRAIN mean\n\nMean plot.\n\n## Local performance\n\nKeep.\n',encoding='utf-8')
            insert_variation_section(report,entries);insert_variation_section(report,entries)
            text=report.read_text(encoding='utf-8');self.assertEqual(text.count('## Prediction variation and collapse'),1)
            self.assertLess(text.index('## Prediction variation'),text.index('## Local performance'))
            values=pd.read_csv(root/'prediction-variation.csv').set_index('variant')
            self.assertEqual(values.loc['constant','ratio'],0.)
            self.assertAlmostEqual(values.loc['wrong_order','ratio'],1.)

    def test_refuses_mismatched_specimen_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);entries=[]
            for i in range(2):
                path=root/str(i);(path/'results').mkdir(parents=True)
                np.savez(path/'results/predictions.npz',sample_ids=np.array(['a',str(i)]),UT_val_curve_truth=np.array([[0.,0.],[1.,1.]]),UT_val_curve_outputs=np.array([[0.,0.],[1.,1.]]))
                entries.append(dict(variant=str(i),path=path))
            report=root/'README.md';report.write_text('## Global performance versus TRAIN mean\n\n',encoding='utf-8')
            with self.assertRaises(ValueError):insert_variation_section(report,entries)


if __name__=='__main__':unittest.main()
