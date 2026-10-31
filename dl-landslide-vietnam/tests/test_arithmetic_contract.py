"""Independent small examples for arithmetic, serialization and configuration."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import math

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from landslide.config import DEFAULTS, BANDS, load, validate
from landslide.data import PatchDataset
from landslide.metrics import confusion, pixel_metrics, object_metrics, aggregate_tiles
from landslide.reporting import export_tables
from landslide.io import write_json, write_csv
from landslide.models import FocalTverskyLoss


class ArithmeticContractTests(unittest.TestCase):
    def test_loss_matches_small_probability_example(self):
        # p=[0.8,0.2], y=[1,0]: TP=0.8, FP=FN=0.2, TI=0.8.
        p = torch.tensor([[[[0.8, 0.2]]]], dtype=torch.float32)
        y = torch.tensor([[[[1., 0.]]]])
        expected_focal = -0.5 * (0.2**2) * math.log(0.8)
        expected = 0.45*expected_focal + 0.55*(1-0.8)
        actual = FocalTverskyLoss(DEFAULTS)(torch.logit(p), y).item()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        # Without positive labels SMP suppresses the Tversky term.
        p = torch.full((1,1,1,2), 0.2)
        actual = FocalTverskyLoss(DEFAULTS)(torch.logit(p), torch.zeros_like(p)).item()
        np.testing.assert_allclose(actual, 0.45*expected_focal, rtol=1e-5, atol=1e-6)

    def test_hand_calculated_counts_and_scores(self):
        gt = np.array([1, 1, 1, 0, 0, 0], bool)
        pred = np.array([1, 1, 0, 1, 0, 0], bool)
        c = confusion(pred, gt)
        self.assertEqual(c, dict(tp=2, fp=1, fn=1, tn=2))
        self.assertEqual(sum(c.values()), gt.size)
        expected = dict(iou=1/2, dice=2/3, precision=2/3, recall=2/3,
                        f1=2/3, f2=2/3, fnr=1/3, fdr=1/3, accuracy=2/3, mcc=1/3)
        actual = pixel_metrics(c)
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)

    def test_nominal_area_bin_counts_and_matching(self):
        # At 10 m/pixel, disconnected objects occupy 1000, 5000 and 5100 m².
        gt = np.zeros((20, 70), bool)
        gt[1, 1:11] = True
        gt[5, 1:51] = True
        gt[9, 1:52] = True
        pred = gt.copy(); pred[5] = False
        counts, sizes = object_metrics(pred, gt, DEFAULTS)
        self.assertEqual(counts, dict(obj_tp=2, obj_fp=0, obj_fn=1))
        self.assertEqual([x['n_gt'] for x in sizes], [1, 1, 1])
        self.assertEqual([x['n_detected'] for x in sizes], [1, 0, 1])
        self.assertEqual(sum(x['n_gt'] for x in sizes), counts['obj_tp']+counts['obj_fn'])

    def test_undefined_values_remain_missing_after_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            values = dict(undefined=float('nan'), zero=0, count=3)
            write_json(p/'a.json', values); write_csv(p/'a.csv', [values])
            self.assertEqual(json.loads((p/'a.json').read_text()), dict(undefined=None, zero=0, count=3))
            with (p/'a.csv').open(newline='') as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row, dict(undefined='', zero='0', count='3'))

    def test_exported_population_counts_and_means(self):
        rows = [dict(gt_pixels=1, pred_pixels=1, iou=0.5, dice=2/3,
                     boundary_f1=1, assd_m=0, hd95_m=0),
                dict(gt_pixels=50, pred_pixels=0, iou=0, dice=0,
                     boundary_f1=float('nan'), assd_m=float('nan'), hd95_m=float('nan'))]
        totals = aggregate_tiles(rows)
        self.assertEqual(totals['n_reference_positive'], 2)
        self.assertEqual(totals['n_overlap_eligible']+totals['n_empty_predictions_on_positive'], 2)
        summary = dict(model='synthetic', threshold=0.5, pixel_pooled=pixel_metrics(dict(tp=1,fp=1,fn=50,tn=2)),
                       object_precision=0.5, object_recall=0.5, object_f1=0.5, **totals)
        with tempfile.TemporaryDirectory() as tmp:
            export_tables(tmp, summary, [dict(bin='small',n_gt=2,n_detected=1,recall=0.5)])
            with (Path(tmp)/'Table_MainResults.csv').open(newline='') as f:
                main = next(csv.DictReader(f))
            self.assertEqual(int(main['n_reference_positive']), 2)
            self.assertEqual(int(main['n_empty_predictions']), 1)
            np.testing.assert_allclose([float(main['iou']),float(main['empty_prediction_rate'])],
                                       [0.5,0.5], rtol=1e-12, atol=1e-12)

    def test_scale_must_be_explicit_finite_positive(self):
        cfg = DEFAULTS | dict(name='synthetic', arch='unet', encoder='resnet50')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            missing = cfg.copy(); del missing['pixel_size_m']
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, 'Explicit pixel_size_m'):
                load(path)
        for value in [0, -1, float('nan'), float('inf'), None, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(cfg | dict(pixel_size_m=value))

    def test_paired_grid_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind, count in [('STACK',20),('LABEL',1)]:
                folder = Path(tmp)/'TRAIN_256'/kind; folder.mkdir(parents=True)
                with rasterio.open(folder/'a.tif','w',driver='GTiff',width=256,height=256,
                                   count=count,dtype='float32',
                                   transform=from_origin(100 if kind=='STACK' else 101,100,10,10)) as dst:
                    dst.write(np.zeros((count,256,256),np.float32))
                    if count==20: dst.descriptions=tuple(BANDS)
            with self.assertRaisesRegex(ValueError,'spatial metadata'):
                PatchDataset(tmp,'train')[0]


if __name__ == '__main__':
    unittest.main()
