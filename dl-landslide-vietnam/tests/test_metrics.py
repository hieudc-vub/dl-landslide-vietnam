import unittest
import numpy as np
from landslide.config import DEFAULTS
from landslide.metrics import components, confusion, pixel_metrics, object_metrics, boundary_metrics, aggregate_tiles, tile_overlap


class MetricTests(unittest.TestCase):
    def test_hd95_is_directional_maximum(self):
        from scipy import ndimage as ndi
        gt = np.zeros((64,64), bool); pred = gt.copy()
        gt[20:40,20:40] = True
        pred[20:40,20:40] = True; pred[2:8,2:8] = True
        p = pred ^ ndi.binary_erosion(pred, structure=np.ones((3,3)), border_value=0)
        g = gt ^ ndi.binary_erosion(gt, structure=np.ones((3,3)), border_value=0)
        a = ndi.distance_transform_edt(~g, sampling=10)[p]
        b = ndi.distance_transform_edt(~p, sampling=10)[g]
        expected = max(np.percentile(a,95), np.percentile(b,95))
        self.assertNotAlmostEqual(expected, np.percentile(np.concatenate([a,b]),95))
        self.assertAlmostEqual(boundary_metrics(pred,gt,DEFAULTS)['hd95_m'], expected)

    def test_test_population_is_not_val_filter(self):
        # The first tile has only ONE reference pixel and must still be included.
        good = dict(gt_pixels=1, pred_pixels=1, iou=1., dice=1., boundary_f1=1., assd_m=0., hd95_m=0.)
        missed = dict(gt_pixels=50, pred_pixels=0, iou=0., dice=0., boundary_f1=np.nan, assd_m=np.nan, hd95_m=np.nan)
        background = dict(good, gt_pixels=0, pred_pixels=2)
        result = aggregate_tiles([good, missed, background])
        self.assertEqual(result['n_reference_positive'], 2)
        self.assertEqual(result['n_overlap_eligible'], 1)
        self.assertEqual(result['overlap_conditional']['iou']['mean'], 1.)
        self.assertEqual(result['empty_prediction_rate'], .5)

    def test_overlap_smoothing(self):
        c = dict(tp=0, fp=0, fn=1, tn=0)
        self.assertAlmostEqual(tile_overlap(c)['iou'], 1e-6/(1+1e-6))

    def test_diagonal_connectivity(self):
        a = np.eye(2, dtype=bool)
        self.assertEqual(components(a, 8)[1], 1)
        self.assertEqual(components(a, 4)[1], 2)

    def test_exact_iou_threshold_is_inclusive(self):
        gt = np.zeros((8, 12), bool)
        pred = gt.copy()
        gt[2, 1:11] = True
        pred[2, 1:4] = True  # Exactly 3/10, must match at >= 0.30.
        result, _ = object_metrics(pred, gt, DEFAULTS)
        self.assertEqual(result, dict(obj_tp=1, obj_fp=0, obj_fn=0))

    def test_one_prediction_cannot_match_two_objects(self):
        gt = np.zeros((8, 12), bool)
        pred = gt.copy()
        gt[3, 1:4] = True
        gt[3, 6:9] = True
        pred[3, 1:9] = True
        result, _ = object_metrics(pred, gt, DEFAULTS)
        self.assertEqual(result, dict(obj_tp=1, obj_fp=0, obj_fn=1))

    def test_size_bin_edges(self):
        gt = np.zeros((20, 80), bool)
        gt[1, 1:11] = True  # 1000 m2
        gt[4, 1:51] = True  # 5000 m2
        gt[8, 1:52] = True
        _, rows = object_metrics(gt, gt, DEFAULTS)
        self.assertEqual([r['n_gt'] for r in rows], [1, 1, 1])

    def test_confusion_and_empty_prediction(self):
        gt = np.array([[1, 0], [1, 0]], bool)
        pred = np.array([[1, 1], [0, 0]], bool)
        c = confusion(pred, gt)
        self.assertEqual(c, dict(tp=1, fp=1, fn=1, tn=1))
        self.assertAlmostEqual(pixel_metrics(c)['iou'], 1/3)
        self.assertEqual(pixel_metrics(confusion(np.zeros_like(gt), gt))['iou'], 0)

    def test_boundaries_and_empty_undefined(self):
        gt = np.zeros((10, 10), bool)
        gt[3:6, 3:6] = True
        b = boundary_metrics(gt, gt, DEFAULTS)
        self.assertEqual(b, dict(boundary_f1=1, assd_m=0, hd95_m=0))
        self.assertTrue(np.isnan(boundary_metrics(np.zeros_like(gt), gt, DEFAULTS)['hd95_m']))


if __name__ == '__main__':
    unittest.main()
