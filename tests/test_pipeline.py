import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from landslide.config import BANDS, DEFAULTS, load
from landslide.data import PatchDataset
from landslide.engine import load_model, read_threshold, threshold_sweep
from landslide.io import sha256, write_json
from landslide.models import LandslideModel, FocalTverskyLoss


class PipelineTests(unittest.TestCase):
    def test_registry_has_13_configs(self):
        paths = list((Path(__file__).resolve().parents[1] / 'configs/models').glob('*.json'))
        self.assertEqual(len(paths), 13)
        for path in paths:
            c = load(path)
            self.assertEqual((c['lr'], c['patience'], c['connectivity']), (1e-4, 20, 8))

    def test_pair_validation_and_band_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'TRAIN_256'
            for kind in ['STACK', 'LABEL']:
                (base/kind).mkdir(parents=True)
                count = 20 if kind == 'STACK' else 1
                with rasterio.open(base/kind/'a.tif', 'w', driver='GTiff', width=256, height=256,
                                   count=count, dtype='float32' if count == 20 else 'uint8',
                                   transform=from_origin(100, 100, 1, 1)) as dst:
                    dst.write(np.zeros((count, 256, 256), dtype=dst.dtypes[0]))
                    if count == 20:
                        dst.descriptions = tuple(BANDS)
            ds = PatchDataset(tmp, 'train')
            x, y, name = ds[0]
            self.assertEqual(tuple(x.shape), (20, 256, 256))
            self.assertEqual(tuple(y.shape), (1, 256, 256))
            (base/'LABEL/a.tif').rename(base/'LABEL/b.tif')
            with self.assertRaises(ValueError):
                PatchDataset(tmp, 'train')

    def test_calibration_only_uses_supplied_val_examples(self):
        class Identity(nn.Module):
            def forward(self, x):
                return x, None
        probabilities = torch.tensor([[[[0.4, 0.6], [0.2, 0.1]]]])
        y = torch.tensor([[[[1., 1.], [0., 0.]]]])
        dl = DataLoader(TensorDataset(torch.logit(probabilities), y, torch.arange(1)), batch_size=1)
        cfg = DEFAULTS | dict(thresholds=[0.3, 0.5, 0.7], pos_pixel_min=0)
        best, _ = threshold_sweep(Identity(), dl, torch.device('cpu'), cfg, 'macro')
        self.assertEqual(best['threshold'], 0.3)

    def test_forward_backward_checkpoint_and_binding(self):
        torch.set_num_threads(2)
        cfg = DEFAULTS | dict(name='smoke', arch='deeplabv3plus', encoder='mobilenet_v2', encoder_weights=None)
        model = LandslideModel(cfg).eval()
        x = torch.randn(1, 20, 256, 256)
        logits, weights = model(x)
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        self.assertEqual(tuple(weights.shape), (1, 20))
        loss = FocalTverskyLoss(cfg)(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(model.input_attention.fc[0].weight.grad).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pth'
            torch.save(dict(format_version=1, config=cfg, model=model.state_dict()), path)
            restored, restored_cfg = load_model(path, torch.device('cpu'))
            with torch.inference_mode():
                actual, _ = restored(x)
            torch.testing.assert_close(actual, logits.detach(), rtol=1e-5, atol=1e-6)
            artifact = Path(tmp)/'threshold.json'
            write_json(artifact, dict(threshold=0.3, checkpoint_sha256=sha256(path), config=cfg))
            self.assertEqual(read_threshold(artifact, path, restored_cfg), 0.3)
            obj = json.loads(artifact.read_text())
            obj['checkpoint_sha256'] = 'wrong'
            write_json(artifact, obj)
            with self.assertRaises(ValueError):
                read_threshold(artifact, path, restored_cfg)


if __name__ == '__main__':
    unittest.main()
