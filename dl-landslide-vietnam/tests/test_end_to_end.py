"""Exercise all production stages on tiny synthetic patches, without downloads."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from landslide.config import DEFAULTS, BANDS
from landslide.engine import train, calibrate, evaluate, predict, benchmark
from landslide.reporting import summarize


class EndToEndTests(unittest.TestCase):
    def test_train_to_prediction_on_synthetic_data(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base/'data'
            rng = np.random.default_rng(42)
            for split in ['TRAIN_256', 'VAL_256', 'TEST_256']:
                for kind in ['STACK', 'LABEL']:
                    (data/split/kind).mkdir(parents=True)
                for i in range(2):
                    x = rng.uniform(-0.5, 0.5, (20, 256, 256)).astype('float32')
                    y = np.zeros((1, 256, 256), dtype='uint8')
                    y[:, 80:100, 90:110] = 1
                    for kind, values in [('STACK', x), ('LABEL', y)]:
                        with rasterio.open(data/split/kind/f'{i}.tif', 'w', driver='GTiff',
                            count=len(values), width=256, height=256, dtype=str(values.dtype),
                            transform=from_origin(100, 100, 1, 1)) as dst:
                            dst.write(values)
                            if kind == 'STACK':
                                dst.descriptions = tuple(BANDS)
            cfg = DEFAULTS | dict(name='synthetic', arch='deeplabv3plus', encoder='mobilenet_v2',
                encoder_weights=None, epochs=1, batch_size=2, thresholds=[0.3, 0.5])
            run = base/'run'
            device = torch.device('cpu')
            with redirect_stdout(io.StringIO()):
                train(cfg, data, run, device)
            self.assertTrue((run/'last_model.pth').is_file())
            cp = run/'best_model.pth'
            artifact = calibrate(cp, data, run/'calibration', device)
            self.assertEqual(len(artifact['val_filenames']), 2)
            summary = evaluate(cp, run/'calibration/threshold.json', data, run/'evaluation', device)
            self.assertEqual(summary['n_test'], 2)
            self.assertEqual(summary['evaluation_precision'], 'fp32')
            self.assertTrue((run/'evaluation/Table_MainResults.csv').is_file())
            self.assertTrue((run/'evaluation/Table_Boundary.csv').is_file())
            self.assertEqual(sum(summary['confusion'].values()), 2*256*256)
            predict(cp, run/'calibration/threshold.json', data/'TEST_256/STACK', run/'predictions', device)
            self.assertEqual(len(list((run/'predictions').glob('*_mask.tif'))), 2)
            with rasterio.open(run/'predictions/0_mask.tif') as mask:
                self.assertEqual(mask.count, 1)
                self.assertEqual(mask.dtypes, ('uint8',))
                self.assertTrue(np.isin(mask.read(1), [0, 1]).all())
            timing = benchmark(cp, run/'benchmark.json', device, warmup=1, repeats=2)
            self.assertGreater(timing['gflops'], 0)
            self.assertGreater(timing['fps'], 0)
            summarize(base, base/'tables')
            self.assertEqual(len(list((base/'tables').glob('Table_*.csv'))), 4)


if __name__ == '__main__':
    unittest.main()
