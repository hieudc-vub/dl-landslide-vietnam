import unittest
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from landslide.config import DEFAULTS
from landslide.engine import iterate
from landslide.stopping import EarlyStopping


class ProtocolTests(unittest.TestCase):
    def test_early_stopping_after_exactly_20_non_improvements(self):
        stop = EarlyStopping(20)
        self.assertEqual(stop.update(.5), (True, False))
        for _ in range(19):
            self.assertEqual(stop.update(.5), (False, False))
        self.assertEqual(stop.update(.4), (False, True))
        self.assertEqual(stop.update(.6), (True, False))
        self.assertEqual(stop.wait, 0)

    def test_evaluation_does_not_enable_training_autocast(self):
        class Recorder(nn.Module):
            def forward(self, x):
                if torch.is_autocast_enabled('cuda'):
                    raise AssertionError('Evaluation enabled CUDA autocast')
                assert x.dtype == torch.float32
                return x, None
        x = torch.zeros(1,1,2,2)
        dl = DataLoader(TensorDataset(x,x,torch.arange(1)))
        outputs = list(iterate(Recorder(),dl,torch.device('cpu'),DEFAULTS))
        self.assertEqual(len(outputs),1)


if __name__ == '__main__':
    unittest.main()
