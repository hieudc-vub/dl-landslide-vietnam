"""Offline CPU forward check of all 13 architectures, one process per model.

Run from the repository root after installing the package. No pretrained download.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--out', default='runs/model_registry_check.json')
    args = parser.parse_args()
    if args.config:
        import torch
        from landslide.config import load
        from landslide.models import LandslideModel
        torch.set_num_threads(2)
        cfg = load(args.config)
        model = LandslideModel(cfg, pretrained=False).eval()
        with torch.inference_mode():
            logits, weights = model(torch.zeros(1, 20, 256, 256))
        assert logits.shape == (1, 1, 256, 256)
        assert weights.shape == (1, 20)
        assert torch.isfinite(logits).all()
        print(json.dumps(dict(model=cfg['name'], status='passed')))
        return
    rows = []
    for path in sorted(Path('configs/models').glob('*.json')):
        completed = subprocess.run([sys.executable, '-B', __file__, '--config', str(path)],
                                   capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"{path.name} failed:\n{completed.stderr}")
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        print(rows[-1]['model'], 'passed', flush=True)
    if len(rows) != 13:
        raise RuntimeError('Expected exactly 13 models')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
