"""Read already packed 20-channel patches without further normalization."""
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .config import BANDS
from .io import sha256, write_csv, write_json

SPLITS = {"train": "TRAIN_256", "val": "VAL_256", "test": "TEST_256"}


def read_stack(path):
    with rasterio.open(path) as src:
        x = src.read(out_dtype="float32")
        descriptions = list(src.descriptions)
        if x.shape != (20, 256, 256):
            raise ValueError(f"{Path(path).name}: expected (20,256,256), got {x.shape}")
        if descriptions != BANDS:
            raise ValueError(f"{Path(path).name}: missing or unexpected band descriptions/order")
        if np.any(src.read_masks() == 0):
            raise ValueError(f"{Path(path).name}: masked pixels require a separate validity protocol")
    if not np.isfinite(x).all() or x.min() < -1.00001 or x.max() > 1.00001:
        raise ValueError(f"{Path(path).name}: expected finite pre-normalized values in [-1,1]")
    return x


class PatchDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / SPLITS[split]
        stacks = {p.name: p for p in (self.root / "STACK").glob("*.tif")}
        labels = {p.name: p for p in (self.root / "LABEL").glob("*.tif")}
        if not stacks or stacks.keys() != labels.keys():
            raise ValueError(f"{split}: empty dataset or unequal STACK/LABEL filenames")
        self.files = sorted(stacks.values())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        p = self.files[index]
        x = read_stack(p)
        with rasterio.open(p) as stack:
            stack_grid = (stack.crs, stack.transform)
        with rasterio.open(self.root / "LABEL" / p.name) as src:
            if (src.crs, src.transform) != stack_grid:
                raise ValueError(f"{p.name}: STACK/LABEL spatial metadata do not match")
            if src.count != 1 or (src.height, src.width) != (256, 256):
                raise ValueError(f"{p.name}: invalid LABEL shape")
            y = src.read(1)
            if np.any(src.read_masks(1) == 0):
                raise ValueError(f"{p.name}: LABEL contains invalid pixels")
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{p.name}: expected binary labels 0/1")
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)[None]), p.name


def validate_dataset(root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, fingerprints, counts = [], {}, {}
    for split in SPLITS:
        ds = PatchDataset(root, split)
        counts[split] = len(ds)
        for i, stack in enumerate(ds.files):
            x, y, name = ds[i]
            # A paired decoded fingerprint catches an exact reused example across splits.
            import hashlib
            fingerprint = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
            previous = fingerprints.get(fingerprint)
            if previous and previous != split:
                raise ValueError(f"Decoded example repeated across {previous}/{split}: {name}")
            fingerprints[fingerprint] = split
            for kind in ["STACK", "LABEL"]:
                path = ds.root / kind / name
                rows.append(dict(path=path.relative_to(Path(root)).as_posix(), size_bytes=path.stat().st_size,
                                 sha256=sha256(path), positive_pixels=int(y.sum())))
            if (i+1) % 250 == 0 or i+1 == len(ds):
                print(f"Checking {split}: {i+1}/{len(ds)} pairs", flush=True)
    write_csv(out / "dataset_manifest.csv", rows)
    write_json(out / "validation.json", dict(counts=counts, bands=BANDS, status="passed",
        limitation="Checks exact paired pixel duplicates, not geographic/source-group independence."))
    return counts
