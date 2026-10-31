"""Artifact writing, provenance and explicit random seeds."""
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import platform
from pathlib import Path

import numpy as np
import torch


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write empty CSV without a schema")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([clean(row) for row in rows])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime(device, precision):
    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, device=str(device), precision=precision,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()})
