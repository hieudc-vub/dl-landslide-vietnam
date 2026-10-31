"""Strict, portable JSON configuration shared by every pipeline stage."""
import json
import math
from pathlib import Path

BANDS = [
    "post_B4", "post_B3", "post_B2", "post_B8", "post_B7", "post_B11", "post_B12", "post_BSI",
    "pre_B4", "pre_B3", "pre_B2", "slope_norm", "D_B2_norm", "D_B3_norm", "D_B4_norm",
    "D_B5_norm", "D_B8_norm", "D_B12_norm", "D_NDVI_norm", "D_BSI_norm",
]
DEFAULTS = dict(
    seed=42, batch_size=16, lr=1e-4, weight_decay=0.02, epochs=150, patience=20,
    num_workers=0, precision="fp16", encoder_weights="imagenet", patch_size=256,
    bands=BANDS, thresholds=[i / 100 for i in range(5, 100, 5)], pos_pixel_min=30,
    object_iou_match=0.30, connectivity=8, size_bins_m2=[1000.0, 5000.0],
    pixel_size_m=10.0, boundary_tolerance_m=20.0, online_augmentation=False,
    weighted_sampler=False, scheduler_t0=10, scheduler_tmult=2, scheduler_eta_min=1e-6,
    focal_alpha=0.5, focal_gamma=2.0, tversky_alpha=0.6, tversky_beta=0.4,
    focal_weight=0.45, tversky_weight=0.55,
)


def validate(cfg):
    required = {"name", "arch", "encoder"} | set(DEFAULTS)
    if set(cfg) != required:
        raise ValueError(f"Invalid configuration keys: missing={required-set(cfg)}, extra={set(cfg)-required}")
    # Reject non-finite scientific settings before comparisons can hide NaN.
    for key, value in cfg.items():
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain only finite numbers")
    if cfg["arch"] not in {"unet", "unetplusplus", "deeplabv3plus", "upernet", "segformer"}:
        raise ValueError("Unsupported architecture")
    if cfg["bands"] != BANDS or cfg["patch_size"] != 256:
        raise ValueError("This release requires the published 20-band, 256x256 schema")
    if cfg["precision"] not in {"fp16", "fp32"} or cfg["connectivity"] not in {4, 8}:
        raise ValueError("Invalid precision/connectivity")
    if cfg["online_augmentation"] or cfg["weighted_sampler"]:
        raise ValueError("Online augmentation and weighted sampling are disabled in this protocol")
    for key in ["batch_size", "epochs", "patience", "scheduler_t0", "scheduler_tmult"]:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["num_workers"] < 0 or cfg["lr"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("Invalid loader/optimizer settings")
    th = cfg["thresholds"]
    if not th or sorted(set(th)) != th or not all(0 < x < 1 for x in th):
        raise ValueError("Thresholds must be unique, increasing and between zero and one")
    if not 0 < cfg["object_iou_match"] <= 1:
        raise ValueError("Invalid object IoU threshold")
    if any(isinstance(cfg[k], bool) or not isinstance(cfg[k], (int, float)) for k in ["pixel_size_m", "boundary_tolerance_m"]):
        raise ValueError("Nominal pixel scale and tolerance must be numeric")
    if cfg["pixel_size_m"] <= 0 or cfg["boundary_tolerance_m"] < 0:
        raise ValueError("Invalid nominal pixel scale/tolerance")
    if len(cfg["size_bins_m2"]) != 2 or not 0 < cfg["size_bins_m2"][0] < cfg["size_bins_m2"][1]:
        raise ValueError("Expected two increasing positive size bins")
    return cfg


def load(path):
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "pixel_size_m" not in values:
        raise ValueError("Explicit pixel_size_m is required; no spatial scale is inferred from patch CRS")
    return validate(DEFAULTS | values)
