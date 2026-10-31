"""Training, VAL-only calibration, TEST evaluation and checkpoint inference."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from .config import validate
from .data import PatchDataset, read_stack
from .io import runtime, seed_all, sha256, write_csv, write_json
from .metrics import boundary_metrics, confusion, mean_defined, object_metrics, pixel_metrics, ratio, tile_overlap, aggregate_tiles
from .models import FocalTverskyLoss, LandslideModel
from .stopping import EarlyStopping
from .reporting import export_tables, plot_evaluation, plot_training


def device_for(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def amp(cfg, device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and cfg["precision"] == "fp16" else nullcontext()


def loader(root, split, cfg, shuffle=False):
    return DataLoader(PatchDataset(root, split), batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], generator=torch.Generator().manual_seed(cfg["seed"]),
                      pin_memory=False, drop_last=shuffle)


@torch.inference_mode()
def iterate(model, dl, device, cfg):
    model.eval()
    for x, y, names in dl:
        # Validation and export are deliberately FP32, independent of training AMP.
        logits, _ = model(x.to(device).float())
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite inference probabilities")
        for prob, gt, name in zip(probabilities, y.numpy()[:, 0] > 0.5, names):
            yield prob, gt, name


def threshold_sweep(model, dl, device, cfg, objective):
    thresholds = cfg["thresholds"]
    pooled = [dict(tp=0, fp=0, fn=0, tn=0) for _ in thresholds]
    ious, dices = [[] for _ in thresholds], [[] for _ in thresholds]
    for prob, gt, _ in iterate(model, dl, device, cfg):
        for i, threshold in enumerate(thresholds):
            c = confusion(prob > threshold, gt)
            for k in c:
                pooled[i][k] += c[k]
            if gt.sum() > cfg["pos_pixel_min"]:
                pm = tile_overlap(c)
                ious[i].append(pm["iou"])
                dices[i].append(pm["dice"])
    rows = [dict(threshold=t, pooled_iou=pixel_metrics(c)["iou"],
                 val_iou_pos_mean=mean_defined(ii)["mean"], val_dice_pos_mean=mean_defined(dd)["mean"],
                 n_val_pos=len(ii)) for t, c, ii, dd in zip(thresholds, pooled, ious, dices)]
    key = "pooled_iou" if objective == "pooled" else "val_iou_pos_mean"
    if not all(np.isfinite(r[key]) for r in rows):
        raise ValueError("Threshold selection is undefined; check VAL positive examples")
    # Ascending grid; deterministic tie breaking: Dice then lower threshold for macro calibration.
    best = max(rows, key=lambda r: (r[key], r["val_dice_pos_mean"] if objective == "macro" else 0, -r["threshold"]))
    return best, rows


def train(cfg, root, out, device):
    validate(cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never mix runs or silently overwrite.
    seed_all(cfg["seed"])
    effective_precision = cfg["precision"] if device.type == "cuda" else "fp32"
    write_json(out / "config.json", cfg)
    write_json(out / "environment.json", runtime(device, effective_precision))
    train_dl, val_dl = loader(root, "train", cfg, True), loader(root, "val", cfg)
    write_json(out / "split_inventory.json", {split: [p.name for p in dl.dataset.files]
                                               for split, dl in [("train", train_dl), ("val", val_dl)]})
    model = LandslideModel(cfg).to(device)
    loss_fn = FocalTverskyLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        T_0=cfg["scheduler_t0"], T_mult=cfg["scheduler_tmult"], eta_min=cfg["scheduler_eta_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=effective_precision == "fp16")
    stopper = EarlyStopping(cfg["patience"])
    history = []
    if len(train_dl) == 0:
        raise ValueError("TRAIN has fewer samples than batch_size with drop_last=True. Use a smaller batch only in a separate demonstration config.")
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        total_loss, samples = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]
        for batch_number, (x, y, _) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp(cfg, device):
                logits, _ = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()*len(x)
            samples += len(x)
            if batch_number % 20 == 0 or batch_number == len(train_dl):
                print(f"Epoch {epoch}: training batch {batch_number}/{len(train_dl)}", flush=True)
        best, _ = threshold_sweep(model, val_dl, device, cfg, "pooled")
        # Report validation loss in FP32. It does not control checkpoint selection.
        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for vx, vy, _ in val_dl:
                vl, _ = model(vx.to(device).float())
                val_total += loss_fn(vl, vy.to(device)).item()*len(vx)
        improved, should_stop = stopper.update(best["pooled_iou"])
        best_score, wait = stopper.best, stopper.wait
        scheduler.step()
        checkpoint = dict(format_version=1, config=cfg, model=model.state_dict(), epoch=epoch,
                          best_score=best_score, epoch_validation_threshold=best["threshold"],
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        torch.save(checkpoint, out / "last_model.pth")
        if improved:
            torch.save(checkpoint, out / "best_model.pth")
        history.append(dict(epoch=epoch, train_loss=total_loss/samples, val_loss=val_total/len(val_dl.dataset), lr=lr,
                            val_pooled_iou=best["pooled_iou"], val_threshold=best["threshold"],
                            improved=improved, epochs_without_improvement=wait))
        write_csv(out / "training_log.csv", history)
        print(f"epoch {epoch}/{cfg['epochs']} loss={total_loss/samples:.5f} VAL IoU={best['pooled_iou']:.5f} wait={wait}", flush=True)
        if should_stop:
            break
    plot_training(out, history)
    write_json(out / "training_summary.json", dict(epochs_completed=epoch, best_val_pooled_iou=best_score,
        early_stopped=wait >= cfg["patience"], checkpoint_sha256=sha256(out / "best_model.pth")))


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Expected a checkpoint produced by this release, not a legacy bare state_dict")
    cfg = validate(payload["config"])
    model = LandslideModel(cfg, pretrained=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), cfg


def calibrate(checkpoint, root, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, cfg = load_model(checkpoint, device)
    best, rows = threshold_sweep(model, loader(root, "val", cfg), device, cfg, "macro")
    write_csv(out / "val_threshold_sweep.csv", rows)
    artifact = dict(threshold=best["threshold"], objective="macro IoU on VAL patches with GT pixels > pos_pixel_min",
                    checkpoint_sha256=sha256(checkpoint), config=cfg,
                    val_filenames=[p.name for p in PatchDataset(root, "val").files])
    write_json(out / "threshold.json", artifact)
    return artifact


def read_threshold(path, checkpoint, cfg):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact["checkpoint_sha256"] != sha256(checkpoint) or artifact["config"] != cfg:
        raise ValueError("Threshold artifact belongs to a different checkpoint/configuration")
    value = float(artifact["threshold"])
    if not 0 < value < 1:
        raise ValueError("Invalid calibrated threshold")
    return value


def evaluate(checkpoint, threshold_path, root, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, cfg = load_model(checkpoint, device)
    threshold = read_threshold(threshold_path, checkpoint, cfg)
    rows, counts = [], dict(tp=0, fp=0, fn=0, tn=0)
    obj = dict(obj_tp=0, obj_fp=0, obj_fn=0)
    size = [dict(bin=k, n_gt=0, n_detected=0) for k in ["small", "medium", "large"]]
    for prob, gt, name in iterate(model, loader(root, "test", cfg), device, cfg):
        pred = prob > threshold
        c = confusion(pred, gt)
        om, bins = object_metrics(pred, gt, cfg)
        for k in c:
            counts[k] += c[k]
        for k in obj:
            obj[k] += om[k]
        for total, row in zip(size, bins):
            total["n_gt"] += row["n_gt"]
            total["n_detected"] += row["n_detected"]
        rows.append(dict(file=name, gt_pixels=int(gt.sum()), pred_pixels=int(pred.sum()),
                         positive=bool(gt.sum() > 0), **c, **(pixel_metrics(c) | tile_overlap(c)), **om,
                         **boundary_metrics(pred, gt, cfg)))
    tp, fp, fn = [obj[k] for k in ["obj_tp", "obj_fp", "obj_fn"]]
    summary = dict(model=cfg["name"], threshold=threshold, n_test=len(rows),
                   checkpoint_sha256=sha256(checkpoint), config=cfg, evaluation_precision="fp32",
                   confusion=counts, pixel_pooled=pixel_metrics(counts), object_counts=obj,
                   object_precision=ratio(tp, tp+fp), object_recall=ratio(tp, tp+fn),
                   object_f1=ratio(2*tp, 2*tp+fp+fn), **aggregate_tiles(rows))
    for row in size:
        row["recall"] = ratio(row["n_detected"], row["n_gt"])
    write_csv(out / "test_per_image.csv", rows)
    write_csv(out / "size_stratified_object_recall.csv", size)
    write_json(out / "test_summary.json", summary)
    export_tables(out, summary, size)
    plot_evaluation(out, summary)
    return summary


@torch.inference_mode()
def predict(checkpoint, threshold_path, input_path, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, cfg = load_model(checkpoint, device)
    threshold = read_threshold(threshold_path, checkpoint, cfg)
    input_path = Path(input_path)
    files = sorted(input_path.glob("*.tif")) if input_path.is_dir() else [input_path]
    if not files:
        raise ValueError("No input TIFFs")
    for path in files:
        x = torch.from_numpy(read_stack(path))[None].to(device)
        logits, _ = model(x.float())
        probabilities = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite prediction")
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            tags = src.tags()
        profile.update(count=1, compress="lzw", nodata=None)
        for suffix, array, dtype in [("probability", probabilities, "float32"),
                                      ("mask", probabilities > threshold, "uint8")]:
            profile.update(dtype=dtype)
            with rasterio.open(out / f"{path.stem}_{suffix}.tif", "w", **profile) as dst:
                dst.write(array.astype(dtype), 1)
                dst.set_band_description(1, suffix)
                dst.update_tags(**tags)
    write_json(out / "prediction_meta.json", dict(threshold=threshold, checkpoint_sha256=sha256(checkpoint),
        input_files=[p.name for p in files], note="Original patch georeferencing preserved; no geographic coordinates are invented."))


@torch.inference_mode()
def benchmark(checkpoint, out, device, warmup=10, repeats=100):
    if Path(out).exists():
        raise FileExistsError(f"Benchmark output already exists: {out}")
    model, cfg = load_model(checkpoint, device)
    seed_all(cfg["seed"])
    x = torch.rand(1, 20, 256, 256, device=device)*2-1
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        model(x)
        synchronize()
        times.append((time.perf_counter()-start)*1000)
    from thop import profile
    # THOP estimates supported operators; two floating operations per MAC.
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    result = dict(model=cfg["name"], params_m=params/1e6, gflops=2*macs/1e9,
        flops_convention="2 x THOP MACs; supported-operator estimate, not exhaustive hardware instructions",
        batch_size=1, input_shape=list(x.shape), warmup=warmup, repeats=repeats, precision="fp32",
        latency_mean_ms=float(np.mean(times)), latency_median_ms=float(np.median(times)),
        fps=1000/float(np.mean(times)), timing_scope="model forward only; excludes IO and transfers",
        checkpoint_sha256=sha256(checkpoint), runtime=runtime(device, "fp32"))
    write_json(out, result)
    return result
