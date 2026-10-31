"""Explicit pixel, object and boundary metrics. Undefined values remain NaN."""
import numpy as np
from scipy import ndimage as ndi


def ratio(a, b):
    return float(a / b) if b else float("nan")


def confusion(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()),
                fn=int((~pred & gt).sum()), tn=int((~pred & ~gt).sum()))


def pixel_metrics(c):
    tp, fp, fn, tn = [c[k] for k in ["tp", "fp", "fn", "tn"]]
    denominator = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return dict(iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn),
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn), f2=ratio(5*tp, 5*tp+4*fn+fp),
                fnr=ratio(fn, tp+fn), fdr=ratio(fp, tp+fp),
                accuracy=ratio(tp+tn, tp+fp+fn+tn), mcc=ratio(tp*tn-fp*fn, denominator))


def tile_overlap(c, epsilon=1e-6):
    """Paper's smoothed per-tile overlap; population selection happens separately."""
    tp, fp, fn = c['tp'], c['fp'], c['fn']
    return dict(iou=(tp + epsilon)/(tp + fp + fn + epsilon),
                dice=(2*tp + epsilon)/(2*tp + fp + fn + epsilon))


def aggregate_tiles(rows):
    """Keep missed positive tiles in pooled/object metrics, but report their rate.

    TEST reference-positive means GT > 0. The VAL-only >30 filter must never
    change the TEST reporting population. No-prediction tiles are excluded only
    from the paper's conditional overlap mean, and are counted explicitly.
    """
    positive = [r for r in rows if r['gt_pixels'] > 0]
    eligible = [r for r in positive if r['pred_pixels'] > 0]
    missed = sum(r['pred_pixels'] == 0 for r in positive)
    return dict(n_reference_positive=len(positive), n_overlap_eligible=len(eligible),
                n_empty_predictions_on_positive=missed,
                empty_prediction_rate=ratio(missed, len(positive)),
                overlap_conditional={k: mean_defined([r[k] for r in eligible]) for k in ['iou', 'dice']},
                boundary_positive={k: mean_defined([r[k] for r in positive])
                                   for k in ['boundary_f1', 'assd_m', 'hd95_m']},
                boundary_undefined_rate=ratio(sum(not np.isfinite(r['hd95_m']) for r in positive), len(positive)))


def components(mask, connectivity):
    return ndi.label(mask, structure=ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1))


def object_metrics(pred, gt, cfg):
    p, npred = components(pred, cfg["connectivity"])
    g, ngt = components(gt, cfg["connectivity"])
    pa, ga = np.bincount(p.ravel()), np.bincount(g.ravel())
    overlap = (p > 0) & (g > 0)
    pairs, intersections = np.unique(np.stack([g[overlap], p[overlap]], axis=1), axis=0, return_counts=True)
    candidates = []
    for (gi, pi), inter in zip(pairs, intersections):
        union = int(ga[gi]) + int(pa[pi]) - int(inter)
        iou = int(inter) / union
        if iou >= cfg["object_iou_match"]:
            candidates.append((-iou, int(gi), int(pi)))
    used_g, used_p = set(), set()
    for _, gi, pi in sorted(candidates):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    sizes = ga[1:] * cfg["pixel_size_m"]**2
    bins = np.searchsorted(cfg["size_bins_m2"], sizes, side="left")
    detected = np.array([i in used_g for i in range(1, ngt+1)], dtype=bool)
    rows = [dict(bin=label, n_gt=int((bins == i).sum()),
                 n_detected=int(((bins == i) & detected).sum()))
            for i, label in enumerate(["small", "medium", "large"])]
    tp = len(used_g)
    return dict(obj_tp=tp, obj_fp=npred-tp, obj_fn=ngt-tp), rows


def boundary_metrics(pred, gt, cfg):
    # Inner one-pixel boundaries with an explicit 3x3 erosion convention.
    p = pred ^ ndi.binary_erosion(pred, structure=np.ones((3, 3)), border_value=0)
    g = gt ^ ndi.binary_erosion(gt, structure=np.ones((3, 3)), border_value=0)
    if not p.any() or not g.any():
        return dict(boundary_f1=float("nan"), assd_m=float("nan"), hd95_m=float("nan"))
    p_to_g = ndi.distance_transform_edt(~g, sampling=cfg["pixel_size_m"])[p]
    g_to_p = ndi.distance_transform_edt(~p, sampling=cfg["pixel_size_m"])[g]
    precision = np.mean(p_to_g <= cfg["boundary_tolerance_m"])
    recall = np.mean(g_to_p <= cfg["boundary_tolerance_m"])
    bf1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    distances = np.concatenate([p_to_g, g_to_p])
    return dict(boundary_f1=float(bf1), assd_m=float(distances.mean()),
                hd95_m=float(max(np.percentile(p_to_g, 95), np.percentile(g_to_p, 95))))


def mean_defined(values):
    a = np.asarray(values, float)
    return dict(mean=float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else float("nan"),
                n_defined=int(np.isfinite(a).sum()), n_total=len(a))
