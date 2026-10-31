# Protocol implemented in version 0.4.0

## Models

All models use an identical input channel gate: mean squared activation per band,
20→5 biased linear layer, ReLU, 5→20 biased linear layer, sigmoid, then input
channel multiplication. SMP returns one
binary logit per pixel; sigmoid is applied for evaluation/inference.

| Config prefix | Decoder | Encoder |
|---|---|---|
| 01 | U-Net | efficientnet-b4 |
| 02 | U-Net | resnet50 |
| 03 | U-Net++ | efficientnet-b4 |
| 04 | U-Net++ | resnet50 |
| 05 | U-Net | tu-hrnet_w18 |
| 06 | U-Net | tu-hrnet_w32 |
| 07 | DeepLabV3+ | resnet50 |
| 08 | DeepLabV3+ | mobilenet_v2 |
| 09 | UPerNet | resnet50 |
| 10 | SegFormer | mit_b0 |
| 11 | SegFormer | mit_b2 |
| 12 | SegFormer | mit_b5 |
| 13 | U-Net | tu-swin_tiny_patch4_window7_224 |

Encoder weights: ImageNet on training initialization. Loading a checkpoint never
downloads encoder weights. Names describe actual decoder/encoder combinations;
no additional claims about family/tier membership are encoded.

Swin-T passes `img_size=256, strict_img_size=False` to timm, enabling runtime
attention masks at the dataset resolution. The `_224` encoder identifier names
the pretrained architecture; input patches are not resized to 224. This explicit
compatibility setting is part of the new implementation, not evidence of how a
historical environment was configured.

## Training

- AdamW over all parameters, including the channel gate; LR 1e-4, weight decay .02.
- Loss = .45 Focal(alpha=.50, gamma=2) + .55 Tversky(alpha=.60 FP, beta=.40 FN),
  using SMP 0.5.0 implementations. Loss is evaluated in FP32.
- CosineAnnealingWarmRestarts: T_0=10 epochs, T_mult=2, eta_min=1e-6;
  scheduler stepped once after each epoch. This follows the notebook scheduler.
- Batch 16; shuffled TRAIN; final short batch dropped (drop_last=True); no weighted sampling,
  online augmentation, extra normalization or synthetic validity masks.
- Seed 42 for Python, NumPy and PyTorch; deterministic cuDNN requested and
  autotuning disabled. Bitwise equivalence across platforms is not promised.
- CUDA autocast explicitly FP16 with GradScaler. CPU fallback FP32 is recorded.
- After each epoch, select the threshold maximizing pooled VAL IoU on the
  .05:.05:.95 grid. Ties use the lower threshold. Save best checkpoint only on
  strict improvement. Stop after 20 consecutive epochs without improvement,
  or at 150 epochs. TEST never controls checkpoint selection.
- Non-finite loss/probabilities and invalid files fail explicitly.

## Final threshold calibration

Reload the best checkpoint. Compute per-patch IoU on VAL patches with strictly
more than 30 positive GT pixels. Choose the threshold maximizing their mean IoU;
ties use mean Dice and then the lower threshold. Per-tile IoU and Dice use additive epsilon 1e-6 in numerator and denominator;
empty predictions on positive patches therefore contribute a near-zero value. Save threshold, configuration, VAL filenames and
checkpoint SHA-256. TEST evaluation refuses a threshold tied to another checkpoint.

## Pixel metrics

Pooled TP/FP/FN/TN produce precision, recall, F1, F2, MCC, FNR, FDR and accuracy.
TEST reference-positive means GT pixels > 0, not the VAL calibration filter >30.
Per-tile IoU = (TP+1e-6)/(TP+FP+FN+1e-6); Dice = (2TP+1e-6)/(2TP+FP+FN+1e-6).
`overlap_conditional` averages only reference-positive tiles with nonempty predictions,
under the documented evaluation convention. The number and rate of excluded empty predictions are reported
explicitly. Pooled pixel and object counts include every TEST tile, including misses.
Undefined denominators are NaN internally and JSON null externally, with counts.

## Objects

Foreground components use explicit **8-connectivity**, including diagonal
neighbors. Compute nonzero intersections and exact intersection/union ratios.
Sort eligible pairs by decreasing IoU; ties by GT component ID, then prediction
ID. Greedily accept one-to-one pairs with IoU **>= .30**. No additive epsilon is
used in this comparison. Object counts are pooled across TEST patches; components
are patch-local and are not deduplicated into a geographic inventory.

GT component area = pixel count × 100 m², using nominal 10 m pixels. Bins are
small <=1000 m², medium (1000,5000] m² and large >5000 m². Recall is matched GT
count / GT count per bin; an empty bin is undefined.

## Boundaries

Inner boundary = foreground XOR 3×3 binary erosion, treating outside the patch
as background. Euclidean distances use nominal 10 m sampling. BF1 uses a 20 m
inclusive tolerance. ASSD is the mean of concatenated distances from both
directions (weighted by boundary-point count); HD95 is the maximum of the two directional 95th
percentiles, not the percentile of the concatenated distances. If either boundary is absent, all three boundary metrics
are undefined. Positive-patch macro means exclude undefined values and report
their counts and the undefined rate.

## Efficiency and reproducibility limits

`benchmark` uses B=1, 20×256×256, evaluation/inference mode, 10 warmup passes,
100 timed passes, CUDA synchronization, and FP32 inference.
Timing excludes dataset IO and device transfers. Runtime/GPU, parameter count and GFLOPs (2 x THOP MACs, a supported-operator
estimate) are recorded. PR/AP curves, bootstrap intervals, robustness experiments,
geographic scene inference and exact interrupted-training resume are not
implemented here; no historical values are substituted for these analyses.

Validation, threshold sweeps, calibration, TEST evaluation and prediction run in FP32 without training autocast. The training loss is FP32 even when the model forward uses FP16 autocast.

Loaded configuration files must explicitly declare a finite positive pixel_size_m. The project files declare 10 m; no fallback is inferred from CRS. Missing spatial metadata does not prove patch registration or split independence. See VALIDATION.md for test tolerances.
