# Reading your results

`training_log.csv` records loss, learning rate, pooled VAL IoU, epoch threshold and early-stopping wait. `best_model.pth` is selected by strict improvement in pooled VAL IoU, not TEST performance. `last_model.pth` is the final epoch checkpoint and need not be best.

`calibration/threshold.json` contains the final VAL-only threshold and checkpoint SHA-256. The per-epoch training threshold is not automatically the final export threshold. A mismatch between threshold and checkpoint is rejected.

`Table_MainResults.csv` reports conditional per-tile overlap separately from pooled pixel and object statistics. Read the empty prediction count/rate alongside conditional IoU/Dice: excluding missed tiles from that mean can raise it. Missed tiles remain in pooled confusion and object counts. Fractions are in [0,1], not percentages; MCC is in [-1,1].

`Table_SizeRecall.csv` uses small <=1000 m², medium (1000,5000] m² and large >5000 m². These areas assume nominal 10 m pixels; they are not recovered from absent CRS information. Objects are patch-local and are not a deduplicated geographic inventory.

`Table_Boundary.csv` reports BF1 (higher is better), ASSD and HD95 in metres (lower is better), defined sample counts and undefined rate. Missing boundaries are undefined rather than zero error. JSON `null` and empty CSV cells must not be converted to perfect scores or zero distances.

`benchmark.json` measures model forward only at batch 1, FP32, with warm-up and device synchronization. It excludes file IO and transfers. FPS is 1000/mean latency in milliseconds. GFLOPs equals twice THOP MACs and may omit unsupported operators; compare measurements only under matching conventions and hardware.

`summarize` writes the three evaluation tables plus `Table_Efficiency.csv` when benchmark files exist. It does not fabricate missing models or substitute historical paper values. Check model counts and configurations before comparing runs.
