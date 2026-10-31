# Running the documented workflow

All commands run from the repository root after the Windows setup. Use the prepared project dataset. Support covers this schema and procedure; modifications for other datasets or methods require the user's own investigation and verification.

## Run all configurations sequentially

```powershell
.\.venv\Scripts\python.exe -m landslide run-all --configs .\configs\models --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\all_models_01 --device cuda
.\.venv\Scripts\python.exe -m landslide summarize --runs .\runs\all_models_01 --out .\runs\all_models_01_tables
```

Each configuration runs in a separate process. On failure, previously completed outputs remain. Run remaining configurations individually into new directories. There is no exact interrupted-training resume. Summary tables include only completed runs actually found; check their identities before interpreting the collection.

## Individual stages

The `run` command performs training, best-checkpoint VAL calibration, TEST evaluation and local forward-efficiency measurement. `--skip-benchmark` omits the final timing stage. `predict` is separate and produces raster outputs. Use `--help` for stage arguments:

```powershell
.\.venv\Scripts\python.exe -m landslide --help
.\.venv\Scripts\python.exe -m landslide train --help
.\.venv\Scripts\python.exe -m landslide calibrate --help
.\.venv\Scripts\python.exe -m landslide evaluate --help
.\.venv\Scripts\python.exe -m landslide predict --help
```

Change input/output paths to match your computer. Changing batch size, precision, thresholds or model settings changes the experimental procedure; such changes should not be represented as the supplied configuration. CPU is suitable for synthetic checks but does not implement the CUDA FP16 training procedure.

Local efficiency outputs are generated for the reader's run only. No measured model statistics, rankings, performance tables or author results are distributed in the source archive.
