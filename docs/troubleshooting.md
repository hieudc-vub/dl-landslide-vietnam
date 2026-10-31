# Troubleshooting on Windows

Always run commands from the folder containing `pyproject.toml`, using `.\.venv\Scripts\python.exe`. Original inputs are not edited. Outputs from a failed run may be incomplete; keep them for diagnosis and choose a new output path for a new run.

| Message or symptom | Action |
|---|---|
| `py` not recognized | Install Python 3.12 with the launcher; reopen PowerShell. |
| Python version is not 3.12 | Use `py -3.12`; do not reuse a virtual environment created by another Python version. Extract a separate project if needed. |
| `.venv\Scripts\python.exe` not found | Confirm you are in the extracted project root; run `scripts\setup_windows.ps1`. |
| CUDA requested but unavailable | Run `check-env`; confirm NVIDIA GPU, driver and CUDA PyTorch wheel. CPU smoke tests remain available; CPU is not the GPU benchmark. |
| GPU out of memory | Close GPU applications and check memory. Do not silently lower the reference batch. Changing the experimental procedure is outside the documented support scope. |
| Output already exists | Change `--out` to a new name. No deletion is necessary. |
| Wrong band count/descriptions/range | Use the prepared 20-band dataset and run validation. Do not apply the old 49-band indices or rename bands to conceal a mismatch. |
| Threshold belongs to another checkpoint | Use `calibration/threshold.json` generated with that exact `best_model.pth`. |
| Internet/download failure | Check connectivity and retry after restoring access. Do not change pretrained initialization to random weights for a reference run. |
| Validation seems slow | It reads approximately 28.6 GB and hashes every raster. Disk speed affects duration. |
| Matplotlib/THOP deprecation warning | A warning can coexist with successful tests. Check the final exit status; a traceback/error or missing outputs needs investigation. |
| Interrupted `run-all` | Completed runs remain intact. Use `run` for remaining configurations, each with a new output directory. Do not rerun into existing model folders. |

For full error details, put `--debug` before the command:

```powershell
.\.venv\Scripts\python.exe -m landslide --debug check-env
```

Normal errors save a traceback under `logs/`. Review it for local paths before attaching it to a GitHub issue. Provide the command, error, Python/PyTorch versions and GPU model. Never share credentials, private inputs or trained checkpoints.

There is no exact interrupted-training resume command in this release. A new run starts with a new initialization and new output directory. CPU synthetic tests do not establish available GPU memory for a full reference batch on every architecture.
