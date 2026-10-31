# Functional checks and arithmetic conventions

The source includes tests of software behavior using synthetic inputs. They do not establish paper-score reproduction, label accuracy or geographic independence. The GitHub workflow reports the status of its particular test execution. Local measured values, hardware statistics and run logs are excluded from the source release.

From the repository root after installation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_docs.py
```

The tests exercise strict data/configuration checks, confusion counts, object matching, area-bin boundaries, directional HD95, undefined values, early stopping, FP32 evaluation and synthetic train/calibrate/evaluate/predict/export stages. No research data or pretrained download is required for these tests.

Integer counts and count identities are checked exactly. Hand-calculated float64 metric cases use rtol=1e-12 and atol=1e-12. Float32 loss examples and checkpoint round-trip comparisons use rtol=1e-5 and atol=1e-6. These are test-specific tolerances, not a bound on differences between full training runs. The input range check allows 1e-5 at the [-1,1] endpoints; it never clips or renormalizes values. Per-tile overlap uses epsilon=1e-6; object matching does not add epsilon to IoU. Display rounding must not be used for threshold decisions or aggregate calculations.

The dataset command checks paired filenames, dimensions, ordered band descriptions, finite/range/binary values, invalid pixels, matching pair CRS/transform, file hashes and exact paired decoded duplicates across splits. Matching missing CRS and identity transforms does not establish physical registration. Exact-duplicate checks cannot establish spatial/source-group independence or rule out augmented variants across splits. The nominal 10 m scale is explicitly declared in the project configuration, not inferred from absent CRS.

Documentation checks compare executable command blocks and section identifiers in the four READMEs and verify local links. They do not certify translation meaning. Supporting documents are in English unless explicitly labelled otherwise.
