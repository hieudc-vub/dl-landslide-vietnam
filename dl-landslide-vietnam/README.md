**English** · [Français](docs/translations/README.fr.md) · [Nederlands](docs/translations/README.nl.md) · [Tiếng Việt](docs/translations/README.vi.md)

# dl-landslide-vietnam

PyTorch workflows for binary landslide segmentation

Code accompanying a study of landslide inventory mapping with bi-temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam. The package provides 13 architecture configurations for prepared patches. It starts from the project dataset schema. Readers train their own models; author-trained checkpoints are not distributed.

[Installation and environment check](#installation) · [Dataset](#dataset) · [Quick start](#quick-start) · [Citation and acknowledgements](#citation)

| Task | Go to |
|---|---|
| Understand the workflow | [Understand the workflow](#overview) |
| Check installation | [Check installation](#installation) |
| Prepare the input | [Prepare the input](#dataset) |
| Run one model | [Run one model](#quick-start) |
| Find generated files | [Find generated files](#outputs) |
| Report a code issue | [Report a code issue](#support) |

<a id="overview"></a>

## Overview

Input: 256 × 256 patches with the project’s ordered 20 bands and binary labels (0 background, 1 landslide). Processing: training, VAL checkpoint selection and threshold calibration, TEST evaluation, and prediction. Outputs are created locally by the user’s run. No paper results are included in this repository.

<a id="before"></a>

## Before you start

The documented workflow uses Windows and Python 3.12 (reference 3.12.8), PyTorch 2.9.0 with CUDA 12.8. NVIDIA CUDA is used for the documented FP16 training workflow. CPU supports the synthetic installation check. Hardware memory requirements vary by architecture; no minimum GPU specification is established here. Internet is needed for installation, the dataset and ImageNet initialization. Dependencies are pinned directly, not a complete transitive environment lock.

| Requirement | Documented setting | Check / guidance |
|---|---|---|
| OS | Windows local | Open PowerShell in the project folder |
| Python | Python 3.12; reference 3.12.8 | Version command below |
| GPU | CUDA for the documented FP16 training | Environment diagnostic after setup |
| Data | Prepared project dataset | Schema and dataset validation below |

<a id="installation"></a>

## Installation and environment check

Download and extract the repository ZIP. Open the folder containing README.md and pyproject.toml. In the File Explorer address bar, type powershell and press Enter. Run the following commands there; they create a separate project environment. Wait for SETUP COMPLETE. The execution-policy option applies only to this process.

If Python is not installed, use the [official Python 3.12.8 page](https://www.python.org/downloads/release/python-3128/) for the reference version. In the Windows installer, enable the Python launcher and the PATH option, then reopen PowerShell. On the [repository page](https://github.com/hieudc-vub/dl-landslide-vietnam), choose **Code → Download ZIP**, then **Extract All** in Windows. Run commands from the extracted project root even when reading a translated README.

```powershell
py -3.12 --version
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Compute cuda
.\.venv\Scripts\python.exe -m landslide check-env
```

Then run the synthetic installation check. It creates temporary data without downloading research data or pretrained weights. Expected final status: OK. This checks software behavior, not research performance. Temporary outputs are removed.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```


<a id="dataset"></a>

## Dataset

Dataset v1.0.0: [Zenodo](https://doi.org/10.5281/zenodo.23838723). Download the three raster ZIP archives; schema and manifests are provided in this repository. See [download and verification instructions](docs/DATASET_RELEASE.md). A full run requires SAMPLE_DATASET_256x256. Place it under data/ as shown, or supply its existing path in quotes. Do not change band order or apply additional normalization. Values must be finite within [-1,1], with an input-check tolerance of 1e-5; invalid/NoData pixels are rejected. STACK and LABEL names and spatial metadata must match.

```text
data/SAMPLE_DATASET_256x256/
├── TRAIN_256/   STACK/ + LABEL/
├── VAL_256/     STACK/ + LABEL/
└── TEST_256/    STACK/ + LABEL/
```

The project configuration explicitly declares a nominal scale of 10 m/pixel. Area is pixel count × 100 m²; this is not a scale recovered from patch CRS. Missing or invalid pixel_size_m is rejected. Schema details and provider attribution are linked below. Data preparation and split creation are performed before using this package.

<a id="quick-start"></a>

## Quick start

Validate the dataset, then run one example configuration. Validation reads and hashes every raster. The run command starts real training, calibrates on VAL, evaluates TEST and writes local outputs; it can take substantial time. Use a new output directory each time. The example architecture is selected to demonstrate commands, not to claim a performance advantage.

```powershell
.\.venv\Scripts\python.exe -m landslide validate-data --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\dataset_validation
.\.venv\Scripts\python.exe -m landslide run --config .\configs\models\04_unetplusplus_resnet50.json --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\04_unetplusplus_resnet50 --device cuda
```

Prediction is a separate command using the checkpoint and threshold produced by your run:

```powershell
.\.venv\Scripts\python.exe -m landslide predict --checkpoint .\runs\04_unetplusplus_resnet50\best_model.pth --threshold .\runs\04_unetplusplus_resnet50\calibration\threshold.json --input .\data\SAMPLE_DATASET_256x256\TEST_256\STACK --out .\runs\04_predictions --device cuda
```


<a id="outputs"></a>

## Generated files

The following tree illustrates file locations only. Checkpoints, tables, figures and raster outputs are generated locally and excluded from Git by default. No measured values or output images are provided here. A JSON null or empty CSV field denotes an undefined metric, not zero.

```text
runs/04_unetplusplus_resnet50/
├── config.json, environment.json, split_inventory.json
├── best_model.pth, last_model.pth
├── training_log.csv, training_summary.json, training_curves.png
├── calibration/threshold.json, val_threshold_sweep.csv
├── evaluation/
│   ├── test_summary.json, test_per_image.csv
│   ├── Table_MainResults.csv, Table_SizeRecall.csv, Table_Boundary.csv
│   ├── size_stratified_object_recall.csv
│   └── confusion_matrix.png
└── benchmark.json
runs/04_predictions/
├── <input-name>_probability.tif
├── <input-name>_mask.tif
└── prediction_meta.json
```


<a id="method"></a>

## Models and method

All 13 configurations use batch size 16, learning rate 1e-4, weight decay 0.02, seed 42, at most 150 epochs, early-stopping patience 20 and num_workers=0. Online augmentation and weighted sampling are disabled. CUDA training uses FP16 autocast; threshold selection, evaluation and export use FP32. The protocol document defines losses, conditional overlap statistics, 8-connected objects and boundary measures.

```text
TRAIN → Training
VAL → Checkpoint selection → Threshold calibration
Best checkpoint + frozen threshold + TEST → Evaluation
Best checkpoint + frozen threshold + STACK → Prediction
```

| ID | Decoder | Encoder |
|---|---|---|
| 01 | unet | efficientnet-b4 |
| 02 | unet | resnet50 |
| 03 | unetplusplus | efficientnet-b4 |
| 04 | unetplusplus | resnet50 |
| 05 | unet | tu-hrnet_w18 |
| 06 | unet | tu-hrnet_w32 |
| 07 | deeplabv3plus | resnet50 |
| 08 | deeplabv3plus | mobilenet_v2 |
| 09 | upernet | resnet50 |
| 10 | segformer | mit_b0 |
| 11 | segformer | mit_b2 |
| 12 | segformer | mit_b5 |
| 13 | unet | tu-swin_tiny_patch4_window7_224 |


<a id="documentation"></a>

## Documentation

Use the linked guides for the schema, metric definitions, output interpretation, all-model commands and troubleshooting. Supporting technical documents are currently in English; the owner upload guide is in Vietnamese.

[Dataset](docs/DATASET.md) · [Protocol](docs/PROTOCOL.md) · [Outputs](docs/interpreting_outputs.md) · [All models](docs/RUNNING.md) · [Troubleshooting](docs/troubleshooting.md) · [Tests](docs/VALIDATION.md) · [Changelog](CHANGELOG.md)


<a id="support"></a>

## Scope, testing and support

Support covers the released code, project dataset schema and documented workflow. Readers adapting other datasets or methods are responsible for their changes and checks; adaptation support is outside this scope. Report released-code bugs through GitHub Issues, after removing private paths and secrets. No response time is guaranteed. Functional test status applies only to the tests and environment used; it does not establish scientific validity, spatial split independence or identical results across machines.

[GitHub Issues](https://github.com/hieudc-vub/dl-landslide-vietnam/issues) · [Contributing](CONTRIBUTING.md)

<a id="citation"></a>

## Citation and acknowledgements

Associated manuscript: *A Systematic Benchmark of Deep Learning Architectures for Landslide Inventory Mapping with Bi-Temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam*. Authors: Cong Hieu Duong, Jonathan Cheung-Wai Chan, Chinh Luu, Matthieu Kervyn. Manuscript preparing for submission; no article DOI is assigned here. Use CITATION.cff and cite the dataset separately using DOI [10.5281/zenodo.23838723](https://doi.org/10.5281/zenodo.23838723). Data-provider notices are documented separately; no institutional endorsement is implied.

[CITATION.cff](CITATION.cff) · [Data sources](docs/THIRD_PARTY_DATA.md)

<a id="licenses"></a>

## Licenses and translations

Project code: MIT. Prepared dataset: CC BY-NC 4.0 within the owner’s rights, with upstream notices retained. Third-party packages and initialization weights retain their own terms. Original polygons are excluded.

[MIT](LICENSE) · [Dataset license](LICENSE-DATA.md)

This README is available in English, French, Dutch and Vietnamese. English is the reference text at the repository root; translations are in docs/translations/. Translation corrections can be proposed through a pull request to the corresponding file. The four texts are aligned with release 0.4.0; this statement is not independent translation certification.
