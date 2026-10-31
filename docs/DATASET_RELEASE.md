# Dataset download, verification and citation

Dataset v1.0.0: [A Systematic Benchmark of Deep Learning Architectures for Landslide Inventory Mapping with Bi-Temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam](https://doi.org/10.5281/zenodo.23838723). DOI: 10.5281/zenodo.23838723.

The three raster archives are distributed on Zenodo. The schema and manifests are supplied with this code repository; no metadata ZIP is required. Retain this repository snapshot with your downloaded dataset.

## Windows instructions

Download these three files from the Zenodo record to `D:\LandslideDownload`:

- dataset-v1.0.0-train.zip
- dataset-v1.0.0-val.zip
- dataset-v1.0.0-test.zip

The checksum file for this three-archive layout is `assets/dataset_SHA256SUMS.txt`. Copy it to `D:\LandslideDownload` and name the copy `SHA256SUMS.txt`. The file lists exactly the three raster archives. An earlier checksum file on Zenodo may also list a metadata ZIP that is not distributed; use the three-entry checksum file supplied with this code snapshot.

From the code repository directory, after installing Python 3.12:

```powershell
py -3.12 .\scripts\verify_dataset.py --archives D:\LandslideDownload
```

Use Windows File Explorer **Extract All** for each ZIP and choose the SAME destination, `D:\LandslideWork`. Each ZIP already contains `dl-landslide-vietnam/data/SAMPLE_DATASET_256x256`. Replace the automatically suggested ZIP-specific extraction destination.

Verify the extracted rasters from the code repository directory:

```powershell
py -3.12 .\scripts\verify_dataset.py --data-root D:\LandslideWork\dl-landslide-vietnam\data\SAMPLE_DATASET_256x256
```

The script reads `metadata/raster_manifest.csv` from this code repository and checks all 9000 raster files. Dataset metadata are not expected to come from a fourth ZIP. Training and validation can use the same full dataset path with `--data-root`; follow the README installation and configuration instructions.

Each pair comprises one 20-band Float32 STACK and one UInt8 LABEL, both 256 by 256 pixels. Binary labels are 0/1. Total: 3600 train, 450 validation, 450 test pairs; 9000 TIFFs and 28624961404 raster bytes. No split changes or additional normalization are applied. Nominal 10 m is the evaluation convention; the patch TIFFs have no geographic CRS. Integrity checks do not establish annotation accuracy or spatial independence.

## Dataset citation

Duong, C. H., Chan, J., Luu, C., & Kervyn, M. (2026). *A Systematic Benchmark of Deep Learning Architectures for Landslide Inventory Mapping with Bi-Temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam* (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23838723

Machine-readable citation: [DATASET_CITATION.cff](../metadata/DATASET_CITATION.cff).

The version-specific DOI above identifies this dataset version. The all-versions DOI is https://doi.org/10.5281/zenodo.23838722. Use the version-specific DOI when reporting use of v1.0.0. Neither DOI identifies the software or a published article. The manuscript title is shared with the dataset; do not change the dataset resource type to an article.

Dataset license: CC BY-NC 4.0 within the owner's rights; source-provider notices remain applicable. Code and helper scripts: MIT. No original polygons, trained checkpoints or research performance results are distributed in this source repository.
