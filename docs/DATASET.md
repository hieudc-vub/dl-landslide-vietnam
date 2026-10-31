# Dataset contract

The expected public dataset contains 3600 TRAIN pairs, 450 VAL pairs and 450 TEST
pairs. Each STACK and LABEL has the same filename within its split. The loader
does not modify the split assignment. Adaptation to other datasets or methods is outside the documented support scope.

STACK: exactly 20 bands, 256×256, finite Float32-compatible values in [-1,1].
The range check permits an endpoint tolerance of 1e-5 without altering values.
Required ordered TIFF band descriptions:

```text
post_B4, post_B3, post_B2, post_B8, post_B7, post_B11, post_B12, post_BSI,
pre_B4, pre_B3, pre_B2, slope_norm,
D_B2_norm, D_B3_norm, D_B4_norm, D_B5_norm, D_B8_norm, D_B12_norm,
D_NDVI_norm, D_BSI_norm
```

Read these 20 packed channels directly, indexed 1..20. Do not apply the old
49-channel canonical indices to these files. Canonical schema documents may
describe upstream inputs; the actual model input contract is the ordered list above.

LABEL: one 256×256 band, values exactly 0/1, where 1 is landslide foreground.
Masks with NoData/invalid pixels are rejected: this release does not infer a
semantic-validity mask or silently turn invalid pixels into background.

Prepared patches include offline processing and repeated source variants. The
validator checks file pairing, raster values, descriptions, matching STACK/LABEL CRS and transform, file SHA-256 and exact
paired pixel duplicates across splits. It does not establish AOI separation,
source-group independence, label provenance or geographical accuracy.

The inspected public-patch format has no CRS and an identity-style grid transform.
Thus nominal 10 m scale is a protocol assumption, not a measurement recovered
from these TIFFs. Preserve these metadata honestly; do not assign a synthetic
geographic CRS or derive original polygons from them for redistribution.

Only STACK/LABEL redistribution is authorized by the dataset owner. Original
polygons are outside this release. The intended dataset license is CC BY-NC 4.0 within the owner's rights, with
upstream notices retained. The dataset v1.0.0 DOI is [10.5281/zenodo.23838723](https://doi.org/10.5281/zenodo.23838723). The three raster archives are distributed on Zenodo; schema and manifests are supplied in this repository. See [download instructions](DATASET_RELEASE.md).
