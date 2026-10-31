[English](../../README.md) · [Français](README.fr.md) · **Nederlands** · [Tiếng Việt](README.vi.md)

# dl-landslide-vietnam

PyTorch-procedures voor binaire segmentatie van aardverschuivingen

Code bij een studie naar het in kaart brengen van aardverschuivingen met bitemporele Sentinel-2 en ALOS AW3D30 in Noord-Vietnam. Het pakket biedt 13 architectuurconfiguraties voor voorbereide beeldpatches volgens het projectschema. Lezers trainen hun eigen modellen; getrainde checkpoints van de auteurs worden niet verspreid.

[Installatie en omgevingscontrole](#installation) · [Dataset](#dataset) · [Aan de slag](#quick-start) · [Citeren en dankbetuiging](#citation)

| Taak | Ga naar |
|---|---|
| Procedure begrijpen | [Procedure begrijpen](#overview) |
| Installatie controleren | [Installatie controleren](#installation) |
| Invoer voorbereiden | [Invoer voorbereiden](#dataset) |
| Eén model uitvoeren | [Eén model uitvoeren](#quick-start) |
| Gegenereerde bestanden vinden | [Gegenereerde bestanden vinden](#outputs) |
| Een codefout melden | [Een codefout melden](#support) |

<a id="overview"></a>

## Overzicht

Invoer: patches van 256 × 256 pixels met de 20 geordende projectbanden en binaire labels (0 achtergrond, 1 aardverschuiving). Verwerking: training, checkpointselectie en drempelkalibratie op VAL, evaluatie op TEST en voorspelling. Uitvoer ontstaat lokaal tijdens de eigen uitvoering. Deze repository bevat geen resultaten uit het artikel.

<a id="before"></a>

## Voorbereiding

De beschreven procedure gebruikt Windows en Python 3.12 (referentie 3.12.8), PyTorch 2.9.0 met CUDA 12.8. NVIDIA CUDA wordt gebruikt voor de beschreven FP16-training. De CPU ondersteunt de synthetische installatiecontrole. Het geheugengebruik verschilt per architectuur; hier is geen minimale GPU-configuratie vastgesteld. Internet is nodig voor installatie, dataset en ImageNet-initialisatie. Directe afhankelijkheden zijn vastgelegd, maar dit is geen volledige vergrendeling van transitieve afhankelijkheden.

| Vereiste | Beschreven instelling | Controle / hulp |
|---|---|---|
| OS | Windows lokaal | Open PowerShell in de projectmap |
| Python | Python 3.12; referentie 3.12.8 | Versieopdracht hieronder |
| GPU | CUDA voor de beschreven FP16-training | Omgevingsdiagnose na installatie |
| Data | Voorbereide projectdataset | Schema en datacontrole hieronder |

<a id="installation"></a>

## Installatie en omgevingscontrole

Download en pak de ZIP van de repository uit. Open de map met README.md en pyproject.toml. Typ powershell in de adresbalk van Windows Verkenner en druk op Enter. Voer onderstaande opdrachten daar uit; ze maken een aparte projectomgeving. Wacht op SETUP COMPLETE. De optie voor het uitvoeringsbeleid geldt alleen voor dit proces.

Als Python ontbreekt, gebruik dan de [officiële Python 3.12.8-pagina](https://www.python.org/downloads/release/python-3128/) voor de referentieversie. Schakel in het Windows-installatieprogramma de Python-launcher en PATH-optie in en open PowerShell opnieuw. Kies op de [repositorypagina](https://github.com/hieudc-vub/dl-landslide-vietnam) **Code → Download ZIP** en daarna **Alles uitpakken** in Windows. Voer opdrachten uit vanuit de hoofdmap van het uitgepakte project, ook bij het lezen van een vertaling.

```powershell
py -3.12 --version
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Compute cuda
.\.venv\Scripts\python.exe -m landslide check-env
```

Voer daarna de synthetische installatiecontrole uit. Deze maakt tijdelijke gegevens zonder onderzoeksdata of vooraf getrainde gewichten te downloaden. Verwachte eindstatus: OK. Dit controleert softwaregedrag, geen wetenschappelijke prestaties. Tijdelijke uitvoer wordt verwijderd.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```


<a id="dataset"></a>

## Dataset

Dataset v1.0.0: [Zenodo](https://doi.org/10.5281/zenodo.23838723). Download de drie raster-ZIP-bestanden; het schema en de manifesten staan in deze repository. Zie de [download- en verificatiehandleiding](../DATASET_RELEASE.md). Voor de volledige procedure is SAMPLE_DATASET_256x256 nodig. Plaats deze onder data/ zoals afgebeeld, of geef het bestaande pad tussen aanhalingstekens op. Verander de bandvolgorde niet en pas geen extra normalisatie toe. Waarden moeten eindig zijn binnen [-1,1], met een controletolerantie van 1e-5; ongeldige/NoData-pixels worden geweigerd. Namen en ruimtelijke metadata van STACK en LABEL moeten overeenkomen.

```text
data/SAMPLE_DATASET_256x256/
├── TRAIN_256/   STACK/ + LABEL/
├── VAL_256/     STACK/ + LABEL/
└── TEST_256/    STACK/ + LABEL/
```

De projectconfiguratie verklaart expliciet een nominale schaal van 10 m/pixel. Oppervlakte is het aantal pixels × 100 m²; deze schaal wordt niet uit het CRS van de patches afgeleid. Een ontbrekende of ongeldige pixel_size_m wordt geweigerd. Schema en bronvermeldingen staan in de gekoppelde documentatie. Datavoorbereiding en indeling in deelverzamelingen vinden vooraf plaats.

<a id="quick-start"></a>

## Aan de slag

Controleer de dataset en voer vervolgens een voorbeeldconfiguratie uit. De controle leest en hasht elk raster. Het run-commando start echte training, kalibreert op VAL, evalueert TEST en schrijft lokale uitvoer; dit kan veel tijd kosten. Gebruik iedere keer een nieuwe uitvoermap. De voorbeeldarchitectuur illustreert opdrachten en impliceert geen prestatievoordeel.

```powershell
.\.venv\Scripts\python.exe -m landslide validate-data --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\dataset_validation
.\.venv\Scripts\python.exe -m landslide run --config .\configs\models\04_unetplusplus_resnet50.json --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\04_unetplusplus_resnet50 --device cuda
```

Voorspelling is een aparte opdracht met het checkpoint en de drempel die uw uitvoering heeft aangemaakt:

```powershell
.\.venv\Scripts\python.exe -m landslide predict --checkpoint .\runs\04_unetplusplus_resnet50\best_model.pth --threshold .\runs\04_unetplusplus_resnet50\calibration\threshold.json --input .\data\SAMPLE_DATASET_256x256\TEST_256\STACK --out .\runs\04_predictions --device cuda
```


<a id="outputs"></a>

## Gegenereerde bestanden

De onderstaande boom toont alleen bestandslocaties. Checkpoints, tabellen, figuren en rasters worden lokaal gegenereerd en standaard van Git uitgesloten. Hier staan geen meetwaarden of resultaatbeelden. JSON null of een leeg CSV-veld betekent dat een metriek niet gedefinieerd is, niet nul.

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

## Modellen en methode

Alle 13 configuraties gebruiken batchgrootte 16, leersnelheid 1e-4, gewichtsverval 0.02, seed 42, maximaal 150 epochs, early-stopping-patience 20 en num_workers=0. Online augmentatie en gewogen steekproeftrekking zijn uitgeschakeld. CUDA-training gebruikt FP16-autocast; drempelselectie, evaluatie en export gebruiken FP32. Het protocol beschrijft verliesfuncties, voorwaardelijke overlapstatistieken, 8-verbonden objecten en grensmaten.

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

## Documentatie

Raadpleeg de gekoppelde schema-, metriek-, uitvoer-, model- en probleemoplossingsgidsen. Aanvullende technische documenten zijn momenteel in het Engels; de uploadgids voor de eigenaar is in het Vietnamees.

[Dataset](../../docs/DATASET.md) · [Protocol](../../docs/PROTOCOL.md) · [Outputs](../../docs/interpreting_outputs.md) · [All models](../../docs/RUNNING.md) · [Troubleshooting](../../docs/troubleshooting.md) · [Tests](../../docs/VALIDATION.md) · [Changelog](../../CHANGELOG.md)


<a id="support"></a>

## Reikwijdte, tests en ondersteuning

Ondersteuning betreft de gepubliceerde code, het projectschema en de beschreven procedure. Aanpassing aan andere datasets of methoden valt onder de verantwoordelijkheid en controles van de lezer en buiten deze ondersteuning. Meld fouten in de gepubliceerde code via GitHub Issues nadat privépaden en geheimen zijn verwijderd. Er is geen gegarandeerde reactietijd. Teststatus geldt alleen voor de uitgevoerde tests en omgeving en bewijst geen wetenschappelijke validiteit, ruimtelijke onafhankelijkheid van deelverzamelingen of identieke resultaten op verschillende machines.

[GitHub Issues](https://github.com/hieudc-vub/dl-landslide-vietnam/issues) · [Contributing](../../CONTRIBUTING.md)

<a id="citation"></a>

## Citeren en dankbetuiging

Bijbehorend manuscript: *A Systematic Benchmark of Deep Learning Architectures for Landslide Inventory Mapping with Bi-Temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam*. Auteurs: Cong Hieu Duong, Jonathan Cheung-Wai Chan, Chinh Luu, Matthieu Kervyn. Het manuscript wordt voorbereid voor indiening; hier is geen artikel-DOI toegekend. Gebruik CITATION.cff en citeer de dataset afzonderlijk met DOI [10.5281/zenodo.23838723](https://doi.org/10.5281/zenodo.23838723). Bronvermeldingen zijn apart beschreven; institutionele goedkeuring wordt niet geïmpliceerd.

[CITATION.cff](../../CITATION.cff) · [Data sources](../../docs/THIRD_PARTY_DATA.md)

<a id="licenses"></a>

## Licenties en vertalingen

Projectcode: MIT. Voorbereide dataset: CC BY-NC 4.0 binnen de rechten van de eigenaar, met behoud van bronvoorwaarden. Pakketten van derden en initialisatiegewichten behouden hun eigen voorwaarden. Oorspronkelijke polygonen zijn uitgesloten.

[MIT](../../LICENSE) · [Dataset license](../../LICENSE-DATA.md)

Deze README is beschikbaar in het Engels, Frans, Nederlands en Vietnamees. De Engelse tekst in de hoofdmap is de referentie; vertalingen staan in docs/translations/. Stel vertaalcorrecties voor via een pull request op het betreffende bestand. De vier teksten horen bij versie 0.4.0; dit is geen onafhankelijke certificering van de vertalingen.
