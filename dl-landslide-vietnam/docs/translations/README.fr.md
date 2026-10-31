[English](../../README.md) · **Français** · [Nederlands](README.nl.md) · [Tiếng Việt](README.vi.md)

# dl-landslide-vietnam

Procédures PyTorch pour la segmentation binaire des glissements de terrain

Code accompagnant une étude de cartographie des glissements de terrain à partir de Sentinel-2 bitemporel et d’ALOS AW3D30 dans le nord du Vietnam. Le logiciel fournit 13 configurations d’architecture pour des tuiles préparées selon le schéma du projet. Les lecteurs entraînent leurs propres modèles ; les poids entraînés des auteurs ne sont pas distribués.

[Installation et vérification de l’environnement](#installation) · [Jeu de données](#dataset) · [Démarrage](#quick-start) · [Citation et remerciements](#citation)

| Objectif | Consulter |
|---|---|
| Comprendre la procédure | [Comprendre la procédure](#overview) |
| Vérifier l’installation | [Vérifier l’installation](#installation) |
| Préparer les entrées | [Préparer les entrées](#dataset) |
| Exécuter un modèle | [Exécuter un modèle](#quick-start) |
| Trouver les fichiers générés | [Trouver les fichiers générés](#outputs) |
| Signaler un défaut du code | [Signaler un défaut du code](#support) |

<a id="overview"></a>

## Présentation

Entrée : tuiles de 256 × 256 pixels avec les 20 bandes ordonnées du projet et des étiquettes binaires (0 arrière-plan, 1 glissement de terrain). Traitement : entraînement, sélection du checkpoint et calibration du seuil sur VAL, évaluation sur TEST et prédiction. Les sorties sont créées localement lors de l’exécution. Ce dépôt ne contient aucun résultat de l’article.

<a id="before"></a>

## Avant de commencer

La procédure documentée utilise Windows et Python 3.12 (référence 3.12.8), avec PyTorch 2.9.0 et CUDA 12.8. Un GPU NVIDIA CUDA est utilisé pour l’entraînement FP16 décrit. Le CPU permet le test d’installation synthétique. Les besoins mémoire varient selon l’architecture ; aucune configuration GPU minimale n’est établie ici. Internet est nécessaire pour l’installation, les données et les poids initiaux ImageNet. Les dépendances directes sont fixées, sans verrouillage complet des dépendances transitives.

| Prérequis | Configuration décrite | Contrôle / aide |
|---|---|---|
| OS | Windows local | Ouvrir PowerShell dans le dossier du projet |
| Python | Python 3.12 ; référence 3.12.8 | Commande de version ci-dessous |
| GPU | CUDA pour l’entraînement FP16 décrit | Diagnostic de l’environnement après installation |
| Data | Jeu de données préparé du projet | Schéma et validation des données ci-dessous |

<a id="installation"></a>

## Installation et vérification de l’environnement

Téléchargez et extrayez le ZIP du dépôt. Ouvrez le dossier contenant README.md et pyproject.toml. Dans la barre d’adresse de l’Explorateur Windows, saisissez powershell et appuyez sur Entrée. Exécutez les commandes suivantes dans ce dossier ; elles créent un environnement propre au projet. Attendez SETUP COMPLETE. L’option de politique d’exécution ne concerne que ce processus.

Si Python manque, consultez la [page officielle Python 3.12.8](https://www.python.org/downloads/release/python-3128/) pour la version de référence. Activez le lanceur Python et l’option PATH dans l’installateur Windows, puis rouvrez PowerShell. Sur la [page du dépôt](https://github.com/hieudc-vub/dl-landslide-vietnam), choisissez **Code → Download ZIP**, puis **Extraire tout** dans Windows. Exécutez les commandes à la racine du projet extrait, même en lisant une traduction.

```powershell
py -3.12 --version
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Compute cuda
.\.venv\Scripts\python.exe -m landslide check-env
```

Exécutez ensuite le test d’installation synthétique. Il crée des données temporaires sans télécharger de données de recherche ni de poids préentraînés. Statut final attendu : OK. Il vérifie le fonctionnement logiciel, pas les performances scientifiques. Les sorties temporaires sont supprimées.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```


<a id="dataset"></a>

## Jeu de données

Jeu de données v1.0.0 : [Zenodo](https://doi.org/10.5281/zenodo.23838723). Téléchargez les trois archives ZIP raster ; le schéma et les manifestes sont fournis dans ce dépôt. Voir le [guide de téléchargement et de vérification](../DATASET_RELEASE.md). Une exécution complète nécessite SAMPLE_DATASET_256x256. Placez-le sous data/ comme indiqué, ou fournissez son chemin existant entre guillemets. Ne modifiez pas l’ordre des bandes et n’appliquez pas de normalisation supplémentaire. Les valeurs doivent être finies dans [-1,1], avec une tolérance de contrôle de 1e-5 ; les pixels invalides/NoData sont refusés. Les noms et métadonnées spatiales STACK/LABEL doivent correspondre.

```text
data/SAMPLE_DATASET_256x256/
├── TRAIN_256/   STACK/ + LABEL/
├── VAL_256/     STACK/ + LABEL/
└── TEST_256/    STACK/ + LABEL/
```

La configuration du projet déclare explicitement une échelle nominale de 10 m/pixel. La surface vaut le nombre de pixels × 100 m² ; cette échelle n’est pas déduite du CRS des tuiles. Une valeur pixel_size_m absente ou invalide est refusée. Le schéma et les mentions des fournisseurs sont détaillés ci-dessous. La préparation des données et leur répartition précèdent l’utilisation du logiciel.

<a id="quick-start"></a>

## Démarrage

Validez les données, puis lancez une configuration d’exemple. La validation lit et calcule l’empreinte de chaque raster. La commande run démarre un entraînement réel, calibre sur VAL, évalue TEST et écrit les sorties locales ; cela peut prendre un temps important. Utilisez un nouveau dossier de sortie à chaque exécution. L’architecture illustre les commandes sans prétendre à un avantage de performance.

```powershell
.\.venv\Scripts\python.exe -m landslide validate-data --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\dataset_validation
.\.venv\Scripts\python.exe -m landslide run --config .\configs\models\04_unetplusplus_resnet50.json --data-root .\data\SAMPLE_DATASET_256x256 --out .\runs\04_unetplusplus_resnet50 --device cuda
```

La prédiction est une commande distincte utilisant le checkpoint et le seuil produits par votre exécution :

```powershell
.\.venv\Scripts\python.exe -m landslide predict --checkpoint .\runs\04_unetplusplus_resnet50\best_model.pth --threshold .\runs\04_unetplusplus_resnet50\calibration\threshold.json --input .\data\SAMPLE_DATASET_256x256\TEST_256\STACK --out .\runs\04_predictions --device cuda
```


<a id="outputs"></a>

## Fichiers générés

L’arborescence suivante illustre uniquement les emplacements des fichiers. Checkpoints, tableaux, figures et rasters sont générés localement et exclus de Git par défaut. Aucune valeur mesurée ni image de résultat n’est fournie ici. Une valeur JSON null ou une cellule CSV vide indique une métrique indéfinie, pas zéro.

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

## Modèles et méthode

Les 13 configurations utilisent un batch de 16, un taux d’apprentissage de 1e-4, une décroissance des poids de 0.02, une graine de 42, au plus 150 époques, une patience de 20 et num_workers=0. Augmentation en ligne et échantillonnage pondéré sont désactivés. L’entraînement CUDA utilise FP16 autocast ; sélection du seuil, évaluation et export utilisent FP32. Le protocole définit les pertes, les statistiques de recouvrement conditionnelles, les objets à connexité 8 et les mesures de contour.

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

Consultez les guides pour le schéma, les métriques, les sorties, les commandes des 13 modèles et le dépannage. Les documents techniques complémentaires sont actuellement en anglais ; le guide de publication du propriétaire est en vietnamien.

[Dataset](../../docs/DATASET.md) · [Protocol](../../docs/PROTOCOL.md) · [Outputs](../../docs/interpreting_outputs.md) · [All models](../../docs/RUNNING.md) · [Troubleshooting](../../docs/troubleshooting.md) · [Tests](../../docs/VALIDATION.md) · [Changelog](../../CHANGELOG.md)


<a id="support"></a>

## Périmètre, tests et assistance

L’assistance couvre le code publié, le schéma du projet et la procédure documentée. Toute adaptation à d’autres données ou méthodes relève du lecteur et de ses propres vérifications ; elle sort de ce périmètre. Signalez les défauts du code publié via GitHub Issues après suppression des chemins privés et secrets. Aucun délai de réponse n’est garanti. Le statut des tests concerne uniquement les tests et l’environnement utilisés ; il n’établit ni validité scientifique, ni indépendance spatiale des ensembles, ni identité des résultats entre machines.

[GitHub Issues](https://github.com/hieudc-vub/dl-landslide-vietnam/issues) · [Contributing](../../CONTRIBUTING.md)

<a id="citation"></a>

## Citation et remerciements

Manuscrit associé : *A Systematic Benchmark of Deep Learning Architectures for Landslide Inventory Mapping with Bi-Temporal Sentinel-2 and ALOS AW3D30 in Northern Vietnam*. Auteurs : Cong Hieu Duong, Jonathan Cheung-Wai Chan, Chinh Luu, Matthieu Kervyn. Manuscrit en préparation pour soumission ; aucun DOI d’article n’est indiqué ici. Utilisez CITATION.cff et citez séparément les données avec le DOI [10.5281/zenodo.23838723](https://doi.org/10.5281/zenodo.23838723). Les mentions des fournisseurs sont documentées séparément ; aucune approbation institutionnelle n’est sous-entendue.

[CITATION.cff](../../CITATION.cff) · [Data sources](../../docs/THIRD_PARTY_DATA.md)

<a id="licenses"></a>

## Licences et traductions

Code du projet : MIT. Données préparées : CC BY-NC 4.0 dans les limites des droits du propriétaire, avec maintien des mentions amont . Les paquets tiers et poids d’initialisation conservent leurs propres conditions. Les polygones originaux sont exclus.

[MIT](../../LICENSE) · [Dataset license](../../LICENSE-DATA.md)

Ce README est disponible en anglais, français, néerlandais et vietnamien. Le texte anglais à la racine sert de référence ; les traductions sont dans docs/translations/. Proposez une correction par pull request sur le fichier concerné. Les quatre textes correspondent à la version 0.4.0 ; cette mention ne constitue pas une certification indépendante des traductions.
