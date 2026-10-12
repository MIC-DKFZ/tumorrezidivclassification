# Dose-Informed Deep Learning for Differentiating In-Field Glioblastoma Recurrence From Radiation-Induced Contrast Enhancement

Code accompanying the manuscript *"Dose-Informed Deep Learning for
Differentiating In-Field Glioblastoma Recurrence From Radiation-Induced
Contrast Enhancement"* (submitted to the International Journal of Radiation
Oncology, Biology, Physics — "Red Journal"). Corresponding author: Lars B.
Wessel (lars.wessel@med.uni-heidelberg.de), Department of Radiation
Oncology, Heidelberg University Hospital.

> Please update this section with the final citation (authors, journal,
> volume/issue, DOI) once the manuscript is accepted and published.

## License and usage restrictions

This code is released for **academic, non-commercial research use only**.
It is licensed under **CC BY-NC 4.0** — see [LICENSE](LICENSE). In
particular, **industrial and commercial use is not permitted**.

This repository contains only code written by this project's authors —
where we needed functionality that exists elsewhere, we import it directly
from the real upstream package rather than vendoring a copy, and two
optional (unused-by-the-published-model) training techniques were removed
entirely rather than republished under our license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exactly what that
means for `datasets/preprocess_3D_data/`, and for links to add the optional
pieces back yourself if you want them.

This code has **not** been validated or certified for clinical use and must
not be used to inform real patient care or diagnosis.

## What this is

This repository contains the imaging preprocessing, model training,
inference, and evaluation code used to differentiate in-field glioblastoma
recurrence from radiation-induced contrast enhancement (RICE) from
longitudinal MRI and the radiotherapy dose distribution. It is the subset
of our internal research codebase that was actually used to produce the
results reported in the manuscript; other experiments, ablations, and
unrelated projects that happened to share the same development repository
were intentionally left out (see THIRD_PARTY_NOTICES.md for a short note on
what was removed and why).

**No patient data is included.** Per the manuscript's data availability
statement, the underlying imaging and clinical data cannot be shared
publicly due to institutional ethics and data-protection requirements;
qualified researchers may request pseudonymized data from the corresponding
author under a data-use agreement. This repository is provided so that the
method is transparent and reproducible on your own, equivalently
structured, cohort.

## Method overview

- **Input**: three co-registered, 1&nbsp;mm isotropic, skull-stripped,
  z-score-normalized 3D volumes per case, cropped/resampled to
  224×224×224: a postoperative contrast-enhanced T1 MRI ("pMRI"), an
  event/index-timepoint contrast-enhanced T1 MRI ("eMRI"), and the
  co-registered RT dose distribution ("RD"). Channels are concatenated
  (early fusion); any subset of the three can be used (see
  `data.module.selected_modalities` below) to reproduce the single-,
  dual-, and tri-modality configurations reported in the paper.
- **Model**: a 3D ResNet-18 (`monai.networks.nets.resnet18`), trained
  **from scratch** (no pretraining/transfer learning), with a dropout
  layer before the final linear classifier (`models/monai_resnet.py`).
- **Training**: 5-fold patient-level cross-validation, Adam, Hydra +
  PyTorch Lightning. See `cli_configs/data/tumorrez.yaml` for the exact
  hyperparameters used for the reported results.
- **Evaluation**: macro F1 (primary), ROC AUC, Matthews correlation
  coefficient, accuracy, with bootstrap confidence intervals.

## Installation

```bash
# Recommended: uv (https://docs.astral.sh/uv/getting-started/installation/)
uv lock      # no uv.lock is checked in; this resolves one for your platform
uv sync
source .venv/bin/activate
```

or with plain pip:

```bash
pip install -r requirements.txt
```

`hd-bet`, `nnunetv2`, and `batchgeneratorsv2` pull in a sizeable scientific
Python / medical-imaging stack; expect the first install to take a while.

## Repository structure

```
main.py                     training entry point (Hydra + PyTorch Lightning)
inference.py                loads trained checkpoint(s) and runs evaluation
base_model.py                LightningModule: losses, metrics, optimizers/schedulers
parsing_utils.py            small OmegaConf resolver helpers used by Hydra configs

models/monai_resnet.py      the model: MONAI 3D ResNet-18 + dropout head

datasets/tumorrez.py        dataset / datamodule for the REZ-vs-RICE cohort
datasets/base_datamodule.py generic LightningDataModule base class
datasets/blosc2io.py        Blosc2 compressed-array I/O helper
datasets/preprocess_3D_data/   mask-centered cropping and Blosc2 I/O
                                helpers specific to this project (resampling,
                                skull-stripping, normalization and k-fold
                                splitting are imported directly from the
                                nnunetv2/hd-bet packages instead, see
                                THIRD_PARTY_NOTICES.md)

data_preparation/           end-to-end scripts turning raw DICOM into the
                             Blosc2 + labels.json/splits.json format the
                             datamodule expects (see "Data preparation")

augmentation/policies/batchgenerators.py   training-time 3D augmentation pipeline
metrics/                    custom torchmetrics (balanced accuracy, confusion matrix logging)

cli_configs/                Hydra configuration (train.yaml, infer.yaml,
                             data/tumorrez.yaml, model/monai_resnet.yaml,
                             env/{local,cluster}.yaml)
```

## Data preparation

Starting from DICOM, the scripts in `data_preparation/` are intended to be
run roughly in this order (each has a `# --- Configuration ---` block at
the top with placeholder paths — edit those for your own environment
before running):

1. `streamlined_preprocessing.py` — DICOM → NIfTI (`dcm2niix`) and
   co-registration/scaling (ANTs) based on a SQLite database of DICOM
   metadata (`sqlite_utilities.py` helps build/merge that database) and a
   CSV of manually- or heuristically-selected series UIDs per case
   (`csv_defined_uid_creation_script.py` can bootstrap that CSV).
2. `create_nnunet_ready_dataset.py` — restructure the co-registered NIfTI
   files into an nnU-Net-style raw dataset directory.
3. `preprocess_v2.py` — skull-stripping (HD-BET), resampling to 1mm
   isotropic, cropping to 224×224×224, z-score normalization, and Blosc2
   compression; also writes a `splits.json` for cross-validation (replace
   with a split that matches your own cohort if class balance matters).
4. `labels_csv_to_json.py` — converts a simple `PatID,label` CSV into the
   `labels.json` the datamodule expects.

`convert_blosc_back_to_nii.py` is a utility to convert a processed `.b2nd`
volume back to NIfTI for visual inspection/debugging.

## Pipeline components, and what to search for if you want alternatives

Rough map of what each stage of the pipeline actually does and which
library performs it here, so that if you want to swap a step out (e.g. you
don't have access to the exact tool we used, or you're on a platform where
it doesn't build) you know what kind of thing to search for instead of
starting from scratch:

| Stage | What it does | Used here | Search for, if you need an alternative |
|---|---|---|---|
| DICOM → NIfTI | Converts raw DICOM series to NIfTI volumes | `dcm2niix` (external binary, not a pip package) | "dcm2niix", or `pydicom`/`SimpleITK`-based DICOM readers if you want to stay in Python |
| Co-registration | Rigid/affine-aligns pMRI, eMRI, and RD to a common reference frame | ANTs (`antsRegistration`/`antsApplyTransforms` binaries) | "ANTs image registration", or `ANTsPy`/`SimpleITK` (`sitk.ImageRegistrationMethod`) for a Python-native option |
| Skull stripping | Removes non-brain tissue from the MRI volumes | **HD-BET** (`hd-bet` pip package, MIC-DKFZ) — a deep-learning brain extraction tool, used via `HD_BET.hd_bet_prediction` | "HD-BET brain extraction", or classical tools like FSL `BET` / `SynthStrip` as alternatives |
| Resampling to isotropic spacing | Resamples all volumes to 1&nbsp;mm isotropic voxel spacing | **nnU-Net v2** (`nnunetv2` pip package) resampling utilities, used via `nnunetv2.preprocessing.resampling.default_resampling` | "nnU-Net resampling", or more generally "anisotropic medical image resampling"; `scipy.ndimage.zoom`/`map_coordinates`, `MONAI`'s `Spacing` transform, or `SimpleITK`'s `Resample` are the usual building blocks |
| Intensity normalization | Z-score normalizes voxel intensities (optionally only inside a brain/foreground mask) | **nnU-Net v2** normalization schemes, used via `nnunetv2.preprocessing.normalization.default_normalization_schemes` | "nnU-Net normalization schemes" / "z-score normalization medical imaging" |
| Mask-centered cropping | Crops the volume to a fixed-size box centered on the brain mask | our own code, `datasets/preprocess_3D_data/crop_to_mask.py` | if reimplementing: MONAI's `CropForeground`/`SpatialCrop`, or `batchgeneratorsv2`'s cropping transforms |
| Compressed on-disk storage of 3D arrays | Stores large preprocessed volumes efficiently as chunked, compressed arrays | **Blosc2** (`blosc2` pip package), see `datasets/blosc2io.py` and `datasets/preprocess_3D_data/blosc_helper.py` | "Blosc2 NDArray", or just use NIfTI/`.npy` directly if storage size/throughput isn't a concern |
| Patient-level k-fold split | Creates the 5 cross-validation folds | `sklearn.model_selection.KFold` (used directly, see `cli_configs/data/tumorrez.yaml` → `data.cv.k`), conceptually the same as nnU-Net's `nnunetv2.utilities.crossval_split.generate_crossval_split` | "scikit-learn KFold" / "nnU-Net cross validation split" |
| Training-time augmentation | Elastic deformation, rotation, scaling, mirroring, Gaussian noise/blur, brightness/contrast/gamma jitter, applied jointly across all input channels | **`batchgeneratorsv2`** (MIC-DKFZ pip package), see `augmentation/policies/batchgenerators.py` | "batchgeneratorsv2", or `MONAI` transforms / `TorchIO` as alternative 3D medical-imaging augmentation libraries |
| 3D classification backbone | 3D ResNet-18 feature extractor + classifier head | **MONAI** (`monai.networks.nets.resnet18`), see `models/monai_resnet.py` | "MONAI resnet18", or `torchvision.models.video.r3d_18` / `timm` 3D backbones as alternatives |
| Experiment configuration & training loop | Config composition/CLI overrides, the actual training/validation loop, checkpointing, logging | **Hydra** + **OmegaConf** (config), **PyTorch Lightning** (training loop), **Weights & Biases** (logging) | these are standard, widely documented frameworks — search their own docs directly |
| (optional, **not** used by the published model) mixup / SAM optimizer | alternative regularization/optimization techniques, disabled by default | not included, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact upstream links | "mixup data augmentation pytorch" / "sharpness-aware minimization pytorch" |

## Training

```bash
export EXPERIMENT_LOCATION=/path/to/experiments
export DATASET_LOCATION=/path/to/preprocessed_dataset
uv run main.py env=cluster data=tumorrez model=monai_resnet \
    data.module.binary_task=1_vs_2 \
    data.module.selected_modalities=[0,1,2]
```

(use `env=local` and set `exp_dir`/`data_dir` directly in
`cli_configs/env/local.yaml` instead, if you're not using environment
variables.)

- `data.module.binary_task`: one of `0_vs_1`, `0_vs_2`, `1_vs_2` to select a
  pairwise comparison between the three classes (class indices follow the
  order in your `labels.json`); the subsampled subset is automatically
  class-balanced.
- `data.module.selected_modalities`: a list of channel indices
  (0=pMRI, 1=eMRI, 2=RD) to reproduce the single-/dual-/tri-modality
  ablations reported in the paper; omit for all channels. Update
  `model.input_channels` to match the number of channels you select.

The config loops over `data.cv.k` folds automatically; checkpoints and
Weights & Biases logs are written under `${exp_dir}/${data.module.name}`.

## Inference

```bash
uv run inference.py exp_dir=/path/to/experiments/Dataset001_GlioblastomaTimeSeries/checkpoints/<run_id>
```

Without `fold=<k>`, predictions from all available fold checkpoints found
under `exp_dir` are ensembled (summed logits).

## Citation

If you use this code, please cite the manuscript above (citation to be
completed once published) and acknowledge the third-party components listed
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) as appropriate.
