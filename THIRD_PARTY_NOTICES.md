# Third-Party Notices

This repository contains only code written by this project's authors. Where
our pipeline needs functionality that originates elsewhere, we import it
directly from the upstream package (already a declared dependency) rather
than vendoring a copy, or we document the external source instead of
including its code. **This means no third-party source code is shipped in
this repository** — but several of our files call into third-party
packages, and some optional features require code we deliberately did not
include. Read this before assuming anything here is "ours" to relicense.

## How we verified this

For every file that looked plausibly non-original (recognizable algorithm,
matching filenames/signatures seen elsewhere, or an explicit attribution
comment already present in the file), we compared it directly against the
actual upstream source on GitHub rather than guessing from familiarity. Two
files were confirmed as original after a web search turned up no matching
public implementation (`datasets/preprocess_3D_data/crop_to_mask.py`,
`datasets/preprocess_3D_data/blosc_helper.py`) — "no match found" is weaker
evidence than a positive match, so treat that determination as provisional
and re-check yourself if you have reason to doubt it.

## Removed: now imported directly from their real upstream package instead

These were previously vendored copies in `datasets/preprocess_3D_data/`. We
compared them line-by-line against the actual upstream files and found
them to be verbatim or near-verbatim copies, so we deleted the local
copies and import the real thing instead — both packages were already
project dependencies:

| Previously vendored as | Now imported from | Upstream source | License |
|---|---|---|---|
| `hd_bet_prediction.py` (`hdbet_predict`, `get_hdbet_predictor`) | `HD_BET.hd_bet_prediction` (the `hd-bet` pip package) | https://github.com/MIC-DKFZ/HD-BET/blob/master/HD_BET/hd_bet_prediction.py — identical function names and signatures | Apache-2.0 |
| `default_resampling.py` (`resample_data_or_seg_to_spacing`, etc.) | `nnunetv2.preprocessing.resampling.default_resampling` (the `nnunetv2` pip package) | https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/resampling/default_resampling.py — identical function names and signatures | Apache-2.0 |
| `normalization.py` (`ZScoreNormalization`) | `nnunetv2.preprocessing.normalization.default_normalization_schemes` | https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/normalization/default_normalization_schemes.py — identical docstring and core logic in `ZScoreNormalization.run()` | Apache-2.0 |
| `cross_validation.py` (`generate_crossval_split`) | `nnunetv2.utilities.crossval_split` | https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/crossval_split.py — identical function signature | Apache-2.0 |

If you need these at a pinned version rather than whatever the installed
`nnunetv2`/`hd-bet` package currently provides, pin the dependency version
in `pyproject.toml` instead of vendoring the file again.

## Removed: optional features not included, obtain them yourself if you want them

These were previously vendored files implementing optional training
techniques that are **not used by the published model**
(`model.mixup`/`model.sam` are `False` by default and not overridden
anywhere in `cli_configs/`). Their own in-file docstrings already credited
an external source, so we removed them rather than republish someone
else's code under our license. `base_model.py` will raise a clear
`ImportError` pointing here if you set `model.mixup=True` or `model.sam=True`
without adding the file back yourself:

| Removed file | Get it from | Used for |
|---|---|---|
| `augmentation/mixup.py` | https://github.com/mzhaoshuai/Divide-and-Co-training/blob/main/model/mixup.py | `mixup_data`/`mixup_criterion`, optional, place at `augmentation/mixup.py` |
| `regularization/sam.py` | https://github.com/davda54/sam/blob/main/sam.py | `SAM` optimizer, optional, place at `regularization/sam.py` |

Check the current license of each repository yourself before using either.

## Kept: not found elsewhere, treated as original

`datasets/preprocess_3D_data/crop_to_mask.py` and
`datasets/preprocess_3D_data/blosc_helper.py` live in the same directory as
the four confirmed-vendored files above and were written by the same
project contributors, but a direct search (MIC-DKFZ nnU-Net, HD-BET, and
the vendored `nnssl` copy that used to be in this repository, plus a general
web search) found no matching public implementation. We kept these as
original work. If you independently know these came from somewhere else,
please remove/replace them and open an issue.

## Not vendored, just a dependency

`augmentation/policies/batchgenerators.py` is original code we wrote, but
it exclusively calls transform classes from the external `batchgeneratorsv2`
package (MIC-DKFZ, installed as an ordinary dependency, not vendored).
Likewise `models/monai_resnet.py` wraps `monai.networks.nets.resnet18` from
the `monai` package. Using a dependency's public API is not the same as
vendoring its code, so these files are listed here only for completeness,
not as "adapted" code.

## Runtime dependencies (not vendored, installed via pip/uv)

Each carries its own license; verify before relying on this list, which
reflects our best knowledge at the time of writing:

| Package | Typical license |
|---|---|
| PyTorch / torchvision | BSD-3-Clause |
| MONAI | Apache-2.0 |
| PyTorch Lightning | Apache-2.0 |
| Hydra / OmegaConf | MIT |
| nnU-Net v2 (`nnunetv2`) | Apache-2.0 |
| HD-BET | Apache-2.0 |
| `batchgeneratorsv2` | Apache-2.0 (MIC-DKFZ) |
| scikit-learn, scikit-image, SimpleITK, NiBabel, Blosc2, pandas, NumPy | BSD / Apache-2.0 / permissive |

## Not included in this release

The original development repository also contained a vendored copy of
**MIC-DKFZ's `nnssl`** self-supervised-pretraining repository (its own
project, licensed CC BY-SA 4.0) and a separate Kaapana clinical-deployment
integration, both removed entirely because they were not used to produce
the results reported in the accompanying publication. See the upstream
`nnssl` repository directly if you are interested in that (unused, in this
work) pretraining approach.
