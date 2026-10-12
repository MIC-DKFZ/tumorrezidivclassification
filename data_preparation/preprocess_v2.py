import sys
import os
import json
import re
from multiprocessing import Pool
from collections import defaultdict
from typing import Tuple, List
import torch

# --- Important: Add project root to path for robust imports ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Building blocks imported directly from their upstream packages (both
# already project dependencies, both Apache-2.0, MIC-DKFZ) rather than
# vendored copies, so this repository only ships our own code. ---
from HD_BET.hd_bet_prediction import hdbet_predict, get_hdbet_predictor
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_spacing
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.utilities.crossval_split import generate_crossval_split

# --- Original helpers for this project (see datasets/preprocess_3D_data/) ---
from datasets.preprocess_3D_data.crop_to_mask import crop_center_with_padding_np, get_mask_center
from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params

# --- Standard library and third-party imports ---
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isfile, subfiles, maybe_mkdir_p


# ==========================================================================================
# --- NEW: hd_bet_predict function from template ---
# ==========================================================================================
def hd_bet_predict(in_folder: str):
    """
    Predicts brain masks for all images in a specified input folder using HD-BET.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing HD-BET predictor")
    # device=0 means using the first available GPU. Change to 'cpu' if you don't have a compatible GPU.
    predictor = get_hdbet_predictor(device=device)
    print("Starting HD-BET mask prediction...")
    hdbet_predict(
        join(in_folder, 'imagesTr'),
        join(in_folder, 'masks'),
        predictor,
        keep_brain_mask=True,
        compute_brain_extracted_image=False
    )
    print("HD-BET prediction complete.")


# ==========================================================================================
# --- OUR CUSTOMIZED run_case FUNCTION (Now with Skull Stripping) ---
# ==========================================================================================
def run_case(
        base_path: str,
        img_id: str,
        target_spacing: Tuple[float, float, float],
        crop_size: Tuple[int, int, int],
        output_dir: str,
        num_modalities: int,
        brain_extract: bool = True  # --- NEW PARAMETER to control stripping ---
) -> None:
    RTDOSE_MODALITY_INDEX = 2
    MAX_EXPECTED_DOSE_GY = 80.0

    mask_path = join(base_path, 'masks', img_id + '_0000_bet.nii.gz')

    if not isfile(mask_path):
        print(f"WARNING: Mask not found for case {img_id} at {mask_path}. Skipping this case.")
        return

    mask_img = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(mask_img)[np.newaxis, ...]
    original_spacing = mask_img.GetSpacing()[::-1]

    mask_1mm = resample_data_or_seg_to_spacing(mask_data, original_spacing, target_spacing, is_seg=True)
    center = get_mask_center(mask_1mm[0])
    resized_mask = crop_center_with_padding_np(mask_1mm[0], center, crop_size)

    for mod_id in range(num_modalities):
        img_path = join(base_path, 'imagesTr', img_id + f'_{mod_id:04d}.nii.gz')
        if not isfile(img_path):
            print(f"WARNING: Image file not found for case {img_id}, modality {mod_id}. Skipping modality.")
            continue
            continue

        img = sitk.ReadImage(img_path)
        data = sitk.GetArrayFromImage(img)[np.newaxis, ...]
        data_1mm = resample_data_or_seg_to_spacing(data, original_spacing, target_spacing)

        # --- NEW: Apply brain mask (skull stripping) before cropping ---
        if brain_extract:
            print(f"  - Applying brain mask (skull stripping) for {img_id}, modality {mod_id}...")
            # We use the resampled mask to match the resampled image
            data_1mm *= mask_1mm

        resized_img = crop_center_with_padding_np(data_1mm[0], center, crop_size)[np.newaxis, ...]

        if mod_id == RTDOSE_MODALITY_INDEX:
            print(f"  - Processing {img_id}_{mod_id:04d} with Dose Clipping/Scaling...")
            processed_data = np.clip(resized_img, 0, MAX_EXPECTED_DOSE_GY) / MAX_EXPECTED_DOSE_GY
        else:
            print(f"  - Processing {img_id}_{mod_id:04d} with Z-Score Normalization...")
            normalizer = ZScoreNormalization()
            normalizer.use_mask_for_norm = True
            processed_data = normalizer.run(resized_img, resized_mask[np.newaxis, ...])

        block_size_data, chunk_size_data = comp_blosc2_params(resized_img.shape, crop_size, processed_data.itemsize)
        out_path_truncated = join(output_dir, f'{img_id}_{mod_id:04d}')
        save_case(processed_data, out_path_truncated, chunks=chunk_size_data, blocks=block_size_data)


# ==========================================================================================
# --- Main Orchestrator Function (Now with brain_extract parameter) ---
# ==========================================================================================
def load_crop_brainextract_normalize_images(
        in_folder: str,
        out_folder: str,
        target_spacing: List[float],
        patch_size: List[int],
        brain_extract: bool = True, # --- NEW PARAMETER ---
        num_workers: int = 1
) -> None:
    maybe_mkdir_p(out_folder)
    all_images = subfiles(join(in_folder, 'imagesTr'), suffix=".nii.gz")
    unique_ids = list(set(re.sub(r'_\d{4}\.nii\.gz$', '', f.split(os.sep)[-1]) for f in all_images))
    pattern = re.compile(r'(.+?)_(\d{4})\.nii\.gz')
    modalities = defaultdict(set)
    for f in all_images:
        m = pattern.match(f.split(os.sep)[-1])
        if m:
            identifier, mod = m.groups()
            modalities[identifier].add(mod)

    expected_mods = set.union(*modalities.values())

    # --- Pass the new brain_extract flag to run_case ---
    args_list = [
        (in_folder, uid, target_spacing, patch_size, out_folder, len(expected_mods), brain_extract)
        for uid in unique_ids
    ]

    print(f"Starting preprocessing for {len(unique_ids)} cases with {num_workers} worker(s)...")
    with Pool(processes=num_workers) as pool:
        pool.starmap(run_case, args_list)

    print("Preprocessing complete. Generating cross-validation splits...")
    split_file = generate_crossval_split(unique_ids, n_splits=5)
    with open(os.path.join(out_folder, 'splits.json'), "w") as f:
        json.dump(split_file, f, indent=4)
    print(f"Splits file saved to {os.path.join(out_folder, 'splits.json')}")


# ==========================================================================================
# --- SCRIPT ENTRY POINT & CONFIGURATION ---
# ==========================================================================================
if __name__ == '__main__':
    NNUNET_RAW_DATA_DIR = "/path/to/nnUNet_raw"
    NNUNET_PREPROCESSED_DIR = "/path/to/downstream_class_preprocessed"
    DATASET_ID = 2
    DATASET_NAME = "GlioblastomaTimeSeries"

    target_spacing = [1.0, 1.0, 1.0]
    patch_size = [160, 160, 160]
    num_workers = 8

    # --- 3. CONFIGURE NEW FEATURES ---
    # Set to True to perform skull stripping (multiply image by mask).
    # Set to False to only use the mask for centering.
    PERFORM_BRAIN_EXTRACTION = True

    # Set to True to run HD-BET to generate masks for all subjects.
    # This will create/overwrite files in the 'masks' folder.
    # Set to False if you are confident you have all the masks you need.
    RUN_HD_BET = True
    # --- END NEW FEATURE CONFIGURATION ---

    dataset_folder_name = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    in_folder = os.path.join(NNUNET_RAW_DATA_DIR, dataset_folder_name)
    out_folder = os.path.join(NNUNET_PREPROCESSED_DIR, dataset_folder_name)

    print("=" * 50)
    print("Starting Data Preprocessing")
    print(f"Input Dataset: {in_folder}")
    print(f"Output Directory: {out_folder}")
    print(f"Perform skull stripping: {PERFORM_BRAIN_EXTRACTION}")
    print(f"Run HD-BET if needed: {RUN_HD_BET}")
    print("=" * 50)

    if RUN_HD_BET:
        hd_bet_predict(in_folder)

    load_crop_brainextract_normalize_images(
        in_folder=in_folder,
        out_folder=out_folder,
        target_spacing=target_spacing,
        patch_size=patch_size,
        brain_extract=PERFORM_BRAIN_EXTRACTION, # Pass the flag
        num_workers=num_workers
    )

    print("\nPreprocessing script finished successfully!")