import os
import shutil
import json
import re
import numpy as np
import nibabel as nib
from collections import OrderedDict

# ==========================================================================================
# --- CONFIGURATION ---
# ==========================================================================================

# 1. Set the path to your source directory containing the 'subj_XXX' folders.
SOURCE_DATA_DIR = "/path/to/partially_preprocessed_output"

# 2. Set the path where the new nnU-Net V2 dataset will be created.
NNUNET_RAW_DATA_DIR = "/path/to/nnUNet_raw"

# ----------------

# 3. Define your new dataset's name and ID. This will create a folder like 'Dataset0001_Glioblastoma'.
DATASET_ID = 2
DATASET_NAME = "GlioblastomaTimeSeries"
DATASET_PREFIX = f"{DATASET_NAME}"  # The prefix for each case, e.g., GlioblastomaTimeSeries_001

# 4. Define the mapping from source filenames to the nnU-Net channel index.
#    0000 will be channel 0, 0001 will be channel 1, etc.
#
#    Format: (Source Filename, nnU-Net Suffix, Human-Readable Name)
FILE_MAPPING = [
    ("MR_postop_ax_km_reg.nii.gz", "_0000.nii.gz", "T1_postop"),
    ("MR_pseudo_ax_km.nii.gz", "_0001.nii.gz", "T1_pseudo"),
    ("RD_reg_scaled.nii.gz", "_0002.nii.gz", "RTDOSE")
]

# 5. Define the source filename for the skull-stripping mask.
MASK_SOURCE_FILENAME = "MR_pseudo_ax_km_str_mask.nii.gz"

# 6. <<< NEW FLAG >>> Set to True to skip subjects if the mask is missing.
#    Set to False to process subjects even if the mask is not present.
REQUIRE_MASK_FILE = False # Set this to True or False

# ==========================================================================================
# --- SCRIPT LOGIC (No need to edit below this line) ---
# ==========================================================================================

def prepare_dataset():
    """
    Finds, verifies, and reorganizes the source data into the nnU-Net V2 raw data format.
    """
    # Create the main dataset directory path
    dataset_dir = os.path.join(NNUNET_RAW_DATA_DIR, f"Dataset{DATASET_ID:03d}_{DATASET_NAME}")

    # Create the required subdirectories
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    masks_dir = os.path.join(dataset_dir, "masks")  # For the preprocessing script's masks
    labels_tr_dir = os.path.join(dataset_dir, "labelsTr")  # Must exist for nnU-Net, but can be empty

    print(f"Creating directory structure at: {dataset_dir}")
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    # --- Data Processing Loop ---
    successful_cases = []
    print(f"\nScanning source directory: {SOURCE_DATA_DIR}")

    # Find all subject directories
    subject_dirs = sorted([d for d in os.listdir(SOURCE_DATA_DIR) if
                           d.startswith("subj_") and os.path.isdir(os.path.join(SOURCE_DATA_DIR, d))])

    for subject_dir_name in subject_dirs:
        subject_path = os.path.join(SOURCE_DATA_DIR, subject_dir_name)

        # Extract the subject ID number
        match = re.search(r'subj_(\d+)', subject_dir_name)
        if not match:
            print(f"Could not parse subject ID from '{subject_dir_name}'. Skipping.")
            continue
        subject_id = int(match.group(1))

        print(f"\n--- Processing Subject: {subject_dir_name} (ID: {subject_id}) ---")

        # 1. Verify that all required files exist for this subject
        missing_files = []
        # --- MODIFIED LOGIC ---
        # Start with the image files from the mapping
        all_required_files = [item[0] for item in FILE_MAPPING]
        # Only add the mask to the list of required files if the flag is True
        if REQUIRE_MASK_FILE:
            all_required_files.append(MASK_SOURCE_FILENAME)
        # --- END MODIFICATION ---

        for required_file in all_required_files:
            if not os.path.exists(os.path.join(subject_path, required_file)):
                missing_files.append(required_file)

        if missing_files:
            print(f"SKIPPING Subject {subject_id}. Reason: Missing {len(missing_files)} required file(s):")
            for f in missing_files:
                print(f"  - {f}")
            continue

        # 2. If all files are present, copy and rename them
        print("All required files found. Copying and renaming...")

        # nnU-Net case identifier
        case_identifier = f"{DATASET_PREFIX}_{subject_id:04d}"

        for source_filename, new_suffix, _ in FILE_MAPPING:
            source_path = os.path.join(subject_path, source_filename)
            dest_filename = f"{case_identifier}{new_suffix}"
            dest_path = os.path.join(images_tr_dir, dest_filename)

            shutil.copyfile(source_path, dest_path)
            print(f"  Copied '{source_filename}' to '{dest_filename}'")

        # --- MODIFIED LOGIC ---
        # Copy mask file only if it exists. This handles both cases:
        # 1. It's required and present.
        # 2. It's optional and present.
        # If it's optional and not present, this block is safely skipped.
        mask_source_path = os.path.join(subject_path, MASK_SOURCE_FILENAME)
        if os.path.exists(mask_source_path):
            mask_dest_filename = f"{case_identifier}_0000_bet.nii.gz"
            mask_dest_path = os.path.join(masks_dir, mask_dest_filename)
            shutil.copyfile(mask_source_path, mask_dest_path)
            print(f"  Copied '{MASK_SOURCE_FILENAME}' to '{mask_dest_filename}' (for preprocessing)")
        elif not REQUIRE_MASK_FILE:
             # This message is helpful when masks are optional but not found
            print(f"  - INFO: Optional mask file not found. Skipping mask copy.")
        # --- END MODIFICATION ---

        successful_cases.append(case_identifier)

    print(f"\n--- Conversion Summary ---")
    print(f"Successfully processed {len(successful_cases)} subjects.")

    if not successful_cases:
        print("No subjects were processed. Please check your configuration and source data.")
        return

    # --- Generate dataset.json ---
    print("Generating dataset.json...")

    channel_names = {str(i): name for i, (_, _, name) in enumerate(FILE_MAPPING)}

    dataset_json = OrderedDict([
        ("channel_names", channel_names),
        ("labels", {"background": 0}),  # Dummy label, required by nnU-Net
        ("numTraining", len(successful_cases)),
        ("file_ending", ".nii.gz"),
        ("overwrite_image_reader_writer", "SimpleITKIO")
    ])

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"dataset.json created at: {json_path}")
    print("\nDataset preparation complete. You can now run the preprocessing script.")


if __name__ == "__main__":
    prepare_dataset()