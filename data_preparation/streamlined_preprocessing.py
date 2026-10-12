# imports
import os
import shutil
import pandas as pd
import logging
import SimpleITK as sitk
from pathlib import Path
import subprocess
import sqlite3
import warnings
import pydicom
import numpy as np

# --- Configuration ---
# Set these paths for your own environment before running this script.
# None of the paths below point at real data; the clinical source data used in
# the publication is not redistributed with this repository (see README.md).

# Paths
abs_data_path = '/path/to/raw_dicom_data'
raw_data_dir = abs_data_path
preprocessed_data_dir = '/path/to/partially_preprocessed_output'
binaries_dir = '/path/to/preprocessing_binaries'  # must contain dcm2niix and ANTs executables
sqlite_database_path = '/path/to/dicom_metadata.sqlite3'
predefined_IUID_path = "/path/to/defined_IUIDs.csv"
logfile_path = os.path.join(preprocessed_data_dir, 'log.txt')
local_temp_dir_path = "/tmp/tumorrez_preprocessing"

# Flags
flag_use_local_temp_dir = True

flag_rescan_source_dir = False

# Constants
mri_name_list = ["pseudo", "preop", "postop", "bestrahlung", "therapie", "posttherapie"]
modality_name_list = ["ax_km", "flair"]


# --- Utility Functions and Classes ---
def scale_rd_reg(rd_reg_path: str, subject_target_dir: str):
    """
    Reads the RD_reg.nii.gz file, determines the max intensity,
    applies the appropriate scaling factor, and saves the output.
    """
    try:
        # Load the registered RD image and get max value
        img = sitk.ReadImage(rd_reg_path)
        img_arr = sitk.GetArrayFromImage(img)
        max_val = img_arr.max()

        # Determine scaling factor based on smart boundaries
        factor = None
        factor_str = ""

        if max_val < 2.5: # case normal fractionation with single dosage in RTDOSE
            factor = 30.0
            factor_str = "30"
        elif 2.5 <= max_val < 2.845: # case normal fractionated with single dosage in RTDOSE
            factor = 25.0
            factor_str = "25"
        elif 2.845 <= max_val < 8.0: # case hyperfractionation with single dosage in RTDOSE
            factor = 15.0
            factor_str = "15"
        elif 8.0 <= max_val < 15.0: # case Proton treatment currently no scaling
            # Skip scaling for values around 10
            log_message(logger, f"Max RD intensity is {max_val:.2f} (close to 10). Excluding from scaling.",
                        level='warning')
            return
        elif 30.0 <= max_val < 48.0: # case hypofractionation with full dosage in RTDOSE
            factor = 1.0
            factor_str = "1"
        elif 48.0 <= max_val < 55.0:
            factor = 1.2  # equivalent to /25 * 30
            factor_str = "1.2"
        else:
            factor = 1.0
            factor_str = "1"

        # Apply the scaling factor
        scaled_arr = img_arr * factor
        scaled_img = sitk.GetImageFromArray(scaled_arr)
        scaled_img.CopyInformation(img)

        # Define output paths
        out_nifti = os.path.join(subject_target_dir, "RD_reg_scaled.nii.gz")
        out_txt = os.path.join(subject_target_dir, f"RD_scaled_by_{factor_str}.txt")

        # Save scaled NIfTI
        sitk.WriteImage(scaled_img, out_nifti)

        # Create empty tracking text file
        with open(out_txt, 'w') as f:
            pass

        log_message(logger, f"Scaled RD_reg by {factor_str} (Max val was {max_val:.2f}). Saved to {out_nifti}")

    except Exception as e:
        log_message(logger, f"Failed to scale {rd_reg_path}. Error: {e}", level='error')


def rescale_to_isotropic(image: sitk.Image, target_spacing: float | None = None):
    # Get original spacing and size
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    # Check if the image is already isotropic
    if np.allclose(original_spacing, original_spacing[0], rtol=1e-5, atol=1e-8):
        print("Image was already isotropic")
        return image  # Image is already isotropic, return without changes

    # Find the smallest spacing (highest resolution)
    min_spacing = np.min(original_spacing)

    # Use target_spacing if provided, otherwise use min_spacing
    new_spacing = target_spacing if target_spacing is not None else min_spacing

    # Calculate new size
    new_size = np.round(original_size * original_spacing / new_spacing).astype(int)

    # Create a new image with isotropic spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([new_spacing] * 3)
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    isotropic_image = resampler.Execute(image)

    return isotropic_image

def setup_logger(logfile: str) -> logging.Logger:
    """Sets up a logger to file."""
    logger = logging.getLogger('preprocessing_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


def log_message(logger: logging.Logger, message: str, level: str = 'info'):
    """Logs a message with the specified level."""
    if level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    else:
        logger.info(message)


class PreDefinedUIDs:
    def __init__(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path, dtype=str)
            if 'PatID' not in self.df.columns or 'Verschluesselungs ID' not in self.df.columns:
                 raise ValueError("CSV must contain both 'PatID' and 'Verschluesselungs ID' columns.")
            self.df['PatID'] = self.df['PatID'].astype(int)
        except Exception as e:
            log_message(logger, f"Error loading or validating predefined UIDs CSV: {e}", level='error')
            self.df = None

    def get_uid(self, pat_id: int, event: str, modality: str = None) -> str | None:
        if self.df is None: return None
        try:
            pat_id = int(pat_id)
            column_name = f"{event}_{modality}" if modality else event
            patient_row = self.df[self.df['PatID'] == pat_id]
            if patient_row.empty: return None
            if column_name in patient_row.columns:
                uid = patient_row[column_name].iloc[0]
                return uid if pd.notna(uid) else None
            return None
        except Exception as e:
            log_message(logger, f"Could not retrieve UID for PatID {pat_id}, event '{event}', modality '{modality}'. Error: {e}", level='warning')
            return None


def copy_files(file_paths: list, destination_dir: str):
    """Copies a list of files to a destination directory."""
    os.makedirs(destination_dir, exist_ok=True)
    for file_path in file_paths:
        if os.path.exists(file_path):
            shutil.copy(file_path, destination_dir)
        else:
            log_message(logger, f"File not found during copy: {file_path}", level='warning')


def run_command(command: str):
    """Executes a shell command."""
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        log_message(logger, f"Command failed: {e.stderr.decode()}", level='error')


def ants_registration(moving_file_path: str, static_file_path: str, output_prefix: str):
    """Performs rigid registration using the detailed ANTs command from the original script."""
    # This command creates the output file as `output_prefix`.nii.gz and the transform as `output_prefix`0GenericAffine.mat
    output_files_arg = f"[{output_prefix},{output_prefix}.nii.gz]"
    command = (
        f"{binaries_dir}/ants/bin/antsRegistration --verbose 0 --dimensionality 3 --float 0 "
        f"--collapse-output-transforms 1 --output {output_files_arg} --interpolation Linear "
        f"--use-histogram-matching 0 --winsorize-image-intensities [0.005,0.995] "
        f"--initial-moving-transform [{static_file_path},{moving_file_path},1] "
        f"--transform Rigid[0.1] "
        f"--metric MI[{static_file_path},{moving_file_path},1,32,Regular,0.25] "
        f"--convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox"
    )
    run_command(command)


def empty_directory(directory: str):
    """Deletes all contents of a directory."""
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def get_dicom_paths_from_uid(cursor: sqlite3.Cursor, patient_id: str, series_uid: str) -> list:
    """Fetches DICOM file paths for a given SeriesInstanceUID."""
    query = "SELECT file_path FROM file_properties WHERE PatientID = ? AND SeriesInstanceUID = ?"
    cursor.execute(query, (patient_id, series_uid))
    return [row[0] for row in cursor.fetchall()]

def index_properties_to_sqlite(start_path: str | os.PathLike, dbpath: str | os.PathLike,
                               max_workers: int = 16, batch_size: int = 1000) -> None:
    """
    Index DICOM file properties into SQLite database using parallel processing.

    Args:
        start_path: Root directory containing DICOM files
        dbpath: Path to SQLite database file
        max_workers: Maximum number of parallel workers (default: 8)
        batch_size: Number of records to insert in each batch (default: 1000)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import logging
    from tqdm import tqdm
    import time
    from collections import Counter

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize database connection
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()

    try:
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_properties (
            file_path TEXT PRIMARY KEY,
            PatientID TEXT,
            StudyInstanceUID TEXT,
            StudyID TEXT,
            StudyDate TEXT,
            SeriesInstanceUID TEXT,
            Modality TEXT,
            SeriesDescription TEXT,
            ProtocolName TEXT,
            SequenceName TEXT,
            ImageType TEXT,
            DoseSummationType TEXT,
            isProjection BOOL,
            RepetitionTime INTEGER,
            InversionTime INTEGER,
            FlipAngle INTEGER,
            EchoTime REAL
        )
        ''')
        conn.commit()

        # Get existing file paths from database
        cursor.execute('SELECT file_path FROM file_properties')
        existing_files = set(row[0] for row in cursor.fetchall())

        # Track specific error types
        error_details = Counter()

        def process_file(file_path: str) -> tuple[str, tuple | None, str | None]:
            """
            Process a single DICOM file and return its properties.
            Returns tuple of (status, data, error_type) where:
            - status is one of: 'existing', 'success', 'error'
            - data is the DICOM data tuple or None
            - error_type is the error description or None
            """
            if file_path in existing_files:
                return ('existing', None, None)

            try:
                ds = pydicom.read_file(file_path, stop_before_pixels=True)
            except Exception as e:
                error_type = f"DICOM read error: {type(e).__name__}"
                return ('error', None, error_type)

            try:
                # Convert ImageType to string if it exists
                image_type = ','.join(ds.ImageType) if hasattr(ds, 'ImageType') else None
                is_projection = 'PROJECTION IMAGE' in (ds.ImageType if hasattr(ds, 'ImageType') else [])

                return ('success', (
                    file_path,
                    getattr(ds, 'PatientID', None),
                    getattr(ds, 'StudyInstanceUID', None),
                    getattr(ds, 'StudyID', None),
                    getattr(ds, 'StudyDate', None),
                    getattr(ds, 'SeriesInstanceUID', None),
                    getattr(ds, 'Modality', None),
                    getattr(ds, 'SeriesDescription', None),
                    getattr(ds, 'ProtocolName', None),
                    getattr(ds, 'SequenceName', None),
                    image_type,
                    getattr(ds, 'DoseSummationType', None),
                    is_projection,
                    int(ds.RepetitionTime) if hasattr(ds, 'RepetitionTime') else None,
                    int(ds.InversionTime) if hasattr(ds, 'InversionTime') else None,
                    int(ds.FlipAngle) if hasattr(ds, 'FlipAngle') else None,
                    float(ds.EchoTime) if hasattr(ds, 'EchoTime') else None
                ), None)
            except Exception as e:
                error_type = f"Data extraction error: {type(e).__name__}"
                return ('error', None, error_type)

        # Collect all files
        all_files = []
        logger.info("Collecting files...")
        for root, _, files in os.walk(start_path):
            all_files.extend(os.path.join(root, f) for f in files)

        total_files = len(all_files)
        logger.info(f"Found {total_files} files to process")

        # Process files in parallel with progress bar
        current_batch = []
        processed_count = 0
        existing_count = 0
        error_count = 0
        db_error_count = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in all_files}

            with tqdm(total=total_files, desc="Processing files", mininterval=10) as pbar:
                for future in as_completed(futures):
                    status, result, error_type = future.result()

                    if status == 'existing':
                        existing_count += 1
                    elif status == 'error':
                        error_count += 1
                        error_details[error_type] += 1
                    else:  # status == 'success'
                        current_batch.append(result)

                        # Insert batch if it reaches batch_size
                        if len(current_batch) >= batch_size:
                            try:
                                cursor.executemany('''
                                INSERT INTO file_properties (
                                    file_path, PatientID, StudyInstanceUID, StudyID, 
                                    StudyDate, SeriesInstanceUID, Modality, SeriesDescription, 
                                    ProtocolName, SequenceName, ImageType, DoseSummationType,
                                    isProjection, RepetitionTime, InversionTime, FlipAngle, EchoTime
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', current_batch)
                                conn.commit()
                                processed_count += len(current_batch)
                            except Exception as e:
                                error_type = f"Database insertion error: {type(e).__name__}"
                                error_details[error_type] += len(current_batch)
                                db_error_count += len(current_batch)
                                logger.error(f"Error inserting batch: {str(e)}")
                            finally:
                                current_batch = []

                    pbar.update(1)

        # Insert any remaining records
        if current_batch:
            try:
                cursor.executemany('''
                INSERT INTO file_properties (
                    file_path, PatientID, StudyInstanceUID, StudyID, 
                    StudyDate, SeriesInstanceUID, Modality, SeriesDescription, 
                    ProtocolName, SequenceName, ImageType, DoseSummationType,
                    isProjection, RepetitionTime, InversionTime, FlipAngle, EchoTime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', current_batch)
                conn.commit()
                processed_count += len(current_batch)
            except Exception as e:
                error_type = f"Database insertion error: {type(e).__name__}"
                error_details[error_type] += len(current_batch)
                db_error_count += len(current_batch)
                logger.error(f"Error inserting final batch: {str(e)}")

        # Calculate and log statistics
        end_time = time.time()
        duration = end_time - start_time

        logger.info("\nProcessing Summary:")
        logger.info("-" * 50)
        logger.info(f"Total time: {duration:.2f} seconds")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {processed_count} files")
        logger.info(f"Already existing: {existing_count} files")
        logger.info(f"DICOM/extraction errors: {error_count} files")
        logger.info(f"Database insertion errors: {db_error_count} files")
        logger.info(f"Average processing rate: {processed_count / duration:.2f} files/second")

        logger.info("\nDetailed Error Breakdown:")
        logger.info("-" * 50)
        for error_type, count in error_details.most_common():
            logger.info(f"{error_type}: {count} files")

        # Optimize database
        logger.info("\nOptimizing database...")
        conn.execute("VACUUM")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

    finally:
        conn.close()

def scale_and_convert_rtdose(rtdose_path: str, output_dir: str, output_filename: str):
    """Reads an RTDOSE DICOM, applies DoseGridScaling, and saves as NIfTI."""
    try:
        dicom_data = pydicom.dcmread(rtdose_path)
        dose_grid_scaling = float(dicom_data.DoseGridScaling)
        reader = sitk.ImageFileReader()
        reader.SetFileName(rtdose_path)
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        arr = sitk.GetArrayFromImage(image)
        scaled_arr = arr * dose_grid_scaling
        scaled_image = sitk.GetImageFromArray(scaled_arr)
        scaled_image.CopyInformation(image)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename}.nii.gz")
        sitk.WriteImage(scaled_image, output_path)
        log_message(logger, f"Successfully converted RTDOSE to {output_path}")
    except Exception as e:
        log_message(logger, f"Failed to scale and convert RTDOSE file {rtdose_path}. Error: {e}", level='error')

def ants_transformation_with_template(moving_file_path: str, static_file_path: str, mat_path: str, output_file_path: str, is_mask: bool = False):
    """Applies a pre-existing ANTs transformation to a moving image."""
    interpolation = "NearestNeighbor" if is_mask else "Linear"
    command = (
        f"{binaries_dir}/ants/bin/antsApplyTransforms -d 3 -i {moving_file_path} "
        f"-r {static_file_path} -o {output_file_path} -t {mat_path} -n {interpolation}"
    )
    run_command(command)


# --- Main Processing Logic ---

def main():
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    global logger
    logger = setup_logger(logfile_path)

    if flag_rescan_source_dir:
        index_properties_to_sqlite(start_path=abs_data_path,dbpath=sqlite_database_path,max_workers=10)

    predefined_IUIDs = PreDefinedUIDs(csv_path=predefined_IUID_path)
    if predefined_IUIDs.df is None: return

    try:
        conn = sqlite3.connect(sqlite_database_path)
        cur = conn.cursor()
    except sqlite3.Error as e:
        log_message(logger, f"Database connection error: {e}", level='error')
        return

    for index, row in predefined_IUIDs.df.iterrows():
        pat_id = row['PatID']
        subject_id = row['Verschluesselungs ID']
        log_message(logger, f"--- Starting preprocessing for Subject PatID: {pat_id} (PatientID: {subject_id}) ---")

        subject_target_dir = os.path.join(preprocessed_data_dir, f"subj_{int(pat_id):03d}")
        os.makedirs(subject_target_dir, exist_ok=True)
        temp_dir_path = os.path.join(local_temp_dir_path,
                                     f"subj_{int(pat_id):03d}") if flag_use_local_temp_dir else os.path.join(
            subject_target_dir, "tmp")

        # === STEP 1: PREPARE STATIC REFERENCE IMAGE ===
        static_image_path = os.path.join(subject_target_dir, 'MR_pseudo_ax_km.nii.gz')
        pseudo_uid = predefined_IUIDs.get_uid(pat_id, 'pseudo', 'ax_km')
        if not pseudo_uid:
            log_message(logger, f"CRITICAL: No UID for reference image (pseudo_ax_km) for subject {pat_id}. Skipping.",
                        level='error')
            continue

        dicom_paths = get_dicom_paths_from_uid(cur, subject_id, pseudo_uid)
        if not dicom_paths:
            log_message(logger,
                        f"CRITICAL: No DICOM files for reference UID {pseudo_uid} for subject {pat_id}. Skipping.",
                        level='error')
            continue

        empty_directory(temp_dir_path)
        copy_files(dicom_paths, temp_dir_path)
        run_command(f"{binaries_dir}/dcm2niix -f MR_pseudo_ax_km -z y -o {subject_target_dir} {temp_dir_path}")

        if not os.path.exists(static_image_path):
            log_message(logger, f"CRITICAL: Failed to create reference image {static_image_path}. Skipping.",
                        level='error')
            continue

        try:
            log_message(logger, f"Resampling static image to isotropic spacing: {static_image_path}")
            pseudo_img = sitk.ReadImage(static_image_path)
            isotropic_pseudo_img = rescale_to_isotropic(pseudo_img)
            sitk.WriteImage(isotropic_pseudo_img, static_image_path)
        except Exception as e:
            log_message(logger, f"CRITICAL: Failed to resample static image. Error: {e}. Skipping.", level='error')
            continue

        # === STEP 2: PROCESS AND REGISTER MRI & CT IMAGES ===
        images_to_process = mri_name_list.copy()
        images_to_process.remove("pseudo")
        images_to_process.append("CT")

        for name in images_to_process:
            is_mri = name in mri_name_list
            modalities = modality_name_list if is_mri else [""]

            for modality in modalities:
                prefix = f"MR_{name}_{modality}" if is_mri else "CT"
                uid = predefined_IUIDs.get_uid(pat_id, name if is_mri else "ct", modality if is_mri else None)
                if not uid: continue
                dicom_paths = get_dicom_paths_from_uid(cur, subject_id, uid)
                if not dicom_paths: continue

                empty_directory(temp_dir_path)
                copy_files(dicom_paths, temp_dir_path)
                run_command(f"{binaries_dir}/dcm2niix -f {prefix} -z y -o {subject_target_dir} {temp_dir_path}")

                moving_image_path = os.path.join(subject_target_dir, f"{prefix}.nii.gz")
                if os.path.exists(moving_image_path):
                    log_message(logger, f"Registering {moving_image_path} to static image.")
                    output_prefix = os.path.join(subject_target_dir, f"{prefix}_reg")
                    ants_registration(moving_image_path, static_image_path, output_prefix)
                else:
                    log_message(logger, f"Conversion failed for {prefix}. Skipping registration.", level='warning')

        # === STEP 3 from above ===
        rtdose_specifier = predefined_IUIDs.get_uid(pat_id, "rtdose")
        if not rtdose_specifier:
            log_message(logger, f"No RTDOSE specifier found for subject {pat_id}. Skipping RD processing.",
                        level='warning')
        else:
            dicom_paths = []
            if ":" in rtdose_specifier:
                uid_part, filename_part = rtdose_specifier.split(":", 1)
                log_message(logger,
                            f"Searching for specific RTDOSE file: UID={uid_part}, Filename contains={filename_part}")
                query = "SELECT file_path FROM file_properties WHERE PatientID = ? AND SeriesInstanceUID = ? AND file_path LIKE ?"
                params = (subject_id, uid_part, f"%{filename_part}")
                cur.execute(query, params)
                dicom_paths = [row[0] for row in cur.fetchall()]
            else:
                log_message(logger, f"Searching for RTDOSE series UID: {rtdose_specifier}")
                dicom_paths = get_dicom_paths_from_uid(cur, subject_id, rtdose_specifier)
            if not dicom_paths:
                log_message(logger,
                            f"No DICOM file found for RTDOSE specifier '{rtdose_specifier}'. Skipping RD processing.",
                            level='warning')
            else:
                rtdose_dicom_path = dicom_paths[0]
                rd_nifti_path = os.path.join(subject_target_dir, "RD.nii.gz")
                scale_and_convert_rtdose(rtdose_path=rtdose_dicom_path, output_dir=subject_target_dir,
                                         output_filename="RD")
                ct_transform_path = os.path.join(subject_target_dir, "CT_reg0GenericAffine.mat")
                if os.path.exists(rd_nifti_path) and os.path.exists(ct_transform_path):
                    log_message(logger, f"Applying CT transformation to {rd_nifti_path}.")
                    rd_reg_path = os.path.join(subject_target_dir, "RD_reg.nii.gz")
                    ants_transformation_with_template(
                        moving_file_path=rd_nifti_path, static_file_path=static_image_path,
                        mat_path=ct_transform_path, output_file_path=rd_reg_path, is_mask=False
                    )

                if os.path.exists(rd_reg_path):
                    scale_rd_reg(rd_reg_path, subject_target_dir)

                elif not os.path.exists(ct_transform_path):
                    log_message(logger,
                                f"CT transform matrix not found at {ct_transform_path}. Cannot register RD file.",
                                level='warning')

    if os.path.exists(local_temp_dir_path):
        log_message(logger, f"Cleaning up all temporary files in {local_temp_dir_path}...")
        empty_directory(local_temp_dir_path)
        log_message(logger, "Temporary directory cleaned.")

    conn.close()
    log_message(logger, "--- Preprocessing complete ---")


if __name__ == '__main__':
    main()