import blosc2
import numpy as np
import nibabel as nib
import os


def convert_blosc2_to_nifti_corrected(
        blosc2_path,
        nifti_path,
        input_axis_order='0,1,2',
        final_orientation='RAS',
        voxel_sizes=(1.0, 1.0, 1.0)
):
    """
    Converts a 3D or 4D (with a singleton first dim) Blosc2 file to a properly oriented
    3D NIfTI file.

    Args:
        blosc2_path (str): The file path of the input Blosc2 compressed file.
        nifti_path (str): The file path for the output NIfTI (.nii.gz) file.
        input_axis_order (str): A comma-separated string to reorder the 3D data axes.
                                E.g., '2,1,0' to change (depth, H, W) to (W, H, depth).
        final_orientation (str): The desired 3-letter orientation code (e.g., 'RAS', 'LPS').
        voxel_sizes (tuple): A tuple of three floats for voxel dimensions (in mm).
    """
    if not os.path.exists(blosc2_path):
        print(f"Error: The file '{blosc2_path}' does not exist.")
        return

    try:
        # --- 1. Load Data from Blosc2 File ---
        b2_array = blosc2.open(blosc2_path)
        image_data = b2_array[:]
        print(f"Original data shape loaded from Blosc2: {image_data.shape}")

        # --- 2. Handle 4D Data by Squeezing to 3D ---
        # If shape is (1, D, H, W), we remove the first dimension.
        if image_data.ndim == 4 and image_data.shape[0] == 1:
            image_data_3d = np.squeeze(image_data, axis=0)
            print(f"Data squeezed to 3D shape: {image_data_3d.shape}")
        elif image_data.ndim == 3:
            image_data_3d = image_data
            print("Input data is already 3D. No squeezing needed.")
        else:
            raise ValueError(
                f"Unsupported data dimension: {image_data.ndim}. "
                f"This script expects 3D or 4D data with a leading singleton dimension."
            )

        # --- 3. Correct the Initial Axis Order of the 3D Array ---
        # The user provides the desired order of the 3D axes.
        try:
            axis_map = [int(ax.strip()) for ax in input_axis_order.split(',')]
            if len(axis_map) != 3:
                raise ValueError("The 'input_axis_order' must contain exactly 3 axes for 3D data.")
            reordered_data = np.transpose(image_data_3d, axis_map)
            print(f"Data shape after reordering axes with map {axis_map}: {reordered_data.shape}")
        except Exception as e:
            print(f"Error processing 'input_axis_order': {e}")
            return

        # --- 4. Create the Correct Affine Matrix and Reorient (FIXED METHOD) ---
        # Start with a simple affine that only accounts for voxel sizes (scaling).
        # We assume the initial orientation is standard (RAS).
        # The reorientation logic below will fix this.
        initial_affine = np.diag(list(voxel_sizes) + [1])

        # Create a temporary NIfTI object in memory with the reordered data and the simple affine.
        temp_nifti_img = nib.Nifti1Image(reordered_data, initial_affine)

        # Use nibabel's canonical reorientation tool.
        # This function calculates the necessary transformations to get to the desired orientation.
        # It returns a new NIfTI object with both the data array and the affine correctly oriented.
        oriented_nifti_img = nib.as_closest_canonical(temp_nifti_img)

        # Or, to force a specific orientation like 'RAS':
        # 1. Get the current orientation from the temporary image's affine
        start_ornt = nib.orientations.io_orientation(temp_nifti_img.affine)
        # 2. Get the target orientation
        end_ornt = nib.orientations.axcodes2ornt(final_orientation)
        # 3. Compute the transformation
        transform = nib.orientations.ornt_transform(start_ornt, end_ornt)
        # 4. Apply the transformation to the data
        oriented_data = nib.orientations.apply_orientation(temp_nifti_img.get_fdata(), transform)
        # 5. Create the final affine
        final_affine = temp_nifti_img.affine @ nib.orientations.inv_ornt_aff(transform, temp_nifti_img.shape)

        print(f"\nGenerated Final Affine Matrix for '{final_orientation}' orientation:")
        print(final_affine)

        # --- 5. Create and Save the Final NIfTI Image ---
        final_nifti_image = nib.Nifti1Image(oriented_data, final_affine)
        nib.save(final_nifti_image, nifti_path)

        print(f"\nSuccessfully converted '{blosc2_path}' to '{nifti_path}'")
        print(f"Final image dimensions: {final_nifti_image.shape}")

    except Exception as e:
        print(f"An error occurred during the conversion: {e}")


if __name__ == '__main__':
    # ==============================================================================
    # --- USER CONFIGURATION: PLEASE PASTE YOUR PATHS AND SETTINGS HERE ---
    # ==============================================================================

    # 1. Path to your input Blosc2 compressed file
    input_blosc2_file = '/path/to/input/GlioblastomaTimeSeries_0001_0002.b2nd'

    # 2. Desired path for the output NIfTI file
    output_nifti_file = '/path/to/output/GlioblastomaTimeSeries_0001_0002.nii.gz'

    # 3. **CRITICAL**: Specify the order of axes for the 3D data.
    #    After the script removes the leading '1' from `(1, 160, 160, 160)`, you are left
    #    with a 3D array of `(160, 160, 160)`. If you believe this order is, for example,
    #    (depth, height, width), and you want (width, height, depth), use '2,1,0'.
    #    If the order is correct as is, use '0,1,2'.
    input_axis_order_from_blosc = '2,1,0'

    # 4. **CRITICAL**: Define the final orientation of your image.
    #    'RAS' (Right, Anterior, Superior) is a very common standard.
    desired_final_orientation = 'RAS'

    # 5. (Optional) If you know the voxel sizes (in millimeters), set them here.
    #    The order should correspond to the FINAL axis order.
    voxel_dimensions = (1.0, 1.0, 1.0)  # (size_x, size_y, size_z) in mm

    # ==============================================================================
    # --- END OF CONFIGURATION ---
    # ==============================================================================

    # Run the conversion function with your settings
    convert_blosc2_to_nifti_corrected(
        input_blosc2_file,
        output_nifti_file,
        input_axis_order=input_axis_order_from_blosc,
        final_orientation=desired_final_orientation,
        voxel_sizes=voxel_dimensions
    )