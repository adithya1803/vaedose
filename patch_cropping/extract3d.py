import numbers
import nrrd
from sklearn.utils import check_array
import numpy as np
import napari


def _compute_n_patches_3d(i_x, i_y, i_z, p_x, p_y, p_z, max_patches=None):
    n_x = i_x - p_x + 1
    n_y = i_y - p_y + 1
    n_z = i_z - p_z + 1
    all_patches = n_x * n_y * n_z
    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
              and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches(volume, patch_shape, extraction_step):
    patch_strides = volume.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = volume[slices].strides

    patch_indices_shape = ((np.array(volume.shape) - np.array(patch_shape)) // np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = np.lib.stride_tricks.as_strided(volume, shape=shape, strides=strides)
    return patches

def extract_patches_3d(volume_file, patch_size):
    volume, _ = nrrd.read(volume_file)
    v_x, v_y, v_z = volume.shape
    p_x, p_y, p_z = patch_size
    if p_x > v_x or p_y > v_y or p_z > v_z:
        raise ValueError("Dimensions of the patch should be less than the dimensions of the volume.")
    volume = check_array(volume, allow_nd=True)
    extracted_patches = []
    for i in range(0, v_x - p_x + 1, 1):
        for j in range(0, v_y - p_y + 1, 1):
            for k in range(0, v_z - p_z + 1, 1):
                patch = volume[i:i + p_x, j:j + p_y, k:k + p_z]
                extracted_patches.append(patch)
    return extracted_patches


# Specify the paths to your npy files
dose_file = "G:/pythonProject/Resampled dose/resampled_4 RTDOSE RT Dose - fx1hetero.nrrd"

# Define the patch size
patch_size = (8, 8, 8)

# Extract patches from the dose volume
dose_patches = extract_patches_3d(dose_file, patch_size)

# Save the patches to npy files
np.save("dose_patches1.npy", dose_patches)

# Load the patches from the npy files
loaded_dose_patches = np.load("dose_patches1.npy", allow_pickle=True)

# Print the shape of the first patch
print(loaded_dose_patches[0].shape)

# Visualize a 3D patch
patch = loaded_dose_patches[0]

# Create a viewer and add the patch as a layer
viewer = napari.Viewer()
viewer.add_image(patch, colormap='gray')

# Show the viewer
napari.run()