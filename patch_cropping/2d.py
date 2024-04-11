import numbers
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product
import nrrd
def _compute_n_patches_2d(i_x, i_y, p_x, p_y, max_patches=None):
    n_x = i_x - p_x + 1
    n_y = i_y - p_y + 1
    all_patches = n_x * n_y

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

def extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = extract_patches(image, patch_shape=(p_h, p_w, n_colors), extraction_step=(1, 1, 1))

    n_patches = _compute_n_patches_2d(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches

def extract_patches_2d_fromMask(volume, mask, patch_size, max_patches=None, random_state=None):
    v_x, v_y = volume.shape[:2]
    p_x, p_y = patch_size

    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    volume = check_array(volume, allow_nd=True)
    volume = volume.reshape((v_x, v_y, -1))
    n_colors = volume.shape[-1]

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, n_colors), extraction_step=1)

    n_patches = _compute_n_patches_2d(v_x, v_y, p_x, p_y, max_patches)

    M = np.array(np.where(mask[int(p_x / 2): int(v_x - p_x / 2), int(p_y / 2):int(v_y - p_y / 2)] == True)).T

    if max_patches:
        rng = check_random_state(random_state)
        indx = rng.randint(len(M), size=n_patches)
        i_s = M[indx][:, 0]
        j_s = M[indx][:, 1]
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_x, p_y, n_colors)
    patches = np.transpose(patches, axes=[0, 3, 1, 2])

    return patches

def reconstruct_from_patches_2d(patches, image_size):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    n_colors = patches.shape[3]

    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1

    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += np.mean(p, axis=-1)  # Average the channels

    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img

def plot_patches(dose_patches, lung_patches):
    fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(20, 5))

    for i in range(5):
        ax = axs[i]
        ax.imshow(dose_patches[i, :, :, 0], cmap='gray')
        ax.set_title(f'Dose Patch {i}')
        ax.axis('off')

        ax = axs[i+5]
        ax.imshow(lung_patches[i].mean(axis=-1), cmap='gray')  # Take the average across the channel dimension
        ax.set_title(f'Lung Patch {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_volumes(dose_volume, reconstructed_volume):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axs[0].imshow(dose_volume[:, :, 0], cmap='gray')
    axs[0].set_title('Original Dose Volume')

    axs[1].imshow(reconstructed_volume, cmap='gray')
    axs[1].set_title('Reconstructed Dose Volume')

    plt.tight_layout()
    plt.show()

import nrrd
import matplotlib.pyplot as plt

# Load dose volume and corresponding mask
dose_file = "G:/pythonProject/Resampled dose/resampled_4 RTDOSE RT Dose - fx1hetero.nrrd"
mask_file = "G:/pythonProject/Lung masks/2 RTSTRUCT RTOG_CONV.nrrd"

dose_data, dose_header = nrrd.read(dose_file)
mask_data, mask_header = nrrd.read(mask_file)

# Define patch size and extract patches
patch_size = (8, 8)
n_patches = 5
dose_patches = extract_patches_2d(dose_data, patch_size, max_patches=n_patches)
mask_patches = extract_patches_2d(mask_data, patch_size, max_patches=n_patches)

# Reconstruct dose volume from patches
reconstructed_dose = reconstruct_from_patches_2d(dose_patches, dose_data.shape[:2])

print("Reconstructed dose min:", np.min(reconstructed_dose))
print("Reconstructed dose max:", np.max(reconstructed_dose))


plot_volumes(dose_volume=dose_data, reconstructed_volume=reconstructed_dose)

# Plot dose volume patches
plot_patches(dose_patches, mask_patches)


# Plot reconstructed dose volume
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(reconstructed_dose, cmap="gray")
ax.set_title("Reconstructed Dose Volume")
plt.tight_layout()
plt.show()


#Visualize the original dose distribution and mask
# Display dose volume
plt.figure(figsize=(10, 10))
plt.imshow(dose_data[:, :, 0], cmap='gray')
plt.title('Original Dose Volume')
plt.show()

# Display mask
plt.figure(figsize=(10, 10))
plt.imshow(mask_data[:, :, 0], cmap='gray')
plt.title('Original Mask')
plt.show()