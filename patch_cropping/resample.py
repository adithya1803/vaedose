# import os
# import nrrd
# from scipy.ndimage import zoom
#
# # Define the directories
# dose_dir = r"G:\Sample DD"
# lung_dir = r"G:\Lung masks"
#
# # Get the list of files in each directory
# dose_files = sorted(os.listdir(dose_dir))
# lung_files = sorted(os.listdir(lung_dir))
#
# # Make sure there are the same number of files in each directory
# assert len(dose_files) == len(lung_files), "Different number of files in the directories"
#
# # Loop over all pairs of files
# for dose_file, lung_file in zip(dose_files, lung_files):
#     # Load the volumes
#     dose_volume, _ = nrrd.read(os.path.join(dose_dir, dose_file))
#     lung_volume, _ = nrrd.read(os.path.join(lung_dir, lung_file))
#
#     # Calculate the zoom factors
#     zoom_factors = [lung_dim / dose_dim for lung_dim, dose_dim in zip(lung_volume.shape, dose_volume.shape)]
#
#     # Resample the dose volume
#     resampled_dose_volume = zoom(dose_volume, zoom_factors)
#
#     print(f"Shape of resampled {dose_file}: {resampled_dose_volume.shape}")
#
#     # Save the resampled dose volume
#     resampled_dose_filepath = os.path.join(dose_dir, f"resampled_{dose_file}")
#     nrrd.write(resampled_dose_filepath, resampled_dose_volume)



# import nrrd
#
# # Specify the paths to your nrrd files
# dose_file = "C:/Users/adith/PycharmProjects/pythonProject/Sample DD/4 RTDOSE RT Dose - fx1hetero.nrrd"
# mask_file = "C:/Users/adith/PycharmProjects/pythonProject/Lung masks/2 RTSTRUCT RTOG_CONV.nrrd"
#
# # Load the nrrd files
# dose, _ = nrrd.read(dose_file)
# mask, _ = nrrd.read(mask_file)
#
# # Print the shapes of the files
# print("Shape of dose:", dose.shape)
# print("Shape of mask:", mask.shape)

# import numpy as np
#
# # Load the patches from the npy files
# loaded_dose_patches = np.load("dose_patches.npy")
# loaded_lung_patches = np.load("lung_patches.npy")
#
# # Check the shape and size of the arrays
# print("Shape of loaded_dose_patches:", loaded_dose_patches.shape)
# print("Shape of loaded_lung_patches:", loaded_lung_patches.shape)


import matplotlib.pyplot as plt
import numpy as np
# Load the patches from the npy files
loaded_dose_patches = np.load("dose_patches1.npy")

# # Select a specific slice to visualize
# slice_index = 0

# Select the first patch
patch = loaded_dose_patches[0]

# Visualize the patch
plt.imshow(patch)
plt.show()
## Check the shape and size of the arrays
# print("Shape of loaded_dose_patches:", loaded_dose_patches.shape)

# # Plot a slice of the dose patch
# plt.figure(figsize=(6, 6))
# plt.imshow(loaded_dose_patches[slice_index], cmap='gray')
# plt.title('Dose Patch')
# plt.show()
