import numpy as np
import csv

# Load the patches from the .npy file
loaded_dose_patches = np.load("dose_patches1.npy", allow_pickle=True)

# Create a list to store the shapes of the patches
patch_shapes = []

# Print the shape of each patch
for patch in loaded_dose_patches:
    patch_shapes.append(patch.shape)
    print(patch.shape)

# Save the shapes to a CSV file
with open('patch_shapes.csv', mode='w', newline='') as csv_file:
    fieldnames = ['shape']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for shape in patch_shapes:
        writer.writerow({'shape': str(shape)})