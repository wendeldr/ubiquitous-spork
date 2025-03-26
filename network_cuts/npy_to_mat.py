# loop through all folders and subfolders and convert all npy files to mat files. delete the npy files after conversion if the conversion was successful.
import os
import glob
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

# Function to convert .npy file to .mat file
def convert_npy_to_mat(npy_file):
    try:
        # Load .npy file
        data = np.load(npy_file)
        
        # Create .mat filename by replacing extension
        mat_file = npy_file.replace('.npy', '.mat')
        
        # Save as .mat file - use the filename without extension as variable name
        savemat(mat_file, {'measure': data})
        
        # Check if .mat file was created successfully
        if os.path.exists(mat_file):
            # print(f"Successfully converted {npy_file} to {mat_file}")
            return True
        else:
            print(f"Failed to create {mat_file}")
            return False
    except Exception as e:
        print(f"Error converting {npy_file}: {e}")
        return False

# Get the root directory (current directory)
root_dir = "/media/dan/Data2/calculations/connectivity/additional_calcs/julia_thresholded_mats"
print(f"Starting conversion process in: {root_dir}")

# Count variables for summary
total_files = 0
converted_files = 0
failed_files = 0

# Walk through all directories and subdirectories
for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc="Walking through directories"):
    # Find all .npy files in current directory
    npy_files = [os.path.join(dirpath, f) for f in filenames if f.endswith('.npy')]
    
    for npy_file in tqdm(npy_files, desc="Converting .npy files to .mat files"):
        total_files += 1
        # Convert the file
        if convert_npy_to_mat(npy_file):
            # Delete the original .npy file if conversion was successful
            os.remove(npy_file)
            # print(f"Deleted original file: {npy_file}")
            converted_files += 1
        else:
            failed_files += 1

# Print summary
print("\nConversion Summary:")
print(f"Total .npy files found: {total_files}")
print(f"Successfully converted and deleted: {converted_files}")
print(f"Failed conversions: {failed_files}")