import os
import numpy as np

# Define the two base directories
a_dir = "/media/dan/Data2/calculations/connectivity/additional_calcs/thresholded_mats"
b_dir = "/media/dan/Data2/calculations/connectivity/additional_calcs/julia_thresholded_mats"

# Lists to track mismatches and missing files
missing_in_b = []
diff_files = []
checked_files = []

# Walk through all subdirectories/files in a_dir
for root, dirs, files in os.walk(a_dir):
    for file in files:
        checked_files.append(file)
        if file.endswith('.npy'):
            # Get full path for the file in a_dir
            file_a_path = os.path.join(root, file)
            # Determine its relative path from a_dir
            rel_path = os.path.relpath(file_a_path, a_dir)
            # Build the corresponding file path in b_dir
            file_b_path = os.path.join(b_dir, rel_path)
            # Check if the corresponding file exists in b
            if not os.path.exists(file_b_path):
                missing_in_b.append(rel_path)
                continue

            # Load both arrays
            a_array = np.load(file_a_path)
            b_array = np.load(file_b_path)

            # Replace NaNs with 0
            a_array = np.nan_to_num(a_array, nan=0)
            b_array = np.nan_to_num(b_array, nan=0)

            # Compare arrays
            if not np.array_equal(a_array, b_array):
                diff_files.append(rel_path)
            
# Print the results
print(len(checked_files))

if missing_in_b:
    print("Files in a but not in b:")
    print(len(missing_in_b))
else:
    print("All files from a have corresponding files in b.")

if diff_files:
    print("\nFiles that differ:")
    print(len(diff_files))
    for file in diff_files:
        print(file)
else:
    print("\nAll corresponding files are identical after replacing NaNs with 0.")


