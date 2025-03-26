import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import dill
import shutil
import re

"""
This script organizes the calculations into HDF5 files.
Warning: This will delete the existing files if they exist. Additionally, this script will take a long time to run (5+ hours).

The files are organized as follows:

patients.h5
/pid
    /column_name
        /upper
            /soz_soz
            /soz_non
            /non_soz
            /non_non
        /lower
            /soz_soz
            /soz_non
            /non_soz
            /non_non

columns.h5
/column_name
    /upper
        /soz_soz
        /soz_non
        /non_soz
        /non_non
    /lower
        /soz_soz
        /soz_non
        /non_soz
        /non_non

ilae.h5
/ilae_group
    /column_name
        /ilae_value
            /upper
                /soz_soz
                /soz_non
                /non_soz
                /non_non
            /lower
                /soz_soz
                /soz_non
                /non_soz
                /non_non
"""

# Function to write results to HDF5
def save_to_hdf5(path, data, file_path):
    if len(data) == 0:
        return
    if path not in file_path:
        file_path.create_dataset(path, data=data, maxshape=(None,), compression="gzip", chunks=True)
    else:
        dataset = file_path[path]
        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
        dataset[-data.shape[0]:] = data

def get_idxs(idxs, soz_idx):
    soz_soz = []
    soz_non = []
    non_soz = []
    non_non = []
    for x, y in zip(idxs[0], idxs[1]):
        if x in soz_idx and y in soz_idx:
            soz_soz.append((x, y))
        elif x in soz_idx or y in soz_idx:
            if x in soz_idx:
                soz_non.append((x, y))
            else:
                non_soz.append((x, y))
        else:
            non_non.append((x, y))
    return np.array(soz_soz), np.array(soz_non), np.array(non_soz), np.array(non_non)

def safe_slice_and_flatten(measure, idx_array):
    if len(idx_array) == 0:
        return np.array([])  # Return empty array if no indices
    return measure[:, idx_array[:, 0], idx_array[:, 1]].flatten()

def sanitize_filename(filename):
    """
    Convert a string into a valid filename by:
    - Removing invalid characters
    - Converting spaces to underscores
    - Removing leading/trailing periods and spaces
    """
    # Keep only valid filename characters and replace spaces with underscores
    filename = re.sub(r'[^\w\-_.]', '', filename.replace(' ', '_'))
    
    # Remove leading/trailing periods and spaces
    filename = filename.strip('._')
    
    # If filename is empty after sanitization, provide a default
    if not filename:
        filename = 'unnamed_file'
        
    return filename

# output_path = "/media/dan/Data/data/calculations"
# mapping_path = "/media/dan/Big/manuiscript_0001_hfo_rates/data/FULL_composite_patient_info.csv"  
# ilae_path = "/media/dan/Big/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"
# calculation_path = "/home/dan/data/connectivity/pyspi_testing/sixrun/calculations/six_run"

csv_path = "/media/dan/Big/network_mining/calculations/electrodes_used"
output_path = "/media/dan/Data/data/tmp"
mapping_path = "/media/dan/Big/manuiscript_0001_hfo_rates/data/FULL_composite_patient_info.csv"  
ilae_path = "/media/dan/Big/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"
calculation_path = "/media/dan/Big/network_mining/calculations/sixrun/calculations/additional_calculations"
final_path = "/media/dan/Big/network_mining/calculations/additional/ilae"

mappings = pd.read_csv(mapping_path)
ilae = pd.read_csv(ilae_path)
# for each patient in mappings, find the corresponding ilae number. The patient may not be in the ilae dataset but has a designation of seizureFree or not.
# if the patient is not in the ilae dataset, then use the seizureFree column to determine the ilae number where -1 is seizureFree and 100 is not seizureFree
ilae_numbers = {}
for p in mappings["pid"].unique():
    if p in ilae["patient"].values:
        ilae_numbers[p] = ilae[ilae["patient"] == p]["ilae"].values[0]
    else:
        if mappings[mappings["pid"] == p]["seizureFree"].values[0] == True:
            ilae_numbers[p] = -1
        else:
            ilae_numbers[p] = 100

# now we have a dictionary of ilae numbers for each patient. Fill in the mappings dataframe with these numbers which has multiple rows for each patient
ilae_list = []
for p in mappings["pid"]:
    ilae_list.append(ilae_numbers[p])
mappings["ilae"] = ilae_list

files = list(sorted(os.listdir(calculation_path)))
pids = list(sorted(set([int(f.split("_")[0]) for f in files])))

# read the first file from the first patient to get the calculation names. make sure file has "epoch" in the name
first_epoch_file = [f for f in files if "epoch" in f][0]

with open(os.path.join(calculation_path, first_epoch_file, 'calc.pkl'), "rb") as f:
    calc = dill.load(f)
columns = calc.columns.levels[0].unique().values    

# patient_hdf5_path = os.path.join(output_path, "patients.h5")
ilae_hdf5_path = os.path.join(output_path, "ilae.h5")
# columns_hdf5_path = os.path.join(output_path, "columns.h5")

# # delete the files if they exist. prompt user to confirm
# # prompt user to confirm
# if input("This will delete the existing files. Are you sure? (y/n): ") != "y":
#     print("Exiting...")
#     exit()  

# # if os.path.exists(patient_hdf5_path):
# #     os.remove(patient_hdf5_path)
if os.path.exists(ilae_hdf5_path):
    os.remove(ilae_hdf5_path)
# if os.path.exists(columns_hdf5_path):
#     os.remove(columns_hdf5_path)

# Get unique ILAE groups before initialization
unique_ilae_groups = sorted(list(set(ilae_numbers.values())))



for col in tqdm(columns, desc=f"Columns", leave=True):
    with h5py.File(ilae_hdf5_path, "w", libver="latest") as hdf5_file:
        # col_group = columns_hdf5_file.create_group(str(col))
        # col_group.create_group("upper")
        # col_group.create_group("lower")
        for p, pid in enumerate(tqdm(pids, desc="Patients", leave=True)):

            pid_files = list(sorted([f for f in files if f.startswith(f"{pid:03}")]))
            try:
                chnames_idx = pid_files.index(f"{pid:03}_chnames.csv")
                chnames = pd.read_csv(os.path.join(csv_path, pid_files[chnames_idx]))['0'].values
                pid_files.pop(chnames_idx)
            except:
                try:
                    chnames = pd.read_csv(os.path.join(csv_path, f"{pid:03}_chnames.csv"))['0'].values
                except:
                    print(f"No chnames file for {pid}")
                    continue

            pid_mappings = mappings[mappings["pid"] == pid]
            pid_mappings = pid_mappings[pid_mappings["electrode"].isin(chnames)]
            pid_mappings = pid_mappings.set_index("electrode").reindex(chnames).reset_index()

            soz_idx = pid_mappings.index[pid_mappings["soz"] == 1].values
            ilae_group = pid_mappings["ilae"].iloc[0]

            if len(soz_idx) == 0:
                continue

            data = []
            skip = False
            for file in pid_files:
                with open(os.path.join(calculation_path, file, 'calc.pkl'), "rb") as f:
                    try:
                        data.append(dill.load(f))
                    except:
                        print(f"Error loading {file}")
                        skip = True
                        break
            if skip:
                continue

            upper_idx = np.triu_indices(len(chnames), 1)
            lower_idx = np.tril_indices(len(chnames), -1)

            U_soz_soz, U_soz_non, U_non_soz, U_non_non = get_idxs(upper_idx, soz_idx)
            L_soz_soz, L_soz_non, L_non_soz, L_non_non = get_idxs(lower_idx, soz_idx)

            measure = []
            for r in data:
                full = r[col].values
                measure.append(full)
            measure = np.array(measure)

            # # Save to patient-level HDF5
            # save_to_hdf5(f"{pid}/{col}/upper/soz_soz", safe_slice_and_flatten(measure, U_soz_soz), patient_hdf5_path)
            # save_to_hdf5(f"{pid}/{col}/upper/soz_non", safe_slice_and_flatten(measure, U_soz_non), patient_hdf5_path)
            # save_to_hdf5(f"{pid}/{col}/upper/non_soz", safe_slice_and_flatten(measure, U_non_soz), patient_hdf5_path)
            # save_to_hdf5(f"{pid}/{col}/upper/non_non", safe_slice_and_flatten(measure, U_non_non), patient_hdf5_path)

            # save_to_hdf5(f"{pid}/{col}/lower/soz_soz", safe_slice_and_flatten(measure, L_soz_soz), patient_hdf5_path)
            # save_to_hdf5(f"{pid}/{col}/lower/soz_non", safe_slice_and_flatten(measure, L_soz_non), patient_hdf5_path)
            # save_to_hdf5(f"{pid}/{col}/lower/non_soz", safe_slice_and_flatten(measure, L_non_soz), patient_hdf5_path)
            # save_to_hdf5(f"{pid}/{col}/lower/non_non", safe_slice_and_flatten(measure, L_non_non), patient_hdf5_path)

            # Save to ilae-level HDF5
            save_to_hdf5(f"{ilae_group}/{col}/upper/soz_soz", safe_slice_and_flatten(measure, U_soz_soz), hdf5_file)
            save_to_hdf5(f"{ilae_group}/{col}/upper/soz_non", safe_slice_and_flatten(measure, U_soz_non), hdf5_file)
            save_to_hdf5(f"{ilae_group}/{col}/upper/non_soz", safe_slice_and_flatten(measure, U_non_soz), hdf5_file)
            save_to_hdf5(f"{ilae_group}/{col}/upper/non_non", safe_slice_and_flatten(measure, U_non_non), hdf5_file)

            save_to_hdf5(f"{ilae_group}/{col}/lower/soz_soz", safe_slice_and_flatten(measure, L_soz_soz), hdf5_file)
            save_to_hdf5(f"{ilae_group}/{col}/lower/soz_non", safe_slice_and_flatten(measure, L_soz_non), hdf5_file)
            save_to_hdf5(f"{ilae_group}/{col}/lower/non_soz", safe_slice_and_flatten(measure, L_non_soz), hdf5_file)
            save_to_hdf5(f"{ilae_group}/{col}/lower/non_non", safe_slice_and_flatten(measure, L_non_non), hdf5_file)

            # Save to columns-level HDF5
            # save_to_hdf5(f"{col}/upper/soz_soz", safe_slice_and_flatten(measure, U_soz_soz), columns_hdf5_file)
            # save_to_hdf5(f"{col}/upper/soz_non", safe_slice_and_flatten(measure, U_soz_non), columns_hdf5_file)
            # save_to_hdf5(f"{col}/upper/non_soz", safe_slice_and_flatten(measure, U_non_soz), columns_hdf5_file)
            # save_to_hdf5(f"{col}/upper/non_non", safe_slice_and_flatten(measure, U_non_non), columns_hdf5_file)
            
            # save_to_hdf5(f"{col}/lower/soz_soz", safe_slice_and_flatten(measure, L_soz_soz), columns_hdf5_file)
            # save_to_hdf5(f"{col}/lower/soz_non", safe_slice_and_flatten(measure, L_soz_non), columns_hdf5_file)
            # save_to_hdf5(f"{col}/lower/non_soz", safe_slice_and_flatten(measure, L_non_soz), columns_hdf5_file)
            # save_to_hdf5(f"{col}/lower/non_non", safe_slice_and_flatten(measure, L_non_non), columns_hdf5_file)

    name = col
    # make sure name is a valid filename
    name = sanitize_filename(name)

    # move the file to a new directory
    shutil.move(ilae_hdf5_path, os.path.join(final_path, f"ilae_{name}.h5"))