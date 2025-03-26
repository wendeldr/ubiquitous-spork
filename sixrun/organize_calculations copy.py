
import dill
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import pickle
import time


output_path = "/home/dan/data/connectivity/pyspi_testing/sixrun/calculations/six_run_organized"


def get_idxs(idxs, soz_idx):
    soz_soz = []
    soz_non = []
    non_soz = []
    non_non = []
    for x,y in zip(idxs[0], idxs[1]):
        if x in soz_idx and y in soz_idx:
            soz_soz.append((x,y))
        elif x in soz_idx or y in soz_idx:
            if x in soz_idx:
                soz_non.append((x,y))
            else:
                non_soz.append((x,y))
        else:
            non_non.append((x,y))
    return np.array(soz_soz), np.array(soz_non), np.array(non_soz), np.array(non_non)


mapping_path = "/media/dan/Big/manuiscript_0001_hfo_rates/data/FULL_composite_patient_info.csv"  
ilae_path = "/media/dan/Big/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"
bad_channels_path = "/media/dan/Big/manuiscript_0001_hfo_rates/data/bad_ch_review.xlsx"
edf_path = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs"

pid_source_path = "/media/dan/Big/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"


# get metadata
mappings = pd.read_csv(mapping_path)
ilae = pd.read_csv(ilae_path)
bad_channels = pd.read_excel(bad_channels_path)
bad_channels["use"] = bad_channels["use"].fillna(1)
bad_channels["use2"] = bad_channels["use2"].fillna(1)
bad_channels["use"] = bad_channels["use"].astype(bool)
bad_channels["use2"] = bad_channels["use2"].astype(bool)

# OR bad_channel columns
bad_channels["bad_channel"] = ~(bad_channels["use"] & bad_channels["use2"])

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


# Perform the merge as before
mappings = mappings.merge(
    bad_channels[['pid', 'ch', 'bad_channel']],
    left_on=['pid', 'electrode'],
    right_on=['pid', 'ch'],
    how='left'
)

# Drop the 'ch' column if needed
mappings = mappings.drop(columns=['ch'])

# Fill NaN values in 'bad_channel' with 0
mappings['bad_channel'] = mappings['bad_channel'].fillna(0)

mappings.loc[(mappings["miccai"].isna() & mappings["aal"].isna()), "bad_channel"] = 1

# get columns
path = "/home/dan/data/connectivity/pyspi_testing/sixrun/calculations/six_run/001_epoch_0000000000-0000000512/calc.pkl"
with open(path, "rb") as f:
    a = dill.load(f)
columns = a.columns.levels[0].unique().values


folder = "/home/dan/data/connectivity/pyspi_testing/sixrun/calculations/six_run"
files = list(sorted(os.listdir(folder)))
pids = list(sorted(set([int(f.split("_")[0]) for f in files])))
results = {'upper': {}, 'lower': {}}
for col in columns:
    results['upper'][col] = {"mean": {}, "all": {}}
    results['lower'][col] = {"mean": {}, "all": {}}
    for z in ['soz_soz', 'soz_non', 'non_soz', 'non_non']:
        results['upper'][col]["mean"][z] = []
        results['lower'][col]["mean"][z] = []
        results['upper'][col]["all"][z] = []
        results['lower'][col]["all"][z] = []

results_ilae = {}
for ilae in mappings["ilae"].unique():
    results_ilae[ilae] = {'upper': {}, 'lower': {}}
    for col in columns:
        results_ilae[ilae]['upper'][col] = {"mean": {}, "all": {}}
        results_ilae[ilae]['lower'][col] = {"mean": {}, "all": {}}
        for z in ['soz_soz', 'soz_non', 'non_soz', 'non_non']:
            results_ilae[ilae]['upper'][col]["mean"][z] = []
            results_ilae[ilae]['lower'][col]["mean"][z] = []
            results_ilae[ilae]['upper'][col]["all"][z] = []
            results_ilae[ilae]['lower'][col]["all"][z] = []

for pid in tqdm(pids, desc="Patients", leave=True):
    # if os.path.exists(os.path.join(output_path, f"{pid}_results.pkl")):
    #     continue
    pid_files = list(sorted([f for f in files if f.startswith(f"{pid:03}")]))
    chnames_idx = pid_files.index(f"{pid:03}_chnames.csv")
    chnames = pd.read_csv(os.path.join(folder, pid_files[chnames_idx]))['0'].values
    # remove chnames file
    pid_files.pop(chnames_idx)

    # soz locations
    pid_mappings = mappings[mappings["pid"] == pid]
    # remove channels that are not in chnames
    pid_mappings = pid_mappings[pid_mappings["electrode"].isin(chnames)]
    # ensure order of chnames and pid_mappings is the same
    pid_mappings = pid_mappings.set_index("electrode").reindex(chnames).reset_index()

    ilae = pid_mappings["ilae"].values[0]

    soz_idx = pid_mappings.index[pid_mappings["soz"] == 1].values

    if len(soz_idx) == 0:
        # print(f"Patient {pid} has no SOZ channels")
        continue


    data = []
    skip = False
    for file in pid_files:
        with open(os.path.join(folder, file, 'calc.pkl'), "rb") as f:
            try:
                data.append(dill.load(f))
            except:
                print(f"Error loading {file}")
                skip = True
                break
            # data.append(dill.load(f))
    if skip:
        continue
    pid_results = {'upper': {}, 'lower': {}}
    upper_idx = np.triu_indices(len(chnames), 1)
    lower_idx = np.tril_indices(len(chnames), -1)

    U_soz_soz, U_soz_non, U_non_soz, U_non_non = get_idxs(upper_idx, soz_idx)
    L_soz_soz, L_soz_non, L_non_soz, L_non_non = get_idxs(lower_idx, soz_idx)

    for col in tqdm(data[0].columns.levels[0].unique(), desc=f"Columns for {pid}", leave=False):

        if col not in pid_results['upper']:
            pid_results['upper'][col] = {"mean": {}, "all": {}}
            pid_results['lower'][col] = {"mean": {}, "all": {}}
            for z in ['soz_soz', 'soz_non', 'non_soz', 'non_non']:
                pid_results['upper'][col]["mean"][z] = []
                pid_results['lower'][col]["mean"][z] = []
                pid_results['upper'][col]["all"][z] = []
                pid_results['lower'][col]["all"][z] = []

        # get full data
        measure = []
        for r in data:
            full = r[col].values
            measure.append(full)
        measure = np.array(measure)

        # results
        try:
            results['upper'][col]['all']['soz_soz'].extend(measure[:, U_soz_soz[:, 0], U_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['all']['soz_non'].extend(measure[:, U_soz_non[:, 0], U_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['all']['non_soz'].extend(measure[:, U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['all']['non_soz'].extend(measure[:, U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['all']['non_non'].extend(measure[:, U_non_non[:, 0], U_non_non[:, 1]].flatten())
        except:
            pass

        try:
            results['lower'][col]['all']['soz_soz'].extend(measure[:, L_soz_soz[:, 0], L_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['lower'][col]['all']['soz_non'].extend(measure[:, L_soz_non[:, 0], L_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results['lower'][col]['all']['non_soz'].extend(measure[:, L_non_soz[:, 0], L_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['lower'][col]['all']['non_non'].extend(measure[:, L_non_non[:, 0], L_non_non[:, 1]].flatten())
        except:
            pass

        # ilae results
        try:
            results_ilae[ilae]['upper'][col]['all']['soz_soz'].extend(measure[:, U_soz_soz[:, 0], U_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['upper'][col]['all']['soz_non'].extend(measure[:, U_soz_non[:, 0], U_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['upper'][col]['all']['non_soz'].extend(measure[:, U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['upper'][col]['all']['non_non'].extend(measure[:, U_non_non[:, 0], U_non_non[:, 1]].flatten())
        except:
            pass

        try:
            results_ilae[ilae]['lower'][col]['all']['soz_soz'].extend(measure[:, L_soz_soz[:, 0], L_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['lower'][col]['all']['soz_non'].extend(measure[:, L_soz_non[:, 0], L_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['lower'][col]['all']['non_soz'].extend(measure[:, L_non_soz[:, 0], L_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['lower'][col]['all']['non_non'].extend(measure[:, L_non_non[:, 0], L_non_non[:, 1]].flatten())
        except:
            pass

        # pid results
        try:
            pid_results['upper'][col]['all']['soz_soz'].extend(measure[:, U_soz_soz[:, 0], U_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['upper'][col]['all']['soz_non'].extend(measure[:, U_soz_non[:, 0], U_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['upper'][col]['all']['non_soz'].extend(measure[:, U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['upper'][col]['all']['non_non'].extend(measure[:, U_non_non[:, 0], U_non_non[:, 1]].flatten())
        except:
            pass

        try:
            pid_results['lower'][col]['all']['soz_soz'].extend(measure[:, L_soz_soz[:, 0], L_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['lower'][col]['all']['soz_non'].extend(measure[:, L_soz_non[:, 0], L_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['lower'][col]['all']['non_soz'].extend(measure[:, L_non_soz[:, 0], L_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['lower'][col]['all']['non_non'].extend(measure[:, L_non_non[:, 0], L_non_non[:, 1]].flatten())
        except:
            pass


        # mean results
        me = np.mean(measure, axis=0)

        # results
        try:
            results['upper'][col]['mean']['soz_soz'].extend(me[U_soz_soz[:, 0], U_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['mean']['soz_non'].extend(me[U_soz_non[:, 0], U_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['mean']['non_soz'].extend(me[U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['mean']['non_soz'].extend(me[U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['upper'][col]['mean']['non_non'].extend(me[U_non_non[:, 0], U_non_non[:, 1]].flatten())
        except:
            pass

        try:
            results['lower'][col]['mean']['soz_soz'].extend(me[L_soz_soz[:, 0], L_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['lower'][col]['mean']['soz_non'].extend(me[L_soz_non[:, 0], L_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results['lower'][col]['mean']['non_soz'].extend(me[L_non_soz[:, 0], L_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results['lower'][col]['mean']['non_non'].extend(me[L_non_non[:, 0], L_non_non[:, 1]].flatten())
        except:
            pass

        # ilae results
        try:
            results_ilae[ilae]['upper'][col]['mean']['soz_soz'].extend(me[U_soz_soz[:, 0], U_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['upper'][col]['mean']['soz_non'].extend(me[U_soz_non[:, 0], U_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['upper'][col]['mean']['non_soz'].extend(me[U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['upper'][col]['mean']['non_non'].extend(me[U_non_non[:, 0], U_non_non[:, 1]].flatten())
        except:
            pass

        try:
            results_ilae[ilae]['lower'][col]['mean']['soz_soz'].extend(me[L_soz_soz[:, 0], L_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['lower'][col]['mean']['soz_non'].extend(me[L_soz_non[:, 0], L_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['lower'][col]['mean']['non_soz'].extend(me[L_non_soz[:, 0], L_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            results_ilae[ilae]['lower'][col]['mean']['non_non'].extend(me[L_non_non[:, 0], L_non_non[:, 1]].flatten())
        except:
            pass

        # pid results
        try:
            pid_results['upper'][col]['mean']['soz_soz'].extend(me[U_soz_soz[:, 0], U_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['upper'][col]['mean']['soz_non'].extend(me[U_soz_non[:, 0], U_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['upper'][col]['mean']['non_soz'].extend(me[U_non_soz[:, 0], U_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['upper'][col]['mean']['non_non'].extend(me[U_non_non[:, 0], U_non_non[:, 1]].flatten())
        except:
            pass

        try:
            pid_results['lower'][col]['mean']['soz_soz'].extend(me[L_soz_soz[:, 0], L_soz_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['lower'][col]['mean']['soz_non'].extend(me[L_soz_non[:, 0], L_soz_non[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['lower'][col]['mean']['non_soz'].extend(me[L_non_soz[:, 0], L_non_soz[:, 1]].flatten())
        except:
            pass
        try:
            pid_results['lower'][col]['mean']['non_non'].extend(me[L_non_non[:, 0], L_non_non[:, 1]].flatten())
        except:
            pass
        break

    # save pickle of pid_results
    print("saving pid results")
    with open(os.path.join(output_path, f"{pid}_results.pkl"), "wb") as f:
        pickle.dump(pid_results, f)

# save pickle of results
with open(os.path.join(output_path, "results.pkl"), "wb") as f:
    pickle.dump(results, f)

# save pickle of results_ilae
with open(os.path.join(output_path, "results_ilae.pkl"), "wb") as f:
    pickle.dump(results_ilae, f)


