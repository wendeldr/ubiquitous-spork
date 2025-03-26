import numpy as np
from tqdm import tqdm
import os
from mne import make_fixed_length_epochs


import sys, os
import numpy as np
import dill
from pyspi.calculator import Calculator
import time
from natsort import natsorted
import warnings
import mne
import pandas as pd

def norm(y):
    # normalize signal
    q25, q50, q75 = np.percentile(y, [25, 50, 75])
    whisker_lim = 1.5 * (q75 - q25)
    high_lim = q75 + whisker_lim
    low_lim = q25 - whisker_lim
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(y)
    # ax[0].axhline(high_lim, color="r")
    # ax[0].axhline(low_lim, color="r")
    # ax[0].axhline(q50, color="g")
    m = np.mean(y[(y <= high_lim) & (y >= low_lim)])
    s = np.std(y[(y <= high_lim) & (y >= low_lim)])
    # ax[0].axhline(m, color="b")
    z = (y - m) / s
    # ax[1].plot(z)
    # ax[1].axhline(5, color="r")
    # ax[1].axhline(-5, color="r")
    # plt.show()

    return z

def read_edf(path, drop_non_eeg=True, normalize=False, drop_EEG_Prefix=True, preload=False):
    path = str(path)
    non_eeg = [
        "DC1",
        "Baseline",
        "ECG",
        "EKG",
        "EMG",
        "MD/Pic",
        "MD",
        "Pic",
        "Mic",
        "Mic-0",
        "Mic-1",
        "Mic-2",
        "Mic-3",
        "Mic-4",
        "Mic-5",
        "Motor",
        "Music",
        "Noise",
        "Picture",
        "Story",
        "ECG ECG",
        "EEG ECG",
        "Pt Mic",
        "MD Mic",
        "PT Mic",
        "Hand Motor",
        "ECG EKG",
        "EKG ECG",
        "Hand",
        "EDF Annotations",
    ]

    if normalize:
        # data has to be preloaded to normalize
        preload = True


    if drop_non_eeg:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mne_raw = mne.io.read_raw_edf(path, preload=preload, verbose=False, exclude=non_eeg)
        sEEG_picks = mne.pick_types(mne_raw.info, eeg=True, exclude=[])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mne_raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
        sEEG_picks = np.arange(len(mne_raw.ch_names))

    if drop_EEG_Prefix:
        mne_raw.rename_channels(lambda x: x.replace("EEG ", ""))
        # replace spaces with nothing
        mne_raw.rename_channels(lambda x: x.replace(" ", ""))

        # some electrodes have spaces, strip whitespace
        mne_raw.rename_channels(lambda x: x.replace(' ','').strip())

    # monkey patch(s) for patient 98 and 82
    # replace "W-A" with "W~A" because dash is delimiter for bipolar channels
    mne_raw.rename_channels(lambda x: x.replace("-", "~"))
    if '082_' in path:
        mne_raw.rename_channels(lambda x: x.replace("ECGA'11", "A'11")) # one of the files is mislabeled as ECG



    chNames = np.array(mne_raw.ch_names)[sEEG_picks]
    primes = [x for x in chNames if "'" in x]
    non_primes = [x for x in chNames if "'" not in x]
    prime_n = len(primes)
    non_prime_n = len(non_primes)
    primes = list(natsorted(primes))
    non_primes = list(natsorted(non_primes))

    if len(non_primes) > 0:
        non_primes.extend(primes)
        chNames = non_primes
    else:
        chNames = primes
    if len(chNames) != prime_n+non_prime_n:
        raise ValueError("Something went wrong with the channel names")

    z = mne_raw.pick(chNames).reorder_channels(chNames)
        
    if normalize:
        return z.apply_function(norm)
    else:
        return z


mapping_path = "/media/dan/Big/manuiscript_0001_hfo_rates/data/FULL_composite_patient_info.csv"  
ilae_path = "/media/dan/Big/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"
bad_channels_path = "/media/dan/Big/manuiscript_0001_hfo_rates/data/bad_ch_review.xlsx"
edf_path = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs"
output_path = "/media/dan/Data/git/network_miner/connectivity/output"

pid_source_path = "/media/dan/Big/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"


dur_msec = 500
dur_sec = dur_msec / 1000
overlap_msec = 1/2048
overlap_sec = overlap_msec / 1000


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
for pid in mappings["pid"].unique():
    if pid in ilae["patient"].values:
        ilae_numbers[pid] = ilae[ilae["patient"] == pid]["ilae"].values[0]
    else:
        if mappings[mappings["pid"] == pid]["seizureFree"].values[0] == True:
            ilae_numbers[pid] = -1
        else:
            ilae_numbers[pid] = 100

# now we have a dictionary of ilae numbers for each patient. Fill in the mappings dataframe with these numbers which has multiple rows for each patient
ilae_list = []
for pid in mappings["pid"]:
    ilae_list.append(ilae_numbers[pid])
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


# find all cells with "" or "nan" and mark them as bad_channels (assuming outside of brain)
mappings.loc[(mappings["miccai"].isna() & mappings["aal"].isna()), "bad_channel"] = 1

a=1

# raw = read_edf(file, preload=True)

# # filter
# sfreq = raw.info["sfreq"]
# nyq_freq = raw.info["sfreq"] // 2 - 1
# line_freq = 60
# l_freq = 0.5
# h_freq = min(nyq_freq, 300)
# raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
# freqs = np.arange(line_freq, max(h_freq, nyq_freq), line_freq)
# raw = raw.notch_filter(freqs=freqs, method="fir")

# # average reference
# raw = raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

# data = raw.get_data()
# maxidx = data.shape[1]
# digit_count = len(str(abs(maxidx))) 

# # non overlapping epochs
# dur_sec = .5
# overlap_sec = 0
# epochs = make_fixed_length_epochs(raw, duration=dur_sec,overlap=overlap_sec, preload=True)
# epochs = epochs.get_data()


# for i, e in enumerate(tqdm(epochs)): # start 1 minute in only use first 5 seconds
#     start_idx = int(sfreq * dur_sec) * i
#     np.save(os.path.join(output_path, f"034_epoch_{start_idx:0{digit_count}}.npy"), e)

# overlapping epochs with overlap of 1 sample
# start = int(sfreq * 60) # start 1 minute in
# for i in tqdm(range(start, start+int(sfreq * 1))): # start 1 minute in only use first 5 seconds
#     np.save(os.path.join(output_path, f"034_epoch_{i}.npy"), data[:,i:i+int(0.5*sfreq)])







file = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs/034_Baseline.EDF"

raw = read_edf(file, preload=True)

# filter
sfreq = raw.info["sfreq"]
nyq_freq = raw.info["sfreq"] // 2 - 1
line_freq = 60
l_freq = 0.5
h_freq = min(nyq_freq, 300)
raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
freqs = np.arange(line_freq, max(h_freq, nyq_freq), line_freq)
raw = raw.notch_filter(freqs=freqs, method="fir")

# average reference
raw = raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

data = raw.get_data()

start = time.time()
calc = Calculator(dataset=data[:,:1024], configfile='/home/dan/data/connectivity/pyspi_testing/sixrun/sixmeasures.yaml',normalise=True) # instantiate the calculator object
calc.compute()

name = f"034_firstepoch.pkl"
with open(name, 'wb') as f:
    dill.dump(calc, f)

end = time.time()
print(f"{end - start} seconds to compute")
