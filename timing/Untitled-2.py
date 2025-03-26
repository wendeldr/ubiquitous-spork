
import numpy as np
from tqdm import tqdm
import os
from mne import make_fixed_length_epochs
import dill


import sys, os
import numpy as np
import dill
from pyspi.calculator import Calculator
import time
import warnings
import mne
import pandas as pd






file = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs/034_Baseline.EDF"

raw = mne.io.read_raw_edf(file, preload=True)

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
calc = Calculator(dataset=data[:,:1024], configfile='/home/dan/data/connectivity/pyspi_testing/fullconfig.yaml',normalise=True) # instantiate the calculator object
calc.compute()
SPI_res = calc.table
with open('timing.pkl', 'wb') as f:
    dill.dump(SPI_res, f)
end = time.time()
print(f"{end - start} seconds to compute")





