import mne
import h5py


pid = "001"

path = f"/media/dan/Data/data/baseline_patients/baseline_edfs/{pid}_Baseline.EDF"
raw = mne.io.read_raw_edf(path, preload=True)
with h5py.File(f"/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s/{pid}_20250414.h5", 'r') as f:
    soz = f['metadata/patient_info/soz'][()]
    ilae = f['metadata/patient_info/ilae'][()]
    names = f['metadata/patient_info/electrode_data/electrode'][()]
    x = f['metadata/patient_info/electrode_data/x'][()]
    y = f['metadata/patient_info/electrode_data/y'][()]
    z = f['metadata/patient_info/electrode_data/z'][()]
    # byte to string
    names = [name.decode('utf-8') for name in names]
wteeg = [f"EEG {x}" for x in names]
other = [raw.ch_names[i] for i in range(len(raw.ch_names)) if raw.ch_names[i] not in wteeg]
raw = raw.drop_channels(other)

test = raw.set_eeg_reference(ref_channels='average', projection=False)