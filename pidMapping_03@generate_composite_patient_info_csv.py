import pickle
import os
import pandas as pd
import warnings
from natsort import natsorted

from connection_complexity.data.rereference_helpers import _identify_fromAtlas
from connection_complexity.data.raw_data.EDF.edf_helpers import read_edf

import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler = logging.FileHandler('logs/generate_composite_patient_info_csv.log', mode='w')
f_handler.setLevel(logging.INFO)
logger.addHandler(f_handler)

# Create the parser
parser = argparse.ArgumentParser(description="Process some files.")

# Add the arguments
parser.add_argument("PARSED_SOZmapping_file", type=str,  
                    help="Path to the output PARSED_SOZmapping_FILE")

parser.add_argument("matching_pids_file", type=str,  
                    help="Path to the MATCHING_PIDS_FILE")

parser.add_argument("raw_edfs", type=str,  
                    help="Path to the RAW_EDF_DATA_DIR")

parser.add_argument("xyz_files", type=str,
                    help="Path to the XYZ_DATA_DIR")

parser.add_argument("PREPEND_0_NUM", type=int,  
                    help="Path to the RAW_EDF_DATA_DIR")

parser.add_argument("FULL_COMPOSITE_PATIENT_INFO", type=str,  
                    help="Path to the output FULL_COMPOSITE_PATIENT_INFO (contains all electrodes)")

parser.add_argument("COMPOSITE_PATIENT_INFO", type=str,  
                    help="Path to the output COMPOSITE_PATIENT_INFO (contains only electrodes in both xyz and edf files)")

# Parse the arguments
args = parser.parse_args()

PARSED_SOZmapping_file = args.PARSED_SOZmapping_file
matching_pids_file = args.matching_pids_file
edf_path = args.raw_edfs
xyz_path = args.xyz_files
PREPEND_0_NUM = args.PREPEND_0_NUM
FULL_COMPOSITE_PATIENT_INFO = args.FULL_COMPOSITE_PATIENT_INFO
COMPOSITE_PATIENT_INFO = args.COMPOSITE_PATIENT_INFO

# PARSED_SOZmapping_file = "data/PARSED_SOZmapping_file.pkl"
# matching_pids_file = "data/sozmapping_edf_xyz_matching_ids.csv"
# edf_path = "/media/dan/Data/data/iEEG/raw_ieeg/real_patients/baseline_edfs"
# xyz_path = "/media/dan/Data/data/iEEG/raw_ieeg/real_patients/metadata/xyz"
# PREPEND_0_NUM = 3
# FULL_COMPOSITE_PATIENT_INFO = "data/FULL_composite_patient_info.csv"
# COMPOSITE_PATIENT_INFO = "data/composite_patient_info.csv"


with open(PARSED_SOZmapping_file, 'rb') as f:
    soz_mappings = pickle.load(f)
    soz_mappings = soz_mappings

with open(matching_pids_file, 'r') as f:
    ids = f.read().split(',')
    ids = [int(x) for x in ids]




patient_info = []
#edit

for pid in ids:
    pid = int(pid)
    # if pid != 36:
    #     continue
    logger.info('------------------------------------------')
    logger.info('Processing pid: {}'.format(pid))
    pid_edf_path = os.path.join(edf_path,f"{pid:0{PREPEND_0_NUM}}_Baseline.EDF")
    pid_xyz_path = os.path.join(xyz_path,f"{pid:0{PREPEND_0_NUM}}_LabelXYZAtlases.csv")

    # read the xyz data
    xyz_coordinates_df = pd.read_csv(pid_xyz_path)

    # read the raw edf data. Function reads only eeg channels.
    mne_raw = read_edf(pid_edf_path)


    # strip 'EEG' from the channel names just incase
    if xyz_coordinates_df['Electrode'].apply(lambda x: 'EEG ' in x).any():
        xyz_coordinates_df['Electrode'] = xyz_coordinates_df['Electrode'].apply(lambda x: x.replace('EEG ', ''))
    
    if xyz_coordinates_df['Electrode'].apply(lambda x: 'ECG' in x).any() and pid == 82:
        xyz_coordinates_df['Electrode'] = xyz_coordinates_df['Electrode'].apply(lambda x: x.replace('ECG', ''))
    
    # replace spaces with nothing
    xyz_coordinates_df['Electrode'] = xyz_coordinates_df['Electrode'].apply(lambda x: x.replace(" ", ""))

    # monkey patch(s) for patient 98 and 82
    # replace "W-A" with "W~A" because dash is delimiter for bipolar channels
    xyz_coordinates_df['Electrode'] = xyz_coordinates_df['Electrode'].apply(lambda x: x.replace("-", "~"))
    if pid == 82:
        mne_raw.rename_channels(lambda x: x.replace("ECGA'11", "A'11")) # one of the files is mislabeled as ECG

    if any([True if 'EEG ' in x else False for x in mne_raw.ch_names]):
        tmp = {x:x.replace('EEG ', '') for x in mne_raw.ch_names}
        mne_raw.rename_channels(tmp)
    
    soz =  soz_mappings[pid].seizureOnsetZones
    seizureFree = soz_mappings[pid].seizureFree
    soz = list(soz)
    age_days_at_recording = soz_mappings[pid].age_days_at_recording
    age_years_at_recording = soz_mappings[pid].age_years_at_recording

    # strip 'EEG' from the channel names. Should be done in the soz_mappings file.
    # but just in case it isn't, do it here.
    if any([True if 'EEG ' in x else False for x in soz]):
        soz = [x.replace('EEG ', '') for x in soz]

    # check if 'Cannot be determined' in soz
    if 'Cannot be determined' in soz:
        logger.warning(f'SOZ cannot be determined for pid {pid}. Skipping...')
        continue

    if len(soz) == 0:
        raise(ValueError('No SOZ found for pid: {}'.format(pid)))

    # check that channel names are the same across all files
    raw_channel_names = set(mne_raw.ch_names)

    raw_channel_names = list(raw_channel_names)
    # raw_channel_names.append('FAKE-1')
    # raw_channel_names.append('FAKE-2')
    raw_channel_names = set(raw_channel_names)

    xyz_channel_names = set(xyz_coordinates_df['Electrode'])

    # counts prime vs non-prime
    raw_prime_N = len([x for x in raw_channel_names if "'" in x])
    xyz_prime_N = len([x for x in xyz_channel_names if "'" in x])
    raw_nonprime_N = len([x for x in raw_channel_names if "'" not in x])
    xyz_nonprime_N = len([x for x in xyz_channel_names if "'" not in x])
    soz = set(soz)

    logger.debug(f'EDF channel names (N:{len(raw_channel_names)}) (prime N: {raw_prime_N}) (non-prime N: {raw_nonprime_N}):')
    logger.debug(natsorted(raw_channel_names))
    logger.debug(f'XYZ channel names (N:{len(xyz_channel_names)}) (prime N: {xyz_prime_N}) (non-prime N: {xyz_nonprime_N}):')
    logger.debug(natsorted(xyz_channel_names))

    soz_names_sans_all = set([x for x in soz if '-all' not in x.lower()])
    soz_names_withall = set([x for x in soz if '-all' in x.lower()])
    
    overlapping_soz = soz.intersection(raw_channel_names.intersection(xyz_channel_names))
    diff = soz_names_sans_all.difference(raw_channel_names.intersection(xyz_channel_names))
    if len(diff) > 0 :
        # there are soz channels that are not in the raw data (likely due to a second surgery and baseline was not recorded)
        logger.warning(f'SOZ markings exist that are not in raw data for pid {pid}. Likely a second surgery and baseline was not recorded.')
        if len(overlapping_soz) > 0:
            logger.warning(f'Removing non-overlapping SOZ channels from SOZ list: {natsorted(diff)}')
            soz_names_sans_all = overlapping_soz
        else:
            ValueError(f'No overlapping SOZ channels between raw and xyz files for pid {pid}.')


    extra_raw_channels = []
    extra_xyz_channels = []
    if raw_channel_names != xyz_channel_names:

        if len(set(raw_channel_names).difference(set(xyz_channel_names))) > 0:
            # raw has channels that xyz does not
            extra_raw_channels = list(set(raw_channel_names).difference(set(xyz_channel_names)))
            string = f'Raw has channels that xyz does not: {natsorted(extra_raw_channels)}'
            logger.warning(string)

        elif len(set(xyz_channel_names).difference(set(raw_channel_names))) > 0:
            # xyz has channels that raw does not
            extra_xyz_channels = list(set(xyz_channel_names).difference(set(raw_channel_names)))
            string = f'XYZ has channels that raw does not: {natsorted(extra_xyz_channels)}'
            logger.warning(string)

    # find "brodcasted" all channel union with shared channels btw raw and xyz
    intersection = raw_channel_names.intersection(xyz_channel_names)
    soz_names_withall_stripped = set([x.replace('-all', '') for x in soz_names_withall])
    
    # find the electrode names from individual electrodes
    electrode_names = set([''.join([i for i in s if not i.isdigit()]) for s in intersection])

    if len(soz_names_withall_stripped.difference(electrode_names)) > 0:
        raise ValueError(f'SOZ designated "all" not in data!\nSOZ: {natsorted(soz_names_withall_stripped)}\nElectrodes: {natsorted(electrode_names)}')

    soz_raw = []
    soz_xyz = []
    for e in soz_names_withall_stripped:
        for r in raw_channel_names:
            ch = ''.join([i for i in r if not i.isdigit()])
            if e == ch:
                soz_raw.append(r)

        for x in xyz_channel_names:
            ch = ''.join([i for i in x if not i.isdigit()])
            if e == ch:
                soz_xyz.append(x)
    soz_raw.extend(list(soz_names_sans_all))
    soz_xyz.extend(list(soz_names_sans_all))

    soz_raw = set(soz_raw)
    soz_xyz = set(soz_xyz)

    soz_final = set(soz_raw).union(soz_xyz)

    # determine the white matter / outside brain electrodes. also strip the 'EEG ' if present from the channel names to match.
    whiteMatter = _identify_fromAtlas(pid_xyz_path, outside_brain=True, white_matter=True)
    if any([True if 'EEG ' in x else False for x in whiteMatter]):
        whiteMatter = [x.replace('EEG ', '') for x in whiteMatter]

    # create df for patient, start with xyz coordinates since we want a lot of the columns
    pt_df = xyz_coordinates_df.copy()
    pt_df.columns = [x.lower() for x in pt_df.columns.values]

    new_df = pd.DataFrame(columns=pt_df.columns.values)
    new_ec_contacts = []
    # add any electrodes that dont overlap in raw and xyz files
    tmp = list(xyz_channel_names.symmetric_difference(raw_channel_names))
    new_ec_contacts += tmp

    # add any electrodes that dont overlap in updated df and whitematter.  Note using difference now.
    tmp = list(set(whiteMatter).difference(set(pt_df.electrode)))
    new_ec_contacts += tmp

    # add any electrodes that dont overlap in updated df and whitematter.  Note using difference now.
    tmp = set(soz_names_sans_all).difference(set(pt_df.electrode))
    new_ec_contacts += tmp

    new_ec_contacts = list(set(new_ec_contacts))
    new_df.electrode = new_ec_contacts
    pt_df = pd.concat([pt_df, new_df], ignore_index=True)
    pt_df.drop_duplicates(subset=['electrode'], inplace=True)

    pt_df['pid'] = pid
    pt_df['age_days_at_recording'] = age_days_at_recording
    pt_df['age_years_at_recording'] = age_years_at_recording
    pt_df['seizureFree'] = seizureFree
    pt_df['white_matter'] = False
    pt_df['soz'] = False
    # pt_df['in_soz_file'] = False
    pt_df['in_xyz_file'] = False
    pt_df['in_edf_file'] = False
    # pt_df['in_whitematter_file'] = False


    pt_df.loc[pt_df['electrode'].isin(whiteMatter), 'white_matter'] = True
    pt_df.loc[pt_df['electrode'].isin(soz_final), 'soz'] = True
    pt_df.loc[pt_df['electrode'].isin(xyz_channel_names), 'in_xyz_file'] = True
    pt_df.loc[pt_df['electrode'].isin(raw_channel_names), 'in_edf_file'] = True
    # pt_df.loc[pt_df['electrode'].isin(whiteMatter), 'in_whitematter_file'] = True
    # pt_df.loc[pt_df['electrode'].isin(soz), 'in_soz_file'] = True

    logger.debug(f'Original SOZ channels (N={len(soz)} (N_sans: {len(soz_names_sans_all)}) (N_with: {len(soz_names_withall)})):')
    logger.debug(f'sans_all: {natsorted(soz_names_sans_all)}')
    logger.debug(f'with_all: {natsorted(soz_names_withall)}')
    logger.debug(f"Matched soz channels (N={len(soz_final)}):")
    logger.debug(natsorted(soz_final))

    unmatched_soz = soz_final.union(diff).difference(raw_channel_names.intersection(xyz_channel_names))
    if len(unmatched_soz) > 0:
        logger.error(f"*** UNMATCHED SOZ CHANNELS *** (N={len(unmatched_soz)}): {natsorted(unmatched_soz)}")


    patient_info.append(pt_df)
df = pd.concat(patient_info)

df.to_csv(FULL_COMPOSITE_PATIENT_INFO, index=False)
df[(df['in_xyz_file'] == True) & (df['in_edf_file'] == True)].to_csv(COMPOSITE_PATIENT_INFO, index=False)