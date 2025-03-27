"""

-- 1. Patients table: basic metadata per patient
CREATE TABLE patients (
    patient_id INT PRIMARY KEY,            -- using provided pid as the unique patient id
    age_days_at_recording INTEGER,
    age_years_at_recording FLOAT,
    sex VARCHAR(10),
    ilae_score FLOAT,                     -- if available; otherwise use seizure_free flag
    seizure_free BOOLEAN,
    etiology_group VARCHAR(50),
    etiology_subgroup VARCHAR(100),         -- additional subgroup if applicable
    etiology_detailed VARCHAR(100)
);

-- 2. Recordings table: each recording session linked to a patient
CREATE TABLE recordings (
    recording_id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(patient_id),
    recording_type VARCHAR(50) NOT NULL,    -- e.g. baseline, cceps, story_listening
    UNIQUE (patient_id, recording_type)     -- ensure one recording type per patient
);

-- 3. Electrodes table: static contact metadata
CREATE TABLE electrodes (
    electrode_id SERIAL PRIMARY KEY,
    patient_id INT NOT NULL REFERENCES patients(patient_id),
    electrode_idx INT NOT NULL, 
    ascii_name TEXT NOT NULL, 
    x real NOT NULL,
    y real NOT NULL,
    z real NOT NULL,
    hemisphere VARCHAR(10), 
    noise_designation VARCHAR(10), 
    soz_label BOOLEAN, 
    UNIQUE (patient_id, electrode_idx)
);

-- 4. Electrode features table: time-varying computed features per electrode per recording
CREATE TABLE electrode_features (
    feature_id BIGSERIAL PRIMARY KEY,
    electrode_id INT REFERENCES electrodes(electrode_id),
    recording_id INT REFERENCES recordings(recording_id),
    time INTEGER NOT NULL,   
    f000000001 NUMERIC,
    f000000002 NUMERIC,
    f000000003 NUMERIC,
    UNIQUE(electrode_id,recording_id,time)
);

-- 5. Electrode atlas labels: stores multiple atlas assignments per electrode
CREATE TABLE electrode_atlas_labels (
    electrode_id INT REFERENCES electrodes(electrode_id),
    atlas_name VARCHAR(50),         -- e.g. 'aal', 'aal2', 'brainnetome', etc.
    region_label TEXT,              -- the region assigned in that atlas
    PRIMARY KEY (electrode_id, atlas_name)
);
"""

import psycopg2
import pandas as pd
import numpy as np
import os

from connection_complexity.data.raw_data.excel.ieeg_mapping import load_mapping
from psycopg2.extras import execute_values

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5433'
}

# Function to get database connection
def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

# Function to upsert patient data
def upsert_patients(df):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Convert data types to match database schema
    df['patient_id'] = df['patient_id'].astype(int)
    
    # Handle numeric conversions with proper rounding
    df['age_days_at_recording'] = pd.to_numeric(df['age_days_at_recording'], errors='coerce')
    df['age_days_at_recording'] = df['age_days_at_recording'].astype('int')
    
    df['age_years_at_recording'] = pd.to_numeric(df['age_years_at_recording'], errors='coerce')
    
    # df['ilae_score'] = pd.to_numeric(df['ilae_score'], errors='coerce')
    # df['ilae_score'] = df['ilae_score'].round().astype('int')
    
    df['seizure_free'] = df['seizure_free'].astype(bool)
    
    # Convert string columns to appropriate length
    df['sex'] = df['sex'].astype(str).str[:10]
    df['etiology_group'] = df['etiology_group'].astype(str).str[:50]
    df['etiology_subgroup'] = df['etiology_subgroup'].astype(str).str[:100]
    df['etiology_detailed'] = df['etiology_detailed'].astype(str).str[:100]
    
    # # Print data types for debugging
    # print("\nData types before insertion:")
    # print(df.dtypes)
    
    # # Print any null values
    # print("\nNull values in each column:")
    # print(df.isnull().sum())
    
    # # Print sample of problematic values
    # print("\nSample of values in age_days_at_recording:")
    # print(df['age_days_at_recording'].head())
    
    # Prepare the data for insertion
    patients_data = df[[
        'patient_id', 'age_days_at_recording', 'age_years_at_recording',
        'sex', 'ilae_score', 'seizure_free', 'etiology_group',
        'etiology_subgroup', 'etiology_detailed'
    ]].values.tolist()
    
    # SQL for upsert
    sql = """
    INSERT INTO patients (
        patient_id, age_days_at_recording, age_years_at_recording,
        sex, ilae_score, seizure_free, etiology_group,
        etiology_subgroup, etiology_detailed
    ) VALUES %s
    ON CONFLICT (patient_id) DO UPDATE SET
        age_days_at_recording = EXCLUDED.age_days_at_recording,
        age_years_at_recording = EXCLUDED.age_years_at_recording,
        sex = EXCLUDED.sex,
        ilae_score = EXCLUDED.ilae_score,
        seizure_free = EXCLUDED.seizure_free,
        etiology_group = EXCLUDED.etiology_group,
        etiology_subgroup = EXCLUDED.etiology_subgroup,
        etiology_detailed = EXCLUDED.etiology_detailed
    """
    
    try:
        execute_values(cur, sql, patients_data)
        conn.commit()
        print(f"Successfully upserted {len(patients_data)} patients")
    except Exception as e:
        print(f"Error upserting patients: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

# import the data

path = "/media/dan/Data/data/iEEG_mapping_SOZ_Outcome_full.xlsx"
soz_outcomes = load_mapping(path)

rows = []
for patient in soz_outcomes:
    rows.append({
        "patient_id": patient.id,
        "age_days_at_recording": patient.age_days_at_recording,
        "age_years_at_recording": patient.age_years_at_recording,
        "seizure_free": patient.seizureFree,
        "seizure_onset_zones": patient.seizureOnsetZones,
    })

df = pd.DataFrame(rows)
ilae_df = pd.read_excel("/media/dan/Data/data/patient_demographics.xlsx")

for index, row in ilae_df.iterrows():
    df.loc[df["patient_id"] == row["pt_id"], "ilae_score"] = row["ILAE_outcome"]

# Define the mapping (old to hfo_numbers (has etiology))
old = [1, 6, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 22, 26, 27, 28, 
            30, 31, 33, 34, 35, 36, 39, 40, 47, 51, 55, 62, 64, 69, 73, 
            75, 77, 78, 79, 81, 82, 83, 86, 87, 89, 90, 91, 92, 94, 95, 
            96, 98, 99, 100, 101, 102, 105, 106, 108, 109, 111, 112, 113]
new = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

mapping = dict(zip(old, new))

etiologies = pd.read_csv('/media/dan/Data/git/ubiquitous-spork/data_master1.csv')

for index, row in etiologies.iterrows():
    df.loc[df["patient_id"] == row["patient"], "etiology_group"] = row["etiol_group"]
    df.loc[df["patient_id"] == row["patient"], "etiology_subgroup"] = row["etiol_group"]
    df.loc[df["patient_id"] == row["patient"], "etiology_detailed"] = row["etiology"]

    df.loc[df["patient_id"] == row["patient"], "sex"] = row["sex"]
    df.loc[df["patient_id"] == row["patient"], "age_years_at_recording"] = row["age_yr"]
    df.loc[df["patient_id"] == row["patient"], "followup_years"] = row["fu_yrs"]
  

    ilae = df.loc[df["patient_id"] == row["patient"], "ilae_score"].values[0]
    if ilae is None or np.isnan(ilae):
        df.loc[df["patient_id"] == row["patient"], "ilae_score"] = row["ILAE"] # never hits

# remove patients with 'cannotbedetermined' in seizure_onset_zones list
df = df[~df['seizure_onset_zones'].apply(lambda x: any('cannotbedetermined' in s.lower() for s in x))]

# write to database
upsert_patients(df)

files_with_data = os.listdir("/media/dan/Data/data/electrodes_used")
# match the files with the patient_id in the df files are named like 001_channels.csv
baseline_data = []
for file in files_with_data:
    patient_id = int(file.split("_")[0])
    if patient_id in df["patient_id"].values:
        baseline_data.append(patient_id)

# update the recordings table to have all the patient_ids that have baseline data
def update_recordings(baseline_data):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # First, ensure the unique constraint exists
    try:
        cur.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = 'recordings_patient_id_recording_type_key'
                ) THEN
                    ALTER TABLE recordings 
                    ADD CONSTRAINT recordings_patient_id_recording_type_key 
                    UNIQUE (patient_id, recording_type);
                END IF;
            END $$;
        """)
        conn.commit()
    except Exception as e:
        print(f"Error adding constraint: {e}")
        conn.rollback()
        return
    
    # Prepare the data for insertion
    recordings_data = [(patient_id, 'baseline') for patient_id in baseline_data]
    
    # SQL for upsert
    sql = """
    INSERT INTO recordings (
        patient_id, recording_type
    ) VALUES %s
    ON CONFLICT (patient_id, recording_type) DO NOTHING
    """
    
    try:
        execute_values(cur, sql, recordings_data)
        conn.commit()
        print(f"Successfully inserted {len(recordings_data)} baseline recordings")
    except Exception as e:
        print(f"Error inserting recordings: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

# Update recordings table with baseline data
update_recordings(baseline_data)

print(df)






